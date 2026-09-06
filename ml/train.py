from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from app.model import CLASSES
from ml.network import DefectCNN
from ml.synthetic_data import SyntheticDefectDataset


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor) -> float:
    error = torch.tensor(0.0)
    boundaries = torch.linspace(0, 1, 11)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += mask.float().mean() * (
                correct[mask].float().mean() - confidence[mask].mean()
            ).abs()
    return round(error.item(), 6)


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, object]:
    model.eval()
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    confidences, correctness = [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images).softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)
            for truth, prediction in zip(labels, predictions):
                confusion[truth, prediction] += 1
            confidences.append(confidence)
            correctness.append(predictions.eq(labels))

    per_class = {}
    for index, label in enumerate(CLASSES):
        true_positive = confusion[index, index].item()
        predicted = confusion[:, index].sum().item()
        actual = confusion[index, :].sum().item()
        per_class[label] = {
            "precision": round(true_positive / predicted, 6) if predicted else 0.0,
            "recall": round(true_positive / actual, 6) if actual else 0.0,
            "support": actual,
        }
    confidence = torch.cat(confidences)
    correct = torch.cat(correctness)
    return {
        "accuracy": round(correct.float().mean().item(), 6),
        "expected_calibration_error": expected_calibration_error(confidence, correct),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def train(epochs: int, samples_per_class: int, output_dir: Path, seed: int) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = SyntheticDefectDataset(samples_per_class=samples_per_class, seed=seed)
    train_size = int(len(dataset) * 0.8)
    train_set, test_set = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)
    model = DefectCNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_function = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimiser.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimiser.step()

    metrics = evaluate(model, test_loader)
    report = {
        "dataset": "procedurally generated synthetic textures",
        "production_evidence": False,
        "seed": seed,
        "epochs": epochs,
        "samples_per_class": samples_per_class,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        **metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": list(CLASSES), "report": report},
        output_dir / "synthetic-defect-cnn.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train(args.epochs, args.samples_per_class, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
