import torch

from app.model import CLASSES
from ml.network import DefectCNN
from ml.synthetic_data import SyntheticDefectDataset


def test_dataset_is_reproducible() -> None:
    first = SyntheticDefectDataset(samples_per_class=2, seed=9)
    second = SyntheticDefectDataset(samples_per_class=2, seed=9)
    image_a, label_a = first[3]
    image_b, label_b = second[3]
    assert label_a == label_b
    assert torch.equal(image_a, image_b)


def test_model_returns_six_logits() -> None:
    model = DefectCNN()
    output = model(torch.zeros((2, 1, 64, 64)))
    assert output.shape == (2, len(CLASSES))
