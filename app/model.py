from __future__ import annotations

import hashlib
from dataclasses import dataclass

CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)


@dataclass(frozen=True)
class Prediction:
    scores: dict[str, float]


class SyntheticModelAdapter:
    """Deterministic adapter used for contract testing without model weights."""

    name = "synthetic-deterministic-v1"

    def predict(self, image_bytes: bytes) -> Prediction:
        digest = hashlib.sha256(image_bytes).digest()
        raw = [digest[index] + 1 for index in range(len(CLASSES))]
        total = float(sum(raw))
        scores = {label: round(value / total, 6) for label, value in zip(CLASSES, raw)}
        return Prediction(scores=scores)
