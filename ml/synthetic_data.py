from __future__ import annotations

import math
import random

import torch
from torch.utils.data import Dataset

from app.model import CLASSES


class SyntheticDefectDataset(Dataset[tuple[torch.Tensor, int]]):
    """Procedural textures for reproducible pipeline testing.

    These images are not substitutes for real inspection data. Each class uses
    a distinct geometric pattern so that training, evaluation and calibration
    code can be exercised end to end without distributing third-party data.
    """

    def __init__(self, samples_per_class: int = 64, image_size: int = 64, seed: int = 17):
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return len(CLASSES) * self.samples_per_class

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = index // self.samples_per_class
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand((1, self.image_size, self.image_size), generator=generator) * 0.12
        image += 0.42
        self._draw_pattern(image[0], label, random.Random(self.seed + index))
        image += torch.randn(image.shape, generator=generator) * 0.025
        return image.clamp(0, 1), label

    def _draw_pattern(self, canvas: torch.Tensor, label: int, rng: random.Random) -> None:
        size = self.image_size
        if label == 0:  # branching cracks
            for x in range(5, size - 5):
                y = int(size / 2 + 8 * math.sin(x / 7) + rng.randint(-1, 1))
                canvas[max(0, y - 1) : min(size, y + 2), x] = 0.92
        elif label == 1:  # inclusions
            for _ in range(12):
                y, x = rng.randrange(size), rng.randrange(size)
                canvas[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = 0.08
        elif label == 2:  # patches
            for _ in range(4):
                y, x = rng.randrange(size - 14), rng.randrange(size - 14)
                canvas[y : y + 14, x : x + 14] = 0.72
        elif label == 3:  # pitted surface
            for _ in range(28):
                y, x = rng.randrange(size), rng.randrange(size)
                canvas[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 0.03
        elif label == 4:  # rolled-in scale
            for y in range(3, size, 8):
                canvas[y : y + 2, :] = 0.82
        else:  # scratches
            for offset in (-14, -4, 7, 17):
                for x in range(size):
                    y = x + offset
                    if 0 <= y < size:
                        canvas[y, x] = 0.96
