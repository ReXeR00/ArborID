from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class BarkNetDataset(Dataset):

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        class_to_idx: dict[str, int],
        transform=None,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(self, idx: int):

        img_path, target = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def build_idx_to_class(class_to_idx: dict[str, int]) -> dict[int, str]:

    return {idx: cls_name for cls_name, idx in class_to_idx.items()}