from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json
import random

from PIL import Image
from torch.utils.data import Dataset

from src.data.schemas import BarkNetRecord


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_label(label: str) -> str:
    """
    Czyści nazwę klasy:
    - usuwa spacje z początku i końca
    - zamienia litery na małe
    - usuwa nadmiarowe spacje w środku
    """
    return " ".join(label.strip().lower().split())


def load_barknet_records(root: str | Path) -> list[BarkNetRecord]:
    """
    Czyta surowe foldery klas, np.:

    root/
      CHR/
      EPB/
      EPN/

    i zwraca listę BarkNetRecord.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}. "
            "Expected a directory containing one subdirectory per class."
        )
    if not root.is_dir():
        raise NotADirectoryError(
            f"Dataset root is not a directory: {root}."
        )

    records: list[BarkNetRecord] = []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        species = class_dir.name

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue

            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            stem = img_path.stem
            parts = stem.split("_")

            if len(parts) < 2:
                print(f"[WARN] Unexpected filename format: {img_path.name}")
                continue

            tree_id = parts[0]

            records.append(
                BarkNetRecord(
                    path=str(img_path),
                    species=species,
                    tree_id=tree_id,
                )
            )

    return records


def filter_records(
    records: list[BarkNetRecord],
    allowed_classes: list[str] | None = None,
    max_images_per_class: int | None = None,
    seed: int = 42,
) -> list[BarkNetRecord]:
    """
    Filtruje rekordy:
    - zostawia tylko wybrane klasy
    - ogranicza liczbę zdjęć na klasę
    - losuje w obrębie klasy przed obcięciem
    """
    normalized_allowed = None
    if allowed_classes is not None:
        normalized_allowed = {normalize_label(cls) for cls in allowed_classes}

    grouped = defaultdict(list)

    for record in records:
        species = normalize_label(record.species)

        if normalized_allowed is not None and species not in normalized_allowed:
            continue

        cleaned_record = BarkNetRecord(
            path=record.path,
            species=species,
            tree_id=record.tree_id,
        )
        grouped[species].append(cleaned_record)

    rng = random.Random(seed)
    filtered: list[BarkNetRecord] = []

    for _, species_records in grouped.items():
        species_records = species_records[:]
        rng.shuffle(species_records)

        if max_images_per_class is not None:
            species_records = species_records[:max_images_per_class]

        filtered.extend(species_records)

    return filtered


def build_class_to_idx(records: list[BarkNetRecord]) -> dict[str, int]:
    """
    Buduje mapowanie:
    class_name -> class_idx
    """
    classes = sorted({normalize_label(record.species) for record in records})
    return {class_name: idx for idx, class_name in enumerate(classes)}


def split_records(
    records: list[BarkNetRecord],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 44,
) -> tuple[list[BarkNetRecord], list[BarkNetRecord], list[BarkNetRecord]]:

    grouped = defaultdict(list)
    for record in records:
        grouped[normalize_label(record.species)].append(record)

    rng = random.Random(seed)

    train_records: list[BarkNetRecord] = []
    val_records: list[BarkNetRecord] = []
    test_records: list[BarkNetRecord] = []

    for species, species_records in grouped.items():
        trees = defaultdict(list)
        for r in species_records:
            trees[r.tree_id].append(r)

        tree_ids = sorted(trees.keys())
        rng.shuffle(tree_ids)

        total = len(tree_ids)
        if total == 1:
            train_trees = tree_ids
            val_trees = []
            test_trees = []
        else:
            test_count = int(round(total * test_size))
            val_count = int(round(total * val_size))

            if total >= 3 and test_size > 0:
                test_count = max(1, test_count)
            if total >= 4 and val_size > 0:
                val_count = max(1, val_count)

            test_count = min(test_count, total - 1)
            val_count = min(val_count, total - test_count - 1)

            if test_count < 0:
                test_count = 0
            if val_count < 0:
                val_count = 0

            train_trees = tree_ids[test_count + val_count:]
            if not train_trees:
                if val_count >= test_count and val_count > 0:
                    val_count -= 1
                elif test_count > 0:
                    test_count -= 1
                train_trees = tree_ids[test_count + val_count:]

            test_trees = tree_ids[:test_count]
            val_trees = tree_ids[test_count:test_count + val_count]

        for tid in train_trees:
            train_records.extend(trees[tid])
        for tid in val_trees:
            val_records.extend(trees[tid])
        for tid in test_trees:
            test_records.extend(trees[tid])

    return train_records, val_records, test_records


def count_per_class(records: list[BarkNetRecord]) -> dict[str, int]:

    counts = defaultdict(int)
    for record in records:
        counts[normalize_label(record.species)] += 1
    return dict(sorted(counts.items()))


def encode_samples(
    records: list[BarkNetRecord],
    class_to_idx: dict[str, int],
) -> list[tuple[str, int]]:

    samples = []
    for record in records:
        species = normalize_label(record.species)
        class_idx = class_to_idx[species]
        samples.append((record.path, class_idx))
    return samples


def save_json(data: dict, output_path: str | Path) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class BarkNetDataset(Dataset):


    def __init__(
        self,
        samples: list[tuple[str, int]],
        class_to_idx: dict[str, int],
        transform=None,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.classes = [cls for cls, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        with Image.open(img_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def prepare_barknet_dataset(
    root: str | Path,
    train_transform=None,
    val_transform=None,
    allowed_classes: list[str] | None = None,
    max_images_per_class: int | None = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    artifacts_dir: str | Path | None = None,
):

    records = load_barknet_records(root)
    if not records:
        raise ValueError(
            f"No images found under {root}. "
            "Expected a layout like <root>/<class_name>/<image>.jpg."
        )

    filtered_records = filter_records(
        records=records,
        allowed_classes=allowed_classes,
        max_images_per_class=max_images_per_class,
        seed=seed,
    )

    class_to_idx = build_class_to_idx(filtered_records)
    classes = [cls for cls, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    train_records, val_records, test_records = split_records(
        filtered_records,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    train_samples = encode_samples(train_records, class_to_idx)
    val_samples = encode_samples(val_records, class_to_idx)
    test_samples = encode_samples(test_records, class_to_idx)

    train_ds = BarkNetDataset(
        samples=train_samples,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )
    val_ds = BarkNetDataset(
        samples=val_samples,
        class_to_idx=class_to_idx,
        transform=val_transform,
    )
    test_ds = BarkNetDataset(
        samples=test_samples,
        class_to_idx=class_to_idx,
        transform=val_transform,
    )

    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)

        save_json(class_to_idx, artifacts_dir / "class_to_idx.json")
        save_json(
            {
                "num_records_total": len(filtered_records),
                "num_train": len(train_samples),
                "num_val": len(val_samples),
                "num_test": len(test_samples),
                "classes": classes,
                "counts_total": count_per_class(filtered_records),
                "counts_train": count_per_class(train_records),
                "counts_val": count_per_class(val_records),
                "counts_test": count_per_class(test_records),
            },
            artifacts_dir / "dataset_info.json",
        )

    return train_ds, val_ds, test_ds, classes
