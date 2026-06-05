from __future__ import annotations

from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from src.data.prep_data import (
    count_dataset_samples_by_class,
    has_any_images,
    prepare_split_dataset,
    save_json,
)
from src.data.transforms import get_transforms


def make_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    """
    Sample rare classes more often to reduce class imbalance during training.
    """
    targets = [target for _, target in dataset.samples]
    class_counts = Counter(targets)
    sample_weights = [1.0 / class_counts[target] for target in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("[SAMPLER] WeightedRandomSampler enabled")
    print("[SAMPLER] class counts:", dict(sorted(class_counts.items())))
    return sampler


def _save_dataset_info(
    artifacts_dir: Path,
    train_ds: ImageFolder,
    val_ds: ImageFolder,
    test_ds: ImageFolder,
    *,
    raw_dir: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    split_counts: dict[str, dict[str, int]],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    class_to_idx = dict(train_ds.class_to_idx)
    classes = [class_name for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    save_json(class_to_idx, artifacts_dir / "class_to_idx.json")
    save_json(
        {
            "raw_dir": str(raw_dir),
            "train_dir": str(train_dir),
            "val_dir": str(val_dir),
            "test_dir": str(test_dir),
            "num_records_total": len(train_ds.samples) + len(val_ds.samples) + len(test_ds.samples),
            "num_train": len(train_ds.samples),
            "num_val": len(val_ds.samples),
            "num_test": len(test_ds.samples),
            "classes": classes,
            "counts_train": count_dataset_samples_by_class(train_ds.samples, classes),
            "counts_val": count_dataset_samples_by_class(val_ds.samples, classes),
            "counts_test": count_dataset_samples_by_class(test_ds.samples, classes),
            "prepared_counts": split_counts,
            "source_layout": "rawdata_to_presplit_imagefolder",
        },
        artifacts_dir / "dataset_info.json",
    )


def _validate_loaded_datasets(train_ds: ImageFolder, val_ds: ImageFolder, test_ds: ImageFolder) -> None:
    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError("class_to_idx differs across train/validation/test datasets.")

    train_paths = {path for path, _ in train_ds.samples}
    val_paths = {path for path, _ in val_ds.samples}
    test_paths = {path for path, _ in test_ds.samples}

    if train_paths & val_paths:
        raise ValueError("train and validation datasets share image paths.")
    if train_paths & test_paths:
        raise ValueError("train and test datasets share image paths.")
    if val_paths & test_paths:
        raise ValueError("validation and test datasets share image paths.")


def get_loader(
    *,
    raw_dir: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    artifacts_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    img_size: int = 224,
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    split_if_missing: bool = True,
    copy_files: bool = True,
    force_resplit: bool = False,
    max_images_per_class: int | None = None,
    seed: int = 42,
    model_name: str = "resnet50",
    use_weighted_sampler: bool = False,
):
    raw_dir = Path(raw_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    test_dir = Path(test_dir)
    artifacts_dir = Path(artifacts_dir)

    train_tfms, val_tfms = get_transforms(model_name=model_name, img_size=img_size)

    split_dirs_ready = train_dir.exists() and val_dir.exists() and test_dir.exists()
    split_has_images = split_dirs_ready and all(
        has_any_images(split_dir) for split_dir in (train_dir, val_dir, test_dir)
    )

    if force_resplit or not split_has_images:
        if not split_if_missing and not force_resplit:
            raise FileNotFoundError(
                "Prepared dataset split is missing. "
                "Enable split_if_missing or create train/validation/test first."
            )

        split_counts = prepare_split_dataset(
            raw_dir=raw_dir,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            copy_files=copy_files,
            force_resplit=force_resplit,
            max_images_per_class=max_images_per_class,
        )
    else:
        print("[DATA] Using existing prepared split from train/validation/test directories.")
        split_counts = {}

    train_ds = ImageFolder(root=str(train_dir), transform=train_tfms)
    val_ds = ImageFolder(root=str(val_dir), transform=val_tfms)
    test_ds = ImageFolder(root=str(test_dir), transform=val_tfms)

    _validate_loaded_datasets(train_ds, val_ds, test_ds)

    classes = train_ds.classes
    print(f"[DATA] Loaded ImageFolder datasets with {len(classes)} classes.")

    _save_dataset_info(
        artifacts_dir=artifacts_dir,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        raw_dir=raw_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        split_counts=split_counts,
    )

    train_sampler = make_weighted_sampler(train_ds) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, classes
