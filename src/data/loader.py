from pathlib import Path
from collections import Counter

from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.transforms import get_transforms
from src.data.prep_data import (
    BarkNetDataset,
    build_class_to_idx,
    count_per_class,
    encode_samples,
    load_barknet_records,
    prepare_barknet_dataset,
    save_json,
)
def normalize_records_labels(records):
    normalized = []

    for record in records:
        if isinstance(record, dict):
            record = record.copy()
            record["species"] = str(record["species"]).strip().lower()
            normalized.append(record)
        else:
            record.species = str(record.species).strip().lower()
            normalized.append(record)

    return normalized

def _prepare_presplit_dataset(
    root: Path,
    artifacts_dir: Path,
    train_transform=None,
    val_transform=None,
):
    train_root = root / "train"
    val_root = root / "validate"
    test_root = root / "test"

    train_records = normalize_records_labels(load_barknet_records(train_root))
    val_records = normalize_records_labels(load_barknet_records(val_root))
    test_records = normalize_records_labels(load_barknet_records(test_root))

    if len(train_records) == 0:
        raise ValueError(f"No train records found in {train_root}")

    if len(val_records) == 0:
        raise ValueError(f"No validation records found in {val_root}")

    if len(test_records) == 0:
        raise ValueError(f"No test records found in {test_root}")

    all_records = train_records + val_records + test_records



    if not all_records:
        raise ValueError(
            f"No images found in pre-split dataset under {root}. "
            "Expected train/validate/test folders with class subdirectories."
        )

    class_to_idx = build_class_to_idx(all_records)
    print("[DATA] class_to_idx:", class_to_idx)
    print("[DATA] train counts:", count_per_class(train_records))
    print("[DATA] val counts:", count_per_class(val_records))
    print("[DATA] test counts:", count_per_class(test_records))


    classes = [cls for cls, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

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

    save_json(class_to_idx, artifacts_dir / "class_to_idx.json")
    save_json(
        {
            "num_records_total": len(all_records),
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "num_test": len(test_samples),
            "classes": classes,
            "counts_train": count_per_class(train_records),
            "counts_val": count_per_class(val_records),
            "counts_test": count_per_class(test_records),
            "source_layout": "presplit",
        },
        artifacts_dir / "dataset_info.json",
    )

    return train_ds, val_ds, test_ds, classes

def make_weighted_sampler(dataset):
    """
    Creates a sampler that balances classes by sampling rare classes more often.
    Works with datasets that have .samples = [(path, class_idx), ...].
    """
    targets = [y for _, y in dataset.samples]
    class_counts = Counter(targets)

    sample_weights = [1.0 / class_counts[y] for y in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("[SAMPLER] WeightedRandomSampler enabled")
    print("[SAMPLER] class counts:", dict(sorted(class_counts.items())))

    return sampler

def get_loader(
    root: Path,
    artifacts_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    img_size: int = 224,
    max_images_per_class: int | None = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    model_name: str = "resnet50",
    use_weighted_sampler: bool = False,
):

    root = Path(root)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        raise FileNotFoundError(
            f"Configured dataset root does not exist: {root}. "
            "Check cfg.data.root in your Hydra config."
        )

    train_tfms, val_tfms = get_transforms(model_name=model_name, img_size=img_size)

    if (root / "train").is_dir() and (root / "validate").is_dir() and (root / "test").is_dir():
        train_ds, val_ds, test_ds, classes = _prepare_presplit_dataset(
            root=root,
            artifacts_dir=artifacts_dir,
            train_transform=train_tfms,
            val_transform=val_tfms,
        )
    else:
        train_ds, val_ds, test_ds, classes = prepare_barknet_dataset(
            root=root,
            train_transform=train_tfms,
            val_transform=val_tfms,
            allowed_classes=None,
            max_images_per_class=max_images_per_class,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            artifacts_dir=artifacts_dir,
        )
    train_sampler = make_weighted_sampler(train_ds) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
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
    
