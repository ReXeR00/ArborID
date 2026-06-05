from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitPaths:
    raw_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path


def save_json(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def resolve_split_paths(
    raw_dir: str | Path,
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
) -> SplitPaths:
    return SplitPaths(
        raw_dir=Path(raw_dir),
        train_dir=Path(train_dir),
        val_dir=Path(val_dir),
        test_dir=Path(test_dir),
    )


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.10f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    for name, value in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}.")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_class_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())


def has_any_images(root: str | Path) -> bool:
    root = Path(root)
    if not root.exists():
        return False
    return any(is_image_file(path) for path in root.rglob("*"))


def _clear_existing_split_dir(split_dir: Path) -> None:
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)


def _relative_output_path(raw_class_dir: Path, source_path: Path, destination_class_dir: Path) -> Path:
    relative_path = source_path.relative_to(raw_class_dir)
    return destination_class_dir / relative_path


def _copy_or_move_file(source_path: Path, destination_path: Path, copy_files: bool) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source_path, destination_path)
    else:
        shutil.move(str(source_path), str(destination_path))


def _compute_split_counts(
    total_images: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    raw_counts = {
        "train": total_images * train_ratio,
        "validation": total_images * val_ratio,
        "test": total_images * test_ratio,
    }
    counts = {name: int(math.floor(value)) for name, value in raw_counts.items()}
    remainder = total_images - sum(counts.values())

    if remainder > 0:
        order = sorted(
            raw_counts.keys(),
            key=lambda name: (raw_counts[name] - counts[name], raw_counts[name]),
            reverse=True,
        )
        for index in range(remainder):
            counts[order[index % len(order)]] += 1

    if total_images >= 3:
        for split_name in ("validation", "test"):
            if raw_counts[split_name] > 0 and counts[split_name] == 0 and counts["train"] > 1:
                counts["train"] -= 1
                counts[split_name] += 1

    if counts["train"] <= 0:
        raise ValueError(
            f"Class with {total_images} images would produce an empty train split. "
            "Increase the class size or adjust the ratios."
        )

    if sum(counts.values()) != total_images:
        raise AssertionError("Split count computation did not preserve the image total.")

    return counts["train"], counts["validation"], counts["test"]


def collect_raw_class_images(
    raw_dir: str | Path,
    *,
    seed: int = 42,
    max_images_per_class: int | None = None,
    min_images_per_class: int = 3,
) -> dict[str, list[Path]]:
    raw_dir = Path(raw_dir)
    class_images: dict[str, list[Path]] = {}
    rng = random.Random(seed)

    for class_dir in iter_class_dirs(raw_dir):
        image_paths = sorted(path for path in class_dir.rglob("*") if is_image_file(path))
        if not image_paths:
            continue

        shuffled_paths = image_paths[:]
        rng.shuffle(shuffled_paths)

        if max_images_per_class is not None:
            shuffled_paths = shuffled_paths[:max_images_per_class]

        if len(shuffled_paths) < min_images_per_class:
            raise ValueError(
                f"Class '{class_dir.name}' has only {len(shuffled_paths)} image(s); "
                f"at least {min_images_per_class} are required."
            )

        class_images[class_dir.name] = shuffled_paths

    if not class_images:
        raise ValueError(
            f"No supported images found under {raw_dir}. "
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    return dict(sorted(class_images.items()))


def validate_prepared_split_dirs(
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
) -> dict[str, dict[str, int]]:
    split_dirs = {
        "train": Path(train_dir),
        "validation": Path(val_dir),
        "test": Path(test_dir),
    }

    split_class_names: dict[str, set[str]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    split_paths: dict[str, set[str]] = {}

    for split_name, split_dir in split_dirs.items():
        if not split_dir.exists():
            raise FileNotFoundError(f"Prepared split directory does not exist: {split_dir}")

        class_names = {path.name for path in iter_class_dirs(split_dir) if any(is_image_file(p) for p in path.rglob('*'))}
        if not class_names:
            raise ValueError(f"Prepared split directory is empty: {split_dir}")

        counts: dict[str, int] = {}
        paths: set[str] = set()
        for class_name in sorted(class_names):
            class_dir = split_dir / class_name
            image_paths = sorted(path for path in class_dir.rglob("*") if is_image_file(path))
            if not image_paths:
                continue
            counts[class_name] = len(image_paths)
            for image_path in image_paths:
                normalized = str(image_path.resolve())
                if normalized in paths:
                    raise ValueError(f"Duplicate prepared image path detected in {split_name}: {image_path}")
                paths.add(normalized)

        split_class_names[split_name] = set(counts.keys())
        split_counts[split_name] = counts
        split_paths[split_name] = paths

    expected_classes = split_class_names["train"]
    for split_name, class_names in split_class_names.items():
        if class_names != expected_classes:
            raise ValueError(
                "Class folders must match across train/validation/test. "
                f"Train has {sorted(expected_classes)}, {split_name} has {sorted(class_names)}."
            )

    if split_paths["train"] & split_paths["validation"]:
        raise ValueError("Prepared train and validation splits share image paths.")
    if split_paths["train"] & split_paths["test"]:
        raise ValueError("Prepared train and test splits share image paths.")
    if split_paths["validation"] & split_paths["test"]:
        raise ValueError("Prepared validation and test splits share image paths.")

    return split_counts


def prepare_split_dataset(
    raw_dir: str | Path,
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
    *,
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
    copy_files: bool = True,
    force_resplit: bool = False,
    max_images_per_class: int | None = None,
    min_images_per_class: int = 3,
) -> dict[str, dict[str, int]]:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    paths = resolve_split_paths(raw_dir, train_dir, val_dir, test_dir)

    split_dirs = [paths.train_dir, paths.val_dir, paths.test_dir]
    split_status = {split_dir.name: has_any_images(split_dir) for split_dir in split_dirs}
    existing_split_has_images = any(split_status.values())
    all_split_dirs_ready = all(split_status.values())

    if existing_split_has_images and not all_split_dirs_ready and not force_resplit:
        raise ValueError(
            "A partial prepared split already exists. "
            f"Current split status: {split_status}. "
            "Use force_resplit=True to delete and recreate train/validation/test."
        )

    if all_split_dirs_ready and not force_resplit:
        print("[DATA] Existing train/validation/test split detected; skipping split creation.")
        return validate_prepared_split_dirs(paths.train_dir, paths.val_dir, paths.test_dir)

    if force_resplit:
        print("[DATA] force_resplit=True, deleting existing prepared split directories.")

    for split_dir in split_dirs:
        _clear_existing_split_dir(split_dir)

    class_images = collect_raw_class_images(
        paths.raw_dir,
        seed=seed,
        max_images_per_class=max_images_per_class,
        min_images_per_class=min_images_per_class,
    )

    print(f"[DATA] Found {len(class_images)} classes in {paths.raw_dir}")

    manifest: dict[str, dict[str, list[str]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    assigned_source_paths: set[str] = set()
    split_counts: dict[str, dict[str, int]] = {
        "train": {},
        "validation": {},
        "test": {},
    }

    for class_name, image_paths in class_images.items():
        class_rng = random.Random(f"{seed}:{class_name}")
        class_paths = image_paths[:]
        class_rng.shuffle(class_paths)

        train_count, val_count, test_count = _compute_split_counts(
            len(class_paths),
            train_ratio,
            val_ratio,
            test_ratio,
        )

        train_paths = class_paths[:train_count]
        val_paths = class_paths[train_count:train_count + val_count]
        test_paths = class_paths[train_count + val_count:]

        if len(test_paths) != test_count:
            raise AssertionError(f"Unexpected split allocation for class {class_name}.")

        raw_class_dir = paths.raw_dir / class_name
        targets = [
            ("train", paths.train_dir / class_name, train_paths),
            ("validation", paths.val_dir / class_name, val_paths),
            ("test", paths.test_dir / class_name, test_paths),
        ]

        for split_name, destination_class_dir, split_paths_for_class in targets:
            destination_class_dir.mkdir(parents=True, exist_ok=True)
            manifest[split_name][class_name] = []
            split_counts[split_name][class_name] = len(split_paths_for_class)

            for source_path in split_paths_for_class:
                source_key = str(source_path.resolve())
                if source_key in assigned_source_paths:
                    raise ValueError(f"Raw image was assigned to more than one split: {source_path}")
                assigned_source_paths.add(source_key)

                destination_path = _relative_output_path(raw_class_dir, source_path, destination_class_dir)
                _copy_or_move_file(source_path, destination_path, copy_files=copy_files)
                manifest[split_name][class_name].append(str(source_path))

        print(
            f"[DATA] {class_name}: "
            f"{train_count} train, {val_count} validation, {test_count} test"
        )

    counts = validate_prepared_split_dirs(paths.train_dir, paths.val_dir, paths.test_dir)
    for split_name, per_class_counts in counts.items():
        if set(per_class_counts.keys()) != set(class_images.keys()):
            raise ValueError(f"Prepared {split_name} split is missing one or more class folders.")

    print("[DATA] Dataset split completed.")
    save_json(
        {
            "raw_dir": str(paths.raw_dir),
            "train_dir": str(paths.train_dir),
            "val_dir": str(paths.val_dir),
            "test_dir": str(paths.test_dir),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "copy_files": copy_files,
            "force_resplit": force_resplit,
            "max_images_per_class": max_images_per_class,
            "manifest": manifest,
            "counts": counts,
        },
        paths.train_dir.parent / "split_manifest.json",
    )
    return counts


def build_class_to_idx_from_dir(train_dir: str | Path) -> dict[str, int]:
    train_dir = Path(train_dir)
    class_names = [path.name for path in iter_class_dirs(train_dir) if any(is_image_file(p) for p in path.rglob("*"))]
    if not class_names:
        raise ValueError(f"No class folders with images found under {train_dir}")
    return {class_name: index for index, class_name in enumerate(sorted(class_names))}


def count_dataset_samples_by_class(samples: list[tuple[str, int]], classes: list[str]) -> dict[str, int]:
    counts = Counter(target for _, target in samples)
    return {class_name: counts.get(index, 0) for index, class_name in enumerate(classes)}
