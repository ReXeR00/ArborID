from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prep_data import prepare_split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic train/validation/test splits from BarkNet rawdata."
    )
    parser.add_argument("--raw_dir", type=Path, required=True, help="Directory with class folders.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Base output directory where train/validation/test will be created.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images_per_class", type=int, default=None)
    parser.add_argument("--min_images_per_class", type=int, default=3)
    parser.add_argument(
        "--move_files",
        action="store_true",
        help="Move files instead of copying them. Copy is the default behavior.",
    )
    parser.add_argument(
        "--force_resplit",
        action="store_true",
        help="Delete existing train/validation/test folders before creating a new split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_split_dataset(
        raw_dir=args.raw_dir,
        train_dir=args.out_dir / "train",
        val_dir=args.out_dir / "validation",
        test_dir=args.out_dir / "test",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_files=not args.move_files,
        force_resplit=args.force_resplit,
        max_images_per_class=args.max_images_per_class,
        min_images_per_class=args.min_images_per_class,
    )


if __name__ == "__main__":
    main()
