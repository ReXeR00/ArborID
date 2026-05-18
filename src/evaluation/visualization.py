# src/visualization.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths (robust regardless of cwd)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "eval"
DEFAULT_OUT_DIR = PROJECT_ROOT / "assets" / "plots"


# -----------------------------
# Helpers
# -----------------------------
def parse_eval_run_name(folder_name: str) -> Tuple[str, Optional[int]]:
    """
    Expected: "<run_id>-p<patches>" e.g. "17-41-59-p1", "18-39-21-p20"
    Returns: (run_id, patches or None)
    """
    patches = None
    run_id = folder_name

    if "-p" in folder_name:
        left, right = folder_name.rsplit("-p", 1)
        run_id = left
        try:
            patches = int(right)
        except ValueError:
            patches = None

    return run_id, patches


def load_eval_csv(csv_path: Path) -> pd.DataFrame:
    """
    Expected columns (from your eval_folder.py):
      path,true_label,pred_label,pred_prob,uncertain,topk_labels,topk_probs
    """
    df = pd.read_csv(csv_path)
    # normalize types
    if "uncertain" in df.columns:
        # could be 0/1 or True/False as str
        df["uncertain"] = df["uncertain"].astype(int)
    return df


def compute_metrics(df: pd.DataFrame, topk: int = 3) -> dict:
    """
    Computes:
      - top1_acc: pred_label == true_label
      - topk_acc: true_label in topk_labels (pipe-separated string)
      - uncertain_rate: mean(uncertain==1)
      - avg_conf: mean(pred_prob)
    """
    if df.empty:
        return {
            "n": 0,
            "top1_acc": 0.0,
            f"top{topk}_acc": 0.0,
            "uncertain_rate": 0.0,
            "avg_conf": 0.0,
        }

    top1 = (df["pred_label"] == df["true_label"]).mean()

    # topk_labels e.g. "oak|linden|alder"
    def in_topk(row) -> bool:
        labels = str(row.get("topk_labels", "")).split("|")
        true = str(row.get("true_label", ""))
        return true in labels[:topk]  # safety if someone changes formatting

    topk_acc = df.apply(in_topk, axis=1).mean()

    uncertain_rate = df["uncertain"].mean() if "uncertain" in df.columns else 0.0
    avg_conf = df["pred_prob"].mean() if "pred_prob" in df.columns else 0.0

    return {
        "n": int(len(df)),
        "top1_acc": float(top1),
        f"top{topk}_acc": float(topk_acc),
        "uncertain_rate": float(uncertain_rate),
        "avg_conf": float(avg_conf),
    }


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plotting
# -----------------------------
def make_bar_chart(summary_df: pd.DataFrame, out_path: Path, topk: int = 3) -> None:
    """
    Creates a grouped bar chart:
      - Top-1
      - Top-k
      - Uncertain rate
    per eval run (folder).
    """
    ensure_dir(out_path.parent)

    # order by run_id then patches
    df = summary_df.copy()
    df["patches_sort"] = df["patches"].fillna(-1)
    df = df.sort_values(["run_id", "patches_sort"]).reset_index(drop=True)

    x_labels = df["eval_name"].tolist()
    x = list(range(len(x_labels)))

    top1_vals = (df["top1_acc"] * 100).tolist()
    topk_vals = (df[f"top{topk}_acc"] * 100).tolist()
    unc_vals = (df["uncertain_rate"] * 100).tolist()

    width = 0.26

    plt.figure(figsize=(max(10, len(x_labels) * 1.2), 6))
    plt.bar([i - width for i in x], top1_vals, width=width, label="Top-1 acc (%)")
    plt.bar(x, topk_vals, width=width, label=f"Top-{topk} acc (%)")
    plt.bar([i + width for i in x], unc_vals, width=width, label="Uncertain rate (%)")

    plt.xticks(x, x_labels, rotation=35, ha="right")
    plt.ylabel("Percent (%)")
    plt.title(f"External eval summary (Top-1 / Top-{topk} / Uncertain)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()


def make_acc_only_chart(summary_df: pd.DataFrame, out_path: Path, topk: int = 3) -> None:
    """
    Cleaner chart for README: only Top-1 and Top-k.
    """
    ensure_dir(out_path.parent)

    df = summary_df.copy()
    df["patches_sort"] = df["patches"].fillna(-1)
    df = df.sort_values(["run_id", "patches_sort"]).reset_index(drop=True)

    x_labels = df["eval_name"].tolist()
    x = list(range(len(x_labels)))

    top1_vals = (df["top1_acc"] * 100).tolist()
    topk_vals = (df[f"top{topk}_acc"] * 100).tolist()

    width = 0.35

    plt.figure(figsize=(max(10, len(x_labels) * 1.2), 6))
    plt.bar([i - width / 2 for i in x], top1_vals, width=width, label="Top-1 acc (%)")
    plt.bar([i + width / 2 for i in x], topk_vals, width=width, label=f"Top-{topk} acc (%)")

    plt.xticks(x, x_labels, rotation=35, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title(f"External eval accuracy (Top-1 / Top-{topk})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate eval/*.csv and create plots for README")

    p.add_argument(
        "--eval_dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Folder containing eval subfolders like 17-41-59-p1/ with eval_folder.csv inside",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to save plots (default: assets/plots)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k used in eval_folder.csv interpretation (default 3)",
    )
    p.add_argument(
        "--save_csv",
        action="store_true",
        help="Save aggregated CSV next to plots",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir: Path = args.eval_dir
    out_dir: Path = args.out_dir
    topk: int = int(args.topk)

    print(f"[INFO] project_root: {PROJECT_ROOT}")
    print(f"[INFO] eval_dir:      {eval_dir}")
    print(f"[INFO] out_dir:       {out_dir}")
    print(f"[INFO] topk:          {topk}")

    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    rows = []
    missing = []

    for sub in sorted(eval_dir.iterdir()):
        if not sub.is_dir():
            continue

        csv_path = sub / "eval_folder.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue

        eval_name = sub.name
        run_id, patches = parse_eval_run_name(eval_name)

        df = load_eval_csv(csv_path)
        metrics = compute_metrics(df, topk=topk)

        rows.append(
            {
                "eval_name": eval_name,
                "run_id": run_id,
                "patches": patches,
                "n": metrics["n"],
                "top1_acc": metrics["top1_acc"],
                f"top{topk}_acc": metrics[f"top{topk}_acc"],
                "uncertain_rate": metrics["uncertain_rate"],
                "avg_conf": metrics["avg_conf"],
                "csv_path": str(csv_path.as_posix()),
            }
        )

    if not rows:
        print("[WARN] No eval_folder.csv found. Nothing to plot.")
        if missing:
            print("[WARN] Missing files (first 10):")
            for m in missing[:10]:
                print("  -", m)
        return

    summary_df = pd.DataFrame(rows)

    # Save aggregated table (handy for debugging / README tables)
    ensure_dir(out_dir)
    agg_csv = out_dir / "eval_aggregate.csv"
    if args.save_csv:
        summary_df.to_csv(agg_csv, index=False)
        print(f"[INFO] saved: {agg_csv}")

    # Plots
    plot1 = out_dir / "external_eval_summary.png"
    plot2 = out_dir / "external_eval_accuracy.png"

    make_bar_chart(summary_df, out_path=plot1, topk=topk)
    make_acc_only_chart(summary_df, out_path=plot2, topk=topk)

    print("[INFO] saved plots:")
    print("  -", plot1)
    print("  -", plot2)

    # Console quick view
    show_cols = ["run_id", "patches", "eval_name", "n", "top1_acc", f"top{topk}_acc", "uncertain_rate", "avg_conf"]
    print("\n=== Aggregate ===")
    print(
        summary_df[show_cols]
        .sort_values(["run_id", "patches"], na_position="first")
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()
