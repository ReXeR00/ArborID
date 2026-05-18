from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model import model_from_cfg
from src.data.transforms import get_transforms


# ----------------------------
# Utilities
# ----------------------------
def set_deterministic(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic helps reproducibility, but can be slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_class_to_idx(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): int(v) for k, v in obj.items()}


def invert_class_map(class_to_idx: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in class_to_idx.items()}


def pick_model_cfg(cfg: DictConfig) -> DictConfig:
    model_cfg = OmegaConf.select(cfg, "model")
    if model_cfg is None:
        model_cfg = OmegaConf.select(cfg, "train.model")
    if model_cfg is None:
        model_cfg = OmegaConf.create(
            {"name": "resnet50", "pretrained": False, "freeze_backbone": False, "dropout": 0.0}
        )
    return model_cfg


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    obj = torch.load(checkpoint_path, map_location=device)

    # handle both formats:
    # 1) state_dict directly
    # 2) checkpoint dict with key "model"/"state_dict"
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj  # already a state_dict-like mapping
    
    
    
    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and is_image_file(p)])


# ----------------------------
# Model wrapper
# ----------------------------
@dataclass(frozen=True)
class EvalResult:
    path: str
    true_label: str
    pred_label: str
    pred_prob: float
    topk_labels: list[str]
    topk_probs: list[float]
    uncertain: bool


class Engine:
    def __init__(
        self,
        model: torch.nn.Module,
        idx_to_class: dict[int, str],
        class_to_idx: dict[str, int],
        device: torch.device,
    ):
        self.model = model
        self.idx_to_class = idx_to_class
        self.class_to_idx = class_to_idx
        self.device = device

        # IMPORTANT: use the same val preprocessing as training/inference
        _, self.val_tfms = get_transforms()
        self.patch_mean = getattr(self.val_tfms, "mean", None)
        self.patch_std = getattr(self.val_tfms, "std", None)

    @classmethod
    def from_run_dir(cls, run_dir: Path, device: torch.device | None = None) -> "Engine":
        run_dir = Path(run_dir)

        ckpt = run_dir / "best_model.pt"
        cfg_path = run_dir / ".hydra" / "config.yaml"
        class_map = run_dir / "artifacts" / "class_to_idx.json"

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing hydra config: {cfg_path}")
        if not class_map.exists():
            raise FileNotFoundError(f"Missing class map: {class_map}")

        cfg = OmegaConf.load(cfg_path)
        model_cfg = pick_model_cfg(cfg)

        class_to_idx = load_class_to_idx(class_map)
        idx_to_class = invert_class_map(class_to_idx)
        num_classes = len(class_to_idx)

        net = model_from_cfg(model_cfg, num_classes=num_classes)
        sd = load_state_dict(ckpt, device=device)
        missing, unexpected = net.load_state_dict(sd, strict=False)

        if missing or unexpected:
            print("[WARN] load_state_dict not exact.")
            print("  missing   :", missing[:20], "…", len(missing))
            print("  unexpected:", unexpected[:20], "…", len(unexpected))

        net.to(device)
        net.eval()

        return cls(model=net, idx_to_class=idx_to_class, class_to_idx=class_to_idx, device=device)

    def _predict_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch: [N, 3, H, W] on device
        returns probs: [N, C]
        """
        with torch.inference_mode():
            logits = self.model(batch)
            return torch.softmax(logits, dim=1)

    def predict_pil(self, img: Image.Image) -> torch.Tensor:
        """
        returns probs: [C] on CPU
        """
        x = self.val_tfms(img.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self._predict_tensor_batch(x)[0].detach().cpu()
        return probs

    def _preprocess_patch(self, crop: Image.Image, crop_size: int) -> torch.Tensor:
        expected_crop = getattr(self.val_tfms, "crop_size", None)
        if isinstance(expected_crop, list) and len(expected_crop) == 1:
            expected_crop = expected_crop[0]

        if self.patch_mean is not None and self.patch_std is not None and crop_size == expected_crop:
            x = TF.pil_to_tensor(crop)
            x = TF.convert_image_dtype(x, torch.float32)
            return TF.normalize(x, mean=self.patch_mean, std=self.patch_std)

        return self.val_tfms(crop)

    def predict_pil_patch_voting(
        self,
        img: Image.Image,
        patches: int,
        crop_size: int = 224,
        patch_batch_size: int = 32,
    ) -> torch.Tensor:

        if patches <= 1:
            return self.predict_pil(img)

        img = img.convert("RGB")
        w, h = img.size

        # Upscale small images to avoid invalid crops
        if min(w, h) < crop_size:
            scale = crop_size / float(min(w, h))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            w, h = img.size

        patch_batch_size = max(1, int(patch_batch_size))
        prob_sum = None
        processed = 0
        crops = []

        def flush_crops() -> None:
            nonlocal crops, prob_sum, processed
            if not crops:
                return
            batch = torch.stack(crops, dim=0).to(self.device)
            probs = self._predict_tensor_batch(batch).sum(dim=0)
            prob_sum = probs if prob_sum is None else prob_sum + probs
            processed += len(crops)
            crops = []

        for _ in range(patches):
            left = random.randint(0, max(0, w - crop_size))
            top = random.randint(0, max(0, h - crop_size))
            crop = img.crop((left, top, left + crop_size, top + crop_size))
            crops.append(self._preprocess_patch(crop, crop_size=crop_size))
            if len(crops) >= patch_batch_size:
                flush_crops()

        flush_crops()
        return (prob_sum / processed).detach().cpu()


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_folder(
    engine: Engine,
    data_dir: Path,
    topk: int = 3,
    threshold: float = 0.45,
    patches: int = 0,
    crop_size: int = 224,
    patch_batch_size: int = 32,
    progress_interval: int = 10,
) -> tuple[list[EvalResult], dict[str, Any]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        suggested_dir = PROJECT_ROOT / "src" / data_dir
        if suggested_dir.exists():
            print(f"[INFO] data_dir not found at {data_dir}; using {suggested_dir} instead.")
            data_dir = suggested_dir
        else:
            suggestion = f" Did you mean: {suggested_dir}?"
            raise FileNotFoundError(
                f"Evaluation data directory does not exist: {data_dir}. "
                "Pass --data_dir to a folder containing class subfolders."
                f"{suggestion}"
            )
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Evaluation data path is not a directory: {data_dir}.")

    # Only evaluate folders that exist in class_to_idx
    class_folders = [p for p in data_dir.iterdir() if p.is_dir()]
    class_folders = sorted(class_folders, key=lambda p: p.name)

    lower_to_class = {k.lower(): k for k in engine.class_to_idx}

    eval_folders: list[tuple[Path, str]] = []
    skipped = []
    for p in class_folders:
        canonical = lower_to_class.get(p.name.lower())
        if canonical is not None:
            eval_folders.append((p, canonical))
        else:
            skipped.append(p.name)

    if skipped:
        print("[WARN] Skipping folders not present in class_to_idx.json:", skipped)

    if not eval_folders:
        print("[DEBUG] class_folders found:", [p.name for p in class_folders])
        print("[DEBUG] class_to_idx keys:  ", sorted(engine.class_to_idx.keys()))
        found = ", ".join(p.name for p in class_folders) or "<none>"
        expected = ", ".join(sorted(engine.class_to_idx.keys()))
        raise ValueError(
            "No valid class folders found to evaluate. "
            f"Found folders: {found}. Expected folders matching the checkpoint classes: {expected}."
        )

    eval_classes = [class_name for _, class_name in eval_folders]

    # gather samples
    samples: list[tuple[Path, str]] = []
    for folder_path, cls_name in eval_folders:
        for img_path in list_images(folder_path):
            samples.append((img_path, cls_name))

    if not samples:
        raise ValueError(f"No images found under: {data_dir}")

    C = len(engine.class_to_idx)
    conf = np.zeros((C, C), dtype=np.int64)

    total = 0
    correct1 = 0
    correct3 = 0
    uncertain_cnt = 0

    per_class_total = {c: 0 for c in eval_classes}
    per_class_correct1 = {c: 0 for c in eval_classes}
    per_class_correct3 = {c: 0 for c in eval_classes}

    results: list[EvalResult] = []

    for sample_idx, (img_path, true_label) in enumerate(samples, start=1):
        if progress_interval > 0 and (sample_idx == 1 or sample_idx % progress_interval == 0 or sample_idx == len(samples)):
            print(f"[INFO] Evaluating {sample_idx}/{len(samples)}: {img_path}")
        img = Image.open(img_path)

        if patches and patches > 1:
            probs = engine.predict_pil_patch_voting(
                img,
                patches=patches,
                crop_size=crop_size,
                patch_batch_size=patch_batch_size,
            )
        else:
            probs = engine.predict_pil(img)

        true_idx = engine.class_to_idx[true_label]

        k = min(topk, probs.numel())
        top_probs, top_idxs = torch.topk(probs, k=k)

        pred_idx = int(top_idxs[0].item())
        pred_prob = float(top_probs[0].item())
        pred_label = engine.idx_to_class[pred_idx]

        topk_labels = [engine.idx_to_class[int(i)] for i in top_idxs.tolist()]
        topk_probs = [float(x) for x in top_probs.tolist()]

        is_uncertain = pred_prob < threshold
        if is_uncertain:
            uncertain_cnt += 1

        total += 1
        per_class_total[true_label] += 1

        if pred_idx == true_idx:
            correct1 += 1
            per_class_correct1[true_label] += 1

        if true_label in topk_labels:
            correct3 += 1
            per_class_correct3[true_label] += 1

        conf[true_idx, pred_idx] += 1

        results.append(
            EvalResult(
                path=str(img_path),
                true_label=true_label,
                pred_label=pred_label,
                pred_prob=pred_prob,
                topk_labels=topk_labels,
                topk_probs=topk_probs,
                uncertain=is_uncertain,
            )
        )

    summary = {
        "total": total,
        "top1_acc": correct1 / total if total else 0.0,
        "top3_acc": correct3 / total if total else 0.0,
        "uncertain_rate": uncertain_cnt / total if total else 0.0,
        "threshold": threshold,
        "topk": topk,
        "patches": patches,
        "crop_size": crop_size,
        "patch_batch_size": patch_batch_size,
        "eval_classes": eval_classes,
        "per_class": {
            c: {
                "n": per_class_total[c],
                "top1": (per_class_correct1[c] / per_class_total[c]) if per_class_total[c] else 0.0,
                "top3": (per_class_correct3[c] / per_class_total[c]) if per_class_total[c] else 0.0,
            }
            for c in eval_classes
        },
        "confusion_matrix": conf,
    }

    return results, summary


def print_confusion_matrix(conf: np.ndarray, idx_to_class: dict[int, str], only_labels: list[str] | None = None) -> None:
    """
    Prints a readable confusion matrix.
    If only_labels is provided, prints only those rows/cols (in idx order).
    """
    if only_labels is not None:
        # build list of indices we want
        wanted = []
        for i, name in idx_to_class.items():
            if name in only_labels:
                wanted.append(i)
        wanted = sorted(wanted)

        conf = conf[np.ix_(wanted, wanted)]
        labels = [idx_to_class[i] for i in wanted]
    else:
        labels = [idx_to_class[i] for i in range(conf.shape[0])]

    # header
    col_w = max(8, max(len(x) for x in labels) + 2)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("".ljust(col_w) + "".join(l[:col_w - 2].ljust(col_w) for l in labels))

    for r, lab in enumerate(labels):
        row = conf[r]
        print(lab[:col_w - 2].ljust(col_w) + "".join(str(int(x)).ljust(col_w) for x in row))


def save_csv(results: list[EvalResult], out_path: Path) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "pred_label", "pred_prob", "uncertain", "topk_labels", "topk_probs"])
        for r in results:
            w.writerow(
                [
                    r.path,
                    r.true_label,
                    r.pred_label,
                    f"{r.pred_prob:.6f}",
                    int(r.uncertain),
                    "|".join(r.topk_labels),
                    "|".join(f"{p:.6f}" for p in r.topk_probs),
                ]
            )


def save_worst_confident_mistakes(results: list[EvalResult], out_path: Path, min_prob: float = 0.60, limit: int = 50) -> None:
    """
    Saves mistakes where model was confident (pred_prob >= min_prob) but wrong.
    These are perfect "hard cases" to add to training later.
    """
    mistakes = [r for r in results if (r.pred_label != r.true_label) and (r.pred_prob >= min_prob)]
    mistakes.sort(key=lambda r: r.pred_prob, reverse=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "pred_label", "pred_prob", "topk_labels", "topk_probs"])
        for r in mistakes[:limit]:
            w.writerow(
                [
                    r.path,
                    r.true_label,
                    r.pred_label,
                    f"{r.pred_prob:.6f}",
                    "|".join(r.topk_labels),
                    "|".join(f"{p:.6f}" for p in r.topk_probs),
                ]
            )


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Hydra output dir, e.g. outputs/2026-01-10/16-22-10")
    p.add_argument("--data_dir", type=str, required=True, help="Folder like data/external with class subfolders")
    p.add_argument("--topk", type=int, default=3, help="Top-k used for top-k accuracy and saved outputs")
    p.add_argument("--threshold", type=float, default=0.45, help="Uncertain threshold for max prob")
    p.add_argument("--patches", type=int, default=0, help="If >1, use random patch voting with this many patches")
    p.add_argument("--crop_size", type=int, default=224, help="Patch size for patch voting")
    p.add_argument("--patch_batch_size", type=int, default=32, help="Patch-voting inference batch size")
    p.add_argument("--progress_interval", type=int, default=10, help="Print progress every N images (0 disables)")
    p.add_argument("--seed", type=int, default=0, help="Seed for reproducibility (also used for patch sampling)")
    p.add_argument("--out_csv", type=str, default="", help="Where to save per-image CSV (default: <run_dir>/eval_folder.csv)")
    p.add_argument("--out_worst_csv", type=str, default="", help="Where to save worst mistakes CSV (default: <run_dir>/eval_worst_confident.csv)")
    p.add_argument("--worst_min_prob", type=float, default=0.60, help="Min prob for 'confident mistake'")
    p.add_argument("--worst_limit", type=int, default=50, help="Max rows in worst mistakes CSV")
    p.add_argument("--eval_root", type=Path, default=Path("eval"), help="Root directory for evaluation outputs (default: eval/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print("[INFO] run_dir:", run_dir)
    print("[INFO] data_dir:", data_dir)

    engine = Engine.from_run_dir(run_dir, device=device)

    results, summary = evaluate_folder(
        engine=engine,
        data_dir=data_dir,
        topk=args.topk,
        threshold=args.threshold,
        patches=args.patches,
        crop_size=args.crop_size,
        patch_batch_size=args.patch_batch_size,
        progress_interval=args.progress_interval,
    )

    print("\n=== Summary ===")
    print(f"Total images      : {summary['total']}")
    print(f"Top-1 accuracy    : {summary['top1_acc']*100:.2f}%")
    print(f"Top-{args.topk} accuracy  : {summary['top3_acc']*100:.2f}%")
    print(f"Uncertain rate(<{args.threshold}) : {summary['uncertain_rate']*100:.2f}%")
    if args.patches and args.patches > 1:
        print(f"Patch voting      : {args.patches} patches (crop_size={args.crop_size})")
    else:
        print("Patch voting      : off")

    print("\nPer-class:")
    for c in summary["eval_classes"]:
        pc = summary["per_class"][c]
        print(f"  - {c:20s} n={pc['n']:3d}  top1={pc['top1']*100:6.2f}%  top{args.topk}={pc['top3']*100:6.2f}%")

    print_confusion_matrix(
        summary["confusion_matrix"],
        idx_to_class=engine.idx_to_class,
        only_labels=summary["eval_classes"],
    )

    run_tag = run_dir.name
    eval_tag = f"{run_tag}-p{args.patches}"
    out_dir = args.eval_root / eval_tag
    out_dir.mkdir(parents=True, exist_ok=True)



    out_csv = Path(args.out_csv) if args.out_csv else (out_dir / "eval_folder.csv")
    out_worst = Path(args.out_worst_csv) if args.out_worst_csv else (out_dir / "eval_worst_confident.csv")

    save_csv(results, out_csv)
    save_worst_confident_mistakes(results, out_worst, min_prob=args.worst_min_prob, limit=args.worst_limit)

    print("\nSaved:")
    print("  - per-image results :", out_csv)
    print("  - worst mistakes    :", out_worst)
    class_to_idx = load_class_to_idx(run_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print("Loaded class_to_idx:", class_to_idx)
    print("idx_to_class:", idx_to_class)
    loaded_class_to_idx = load_class_to_idx(args.run_dir)

    print("Loaded class_to_idx:", loaded_class_to_idx)
if __name__ == "__main__":
    main()
