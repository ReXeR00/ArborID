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

from src.checkpoints import load_state_dict
from src.data.transforms import get_transforms
from src.model.model import model_from_cfg


def set_deterministic(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*") if path.is_file() and is_image_file(path)])


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
        val_tfms,
    ) -> None:
        self.model = model
        self.idx_to_class = idx_to_class
        self.class_to_idx = class_to_idx
        self.device = device
        self.val_tfms = val_tfms
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
        override = getattr(model_cfg, "num_classes_override", None)
        if override is not None:
            num_classes = int(override)

        input_size = int(OmegaConf.select(cfg, "data.img_size", default=224))
        model_name = str(getattr(model_cfg, "name", "resnet50"))
        _, val_tfms = get_transforms(model_name=model_name, img_size=input_size)

        net = model_from_cfg(model_cfg, num_classes=num_classes, input_size=input_size)
        checkpoint, state_dict = load_state_dict(ckpt, device=device)
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("[WARN] load_state_dict not exact.")
            print("  missing   :", missing[:20], "...", len(missing))
            print("  unexpected:", unexpected[:20], "...", len(unexpected))
        else:
            print("[LOAD BEST] epoch:", checkpoint.get("epoch"))
            print("[LOAD BEST] metric:", checkpoint.get("metric_name"), checkpoint.get("best_metric"))

        net.to(device)
        net.eval()

        return cls(
            model=net,
            idx_to_class=idx_to_class,
            class_to_idx=class_to_idx,
            device=device,
            val_tfms=val_tfms,
        )

    def _predict_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits = self.model(batch)
            return torch.softmax(logits, dim=1)

    def predict_pil(self, img: Image.Image) -> torch.Tensor:
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

    class_folders = sorted([path for path in data_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
    lower_to_class = {key.lower(): key for key in engine.class_to_idx}

    eval_folders: list[tuple[Path, str]] = []
    skipped = []
    for path in class_folders:
        canonical = lower_to_class.get(path.name.lower())
        if canonical is not None:
            eval_folders.append((path, canonical))
        else:
            skipped.append(path.name)

    if skipped:
        print("[WARN] Skipping folders not present in class_to_idx.json:", skipped)

    if not eval_folders:
        found = ", ".join(path.name for path in class_folders) or "<none>"
        expected = ", ".join(sorted(engine.class_to_idx.keys()))
        raise ValueError(
            "No valid class folders found to evaluate. "
            f"Found folders: {found}. Expected folders matching the checkpoint classes: {expected}."
        )

    eval_classes = [class_name for _, class_name in eval_folders]

    samples: list[tuple[Path, str]] = []
    for folder_path, cls_name in eval_folders:
        for img_path in list_images(folder_path):
            samples.append((img_path, cls_name))

    if not samples:
        raise ValueError(f"No images found under: {data_dir}")

    class_count = len(engine.class_to_idx)
    conf = np.zeros((class_count, class_count), dtype=np.int64)

    total = 0
    correct1 = 0
    correctk = 0
    uncertain_cnt = 0

    per_class_total = {name: 0 for name in eval_classes}
    per_class_correct1 = {name: 0 for name in eval_classes}
    per_class_correctk = {name: 0 for name in eval_classes}

    results: list[EvalResult] = []

    for sample_idx, (img_path, true_label) in enumerate(samples, start=1):
        if progress_interval > 0 and (sample_idx == 1 or sample_idx % progress_interval == 0 or sample_idx == len(samples)):
            print(f"[INFO] Evaluating {sample_idx}/{len(samples)}: {img_path}")

        with Image.open(img_path) as image_file:
            img = image_file.convert("RGB")

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
        topk_labels = [engine.idx_to_class[int(idx)] for idx in top_idxs.tolist()]
        topk_probs = [float(prob) for prob in top_probs.tolist()]

        is_uncertain = pred_prob < threshold
        if is_uncertain:
            uncertain_cnt += 1

        total += 1
        per_class_total[true_label] += 1

        if pred_idx == true_idx:
            correct1 += 1
            per_class_correct1[true_label] += 1

        if true_label in topk_labels:
            correctk += 1
            per_class_correctk[true_label] += 1

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
        "topk_acc": correctk / total if total else 0.0,
        "uncertain_rate": uncertain_cnt / total if total else 0.0,
        "threshold": threshold,
        "topk": topk,
        "patches": patches,
        "crop_size": crop_size,
        "patch_batch_size": patch_batch_size,
        "eval_classes": eval_classes,
        "per_class": {
            name: {
                "n": per_class_total[name],
                "top1": (per_class_correct1[name] / per_class_total[name]) if per_class_total[name] else 0.0,
                "topk": (per_class_correctk[name] / per_class_total[name]) if per_class_total[name] else 0.0,
            }
            for name in eval_classes
        },
        "confusion_matrix": conf,
    }
    return results, summary


def print_confusion_matrix(conf: np.ndarray, idx_to_class: dict[int, str], only_labels: list[str] | None = None) -> None:
    if only_labels is not None:
        wanted = sorted([idx for idx, name in idx_to_class.items() if name in only_labels])
        conf = conf[np.ix_(wanted, wanted)]
        labels = [idx_to_class[idx] for idx in wanted]
    else:
        labels = [idx_to_class[idx] for idx in range(conf.shape[0])]

    col_w = max(8, max(len(label) for label in labels) + 2)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("".ljust(col_w) + "".join(label[:col_w - 2].ljust(col_w) for label in labels))

    for row_idx, label in enumerate(labels):
        row = conf[row_idx]
        print(label[:col_w - 2].ljust(col_w) + "".join(str(int(value)).ljust(col_w) for value in row))


def save_csv(results: list[EvalResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label", "pred_prob", "uncertain", "topk_labels", "topk_probs"])
        for result in results:
            writer.writerow(
                [
                    result.path,
                    result.true_label,
                    result.pred_label,
                    f"{result.pred_prob:.6f}",
                    int(result.uncertain),
                    "|".join(result.topk_labels),
                    "|".join(f"{prob:.6f}" for prob in result.topk_probs),
                ]
            )


def save_worst_confident_mistakes(
    results: list[EvalResult],
    out_path: Path,
    min_prob: float = 0.60,
    limit: int = 50,
) -> None:
    mistakes = [result for result in results if (result.pred_label != result.true_label) and (result.pred_prob >= min_prob)]
    mistakes.sort(key=lambda result: result.pred_prob, reverse=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label", "pred_prob", "topk_labels", "topk_probs"])
        for result in mistakes[:limit]:
            writer.writerow(
                [
                    result.path,
                    result.true_label,
                    result.pred_label,
                    f"{result.pred_prob:.6f}",
                    "|".join(result.topk_labels),
                    "|".join(f"{prob:.6f}" for prob in result.topk_probs),
                ]
            )


def select_gradcam_samples(
    results: list[EvalResult],
    mode: str,
    limit: int,
    min_confident_prob: float,
) -> list[EvalResult]:
    existing_results = [result for result in results if Path(result.path).exists()]
    missing_count = len(results) - len(existing_results)
    if missing_count:
        print(f"[GRAD-CAM] Skipping {missing_count} missing image files.")

    if mode == "all":
        selected = existing_results
    elif mode == "wrong":
        selected = [result for result in existing_results if result.pred_label != result.true_label]
    elif mode == "correct":
        selected = [result for result in existing_results if result.pred_label == result.true_label]
    elif mode == "worst_confident":
        selected = [
            result
            for result in existing_results
            if result.pred_label != result.true_label and result.pred_prob >= min_confident_prob
        ]
        selected = sorted(selected, key=lambda result: result.pred_prob, reverse=True)
    else:
        raise ValueError(f"Unknown Grad-CAM mode: {mode}")

    if limit > 0:
        selected = selected[:limit]

    return selected


def open_gradcam_after_eval(
    engine: Engine,
    results: list[EvalResult],
    run_dir: Path,
    mode: str,
    limit: int,
    min_confident_prob: float,
) -> None:
    selected = select_gradcam_samples(
        results=results,
        mode=mode,
        limit=limit,
        min_confident_prob=min_confident_prob,
    )

    if not selected:
        print(f"[GRAD-CAM] No samples matched mode='{mode}'. Viewer was not opened.")
        return

    from src.evaluation.gradcam import GradCAM
    from src.evaluation.gradcam_viewer import GradCAMSample, open_gradcam_viewer

    samples = [
        GradCAMSample(
            path=result.path,
            true_label=result.true_label,
            pred_label=result.pred_label,
            pred_prob=result.pred_prob,
            topk_labels=result.topk_labels,
            topk_probs=result.topk_probs,
        )
        for result in selected
    ]

    output_dir = run_dir / "gradcam"
    print(f"[GRAD-CAM] Opening viewer with {len(samples)} samples (mode={mode}).")
    print(f"[GRAD-CAM] Save directory: {output_dir}")
    with GradCAM(model=engine.model, transform=engine.val_tfms, device=engine.device) as gradcam:
        open_gradcam_viewer(gradcam=gradcam, samples=samples, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Hydra output dir, e.g. outputs/2026-01-10/16-22-10")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder like data/external with class subfolders")
    parser.add_argument("--topk", type=int, default=3, help="Top-k used for top-k accuracy and saved outputs")
    parser.add_argument("--threshold", type=float, default=0.45, help="Uncertain threshold for max prob")
    parser.add_argument("--patches", type=int, default=0, help="If >1, use random patch voting with this many patches")
    parser.add_argument("--crop_size", type=int, default=224, help="Patch size for patch voting")
    parser.add_argument("--patch_batch_size", type=int, default=32, help="Patch-voting inference batch size")
    parser.add_argument("--progress_interval", type=int, default=10, help="Print progress every N images (0 disables)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility and patch sampling")
    parser.add_argument("--out_csv", type=str, default="", help="Where to save per-image CSV")
    parser.add_argument("--out_worst_csv", type=str, default="", help="Where to save worst mistakes CSV")
    parser.add_argument("--worst_min_prob", type=float, default=0.60, help="Min prob for confident mistakes")
    parser.add_argument("--worst_limit", type=int, default=50, help="Max rows in worst mistakes CSV")
    parser.add_argument("--eval_root", type=Path, default=Path("eval"), help="Root directory for evaluation outputs")
    parser.add_argument("--show_gradcam", action="store_true", help="Open an interactive Grad-CAM viewer after evaluation")
    parser.add_argument("--gradcam_limit", type=int, default=50, help="Maximum number of samples to show in the Grad-CAM viewer")
    parser.add_argument(
        "--gradcam_mode",
        choices=["all", "wrong", "correct", "worst_confident"],
        default="all",
        help="Which evaluated samples to show in the Grad-CAM viewer",
    )
    return parser.parse_args()


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
    print(f"Top-1 accuracy    : {summary['top1_acc'] * 100:.2f}%")
    print(f"Top-{args.topk} accuracy  : {summary['topk_acc'] * 100:.2f}%")
    print(f"Uncertain rate(<{args.threshold}) : {summary['uncertain_rate'] * 100:.2f}%")
    
    if args.patches and args.patches > 1:
        print(f"Patch voting      : {args.patches} patches (crop_size={args.crop_size})")
    else:
        print("Patch voting      : off")

    print("\nPer-class:")
    for class_name in summary["eval_classes"]:
        per_class = summary["per_class"][class_name]
        print(f"  - {class_name:20s} n={per_class['n']:3d}  top1={per_class['top1'] * 100:6.2f}%  top{args.topk}={per_class['topk'] * 100:6.2f}%")

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

    if args.show_gradcam:
        open_gradcam_after_eval(
            engine=engine,
            results=results,
            run_dir=run_dir,
            mode=args.gradcam_mode,
            limit=args.gradcam_limit,
            min_confident_prob=args.worst_min_prob,
        )


if __name__ == "__main__":
    main()
