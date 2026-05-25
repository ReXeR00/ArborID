from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoints import load_state_dict
from src.data.transforms import get_transforms
from src.model.model import model_from_cfg


@dataclass(frozen=True)
class Prediction:
    label: str
    prob: float


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        idx_to_class: dict[int, str],
        val_tfms,
        device: torch.device,
    ) -> None:
        self.model = model
        self.idx_to_class = idx_to_class
        self.val_tfms = val_tfms
        self.device = device

    @staticmethod
    def _load_class_map(class_to_idx_path: Path) -> dict[int, str]:
        with open(class_to_idx_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)

        idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
        idxs = sorted(idx_to_class.keys())
        if idxs != list(range(len(idxs))):
            raise ValueError(f"Non-contiguous class indices in {class_to_idx_path}: {idxs[:50]}")

        return idx_to_class

    @staticmethod
    def _pick_model_cfg(cfg: DictConfig) -> DictConfig:
        model_cfg = OmegaConf.select(cfg, "model")
        if model_cfg is None:
            model_cfg = OmegaConf.select(cfg, "train.model")

        if model_cfg is None:
            model_cfg = OmegaConf.create(
                {
                    "name": "resnet50",
                    "pretrained": False,
                    "freeze_backbone": False,
                    "dropout": 0.0,
                    "num_classes_override": None,
                }
            )
        return model_cfg

    @classmethod
    def from_run_dir(cls, run_dir: Path, device: torch.device | None = None) -> "InferenceEngine":
        run_dir = Path(run_dir)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = run_dir / "best_model.pt"
        class_map_path = run_dir / "artifacts" / "class_to_idx.json"
        config_path = run_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            alt = run_dir / "config.yaml"
            if alt.exists():
                config_path = alt

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not class_map_path.exists():
            raise FileNotFoundError(f"class_to_idx not found: {class_map_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        cfg: DictConfig = OmegaConf.load(config_path)
        model_cfg = cls._pick_model_cfg(cfg)

        idx_to_class = cls._load_class_map(class_map_path)
        num_classes = len(idx_to_class)

        override = getattr(model_cfg, "num_classes_override", None)
        if override is not None:
            num_classes = int(override)

        input_size = int(OmegaConf.select(cfg, "data.img_size", default=224))
        model_name = str(getattr(model_cfg, "name", "resnet50"))
        _, val_tfms = get_transforms(model_name=model_name, img_size=input_size)

        net = model_from_cfg(model_cfg, num_classes=num_classes, input_size=input_size)
        checkpoint, state_dict = load_state_dict(ckpt_path, device=device)
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "State dict mismatch - refusing to run inference.\n"
                f"Missing (first 20): {missing[:20]}\n"
                f"Unexpected (first 20): {unexpected[:20]}"
            )

        print("[LOAD BEST] epoch:", checkpoint.get("epoch"))
        print("[LOAD BEST] metric:", checkpoint.get("metric_name"), checkpoint.get("best_metric"))

        net.to(device)
        net.eval()

        return cls(model=net, idx_to_class=idx_to_class, val_tfms=val_tfms, device=device)

    def _probs_from_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu()

    def predict_pil(self, img: Image.Image, topk: int = 3) -> list[Prediction]:
        if topk < 1:
            raise ValueError("topk must be >= 1")

        x = self.val_tfms(img.convert("RGB")).unsqueeze(0).to(self.device)

        probs = self._probs_from_tensor_batch(x).squeeze(0)
        k = min(topk, probs.shape[0])
        top_probs, top_idxs = torch.topk(probs, k=k)

        return [
            Prediction(label=self.idx_to_class[int(idx)], prob=float(prob))
            for prob, idx in zip(top_probs.tolist(), top_idxs.tolist())
        ]

    def predict_path(
        self,
        image_path: Path,
        topk: int = 3,
        patches: int = 0,
        crop_size: int = 224,
        seed: int | None = 0,
    ) -> list[Prediction]:
        with Image.open(image_path) as image_file:
            img = image_file.convert("RGB")

        if patches and patches > 1:
            return self.predict_pil_patch_voting(
                img,
                topk=topk,
                patches=patches,
                crop_size=crop_size,
                seed=seed,
            )
        return self.predict_pil(img, topk=topk)

    def predict_pil_patch_voting(
        self,
        img: Image.Image,
        topk: int = 3,
        patches: int = 20,
        crop_size: int = 224,
        seed: int | None = 0,
    ) -> list[Prediction]:
        if topk < 1:
            raise ValueError("topk must be >= 1")
        if patches < 1:
            raise ValueError("patches must be >= 1")

        rng = random.Random(seed)
        img = img.convert("RGB")
        w, h = img.size

        if min(w, h) < crop_size:
            scale = crop_size / float(min(w, h))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            w, h = img.size

        crop_tensors = []
        for _ in range(patches):
            left = rng.randint(0, max(0, w - crop_size))
            top = rng.randint(0, max(0, h - crop_size))
            crop = img.crop((left, top, left + crop_size, top + crop_size))
            crop_tensors.append(self.val_tfms(crop))

        batch = torch.stack(crop_tensors, dim=0).to(self.device)
        probs = self._probs_from_tensor_batch(batch).mean(dim=0)

        k = min(topk, probs.shape[0])
        top_probs, top_idxs = torch.topk(probs, k=k)
        return [
            Prediction(label=self.idx_to_class[int(idx)], prob=float(prob))
            for prob, idx in zip(top_probs.tolist(), top_idxs.tolist())
        ]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for ArborID")

    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory with checkpoint + artifacts")
    parser.add_argument("--image", type=Path, required=True, help="Path to image")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--patches", type=int, default=0, help="If >1, use random patch voting with this many crops.")
    parser.add_argument("--crop_size", type=int, default=224, help="Patch size used for random patch voting.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for patch sampling")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu") if args.cpu else None
    engine = InferenceEngine.from_run_dir(run_dir=Path(args.run_dir), device=device)

    preds = engine.predict_path(
        Path(args.image),
        topk=args.topk,
        patches=args.patches,
        crop_size=args.crop_size,
        seed=args.seed,
    )

    print(f"Image: {args.image}")
    for pr in preds:
        print(f"{pr.label:20s}  {pr.prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
