# src/infer.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import random
from PIL import Image
from omegaconf import OmegaConf, DictConfig

from model.model import model_from_cfg
from data.transforms import get_transforms


# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class Prediction:
    label: str
    prob: float


# ----------------------------
# Inference engine
# ----------------------------
class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        idx_to_class: dict[int, str],
        val_tfms,
        device: torch.device,
    ):
        self.model = model
        self.idx_to_class = idx_to_class
        self.val_tfms = val_tfms
        self.device = device

    @staticmethod
    def _load_class_map(class_to_idx_path: Path) -> dict[int, str]:
        """
        Reads JSON: {"oak": 0, "spruce": 1, ...}
        Returns: {0: "oak", 1: "spruce", ...}
        """
        with open(class_to_idx_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)

        idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

        # sanity: must be contiguous 0..N-1
        idxs = sorted(idx_to_class.keys())
        if idxs != list(range(len(idxs))):
            raise ValueError(f"Non-contiguous class indices in {class_to_idx_path}: {idxs[:50]}")

        return idx_to_class

    @staticmethod
    def _pick_model_cfg(cfg: DictConfig) -> DictConfig:
        """
        Prefer cfg.model; fallback to cfg.train.model; else default.
        """
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

    @staticmethod
    def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
        """
        Supports:
          - state_dict directly (OrderedDict / dict of tensors)
          - dict with {"state_dict": ...} or {"model": ...}
        Also strips "module." prefix if present.
        """
        obj = torch.load(checkpoint_path, map_location=device)
        print("[DEBUG] checkpoint type:", type(obj))

        state_dict: dict[str, Any] | None = None

        if isinstance(obj, dict):
            # checkpoint wrapper formats
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                state_dict = obj["state_dict"]
            elif "model" in obj and isinstance(obj["model"], dict):
                state_dict = obj["model"]
            else:
                # might already be a state_dict
                state_dict = obj
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

        # strip DataParallel prefix if needed
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        keys = list(state_dict.keys())
        print("[DEBUG] checkpoint keys sample:", keys[:20])
        print("[DEBUG] has fc.weight:", "fc.weight" in state_dict)
        print("[DEBUG] has fc.bias  :", "fc.bias" in state_dict)

        # helpful head debug
        head_keys = [k for k in state_dict.keys() if any(s in k for s in ["fc", "classifier", "head", "linear"])]
        print("[DEBUG] ckpt head-like keys:", head_keys[:50])

        return state_dict

    @classmethod
    def from_run_dir(cls, run_dir: Path, device: torch.device | None = None) -> "InferenceEngine":
        """
        Expects run_dir structure:
          run_dir/
            best_model.pt
            artifacts/class_to_idx.json
            .hydra/config.yaml    (Hydra default)
        """
        run_dir = Path(run_dir)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = run_dir / "best_model.pt"
        class_map_path = run_dir / "artifacts" / "class_to_idx.json"

        # Hydra saves config into .hydra/config.yaml by default
        config_path = run_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            # fallback (if you also copied config elsewhere)
            alt = run_dir / "config.yaml"
            if alt.exists():
                config_path = alt

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not class_map_path.exists():
            raise FileNotFoundError(f"class_to_idx not found: {class_map_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path} (expected Hydra .hydra/config.yaml)")

        cfg: DictConfig = OmegaConf.load(config_path)
        model_cfg = cls._pick_model_cfg(cfg)

        idx_to_class = cls._load_class_map(class_map_path)
        num_classes = len(idx_to_class)

        # optional override
        override = getattr(model_cfg, "num_classes_override", None)
        if override is not None:
            num_classes = int(override)
            print(f"[INFO] Overriding num_classes to {num_classes} from cfg.model.num_classes_override")

        model_name = str(getattr(model_cfg, "name", "resnet50"))
        _, val_tfms = get_transforms(model_name=model_name)

        net = model_from_cfg(model_cfg, num_classes=num_classes)
        state_dict = cls._load_state_dict(ckpt_path, device=device)

        # single load, strict=False with hard stop if mismatch
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        print("[DEBUG] missing:", missing[:20], "…", len(missing))
        print("[DEBUG] unexpected:", unexpected[:20], "…", len(unexpected))
        if missing or unexpected:
            raise RuntimeError(
                "State dict mismatch — refusing to run inference.\n"
                f"Missing (first 20): {missing[:20]}\n"
                f"Unexpected (first 20): {unexpected[:20]}"
            )

        net.to(device)
        net.eval()

        # show model head keys for sanity
        model_keys = list(net.state_dict().keys())
        model_head_keys = [k for k in model_keys if any(s in k for s in ["fc", "classifier", "head", "linear"])]
        print("[DEBUG] model head-like keys:", model_head_keys[:50])
        print("[DEBUG] device:", device)
        print("[DEBUG] model.training:", net.training)

        return cls(model=net, idx_to_class=idx_to_class, val_tfms=val_tfms, device=device)

    def predict_pil(self, img: Image.Image, topk: int = 3) -> list[Prediction]:
        if topk < 1:
            raise ValueError("topk must be >= 1")

        img = img.convert("RGB")
        x = self.val_tfms(img).unsqueeze(0).to(self.device)  # [1,3,H,W]

        # debug min/max (useful for catching normalization issues)
        print("[DEBUG] input tensor min/max:", float(x.min()), float(x.max()))

        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

            k = min(topk, probs.shape[1])
            top_probs, top_idxs = torch.topk(probs, k=k, dim=1)

        top_probs = top_probs.squeeze(0).cpu().tolist()
        top_idxs = top_idxs.squeeze(0).cpu().tolist()

        preds: list[Prediction] = []
        for p, idx in zip(top_probs, top_idxs):
            label = self.idx_to_class[int(idx)]
            preds.append(Prediction(label=label, prob=float(p)))

        return preds

    def predict_path(self, image_path: Path, topk: int = 3, patches: int = 0, seed: int | None = 0,) -> list[Prediction]:
        img = Image.open(image_path).convert("RGB")
        if patches and patches > 1:
            return self.predict_pil_patch_voting(img, topk=topk, patches=patches, seed=seed)
        return self.predict_pil(img, topk=topk) 
    
    def probs_form_tenesors_batch(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.interfernece_mode():
            logits = slef.mode(batch)
            probs = torch.softmax(logits, dim = 1)
        return 
    def predict_pil_patch_voting(
            self,
            img: Image.Image,
            topk: int = 3,
            patches: int= 20,
            crop_size: int = 200,
            seed: int | None = 0,
    ) -> list[Prediction]:
        if topk <1:
            raise ValueError("topk must be >= 1")
        if patches < 1:
            raise ValueError("patches must be >= 1")
        img = img.convert('RGB')
        if seed is not None:
            random.seed(seed)
        w, h = img.size
        if min(w,h) < crop_size:
            scale = crop_size / float(min(w, h))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            w, h = img.size
        crop_tensors = []
        for _ in range(patches):
            left = random.randint(0, max(0, w - crop_size))
            top  = random.randint(0, max(0, h - crop_size))
            crop = img.crop((left, top, left + crop_size, top + crop_size))

            x = self.val_tfms(crop)  
            crop_tensors.append(x)

        batch = torch.stack(crop_tensors, dim=0).to(self.device)  # [P,3,H,W]

        probs_batch = self._probs_from_tensor_batch(batch)  # [P,C] na CPU
        probs = probs_batch.mean(dim=0)  # [C] uśrednienie po patchach

        k = min(topk, probs.shape[0])
        top_probs, top_idxs = torch.topk(probs, k=k)

        preds: list[Prediction] = []
        for p, idx in zip(top_probs.tolist(), top_idxs.tolist()):
            label = self.idx_to_class[int(idx)]
            preds.append(Prediction(label=label, prob=float(p)))
        return preds


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Inference for ArborID")

    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory with checkpoint + artifacts")
    parser.add_argument("--image", type=Path, required=True, help="Path to image")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # PATCH VOTING
    parser.add_argument("--patches", type=int, default=0,
                        help="If >1, use random patch voting with this many crops.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for patch sampling")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    image_path = Path(args.image)


    

    device = torch.device("cpu") if args.cpu else None
    run_dir = Path(args.run_dir)
    image_path = Path(args.image)

    engine = InferenceEngine.from_run_dir(run_dir=run_dir, device=device)

    preds = engine.predict_path(image_path, topk=args.topk, patches=args.patches, seed=args.seed)



    print(f"Image: {image_path}")
    for pr in preds:
        print(f"{pr.label:20s}  {pr.prob*100:6.2f}%")


if __name__ == "__main__":
    main()
