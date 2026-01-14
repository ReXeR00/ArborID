from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import torch
from PIL import Image
from omegaconf import OmegaConf, DictConfig

from model import model_from_cfg
from transforms import get_transforms
import torchvision.transforms as T

@dataclass(frozen=True)
class Prediction:
    label: str
    prob: float


class InferenceEngine:
    def __init__(self, model: torch.nn.Module, idx_to_class: dict[int, str], device: torch.device):
        self.model = model
        self.idx_to_class = idx_to_class
        self.device = device

        # stable input pipeline for inference
        _, self.val_tfms = get_transforms()

    @staticmethod
    def _load_class_map(class_to_idx_path: Path) -> dict[int, str]:
        with open(class_to_idx_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)

        # class_to_idx expected: {"oak": 0, "spruce": 1, ...}
        return {int(v): str(k) for k, v in class_to_idx.items()}

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
                }
            )
        return model_cfg

    @staticmethod
    def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
        obj = torch.load(checkpoint_path, map_location=device)
        print("[DEBUG] checkpoint type:", type(obj))
        if isinstance(obj, dict):
            keys = list(obj.keys())
            print("[DEBUG] checkpoint keys sample:", keys[:20])






        # handle both formats:
        # 1) state_dict directly
        # 2) checkpoint dict with key "model"/"state_dict"
        if isinstance(obj, dict):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return obj["state_dict"]
            if "model" in obj and isinstance(obj["model"], dict):
                return obj["model"]

        # assume it's already a state_dict-like mapping
        if isinstance(obj, dict):
            return obj  # type: ignore[return-value]

        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")



    @classmethod
    def from_paths(
        cls,
        checkpoint_path: Path,
        class_to_idx_path: Path,
        config_path: Path,
        device: torch.device | None = None,
    ) -> "InferenceEngine":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not class_to_idx_path.exists():
            raise FileNotFoundError(f"class_to_idx not found: {class_to_idx_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        cfg: DictConfig = OmegaConf.load(config_path)
        model_cfg = cls._pick_model_cfg(cfg)

        idx_to_class = cls._load_class_map(class_to_idx_path)
        num_classes = len(idx_to_class)

        net = model_from_cfg(model_cfg, num_classes=num_classes)
        state_dict = cls._load_state_dict(checkpoint_path, device=device)

        
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch. Missing={missing[:10]}, unexpected={unexpected[:10]}")
        net.to(device)
        net.eval()
        print("[DEBUG] missing:", missing[:20], "…", len(missing))
        print("[DEBUG] unexpected:", unexpected[:20], "…", len(unexpected))
        print("[DEBUG] has fc.weight:", "fc.weight" in state_dict)
        print("[DEBUG] has fc.bias  :", "fc.bias" in state_dict)

        # 1) klucze checkpointu, które wyglądają na head
        head_keys = [k for k in state_dict.keys() if any(s in k for s in ["fc", "classifier", "head", "module.fc", "linear"])]
        print("[DEBUG] ckpt head-like keys:", head_keys[:50])
        
        # 2) klucze modelu (to co model oczekuje)
        model_keys = list(net.state_dict().keys())
        model_head_keys = [k for k in model_keys if any(s in k for s in ["fc", "classifier", "head", "linear"])]
        print("[DEBUG] model head-like keys:", model_head_keys[:50])

        return cls(model=net, idx_to_class=idx_to_class, device=device)

    def predict_pil(self, img: Image.Image, topk: int = 3) -> list[Prediction]:
        if topk < 1:
            raise ValueError("topk must be >= 1")

        img = img.convert("RGB")
        x0 = T.ToTensor()(img)
        print("[DEBUG] toTensor min/max:", float(x0.min()), float(x0.max()))
        x = self.val_tfms(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        print("[DEBUG] val_tfms min/max:", float(x.min()), float(x.max()))


        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

            k = min(topk, probs.shape[1])
            top_probs, top_idxs = torch.topk(probs, k=k, dim=1)

        top_probs = top_probs.squeeze(0).detach().cpu().tolist()
        top_idxs = top_idxs.squeeze(0).detach().cpu().tolist()

        preds: list[Prediction] = []
        for p, idx in zip(top_probs, top_idxs):
            label = self.idx_to_class[int(idx)]
            preds.append(Prediction(label=label, prob=float(p)))
        return preds

    def predict_path(self, image_path: Path, topk: int = 3) -> list[Prediction]:
        img = Image.open(image_path).convert("RGB")
        return self.predict_pil(img, topk=topk)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    checkpoint = project_root / "best_model.pt"
    class_map = project_root / "artifacts" / "class_to_idx.json"
    config = project_root / "configs" / "config.yaml"

    engine = InferenceEngine.from_paths(
        checkpoint_path=checkpoint,
        class_to_idx_path=class_map,
        config_path=config,
    )

    test_img = project_root / "data" / "raw" / "test" / "oak" / "0.jpg"
    preds = engine.predict_path(test_img, topk=3)

    print(f"Image: {test_img}")
    for pr in preds:
        print(f"{pr.label:20s}  {pr.prob*100:6.2f}%")
