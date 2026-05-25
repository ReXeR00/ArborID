from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


TensorTransform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class GradCAMOutput:
    original: Image.Image
    heatmap: np.ndarray
    heatmap_rgb: Image.Image
    overlay: Image.Image
    target_class_idx: int
    predicted_class_idx: int
    predicted_prob: float


def get_default_resnet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "layer4"):
        layer4 = getattr(model, "layer4")
        try:
            return layer4[-1]
        except (TypeError, IndexError) as exc:
            raise ValueError("Grad-CAM target layer must be configured manually; model.layer4[-1] is unavailable.") from exc

    raise ValueError("Grad-CAM target layer must be configured manually; this model has no layer4 attribute.")


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0)
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - min_value) / (max_value - min_value)).astype(np.float32)


def colorize_heatmap(heatmap: np.ndarray, colormap: str = "jet") -> Image.Image:
    cmap = cm.get_cmap(colormap)
    colored = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    return Image.fromarray((colored * 255).astype(np.uint8), mode="RGB")


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    resized = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def overlay_heatmap(original: Image.Image, heatmap_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    original = original.convert("RGB")
    heatmap_rgb = heatmap_rgb.resize(original.size, resample=Image.BILINEAR).convert("RGB")
    return Image.blend(original, heatmap_rgb, alpha=alpha)


class GradCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        transform: TensorTransform,
        device: torch.device,
        target_layer: torch.nn.Module | None = None,
        overlay_alpha: float = 0.45,
    ) -> None:
        self.model = model
        self.transform = transform
        self.device = device
        self.target_layer = target_layer if target_layer is not None else get_default_resnet_target_layer(model)
        self.overlay_alpha = overlay_alpha
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hooks.append(self.target_layer.register_forward_hook(self._forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def generate(
        self,
        image_path: str | Path,
        target_class_idx: int | None = None,
    ) -> GradCAMOutput:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Grad-CAM image not found: {image_path}")

        with Image.open(image_path) as image_file:
            original = image_file.convert("RGB")

        input_tensor = self.transform(original).unsqueeze(0).to(self.device)
        input_tensor = input_tensor.float()

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            predicted_class_idx = int(probs.argmax(dim=1).item())
            class_idx = predicted_class_idx if target_class_idx is None else int(target_class_idx)
            score = logits[:, class_idx].sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        heatmap = self._build_heatmap(self.activations, self.gradients)
        heatmap = resize_heatmap(heatmap, original.size)
        heatmap_rgb = colorize_heatmap(heatmap)
        overlay = overlay_heatmap(original, heatmap_rgb, alpha=self.overlay_alpha)

        return GradCAMOutput(
            original=original,
            heatmap=heatmap,
            heatmap_rgb=heatmap_rgb,
            overlay=overlay,
            target_class_idx=class_idx,
            predicted_class_idx=predicted_class_idx,
            predicted_prob=float(probs[0, predicted_class_idx].detach().cpu().item()),
        )

    @staticmethod
    def _build_heatmap(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).detach().cpu().numpy()
        return normalize_heatmap(cam)
