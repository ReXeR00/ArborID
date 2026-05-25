from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from src.evaluation.gradcam import GradCAM, GradCAMOutput


@dataclass(frozen=True)
class GradCAMSample:
    path: str
    true_label: str
    pred_label: str
    pred_prob: float
    topk_labels: list[str]
    topk_probs: list[float]

    @property
    def correct(self) -> bool:
        return self.true_label == self.pred_label


class GradCAMViewer:
    def __init__(
        self,
        gradcam: GradCAM,
        samples: list[GradCAMSample],
        output_dir: Path,
    ) -> None:
        if not samples:
            raise ValueError("Grad-CAM viewer needs at least one sample.")

        self.gradcam = gradcam
        self.samples = samples
        self.output_dir = Path(output_dir)
        self.index = 0
        self.cache: dict[int, GradCAMOutput] = {}
        self.saved_indices: set[int] = set()

        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 6))
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.16, wspace=0.04)

        self.prev_button = Button(plt.axes([0.25, 0.04, 0.10, 0.06]), "Previous")
        self.next_button = Button(plt.axes([0.37, 0.04, 0.10, 0.06]), "Next")
        self.save_current_button = Button(plt.axes([0.52, 0.04, 0.14, 0.06]), "Save current")
        self.save_all_button = Button(plt.axes([0.68, 0.04, 0.14, 0.06]), "Save all")

        self.prev_button.on_clicked(self._previous)
        self.next_button.on_clicked(self._next)
        self.save_current_button.on_clicked(self._save_current)
        self.save_all_button.on_clicked(self._save_all)

    def show(self) -> None:
        self._draw()
        plt.show()

    def _previous(self, _event) -> None:
        self.index = (self.index - 1) % len(self.samples)
        self._draw()

    def _next(self, _event) -> None:
        self.index = (self.index + 1) % len(self.samples)
        self._draw()

    def _save_current(self, _event) -> None:
        self.save_sample(self.index)
        self._set_status(f"Saved sample {self.index + 1}/{len(self.samples)}")

    def _save_all(self, _event) -> None:
        for index in range(len(self.samples)):
            self.save_sample(index)
        self._set_status(f"Saved {len(self.samples)} Grad-CAM samples")

    def _get_output(self, index: int) -> GradCAMOutput:
        if index not in self.cache:
            self.cache[index] = self.gradcam.generate(self.samples[index].path)
        return self.cache[index]

    def _draw(self) -> None:
        sample = self.samples[self.index]

        try:
            output = self._get_output(self.index)
        except Exception as exc:
            self._draw_error(sample, exc)
            return

        panels = [
            ("Original", output.original),
            ("Grad-CAM", output.heatmap_rgb),
            ("Overlay", output.overlay),
        ]

        for axis, (title, image) in zip(self.axes, panels):
            axis.clear()
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")

        topk = ", ".join(
            f"{label}: {prob * 100:.1f}%"
            for label, prob in zip(sample.topk_labels, sample.topk_probs)
        )
        correctness = "correct" if sample.correct else "wrong"
        filename = Path(sample.path).name
        self.fig.suptitle(
            "\n".join(
                [
                    f"{self.index + 1}/{len(self.samples)} | {filename}",
                    f"true={sample.true_label} | pred={sample.pred_label} | confidence={sample.pred_prob * 100:.2f}% | {correctness}",
                    f"top-k: {topk}",
                    str(sample.path),
                ]
            ),
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

    def _draw_error(self, sample: GradCAMSample, exc: Exception) -> None:
        for axis in self.axes:
            axis.clear()
            axis.axis("off")
        self.axes[1].text(
            0.5,
            0.5,
            f"Could not generate Grad-CAM for:\n{sample.path}\n\n{exc}",
            ha="center",
            va="center",
            wrap=True,
        )
        self.fig.canvas.draw_idle()

    def _set_status(self, text: str) -> None:
        self.fig.suptitle(text, fontsize=10)
        self.fig.canvas.draw_idle()

    def save_sample(self, index: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self._get_output(index)
        prefix = self.output_dir / f"gradcam_{index + 1:04d}"
        output.original.save(prefix.with_name(prefix.name + "_original.png"))
        output.heatmap_rgb.save(prefix.with_name(prefix.name + "_heatmap.png"))
        output.overlay.save(prefix.with_name(prefix.name + "_overlay.png"))
        self.saved_indices.add(index)
        self._write_metadata()

    def _write_metadata(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output_dir / "gradcam_metadata.csv"
        with metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "index",
                    "path",
                    "true_label",
                    "pred_label",
                    "pred_prob",
                    "correct",
                    "topk_labels",
                    "topk_probs",
                    "original_file",
                    "heatmap_file",
                    "overlay_file",
                ]
            )
            for index in sorted(self.saved_indices):
                sample = self.samples[index]
                prefix = f"gradcam_{index + 1:04d}"
                writer.writerow(
                    [
                        index + 1,
                        sample.path,
                        sample.true_label,
                        sample.pred_label,
                        f"{sample.pred_prob:.6f}",
                        int(sample.correct),
                        "|".join(sample.topk_labels),
                        "|".join(f"{prob:.6f}" for prob in sample.topk_probs),
                        f"{prefix}_original.png",
                        f"{prefix}_heatmap.png",
                        f"{prefix}_overlay.png",
                    ]
                )


def open_gradcam_viewer(
    gradcam: GradCAM,
    samples: list[GradCAMSample],
    output_dir: Path,
) -> None:
    viewer = GradCAMViewer(gradcam=gradcam, samples=samples, output_dir=output_dir)
    viewer.show()
