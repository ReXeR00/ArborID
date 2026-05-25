# ArborID

Explainable tree bark classification with PyTorch, ResNet and Grad-CAM.

ArborID is a Computer Vision project for classifying tree species from bark images.
It combines a PyTorch training pipeline, ResNet-based image classification, folder-based evaluation, multi-patch inference, confidence analysis and Grad-CAM visual explanations.

The current version is evaluated on a 5-class subset of the BarkNet dataset.

## Latest Verified Result

| Metric | Result |
|---|---:|
| Top-1 Accuracy | 87.70% |
| Top-3 Accuracy | 99.18% |
| Number of classes | 5 |
| Evaluation split | `src/data/testdata` |
| Model checkpoint | `best_model.pt` |

Current class mapping:

```json
{
  "chr": 0,
  "epb": 1,
  "epn": 2,
  "epo": 3,
  "epr": 4
}
```

These results come from the current 5-class setup. The next major research step is to test whether the same pipeline generalizes to the full BarkNet dataset with more visually similar tree species.

## Features

- PyTorch-based training pipeline
- ResNet50 classifier with a custom classification head
- Hydra-based configuration
- Reproducible training runs
- Best-checkpoint saving based on monitored validation metric
- Macro-F1, Top-1 and Top-k evaluation
- Weighted sampling and optional weighted loss
- Single-image inference
- Folder-based evaluation
- Multi-patch inference / patch voting
- Worst confident mistake export
- Grad-CAM visual explanation viewer
- CSV exports for evaluation and error analysis

## Why This Project Exists

Tree bark classification is difficult because bark texture changes with lighting, weather, tree age, camera type, distance, shadows, damage and species similarity.

A useful model should not only return a class prediction. It should also provide:

- reproducible experiments
- clear evaluation metrics
- confidence information
- error analysis
- visual explanations
- stable training, inference and evaluation scripts

ArborID is built as a practical Machine Learning Engineering pipeline around those requirements.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run default training:

```bash
python src/trainer/train.py
```

Example with Hydra overrides:

```bash
python src/trainer/train.py optimizer=adamw data.use_weighted_sampler=true data.use_class_weights=false
```

Training creates a Hydra output directory:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
|-- best_model.pt
|-- .hydra/
|   `-- config.yaml
`-- artifacts/
    |-- history.json
    |-- class_to_idx.json
    `-- dataset_info.json
```

The most important file is:

```text
best_model.pt
```

This checkpoint is used later by inference, evaluation and Grad-CAM.

## Evaluation

Evaluate a trained model on a folder with class subdirectories:

```bash
python src/evaluation/eval_folder.py \
  --run_dir outputs/YYYY-MM-DD/HH-MM-SS \
  --data_dir src/data/testdata \
  --patches 1 \
  --crop_size 224 \
  --topk 3
```

Evaluation prints:

- Top-1 accuracy
- Top-k accuracy
- uncertainty rate
- per-class accuracy
- confusion matrix

It also saves CSV files under `eval/`:

```text
eval/<run-id>-p<patches>/
|-- eval_folder.csv
`-- eval_worst_confident.csv
```

## Multi-Patch Inference

Patch voting can improve robustness by averaging predictions across random crops:

```bash
python src/evaluation/eval_folder.py \
  --run_dir outputs/YYYY-MM-DD/HH-MM-SS \
  --data_dir src/data/testdata \
  --patches 20 \
  --crop_size 224 \
  --topk 3
```

Use this when full-image predictions are unstable or when bark texture occupies only part of the image.

## Grad-CAM Viewer

ArborID includes a Grad-CAM viewer for visual model explanations.

Run evaluation and open the viewer afterward:

```bash
python src/evaluation/eval_folder.py \
  --run_dir outputs/YYYY-MM-DD/HH-MM-SS \
  --data_dir src/data/testdata \
  --patches 1 \
  --crop_size 224 \
  --topk 3 \
  --show_gradcam \
  --gradcam_limit 30
```

Show only wrong predictions:

```bash
python src/evaluation/eval_folder.py \
  --run_dir outputs/YYYY-MM-DD/HH-MM-SS \
  --data_dir src/data/testdata \
  --patches 1 \
  --crop_size 224 \
  --topk 3 \
  --show_gradcam \
  --gradcam_mode wrong
```

Available Grad-CAM modes:

```text
all
wrong
correct
worst_confident
```

The viewer displays:

- original image
- Grad-CAM heatmap
- overlay image
- true label
- predicted label
- confidence
- top-k predictions
- image path

Saved Grad-CAM outputs are written to:

```text
<run_dir>/gradcam/
|-- gradcam_0001_original.png
|-- gradcam_0001_heatmap.png
|-- gradcam_0001_overlay.png
`-- gradcam_metadata.csv
```

## How Grad-CAM Works Here

For ResNet models, ArborID uses the last convolutional block as the default Grad-CAM target layer:

```python
model.layer4[-1]
```

The process is:

1. Run a forward pass through the CNN.
2. Select the predicted class score.
3. Backpropagate that score.
4. Capture activations from the target convolutional layer.
5. Capture gradients from the same layer.
6. Average gradients spatially to create channel weights.
7. Weight the activations.
8. Sum the weighted activations.
9. Apply ReLU.
10. Normalize the heatmap.
11. Resize it to the original image.
12. Overlay it on the RGB bark image.

This helps verify whether the model focuses on bark texture or gets distracted by background, lighting, shadows or non-bark regions.

## Single-Image Inference

Run inference on one image:

```bash
python src/trainer/infer.py \
  --run_dir outputs/YYYY-MM-DD/HH-MM-SS \
  --image src/data/testdata/CHR/example.jpg
```

Force CPU inference:

```bash
python src/trainer/infer.py \
  --run_dir outputs/YYYY-MM-DD/HH-MM-SS \
  --image src/data/testdata/CHR/example.jpg \
  --cpu
```

Use patch voting for a single image:

```bash
python src/trainer/infer.py \
  --run_dir outputs/YYYY-MM-DD/HH-MM-SS \
  --image src/data/testdata/CHR/example.jpg \
  --patches 20 \
  --crop_size 224
```

## Dataset

The current project uses a 5-class BarkNet subset.

Expected class folders:

```text
CHR
EPB
EPN
EPO
EPR
```

The project supports two dataset layouts.

Pre-split layout:

```text
data/
|-- train/
|   |-- CHR/
|   |-- EPB/
|   |-- EPN/
|   |-- EPO/
|   `-- EPR/
|-- validate/
|   |-- CHR/
|   |-- EPB/
|   |-- EPN/
|   |-- EPO/
|   `-- EPR/
`-- test/
    |-- CHR/
    |-- EPB/
    |-- EPN/
    |-- EPO/
    `-- EPR/
```

Single-root layout:

```text
data/
|-- CHR/
|-- EPB/
|-- EPN/
|-- EPO/
`-- EPR/
```

In the single-root layout, ArborID creates train, validation and test splits internally.

## Model

The main model is a ResNet50 image classifier.

Pipeline:

```text
image -> ResNet backbone -> classification head -> logits -> softmax probabilities
```

The classification head is adapted to the number of classes detected in the dataset.

The model configuration supports:

- pretrained ResNet weights
- optional backbone freezing
- custom dropout
- configurable number of classes
- configurable input size
- simple MLP baseline

## Project Structure

```text
ArborID/
|-- src/
|   |-- checkpoints.py
|   |-- configs/
|   |-- data/
|   |   |-- loader.py
|   |   |-- prep_data.py
|   |   `-- transforms.py
|   |-- evaluation/
|   |   |-- evaluation.py
|   |   |-- eval_folder.py
|   |   |-- gradcam.py
|   |   |-- gradcam_viewer.py
|   |   `-- visualization.py
|   |-- model/
|   |   `-- model.py
|   `-- trainer/
|       |-- train.py
|       |-- infer.py
|       |-- early_stopping.py
|       `-- utils.py
|-- eval/
|-- outputs/
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Reproducibility

Each run stores:

- resolved Hydra configuration
- best model checkpoint
- class mapping
- dataset metadata
- training history
- evaluation CSVs
- optional Grad-CAM outputs

This makes it easier to compare experiments and debug model behavior.

## Current Status

Completed:

- 5-class BarkNet classification pipeline
- training pipeline
- inference script
- folder evaluation
- shared checkpoint loading
- patch voting
- weighted sampling
- optional weighted loss
- Grad-CAM viewer
- CSV result export

Planned:

- full BarkNet training
- architecture comparison
- stronger augmentation experiments
- confusion matrix visualization
- training curve visualization
- Grad-CAM examples in README
- FastAPI inference endpoint
- simple web demo
- deployment-ready inference workflow

## Author

Created by Emil Nowak.

GitHub: https://github.com/ReXeR00
