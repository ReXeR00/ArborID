# ArborID

Tree bark classification project built with PyTorch and Hydra.

## What is included

- training with `ResNet50` or a simple `MLP`
- dataset preparation with tree-level splits
- weighted sampling and optional weighted loss
- validation and test evaluation with macro-F1
- single-image inference and folder evaluation
- Hydra run directories with saved configs and checkpoints

## Project layout

```text
ArborID/
|-- src/
|   |-- checkpoints.py
|   |-- configs/
|   |-- data/
|   |-- evaluation/
|   |-- model/
|   `-- trainer/
|-- eval/
|-- outputs/
`-- requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Default training run:

```bash
python src/trainer/train.py
```

Example Hydra override:

```bash
python src/trainer/train.py optimizer=adamw data.use_weighted_sampler=true
```

## Inference

Single image:

```bash
python src/trainer/infer.py --run_dir outputs/2026-01-01/12-00-00 --image path/to/image.jpg
```

Patch-voting inference:

```bash
python src/trainer/infer.py --run_dir outputs/2026-01-01/12-00-00 --image path/to/image.jpg --patches 20 --crop_size 224
```

## Folder evaluation

```bash
python src/evaluation/eval_folder.py --run_dir outputs/2026-01-01/12-00-00 --data_dir src/data/testdata --patches 20
```

## Notes

- `best_model.pt` stores the best checkpoint according to `train.save_best_by`.
- `class_to_idx.json` and dataset metadata are saved inside each Hydra run.
- `data.use_weighted_sampler` and `data.use_class_weights` are configurable separately.
- expensive duplicate-image hash checks are disabled by default and can be enabled through `train.hash_data_checks=true`.
