# train.py

from pathlib import Path
import hashlib

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd


import torch
import torch.nn as nn

from data import get_loader          
from model import model_from_cfg     
from evaluation import evaluate      
from utils import set_seed, setup_amp

import logging
logger = logging.getLogger(__name__)
from hydra.core.hydra_config import HydraConfig


# -------------------------------------------------
# Main training entry point
# -------------------------------------------------
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / "best_model.pt"

    # Basic config printout
    print("Config:\n", OmegaConf.to_yaml(cfg))
    train_cfg = cfg.train
    model_cfg = cfg.model
    optim_cfg = cfg.optimizer
    data_cfg = cfg.data

    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    

    project_root = Path(get_original_cwd())
    data_root = project_root / data_cfg.root



    
    artifacts_dir = run_dir / "artifacts"

    train_loader, val_loader, test_loader, classes = get_loader(
        root=data_root,
        artifacts_dir=artifacts_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    num_classes = len(classes)
    
    print(f"Detected {num_classes} classes: {classes}")

    # Optional override from model config
    if getattr(cfg.model, "num_classes_override", None) is not None:
        num_classes = int(cfg.model.num_classes_override)
        print(f"[INFO] Overriding num_classes to {num_classes} from cfg.model.num_classes_override")

    # -------------------------------------------------
    # Sanity checks
    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    test_ds  = test_loader.dataset

    print(f"[SANITY] Train size: {len(train_loader.dataset)}")
    print(f"[SANITY] Val size:   {len(val_loader.dataset)}")
    print(f"[SANITY] Test size:  {len(test_loader.dataset)}")
    print("[DEBUG] train class_to_idx:", getattr(train_loader.dataset, "class_to_idx", None))
    print("[DEBUG] val   class_to_idx:", getattr(val_loader.dataset, "class_to_idx", None))
    print("[DEBUG] test  class_to_idx:", getattr(test_loader.dataset, "class_to_idx", None))

    print("[DEBUG] mapping equal train==val:", getattr(train_loader.dataset, "class_to_idx", None) == getattr(val_ds, "class_to_idx", None))
    print("[DEBUG] mapping equal train==test:", getattr(train_loader.dataset, "class_to_idx", None) == getattr(test_ds, "class_to_idx", None))
    print("[DEBUG] train transforms:", getattr(train_loader.dataset, "transform", None))
    print("[DEBUG] val transforms:", getattr(val_loader.dataset, "transform", None))


    # Print a few sample file paths from each split
    def _print_samples(loader, split_name, n=5):
        samples = loader.dataset.samples  
        print(f"[SANITY] {split_name} sample paths (first {min(n, len(samples))}):")
        for i in range(min(n, len(samples))):
            p, y = samples[i]
            print(f"  - {p} (class_idx={y})")

    _print_samples(train_loader, "TRAIN", n=5)
    _print_samples(val_loader,   "VAL",   n=5)
    _print_samples(test_loader,  "TEST",  n=5)

    # Check for exact path overlap between splits
    train_paths = {str(p) for p, _ in train_loader.dataset.samples}
    val_paths   = {str(p) for p, _ in val_loader.dataset.samples}
    test_paths  = {str(p) for p, _ in test_loader.dataset.samples}

    overlap_train_val  = train_paths & val_paths
    overlap_train_test = train_paths & test_paths
    overlap_val_test   = val_paths & test_paths

    print(f"[SANITY] Path overlap train∩val:  {len(overlap_train_val)}")
    print(f"[SANITY] Path overlap train∩test: {len(overlap_train_test)}")
    print(f"[SANITY] Path overlap val∩test:   {len(overlap_val_test)}")

    # If overlaps exist, print a few examples
    def _print_overlap(name, s, n=10):
        if len(s) == 0:
            return
        print(f"[SANITY][WARNING] Examples of {name} overlap:")
        for i, p in enumerate(list(s)[:n]):
            print(f"  - {p}")

    _print_overlap("train∩val", overlap_train_val)
    _print_overlap("train∩test", overlap_train_test)
    _print_overlap("val∩test", overlap_val_test)
    # Optional: detect duplicates by file content (hash)

    

    def file_md5(path: str, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def build_hash_map(samples):
        # samples: list of (path, class_idx)
        m = {}
        for p, y in samples:
            hp = file_md5(p)
            m.setdefault(hp, []).append((p, y))
        return m

    train_hashes = build_hash_map(train_loader.dataset.samples)
    val_hashes   = build_hash_map(val_loader.dataset.samples)
    test_hashes  = build_hash_map(test_loader.dataset.samples)

    train_set = set(train_hashes.keys())
    val_set   = set(val_hashes.keys())
    test_set  = set(test_hashes.keys())

    h_overlap_train_val  = train_set & val_set
    h_overlap_train_test = train_set & test_set
    h_overlap_val_test   = val_set & test_set

    print(f"[SANITY] Hash overlap train∩val:  {len(h_overlap_train_val)}")
    print(f"[SANITY] Hash overlap train∩test: {len(h_overlap_train_test)}")
    print(f"[SANITY] Hash overlap val∩test:   {len(h_overlap_val_test)}")

    def _print_hash_overlap(name, overlap_hashes, hm_a, hm_b, n=3):
        if len(overlap_hashes) == 0:
            return
        print(f"[SANITY][WARNING] {name} hash overlap examples:")
        for h in list(overlap_hashes)[:n]:
            print(f"  hash={h}")
            print("   A:", hm_a[h][0])
            print("   B:", hm_b[h][0])

    _print_hash_overlap("train∩val", h_overlap_train_val, train_hashes, val_hashes)
    _print_hash_overlap("train∩test", h_overlap_train_test, train_hashes, test_hashes)
    _print_hash_overlap("val∩test", h_overlap_val_test, val_hashes, test_hashes)

    


    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = model_from_cfg(cfg.model, num_classes=num_classes)
    model.to(device)
    print(f"Model:\n{model}")
    trainable = [(n, p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    print("[DEBUG] trainable count:", len(trainable), "params:", sum(x[1] for x in trainable))
    print("[DEBUG] first trainable:", [n for n,_ in trainable[:20]])

    # -------------------------------------------------
    # Loss, optimizer, AMP
    # -------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    opt_name = str(getattr(optim_cfg, "name", "adamw")).lower()
    lr = float(getattr(optim_cfg, "lr", 3e-4))
    weight_decay = float(getattr(optim_cfg, "weight_decay", 0.0))

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adamw":
        betas = tuple(getattr(optim_cfg, "betas", [0.9, 0.999]))
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=lr, weight_decay=weight_decay, betas=betas
        )
    elif opt_name == "sgd":
        momentum = float(getattr(optim_cfg, "momentum", 0.9))
        nesterov = bool(getattr(optim_cfg, "nesterov", False))
        optimizer = torch.optim.SGD(
            params_to_optimize, lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=nesterov
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    use_amp, scaler, amp_dtype = setup_amp(train_cfg.amp_dtype, device)

    # -------------------------------------------------
    # Train loop
    # -------------------------------------------------
    best_metric = -float("inf")
    metric_name = train_cfg.save_best_by  

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if batch_idx % train_cfg.log_interval == 0:
                avg_loss = running_loss / total if total > 0 else 0.0
                acc = correct / total if total > 0 else 0.0
                print(
                    f"Epoch [{epoch}/{train_cfg.epochs}] "
                    f"Step [{batch_idx}/{len(train_loader)}] "
                    f"Train loss: {avg_loss:.4f}, acc: {acc:.4f}"
                )


        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        
        # -------------------------------------------------
        # Validation
        # -------------------------------------------------
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # ---- epoch summary log (NOW variables exist)
        logger.info("Epoch [%d/%d] Train loss: %.4f, acc: %.4f | Val loss: %.4f, acc: %.4f",
            epoch, train_cfg.epochs, train_loss, train_acc, val_loss, val_acc)

        print(
            f"Epoch [{epoch}/{train_cfg.epochs}] "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if metric_name == "val_acc":
            current_metric = val_acc
        elif metric_name == "val_loss":
            current_metric = -val_loss
        else:
            raise ValueError(f"Unknown save_best_by: {metric_name}")

        if current_metric > best_metric:
            best_metric = current_metric
            
            torch.save(model.state_dict(), ckpt_path)
            print(f"[BEST] Saved new best model to {ckpt_path} ({metric_name}={current_metric:.4f})")

    # -------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[TEST] loss: {test_loss:.4f}, acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
