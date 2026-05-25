from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int | None = None,
) -> Tuple[float, float, float]:

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_targets =[]
    all_preds = []

    
    inferred_num_classes = num_classes

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        if inferred_num_classes is None:
            inferred_num_classes = int(outputs.shape[1])
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    if total > 0 and inferred_num_classes is not None:
        labels = list(range(inferred_num_classes))
        val_macro_f1 = f1_score(
            all_targets,
            all_preds,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    else:
        val_macro_f1 = 0.0

    return avg_loss, acc, val_macro_f1
