from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import json
from transforms import get_transforms


def get_loader( root: Path, artifacts_dir: Path, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = True):

    root = Path(root)
    train_tfms, val_tfms = get_transforms()

    train_ds = ImageFolder(Path(root)/"train",    transform=train_tfms)
    val_ds   = ImageFolder(Path(root)/"validate", transform=val_tfms)
    test_ds  = ImageFolder(Path(root)/"test",     transform=val_tfms)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(train_ds.class_to_idx, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, train_ds.classes



