from torchvision.models import ResNet18_Weights
from torchvision.transforms import v2 as T
import torch

def get_transforms(img_size: int = 224):
    weights = ResNet18_Weights.DEFAULT
    mean, std = weights.meta["mean"], weights.meta["std"]

    train_tfms = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std),
    ])
    # walidacja/test dokładnie jak w pretrainie
    val_tfms = weights.transforms()
    return train_tfms, val_tfms
