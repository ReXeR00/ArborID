from __future__ import annotations

from torchvision import transforms as T
from torchvision.models import ResNet50_Weights


def get_transforms(model_name: str = "resnet50", img_size: int = 224):
    name = model_name.lower()
    resize_size = max(img_size + 32, int(round(img_size / 0.875)))

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        mean = weights.meta["mean"] if "mean" in weights.meta else (0.485, 0.456, 0.406)
        std = weights.meta["std"] if "std" in weights.meta else (0.229, 0.224, 0.225)
    elif name == "mlp":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Unsupported model name for transforms: {model_name}")

    normalize = T.Normalize(mean=mean, std=std)

    train_tfms = T.Compose([
        T.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.75, 1.33), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomPerspective(distortion_scale=0.3, p=0.2),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ], p=0.7),
        T.RandomApply([
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        ], p=0.15),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    val_tfms = T.Compose([
        T.Resize(resize_size, antialias=True),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])

    return train_tfms, val_tfms
