# transforms.py
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights

def get_transforms(model_name: str = "resnet50"):
    name = model_name.lower()

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        

        train_tfms = T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.75, 1.33), antialias=True),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            ], p=0.7),
            T.RandomRotation(10),
            T.RandomPerspective(distortion_scale=0.3 ,p=0.2),
            preprocess
        ])
        val_tfms = preprocess
        

    else:
        train_tfms = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor()
        ])
        val_tfms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
    return train_tfms, val_tfms

