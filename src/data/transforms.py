# transforms.py
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights

def get_transforms(model_name: str = "resnet50"):
    name = model_name.lower()

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        normalize = T.Normalize(mean=preprocess.mean, std=preprocess.std)


        train_tfms = T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.75, 1.33), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.3 ,p=0.2),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            ], p=0.7),
            T.RandomApply([
                T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ],p=0.15),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
        val_tfms = preprocess
        

    else:
        val_tfms = T.Compose([
            T.Resize((224, 224)), 
            T.ToTensor(),
            normalize
        ])
    return train_tfms, val_tfms

