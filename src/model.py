from torch import nn
from torchvision.models import resnet50, ResNet50_Weights



def model(num_classes: int, pretrained: bool, freeze_backbone: bool, dropout: float):

    weights = ResNet50_Weights.DEFAULT if pretrained else None

    m  = resnet50(weights=weights)
    

    if freeze_backbone:
        for name, p in m.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    in_feats = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feats, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    return m 

def model_from_cfg(cfg_model, num_classes: int):
    name = cfg_model.name.lower()
    if name == "resnet50":
        return model(
            num_classes=num_classes,
            pretrained=bool(cfg_model.pretrained),
            freeze_backbone=bool(cfg_model.freeze_backbone),
            dropout=float(cfg_model.dropout),
        )
    elif name == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, int(cfg_model.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(cfg_model.dropout)),
            nn.Linear(int(cfg_model.hidden_dim), num_classes),
        )
    else:
        raise ValueError(f"Unknown model name: {name}")