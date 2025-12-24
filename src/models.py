import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN2D(nn.Module):
    """A compact 2D CNN for 2.5D CT slice-triplets (3 channels).
    This is intentionally lightweight to run on CPUs and modest GPUs.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

@torch.no_grad()
def aggregate_logits(model, volume_batch_slices, device):
    """Optional helper if you want volume-level aggregation:
    pass multiple slices from the same volume and average logits.
    Not used in default training loop.
    """
    model.eval()
    logits = []
    for x in volume_batch_slices:
        x = x.to(device)
        logits.append(model(x))
    return torch.stack(logits, dim=0).mean(dim=0)
