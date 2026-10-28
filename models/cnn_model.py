"""
CNN Model Architecture for Burn Injuries Diagnostic System.
Supports Transfer Learning with ResNet-18 / MobileNetV3 for 4-class burn classification.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES = ["degree_1", "degree_2", "degree_3", "healthy"]

CLASS_DISPLAY = {
    "degree_1": "Degree 1 Burn (Superficial)",
    "degree_2": "Degree 2 Burn (Partial Thickness)",
    "degree_3": "Degree 3 Burn (Full Thickness)",
    "healthy": "Healthy Normal Skin"
}

def get_transforms(is_train: bool = False):
    """
    Returns image transformation pipeline for training and inference.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

class BurnCNN(nn.Module):
    """
    Transfer learning CNN using ResNet18 backbone.
    High accuracy, fast CPU/GPU inference, well-suited for Grad-CAM explainability.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(BurnCNN, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_target_layer(self):
        """
        Returns the final convolutional layer for Grad-CAM computation.
        In ResNet-18, this is layer4[-1].conv2.
        """
        return self.backbone.layer4[-1].conv2
