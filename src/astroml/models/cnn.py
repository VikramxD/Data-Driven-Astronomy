"""CNN models for galaxy classification using modern architectures."""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from astroml.models.base import GalaxyClassifier, ModelConfig, PredictionResult


class GalaxyCNNConfig(ModelConfig):
    """Configuration for CNN-based galaxy classifier."""

    backbone: str = "resnet50"  # resnet50, efficientnet_b0, convnext_tiny
    num_classes: int = 5
    input_size: tuple[int, int, int] = (3, 224, 224)
    dropout: float = 0.5
    freeze_backbone: bool = False


class GalaxyCNN(GalaxyClassifier):
    """
    Modern CNN for galaxy classification.

    Supports multiple backbones:
    - ResNet50 (default)
    - EfficientNet-B0
    - ConvNeXt-Tiny
    """

    def __init__(self, config: GalaxyCNNConfig):
        super().__init__(config)
        self.config: GalaxyCNNConfig = config

        # Select backbone
        if config.backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif config.backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif config.backbone == "convnext_tiny":
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout / 2),
            nn.Linear(512, config.num_classes),
        )

        # Grad-CAM hook for attention visualization
        self.gradients = None
        self.activations = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> PredictionResult:
        """Predict galaxy type with confidence."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()

            # Add batch dimension if needed
            if x.ndim == 3:
                x = x.unsqueeze(0)

            x = x.to(self.device)
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        return PredictionResult(
            prediction=self.classes[prediction.item()],
            confidence=confidence.item(),
            probabilities={
                class_name: prob.item()
                for class_name, prob in zip(self.classes, probs[0])
            },
            metadata={
                "model": self.config.backbone,
                "device": str(self.device),
            },
        )

    def predict_with_attention(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> tuple[PredictionResult, np.ndarray]:
        """
        Predict with Grad-CAM attention visualization.

        Returns:
            Prediction result and attention map (H, W)
        """
        # TODO: Implement Grad-CAM
        result = self.predict(x)
        attention_map = np.zeros((224, 224))  # Placeholder
        return result, attention_map

    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """Unfreeze backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class SimpleGalaxyCNN(GalaxyClassifier):
    """Simple CNN for quick experiments."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.num_classes or 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        output = self.classifier(features)
        return output

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> PredictionResult:
        """Predict galaxy type."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if x.ndim == 3:
                x = x.unsqueeze(0)

            x = x.to(self.device)
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        return PredictionResult(
            prediction=self.classes[prediction.item()],
            confidence=confidence.item(),
            probabilities={
                class_name: prob.item()
                for class_name, prob in zip(self.classes, probs[0])
            },
        )

    def predict_with_attention(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> tuple[PredictionResult, np.ndarray]:
        result = self.predict(x)
        attention_map = np.zeros((224, 224))
        return result, attention_map
