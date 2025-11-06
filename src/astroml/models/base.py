"""Base classes for astronomical machine learning models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Base configuration for models."""

    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model (classifier/regressor)")
    num_classes: Optional[int] = Field(None, description="Number of classes for classification")
    input_size: tuple[int, ...] = Field(..., description="Input tensor size")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu")
    pretrained: bool = Field(False, description="Use pretrained weights")

    class Config:
        arbitrary_types_allowed = True


class PredictionResult(BaseModel):
    """Standard prediction result format."""

    prediction: Union[int, float, str]
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    features: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseModel(ABC, nn.Module):
    """Base class for all astronomical ML models."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    @abstractmethod
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> PredictionResult:
        """Make predictions on input data."""
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config.dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        config = ModelConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def to_device(self, device: Optional[str] = None) -> "BaseModel":
        """Move model to device."""
        device = device or self.config.device
        self.to(device)
        return self


class GalaxyClassifier(BaseModel):
    """Base class for galaxy classification models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.classes = ["Elliptical", "Spiral", "Irregular", "Lenticular", "Merger"]

    @abstractmethod
    def predict_with_attention(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> tuple[PredictionResult, np.ndarray]:
        """Predict with attention visualization."""
        pass


class RedshiftPredictor(BaseModel):
    """Base class for redshift prediction models."""

    @abstractmethod
    def predict_with_uncertainty(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> tuple[float, float]:
        """Predict redshift with uncertainty estimate."""
        pass


class EnsembleModel(BaseModel):
    """Ensemble of multiple models for improved predictions."""

    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        # Don't call super().__init__ as we don't need ModelConfig for ensemble
        nn.Module.__init__(self)
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average of model predictions."""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model(x)
            predictions.append(pred * weight)
        return torch.stack(predictions).sum(dim=0)

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> PredictionResult:
        """Ensemble prediction."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(self.models[0].device)
            output = self(x)

        # Aggregate predictions from all models
        all_results = [model.predict(x) for model in self.models]

        # Simple voting/averaging for now
        return all_results[0]  # TODO: Implement proper aggregation
