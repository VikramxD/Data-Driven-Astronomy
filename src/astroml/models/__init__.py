"""Modern machine learning models for astronomy."""

from astroml.models.classical import (
    DecisionTreeClassifier as ClassicalGalaxyClassifier,
    DecisionTreeRegressor as ClassicalRedshiftPredictor,
)
from astroml.models.base import GalaxyClassifier, RedshiftPredictor, EnsembleModel

__all__ = [
    "GalaxyClassifier",
    "RedshiftPredictor",
    "EnsembleModel",
    "ClassicalGalaxyClassifier",
    "ClassicalRedshiftPredictor",
]
