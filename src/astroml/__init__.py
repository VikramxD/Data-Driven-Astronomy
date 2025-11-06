"""
AstroML - Modern Machine Learning for Astronomy

A production-ready machine learning framework for galaxy classification,
photometric redshift prediction, and astronomical data analysis.
"""

__version__ = "2.0.0"
__author__ = "Data-Driven Astronomy Contributors"
__license__ = "MIT"

from astroml.models import (
    GalaxyClassifier,
    RedshiftPredictor,
    EnsembleModel,
)

__all__ = [
    "GalaxyClassifier",
    "RedshiftPredictor",
    "EnsembleModel",
]
