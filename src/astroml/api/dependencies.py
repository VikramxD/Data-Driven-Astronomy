"""Dependency injection for FastAPI."""

from typing import Optional

from fastapi import Depends, HTTPException, status


class ModelManager:
    """Singleton model manager for loading and caching models."""

    _instance: Optional["ModelManager"] = None
    _models: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_name: str, model_path: str):
        """Load a model into memory."""
        # TODO: Implement model loading
        pass

    def get_model(self, model_name: str):
        """Get a loaded model."""
        if model_name not in self._models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found",
            )
        return self._models[model_name]


def get_model_manager() -> ModelManager:
    """Dependency to get model manager."""
    return ModelManager()
