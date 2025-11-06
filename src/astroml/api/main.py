"""
FastAPI application for AstroML inference service.

Modern REST API with:
- Galaxy classification endpoint
- Redshift prediction endpoint
- Batch processing
- Model versioning
- Health checks
- OpenAPI documentation
"""

from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from astroml.models.base import PredictionResult
from astroml.api.dependencies import get_model_manager


# Pydantic models for API
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
    device: str


class GalaxyClassificationRequest(BaseModel):
    """Request for galaxy classification."""

    image_url: Optional[str] = Field(None, description="URL to galaxy image")
    use_attention: bool = Field(False, description="Return attention visualization")


class GalaxyClassificationResponse(BaseModel):
    """Response for galaxy classification."""

    prediction: str
    confidence: float
    probabilities: dict[str, float]
    attention_map: Optional[List[List[float]]] = None
    model_version: str
    processing_time_ms: float


class RedshiftPredictionRequest(BaseModel):
    """Request for redshift prediction."""

    colors: dict[str, float] = Field(
        ..., description="Galaxy colors: u-g, g-r, r-i, i-z"
    )
    additional_features: Optional[dict[str, float]] = None


class RedshiftPredictionResponse(BaseModel):
    """Response for redshift prediction."""

    redshift: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    model_version: str


class BatchClassificationRequest(BaseModel):
    """Request for batch classification."""

    image_urls: List[str]
    max_batch_size: int = Field(32, description="Maximum batch size for processing")


# Lifespan context manager for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("Starting AstroML API server...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Load models here (placeholder)
    # model_manager.load_models()

    yield

    # Cleanup
    logger.info("Shutting down AstroML API server...")


# Create FastAPI app
app = FastAPI(
    title="AstroML API",
    description="Modern REST API for galaxy classification and redshift prediction",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=True,  # TODO: Check actual model status
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AstroML API",
        "version": "2.0.0",
        "description": "Modern machine learning API for astronomy",
        "docs": "/docs",
        "health": "/health",
    }


# Galaxy classification endpoints
@app.post(
    "/v1/classify/galaxy",
    response_model=GalaxyClassificationResponse,
    tags=["Classification"],
    status_code=status.HTTP_200_OK,
)
async def classify_galaxy_image(
    file: UploadFile = File(..., description="Galaxy image file (JPG, PNG)")
) -> GalaxyClassificationResponse:
    """
    Classify a galaxy image into morphological types.

    Supported types:
    - Elliptical
    - Spiral
    - Irregular
    - Lenticular
    - Merger
    """
    import time

    start_time = time.time()

    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG supported.",
            )

        # Read and process image
        contents = await file.read()

        # TODO: Load image and run prediction
        # prediction = model.predict(image)

        # Placeholder response
        processing_time = (time.time() - start_time) * 1000

        return GalaxyClassificationResponse(
            prediction="Spiral",
            confidence=0.92,
            probabilities={
                "Elliptical": 0.02,
                "Spiral": 0.92,
                "Irregular": 0.03,
                "Lenticular": 0.02,
                "Merger": 0.01,
            },
            model_version="resnet50-v1",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}",
        )


@app.post(
    "/v1/classify/batch",
    response_model=List[GalaxyClassificationResponse],
    tags=["Classification"],
)
async def classify_galaxy_batch(
    request: BatchClassificationRequest,
) -> List[GalaxyClassificationResponse]:
    """Batch classification of multiple galaxies."""
    # TODO: Implement batch processing
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Batch processing not yet implemented",
    )


# Redshift prediction endpoints
@app.post(
    "/v1/predict/redshift",
    response_model=RedshiftPredictionResponse,
    tags=["Redshift"],
)
async def predict_redshift(
    request: RedshiftPredictionRequest,
) -> RedshiftPredictionResponse:
    """
    Predict photometric redshift from galaxy colors.

    Input colors: u-g, g-r, r-i, i-z
    """
    try:
        # Validate colors
        required_colors = ["u-g", "g-r", "r-i", "i-z"]
        if not all(color in request.colors for color in required_colors):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required colors. Need: {required_colors}",
            )

        # TODO: Run prediction
        # prediction = redshift_model.predict(features)

        return RedshiftPredictionResponse(
            redshift=0.045,
            uncertainty=0.003,
            confidence=0.95,
            model_version="dt-depth19-v1",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting redshift: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error predicting redshift: {str(e)}",
        )


# Model information endpoints
@app.get("/v1/models/info", tags=["Models"])
async def get_models_info():
    """Get information about available models."""
    return {
        "classification_models": [
            {
                "name": "resnet50-v1",
                "type": "CNN",
                "accuracy": 0.89,
                "parameters": "25M",
                "inference_time_ms": 15,
            },
            {
                "name": "vit-base-v1",
                "type": "Transformer",
                "accuracy": 0.92,
                "parameters": "86M",
                "inference_time_ms": 45,
            },
        ],
        "redshift_models": [
            {
                "name": "dt-depth19-v1",
                "type": "Decision Tree",
                "median_error": 0.014,
                "inference_time_ms": 1,
            }
        ],
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
