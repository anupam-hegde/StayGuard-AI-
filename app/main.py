import os
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    ReservationFeatures,
    BatchReservationRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)
from config.paths_config import MODEL_OUTPUT_PATH
from src.logger import get_logger

# Configure structured application logger
logger = get_logger("app.main")


def determine_risk_level(cancellation_prob: float) -> str:
    """
    Categorizes cancellation probability into actionable risk tiers.
    """
    if cancellation_prob >= 0.70:
        return "High"
    elif cancellation_prob >= 0.40:
        return "Medium"
    return "Low"


# ==============================================================================
# 1. APPLICATION LIFESPAN (MODEL LOADING & RESOURCE LIFECYCLE)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan manager that safely loads the serialized Scikit-Learn
    pipeline artifact during application startup and handles graceful shutdown.
    """
    logger.info("Initializing FastAPI Application Lifespan")
    model_path = os.getenv("MODEL_OUTPUT_PATH", MODEL_OUTPUT_PATH)

    if os.path.exists(model_path):
        try:
            logger.info(f"Loading unified ML model pipeline from: {model_path}")
            app.state.pipeline = joblib.load(model_path)
            app.state.model_loaded = True
            app.state.model_path = model_path
            logger.info("ML model pipeline successfully loaded into application state")
        except Exception as e:
            logger.error(f"Failed to deserialize model pipeline artifact: {e}")
            app.state.pipeline = None
            app.state.model_loaded = False
            app.state.model_path = model_path
    else:
        logger.warning(
            f"Model artifact not found at {model_path}. Endpoints will return 503 until model is trained."
        )
        app.state.pipeline = None
        app.state.model_loaded = False
        app.state.model_path = model_path

    yield

    logger.info("FastAPI Application Lifespan tearing down resources")


# ==============================================================================
# 2. FASTAPI APPLICATION INITIALIZATION & MIDDLEWARE
# ==============================================================================

app = FastAPI(
    title="StayGuard AI - Hotel Reservation Cancellation API",
    description="Enterprise REST API for real-time and batch hotel booking cancellation risk scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """
    HTTP middleware providing:
    - Unique X-Request-ID propagation
    - Latency benchmarking (X-Process-Time)
    - Structured JSON request/response audit logging
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    # Log incoming request event
    log_context = {
        "event": "http_request_received",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host if request.client else "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(json.dumps(log_context))

    try:
        response: Response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.6f}"

        # Log request completion
        completion_context = {
            "event": "http_request_completed",
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_seconds": round(process_time, 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(json.dumps(completion_context))
        return response

    except Exception as exc:
        process_time = time.perf_counter() - start_time
        error_context = {
            "event": "http_request_unhandled_exception",
            "request_id": request_id,
            "error": str(exc),
            "duration_seconds": round(process_time, 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(json.dumps(error_context))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )


# ==============================================================================
# 3. ENDPOINTS
# ==============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint returning service identity and documentation links.
    """
    return {
        "service": "StayGuard AI - Hotel Reservation Cancellation Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Observability"],
)
async def health_check():
    """
    Readiness & liveness health check endpoint detailing model availability.
    """
    is_ready = getattr(app.state, "model_loaded", False)
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_path=getattr(app.state, "model_path", "unknown"),
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_single(features: ReservationFeatures):
    """
    Validates payload and evaluates cancellation risk for a single hotel reservation.
    """
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline artifact is not loaded. Please train or deploy the pipeline.",
        )

    try:
        # Convert Pydantic model to DataFrame matching pipeline feature names
        input_data = features.model_dump()
        input_df = pd.DataFrame([input_data])

        # Run inference through the unified Scikit-Learn pipeline
        pred = int(pipeline.predict(input_df)[0])
        prob = float(pipeline.predict_proba(input_df)[0][1])
        risk = determine_risk_level(prob)

        return PredictionResponse(
            prediction=pred,
            cancellation_probability=round(prob, 4),
            status_label="Canceled" if pred == 1 else "Not_Canceled",
            risk_level=risk,
        )

    except Exception as e:
        logger.error(f"Inference error during single prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_batch(request_payload: BatchReservationRequest):
    """
    Performs vectorized batch inference across a list of reservation records.
    """
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline artifact is not loaded. Please train or deploy the pipeline.",
        )

    try:
        # Vectorized conversion to DataFrame
        records = [item.model_dump() for item in request_payload.items]
        batch_df = pd.DataFrame(records)

        # Batch prediction
        predictions = pipeline.predict(batch_df)
        probabilities = pipeline.predict_proba(batch_df)[:, 1]

        results = []
        for pred, prob in zip(predictions, probabilities):
            p = int(pred)
            pr = float(prob)
            results.append(
                PredictionResponse(
                    prediction=p,
                    cancellation_probability=round(pr, 4),
                    status_label="Canceled" if p == 1 else "Not_Canceled",
                    risk_level=determine_risk_level(pr),
                )
            )

        return BatchPredictionResponse(
            total_records=len(results),
            predictions=results,
        )

    except Exception as e:
        logger.error(f"Inference error during batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
