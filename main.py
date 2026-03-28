"""
main.py
=======
FastAPI server — Full Crop Disease Intelligence Pipeline

Endpoints:
  POST /predict          — accepts leaf image, returns full intelligence report + stores to DB
  GET  /health           — health check
  GET  /classes          — list all 38 supported disease classes
  GET  /docs             — auto-generated Swagger UI (FastAPI built-in)
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Load .env before anything else ────────────────────────────────
load_dotenv()

# ── Internal modules ───────────────────────────────────────────────
from model_loader import get_model, CLASS_NAMES, PlantDiseaseModel
from intelligence_engine import IntelligenceEngine
from db_client import get_db

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN  (load model once at boot)
# ══════════════════════════════════════════════════════════════════

_model: Optional[PlantDiseaseModel] = None
_engine: Optional[IntelligenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _engine
    logger.info("🚀 Starting Crop Disease Intelligence Pipeline...")

    model_path     = os.getenv("MODEL_PATH", "mobilenetv2_plant.pth")
    protocols_path = os.getenv("PROTOCOLS_PATH", "disease_protocols.json")
    weather_key    = os.getenv("OPENWEATHER_API_KEY", "")

    try:
        _model  = get_model(model_path)
        _engine = IntelligenceEngine(
            protocols_path=protocols_path,
            weather_api_key=weather_key,
        )
        logger.info("✅ Pipeline ready.")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield  # ← server runs here

    logger.info("🛑 Shutting down...")


# ══════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="🌿 Crop Disease Intelligence API",
    description=(
        "Multi-layer AI pipeline for crop disease detection.\n\n"
        "**Layer 1** — MobileNetV2 DL model (38 disease classes)\n"
        "**Layer 2** — Intelligence Engine (severity, remedy, weather, economics)\n"
        "**DB**      — Supabase `disease_scans` table\n\n"
        "Upload a leaf image and get a full agronomic intelligence report."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════

class LocationModel(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None

class MarketplaceModel(BaseModel):
    recommended_products: list[str]
    product_type: str
    note: str

class PredictionResponse(BaseModel):
    # Layer 1
    disease:        str
    disease_key:    str
    crop:           str
    confidence:     float = Field(description="Confidence in %")
    confidence_raw: float = Field(description="Raw softmax probability (0–1)")

    # Validation engine outputs
    severity:       str
    severity_level: str
    is_positive:    bool = Field(description="True = diseased, False = healthy")
    disease_type:   str  = Field(description="fungal | bacterial | viral | pest | healthy | unknown")

    # Layer 2 intelligence
    first_aid:      str
    action_plan:    list[str]
    weather_advice: str

    # Economic
    yield_loss_pct:         Optional[float]
    economic_loss_rs:       Optional[float]
    economic_loss_per_acre: Optional[float]

    # Routing
    marketplace:    MarketplaceModel

    # Meta
    location:             LocationModel
    timestamp:            str
    top_k_predictions:    list[dict]

    # DB result
    db_insert_id:   Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    """Server and model health check."""
    return {
        "status": "ok",
        "model_loaded":  _model is not None,
        "engine_loaded": _engine is not None,
        "db_connected":  get_db().is_connected,
    }


@app.get("/classes", tags=["Model Info"])
async def list_classes():
    """Returns all 38 supported disease/healthy class names."""
    return {
        "total": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Upload a leaf image and get the full intelligence report",
)
async def predict(
    file:            UploadFile = File(..., description="Leaf image (JPG/PNG)"),
    latitude:        Optional[float] = Form(None, description="GPS latitude"),
    longitude:       Optional[float] = Form(None, description="GPS longitude"),
    crop_area_acres: float           = Form(1.0,  description="Farm area in acres"),
    market_price:    float           = Form(1500.0, description="Current mandi price ₹/quintal"),
    top_k:           int             = Form(3,    description="Number of top predictions"),
):
    """
    ## Full Pipeline
    1. Upload leaf image
    2. MobileNetV2 runs inference → disease class + confidence
    3. IntelligenceEngine enriches result → severity, remedy, weather, economics
    4. ValidationEngine ensures all fields are consistent
    5. Result stored in Supabase `disease_scans` table
    6. Full JSON response returned

    ### Form Fields
    - **file**: Leaf image file (JPEG or PNG)
    - **latitude / longitude**: GPS coordinates (optional, for weather advice)
    - **crop_area_acres**: Size of farm plot (default: 1.0)
    - **market_price**: Current mandi price in ₹/quintal (default: 1500)
    - **top_k**: Number of top disease predictions (default: 3)
    """
    if _model is None or _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is still starting up.")

    # ── Validate file type ─────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG/WebP accepted.",
        )

    # ── Read image ────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # ── Layer 1: Model inference ──────────────────────────────────
    try:
        disease_key, confidence, top_k_results = _model.predict(image_bytes, top_k=top_k)
        logger.info(f"🔍 Prediction: {disease_key} ({confidence*100:.1f}%)")
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # ── Layer 2: Intelligence Engine + Validation ─────────────────
    location = (latitude, longitude) if latitude is not None and longitude is not None else None
    try:
        result = _engine.analyze(
            disease_key=disease_key,
            confidence=confidence,
            location=location,
            crop_area_acres=crop_area_acres,
            market_price_rs_per_quintal=market_price,
            top_k_predictions=top_k_results,
        )
    except Exception as e:
        logger.error(f"Intelligence engine failed: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence processing failed: {e}")

    # ── DB Insert ─────────────────────────────────────────────────
    db_id = None
    try:
        db = get_db()
        db_record = db.insert_scan(result)
        if db_record and not db_record.get("dry_run"):
            db_id = db_record.get("id")
    except Exception as e:
        logger.warning(f"DB insert failed (non-fatal): {e}")

    result["db_insert_id"] = db_id

    return JSONResponse(content=result)


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
