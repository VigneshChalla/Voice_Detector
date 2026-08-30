"""Voice Cloning Detection API - Main Application."""
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voice_detection_app.api.routes import router
from voice_detection_app.api.websocket_routes import ws_router
from voice_detection_app.config import settings
from voice_detection_app.edge.export_and_infer import EdgeInferenceEngine
from voice_detection_app.services.privacy import PrivacyComplianceModule
from voice_detection_app.services.speaker.enrollment import CrossSessionConsistency, SpeakerEnrollment

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Cloning Detection API",
    description=(
        "AI-powered real-time detection of voice cloning impersonation attacks. "
        "Features: multi-layer acoustic analysis, WebSocket streaming, "
        "speaker enrollment, cross-session verification, edge inference."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ws_router)

# Serve static files (mobile UI)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

speaker_enrollment = SpeakerEnrollment(enrollment_dir=settings.speaker.enrollment_dir)
cross_session = CrossSessionConsistency(enrollment_manager=speaker_enrollment)
edge_engine = EdgeInferenceEngine(onnx_path=settings.edge.onnx_path)
privacy_module = PrivacyComplianceModule()


@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("Voice Cloning Detection API v1.0.0")
    logger.info("=" * 60)
    logger.info("Edge inference: %s", "READY" if edge_engine.is_available else "UNAVAILABLE")
    logger.info("Speaker enrollments: %d", len(speaker_enrollment.list_speakers()))
    logger.info("Privacy: raw_audio=%s, anonymize=%s",
                settings.privacy.store_raw_audio, settings.privacy.anonymize_caller_id)
    logger.info("Rate limits: %d/min, %d/hour",
                settings.auth.rate_limit_minute, settings.auth.rate_limit_hour)
    logger.info("Server: http://%s:%d", settings.host, settings.port)
    logger.info("Docs: http://%s:%d/docs", settings.host, settings.port)
    logger.info("=" * 60)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/")
async def root():
    return {
        "service": "Voice Cloning Detection",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /api/v1/health",
            "detect": "POST /api/v1/detect",
            "detect_stream": "POST /api/v1/detect/stream",
            "risk_score": "POST /api/v1/risk/score",
            "speaker_enroll": "POST /api/v1/speaker/enroll",
            "speaker_verify": "POST /api/v1/speaker/verify",
            "cross_session": "POST /api/v1/speaker/cross-session",
            "websocket": "ws://host:8000/ws/detect",
            "edge_status": "GET /api/v1/edge/status",
        },
        "features": [
            "multi-layer-acoustic-analysis",
            "real-time-risk-scoring",
            "websocket-streaming",
            "speaker-enrollment",
            "cross-session-verification",
            "edge-inference",
            "privacy-compliance",
            "api-authentication",
            "rate-limiting",
        ],
    }


@app.get("/api/v1/metrics")
async def metrics():
    """Basic metrics endpoint for monitoring."""
    return {
        "enrolled_speakers": len(speaker_enrollment.list_speakers()),
        "edge_inference": edge_engine.is_available,
        "privacy": {
            "store_raw_audio": settings.privacy.store_raw_audio,
            "anonymize_caller_id": settings.privacy.anonymize_caller_id,
        },
    }


@app.get("/mobile", response_class=HTMLResponse)
async def mobile_ui():
    """Serve the mobile-friendly web interface."""
    html_path = static_dir / "mobile.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Mobile UI not found</h1>", status_code=404)


if __name__ == "__main__":
    uvicorn.run(
        "voice_detection_app.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
