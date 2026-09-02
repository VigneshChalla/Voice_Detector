"""API routes for voice cloning detection."""
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from voice_detection_app.config import settings
from voice_detection_app.models.detector import VoiceDetector
from voice_detection_app.services.alert_service import AlertService
from voice_detection_app.services.auth import APIKeyConfig, APIAuthMiddleware, auth_middleware
from voice_detection_app.services.audio_processor import AudioProcessor
from voice_detection_app.services.forensic_analyzer import analyze_forensic, hybrid_score
from voice_detection_app.services.privacy import PrivacyComplianceModule
from voice_detection_app.services.risk_scorer import RiskScorer
from voice_detection_app.services.speaker.enrollment import CrossSessionConsistency, SpeakerEnrollment
from voice_detection_app.services.stream_analyzer import StreamAnalyzer

router = APIRouter(prefix="/api/v1", tags=["voice-detection"])

audio_processor = AudioProcessor()
detector = VoiceDetector()
risk_scorer = RiskScorer()
alert_service = AlertService()
stream_analyzer = StreamAnalyzer()
speaker_enrollment = SpeakerEnrollment(enrollment_dir=settings.speaker.enrollment_dir)
cross_session = CrossSessionConsistency(enrollment_manager=speaker_enrollment)
privacy_module = PrivacyComplianceModule()


# --- Response Models ---

class DetectionResponse(BaseModel):
    synthetic_probability: float
    genuine_probability: float
    is_synthetic: bool
    risk_score: float
    risk_level: str
    recommendation: str
    requires_secondary_verification: bool
    should_block_transaction: bool
    caller_id_hash: str = ""
    # Hybrid forensic details
    ml_probability: float = 0.0
    forensic_score: float = 0.0
    final_synthetic_percent: float = 0.0
    final_human_percent: float = 0.0
    human_similarity: float = 0.0
    ai_similarity: float = 0.0
    forensic_factors: dict = {}
    dominant_clues: list = []
    agreement: str = ""
    confidence: float = 0.0
    analysis_summary: str = ""


class StreamingAnalysisResponse(BaseModel):
    streaming_risk: dict
    segments: list
    total_segments: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    edge_inference_available: bool
    enrolled_speakers: int
    privacy_compliance: bool


class EnrollmentResponse(BaseModel):
    speaker_id: str
    label: str
    enrolled_at: float
    num_samples: int


class VerificationResponse(BaseModel):
    verified: bool
    similarity_score: float
    cosine_similarity: float
    feature_deviation_zscore: float
    euclidean_distance: float
    mahalanobis_distance: float
    anomaly_flags: list[str]
    enrolled_samples: int
    threshold: float


class CrossSessionResponse(BaseModel):
    speaker_id: str
    verification: dict
    historical_call_ids: list[str]
    cross_session_assessment: str
    alert: dict | None = None


class APIKeyResponse(BaseModel):
    key: str
    name: str
    tier: str
    rate_limit: dict


# --- Health ---

@router.get("/health", response_model=HealthResponse)
async def health_check():
    from voice_detection_app.app import edge_engine
    return HealthResponse(
        status="ok",
        model_loaded=detector.is_trained,
        edge_inference_available=edge_engine.is_available,
        enrolled_speakers=len(speaker_enrollment.list_speakers()),
        privacy_compliance=True,
    )


# --- Detection ---

@router.post("/detect", response_model=DetectionResponse)
async def detect_voice(
    request: Request,
    file: UploadFile = File(...),
    caller_id: str = "",
    call_type: str = "regular_call",
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    privacy_result = privacy_module.process_detection_request(
        caller_id=caller_id,
        metadata={"call_type": call_type, "file_name": file.filename},
    )

    try:
        y, sr = audio_processor.load_audio_from_bytes(audio_bytes, content_type=file.content_type or "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")

    if len(y) == 0:
        raise HTTPException(status_code=400, detail="No audio data extracted")

    feature_vector, metadata = audio_processor.process_audio(y)
    prediction = detector.predict(feature_vector)
    ml_prob = float(prediction["synthetic_probability"])

    # Forensic multi-factor analysis (frequency stability, energy, spectral, etc.)
    try:
        forensic = analyze_forensic(y, sr)
        forensic_score = float(forensic["forensic_score"])
    except Exception as fe:
        forensic = {"forensic_score": ml_prob, "factors": {}, "dominant_clues": [], "human_similarity": (1-ml_prob)*100, "ai_similarity": ml_prob*100}
        forensic_score = ml_prob

    # Hybrid: combine ML + forensic
    hybrid = hybrid_score(ml_prob, forensic_score, ml_weight=0.70)
    final_prob = float(hybrid["final_synthetic_prob"])

    context = {"caller_id": caller_id, "call_type": call_type}
    risk = risk_scorer.compute_risk_score(final_prob, context)

    privacy_module.log_result(
        features=metadata,
        risk_score=risk["risk_score"],
        risk_level=risk["risk_level"],
        caller_id=caller_id,
        metadata={"call_type": call_type, "ml_prob": ml_prob, "forensic_score": forensic_score, "final_prob": final_prob},
    )

    if risk["risk_level"] in ("HIGH", "MEDIUM"):
        alert_service.create_alert(risk, {"caller_id": caller_id})

    # Build human-readable summary
    if hybrid["is_synthetic"]:
        if hybrid["confidence"] > 60:
            summary = f"Strong AI indicators ({hybrid['final_synthetic_percent']}% AI). " + "; ".join([c["interpretation"] for c in forensic.get("dominant_clues", [])[:2]])
        else:
            summary = f"Likely AI ({hybrid['final_synthetic_percent']}% AI) - moderate confidence. Top clue: {forensic.get('dominant_clues', [{}])[0].get('interpretation','') if forensic.get('dominant_clues') else ''}"
    else:
        if hybrid["confidence"] > 60:
            summary = f"Strong human indicators ({hybrid['final_human_percent']}% human). " + "; ".join([c["interpretation"] for c in forensic.get("dominant_clues", [])[:2]])
        else:
            summary = f"Likely human ({hybrid['final_human_percent']}% human) - moderate confidence."

    return DetectionResponse(
        synthetic_probability=final_prob,
        genuine_probability=1.0 - final_prob,
        is_synthetic=hybrid["is_synthetic"],
        risk_score=risk["risk_score"],
        risk_level=risk["risk_level"],
        recommendation=risk["recommendation"],
        requires_secondary_verification=risk["requires_secondary_verification"],
        should_block_transaction=risk["should_block_transaction"],
        caller_id_hash=privacy_result["caller_id_hash"],
        ml_probability=round(ml_prob, 4),
        forensic_score=round(forensic_score, 4),
        final_synthetic_percent=hybrid["final_synthetic_percent"],
        final_human_percent=hybrid["final_human_percent"],
        human_similarity=round(forensic.get("human_similarity", (1-final_prob)*100), 1),
        ai_similarity=round(forensic.get("ai_similarity", final_prob*100), 1),
        forensic_factors=forensic.get("factors", {}),
        dominant_clues=forensic.get("dominant_clues", []),
        agreement=hybrid.get("agreement", ""),
        confidence=hybrid.get("confidence", 0),
        analysis_summary=summary,
    )


@router.post("/detect/stream", response_model=StreamingAnalysisResponse)
async def detect_voice_streaming(
    request: Request,
    file: UploadFile = File(...),
    caller_id: str = "",
    call_type: str = "regular_call",
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    audio_bytes = await file.read()
    try:
        y, sr = audio_processor.load_audio_from_bytes(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")

    if len(y) == 0:
        raise HTTPException(status_code=400, detail="No audio data extracted")

    context = {"caller_id": caller_id, "call_type": call_type}
    result = stream_analyzer.analyze_segments(y, context)

    if result["streaming_risk"]["risk_level"] in ("HIGH", "MEDIUM"):
        alert_service.create_alert(result["streaming_risk"], {"caller_id": caller_id})

    return StreamingAnalysisResponse(
        streaming_risk=result["streaming_risk"],
        segments=result["segments"],
        total_segments=result["total_segments"],
    )


@router.post("/risk/score")
async def compute_risk_score(
    synthetic_probability: float,
    call_type: str = "regular_call",
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    context = {"call_type": call_type}
    return risk_scorer.compute_risk_score(synthetic_probability, context)


# --- Speaker Management ---

@router.post("/speaker/enroll", response_model=EnrollmentResponse)
async def enroll_speaker(
    request: Request,
    file: UploadFile = File(...),
    speaker_id: str = "",
    label: str = "",
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id is required")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        profile = speaker_enrollment.enroll(speaker_id, audio_bytes, label)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Enrollment failed: {e}")

    privacy_module.auditor.log_access(
        action="speaker_enroll",
        user=speaker_id,
        resource="speaker_enrollment",
    )

    return EnrollmentResponse(**profile)


@router.post("/speaker/add-sample", response_model=EnrollmentResponse)
async def add_speaker_sample(
    file: UploadFile = File(...),
    speaker_id: str = "",
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id is required")

    audio_bytes = await file.read()
    try:
        profile = speaker_enrollment.add_sample(speaker_id, audio_bytes)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed: {e}")

    return EnrollmentResponse(**profile)


@router.get("/speaker/list")
async def list_speakers(
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    return {"speakers": speaker_enrollment.list_speakers()}


@router.get("/speaker/{speaker_id}")
async def get_speaker(
    speaker_id: str,
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    profile = speaker_enrollment.get_profile(speaker_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return profile


@router.delete("/speaker/{speaker_id}")
async def delete_speaker(
    speaker_id: str,
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    deleted = speaker_enrollment.delete(speaker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Speaker not found")

    privacy_module.auditor.log_access(
        action="speaker_delete",
        user=speaker_id,
        resource="speaker_enrollment",
    )

    return {"deleted": True, "speaker_id": speaker_id}


@router.post("/speaker/verify", response_model=VerificationResponse)
async def verify_speaker(
    file: UploadFile = File(...),
    speaker_id: str = "",
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id is required")

    audio_bytes = await file.read()
    try:
        result = cross_session.verify_speaker(speaker_id, audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Verification failed: {e}")

    return VerificationResponse(**result)


@router.post("/speaker/cross-session", response_model=CrossSessionResponse)
async def cross_session_check(
    file: UploadFile = File(...),
    speaker_id: str = "",
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id is required")

    audio_bytes = await file.read()
    try:
        result = cross_session.check_cross_session_anomaly(speaker_id, audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Check failed: {e}")

    return CrossSessionResponse(**result)


# --- Edge Inference ---

@router.get("/edge/status")
async def edge_inference_status(
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    from voice_detection_app.app import edge_engine
    return {
        "available": edge_engine.is_available,
        "onnx_path": settings.edge.onnx_path,
    }


# --- API Key Management (Admin) ---

@router.post("/admin/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    name: str = "",
    tier: str = "standard",
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    from voice_detection_app.services.auth import RateLimitConfig
    key = auth_middleware.key_manager.generate_key(name, tier)
    config = auth_middleware.key_manager.validate_key(key)

    return APIKeyResponse(
        key=key,
        name=config.name,
        tier=config.tier,
        rate_limit={
            "per_minute": config.rate_limit.requests_per_minute,
            "per_hour": config.rate_limit.requests_per_hour,
        },
    )


@router.get("/admin/api-keys")
async def list_api_keys(
    api_key_config: APIKeyConfig = Depends(auth_middleware.authenticate),
):
    keys = []
    for key, config in auth_middleware.key_manager.keys.items():
        keys.append({
            "name": config.name,
            "tier": config.tier,
            "enabled": config.enabled,
            "key_preview": key[:8] + "...",
        })
    return {"keys": keys}
