"""Application configuration with environment variable support."""
import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    val = os.environ.get(key)
    return int(val) if val else default


def _env_float(key: str, default: float = 0.0) -> float:
    val = os.environ.get(key)
    return float(val) if val else default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").lower()
    return val in ("true", "1", "yes") if val else default


def _env_list(key: str, default: str = "") -> list[str]:
    val = os.environ.get(key, default)
    return [x.strip() for x in val.split(",") if x.strip()] if val else []


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    max_duration_sec: float = 30.0
    segment_duration_sec: float = 3.0


@dataclass
class ModelConfig:
    input_features: int = 192
    hidden_sizes: list = field(default_factory=lambda: [512, 512, 256, 256, 128, 64])
    num_classes: int = 2
    dropout: float = 0.3
    model_path: str = "voice_detection_app/trained_model_v3.pth"
    scaler_path: str = "voice_detection_app/scaler_v3.npz"


@dataclass
class RiskConfig:
    high_risk_threshold: float = 0.75
    medium_risk_threshold: float = 0.45
    low_risk_threshold: float = 0.2
    context_multipliers: dict = field(default_factory=lambda: {
        "high_value_transaction": 1.3,
        "privileged_access": 1.25,
        "regular_call": 1.0,
    })


@dataclass
class AlertConfig:
    enabled: bool = True
    channels: list = field(default_factory=lambda: ["ui", "log"])
    webhook_url: str = ""
    email_recipients: list = field(default_factory=list)


@dataclass
class SpeakerConfig:
    enrollment_dir: str = "data/enrollments"
    verification_threshold: float = 0.55
    max_enrollment_samples: int = 10
    cosine_weight: float = 0.35
    zscore_weight: float = 0.25
    euclidean_weight: float = 0.20
    mahalanobis_weight: float = 0.20


@dataclass
class EdgeConfig:
    onnx_path: str = "voice_detection_app/trained_model_v3.onnx"
    torchscript_path: str = "voice_detection_app/trained_model_v3.pt"
    quantized_path: str = "voice_detection_app/trained_model_v3_quantized.pt"
    opset_version: int = 14
    enable_quantization: bool = True


@dataclass
class StreamingConfig:
    segment_duration_sec: float = 3.0
    max_concurrent_sessions: int = 100
    session_timeout_sec: float = 600.0


@dataclass
class AuthConfig:
    api_key: str = "vd_dev_key_2024"
    rate_limit_minute: int = 60
    rate_limit_hour: int = 1000
    rate_limit_burst: int = 10


@dataclass
class PrivacyConfig:
    store_raw_audio: bool = False
    anonymize_caller_id: bool = True
    feature_retention_days: int = 30
    log_retention_days: int = 90


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


def load_config() -> AppConfig:
    """Load configuration from environment variables with defaults."""
    return AppConfig(
        model=ModelConfig(
            model_path=_env("VD_MODEL_PATH", "voice_detection_app/trained_model_v3.pth"),
        ),
        risk=RiskConfig(
            high_risk_threshold=_env_float("VD_HIGH_RISK_THRESHOLD", 0.75),
            medium_risk_threshold=_env_float("VD_MEDIUM_RISK_THRESHOLD", 0.45),
            low_risk_threshold=_env_float("VD_LOW_RISK_THRESHOLD", 0.2),
        ),
        alert=AlertConfig(
            webhook_url=_env("VD_WEBHOOK_URL"),
            email_recipients=_env_list("VD_EMAIL_RECIPIENTS"),
        ),
        auth=AuthConfig(
            api_key=_env("VD_API_KEY", "vd_dev_key_2024"),
            rate_limit_minute=_env_int("VD_RATE_LIMIT_MINUTE", 60),
            rate_limit_hour=_env_int("VD_RATE_LIMIT_HOUR", 1000),
        ),
        privacy=PrivacyConfig(
            store_raw_audio=_env_bool("VD_STORE_RAW_AUDIO", False),
            anonymize_caller_id=_env_bool("VD_ANONYMIZE_CALLER_ID", True),
            feature_retention_days=_env_int("VD_FEATURE_RETENTION_DAYS", 30),
            log_retention_days=_env_int("VD_LOG_RETENTION_DAYS", 90),
        ),
        host=_env("VD_HOST", "0.0.0.0"),
        port=_env_int("VD_PORT", 8000),
        debug=_env_bool("VD_DEBUG", False),
    )


settings = load_config()
