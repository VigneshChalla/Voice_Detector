"""Privacy and compliance module for voice data handling."""
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Data retention configuration."""
    audio_retention_days: int = 0
    feature_retention_days: int = 30
    log_retention_days: int = 90
    enrollment_retention_days: int = 365
    auto_delete_enabled: bool = True


@dataclass
class PrivacyConfig:
    """Privacy and compliance settings."""
    anonymize_caller_id: bool = True
    store_raw_audio: bool = False
    store_features_only: bool = True
    hash_pii: bool = True
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    compliance_log_path: str = "data/compliance_logs"
    audit_enabled: bool = True


class Anonymizer:
    """Anonymizes PII in call metadata."""

    @staticmethod
    def hash_value(value: str, salt: str = "voice_detection_v1") -> str:
        return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]

    @staticmethod
    def anonymize_caller_id(caller_id: str) -> str:
        return Anonymizer.hash_value(caller_id, "caller_salt")

    @staticmethod
    def anonymize_metadata(metadata: dict) -> dict:
        anonymized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                if any(kw in key.lower() for kw in ["phone", "caller", "id", "email", "name"]):
                    anonymized[key] = Anonymizer.hash_value(value)
                else:
                    anonymized[key] = value
            else:
                anonymized[key] = value
        return anonymized

    @staticmethod
    def strip_audio_identifiers(audio: np.ndarray) -> np.ndarray:
        """Remove potential speaker-identifying segments (first/last 0.5s)."""
        sr = 16000
        trim_samples = int(0.5 * sr)
        if len(audio) > trim_samples * 4:
            return audio[trim_samples:-trim_samples]
        return audio


class FeatureOnlyLogger:
    """Logs features instead of raw audio for privacy."""

    def __init__(self, log_dir: str = "data/feature_logs", max_size_mb: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb

    def log_detection(
        self,
        features: dict,
        risk_score: float,
        risk_level: str,
        caller_id_hash: str = "",
        metadata: dict | None = None,
    ):
        import json

        entry = {
            "timestamp": time.time(),
            "caller_id_hash": caller_id_hash,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "features_summary": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in features.items()
                if k.endswith("_mean") or k.endswith("_std")
            },
            "metadata": Anonymizer.anonymize_metadata(metadata or {}),
        }

        log_file = self.log_dir / f"features_{time.strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._cleanup_old_logs()

    def _cleanup_old_logs(self):
        total_size = sum(f.stat().st_size for f in self.log_dir.glob("*.jsonl"))
        if total_size > self.max_size_mb * 1024 * 1024:
            files = sorted(self.log_dir.glob("*.jsonl"))
            for f in files[:-5]:
                f.unlink()
                logger.info("Deleted old log: %s", f.name)


class ComplianceAuditor:
    """Tracks and audits data access for compliance."""

    def __init__(self, audit_dir: str = "data/audit_logs"):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def log_access(self, action: str, user: str, resource: str, details: dict | None = None):
        import json

        entry = {
            "timestamp": time.time(),
            "action": action,
            "user": user,
            "resource": resource,
            "details": details or {},
        }

        log_file = self.audit_dir / f"audit_{time.strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_data_deletion(self, resource_type: str, count: int, reason: str = "retention_policy"):
        self.log_access(
            action="data_deletion",
            user="system",
            resource=resource_type,
            details={"count": count, "reason": reason},
        )

    def log_consent(self, user_id: str, consent_type: str, granted: bool):
        self.log_access(
            action="consent_update",
            user=user_id,
            resource="privacy_consent",
            details={"consent_type": consent_type, "granted": granted},
        )


class DataRetentionPolicy:
    """Enforces data retention policies automatically."""

    def __init__(self, config: PrivacyConfig | None = None):
        self.config = config or PrivacyConfig()

    def enforce_retention(self):
        """Delete data older than retention period."""
        deleted = 0

        if self.config.audit_enabled:
            deleted += self._cleanup_dir(
                "data/feature_logs",
                self.config.retention.feature_retention_days,
            )

        deleted += self._cleanup_dir(
            "data/compliance_logs",
            self.config.retention.log_retention_days,
        )

        deleted += self._cleanup_dir(
            "data/audit_logs",
            self.config.retention.log_retention_days,
        )

        if deleted > 0:
            logger.info("Retention policy: deleted %d old files", deleted)

        return deleted

    def _cleanup_dir(self, dir_path: str, max_days: int) -> int:
        path = Path(dir_path)
        if not path.exists():
            return 0

        deleted = 0
        cutoff = time.time() - (max_days * 86400)

        for f in path.rglob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1

        return deleted


class PrivacyComplianceModule:
    """Main privacy and compliance interface."""

    def __init__(self, config: PrivacyConfig | None = None):
        self.config = config or PrivacyConfig()
        self.anonymizer = Anonymizer()
        self.feature_logger = FeatureOnlyLogger()
        self.auditor = ComplianceAuditor()
        self.retention = DataRetentionPolicy(self.config)

    def process_detection_request(
        self,
        audio: np.ndarray | None = None,
        caller_id: str = "",
        metadata: dict | None = None,
    ) -> dict:
        """Process a detection request with privacy protections."""
        result = {
            "caller_id_hash": "",
            "audio_stored": False,
            "features_logged": False,
            "anonymized_metadata": {},
        }

        if caller_id:
            result["caller_id_hash"] = self.anonymizer.anonymize_caller_id(caller_id)

        if metadata:
            result["anonymized_metadata"] = self.anonymizer.anonymize_metadata(metadata)

        if audio is not None and not self.config.store_raw_audio:
            audio = None

        self.auditor.log_access(
            action="detection_request",
            user=caller_id or "anonymous",
            resource="voice_detection",
            details={"has_audio": audio is not None},
        )

        return result

    def log_result(
        self,
        features: dict,
        risk_score: float,
        risk_level: str,
        caller_id: str = "",
        metadata: dict | None = None,
    ):
        """Log detection result (features only, no raw audio)."""
        caller_hash = self.anonymizer.anonymize_caller_id(caller_id) if caller_id else ""
        self.feature_logger.log_detection(
            features=features,
            risk_score=risk_score,
            risk_level=risk_level,
            caller_id_hash=caller_hash,
            metadata=metadata,
        )

    def run_maintenance(self):
        """Run retention policy cleanup."""
        return self.retention.enforce_retention()
