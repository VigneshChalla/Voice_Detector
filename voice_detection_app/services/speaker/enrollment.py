"""Speaker enrollment and cross-session voiceprint consistency checks."""
import json
import logging
import time
from pathlib import Path

import numpy as np

from voice_detection_app.config import settings
from voice_detection_app.services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class SpeakerEnrollment:
    """Manages speaker voiceprint enrollment and storage."""

    def __init__(self, enrollment_dir: str = "data/enrollments"):
        self.audio_processor = AudioProcessor()
        self.enrollment_dir = Path(enrollment_dir)
        self.enrollment_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        for path in self.enrollment_dir.glob("*.json"):
            speaker_id = path.stem
            with open(path) as f:
                self._cache[speaker_id] = json.load(f)
        logger.info("Loaded %d speaker enrollments", len(self._cache))

    def enroll(
        self,
        speaker_id: str,
        audio_bytes: bytes,
        label: str = "",
    ) -> dict:
        y, sr = self.audio_processor.load_audio_from_bytes(audio_bytes)
        if len(y) == 0:
            raise ValueError("No audio data for enrollment")

        _, aggregated = self.audio_processor.process_audio(y)
        feature_vector = self.audio_processor.get_feature_vector(aggregated, target_length=64)

        profile = {
            "speaker_id": speaker_id,
            "label": label,
            "enrolled_at": time.time(),
            "num_samples": 1,
            "feature_vector_mean": feature_vector.tolist(),
            "feature_vector_std": np.zeros_like(feature_vector).tolist(),
            "feature_vectors": [feature_vector.tolist()],
        }

        self._cache[speaker_id] = profile
        self._save(speaker_id, profile)
        logger.info("Enrolled speaker: %s", speaker_id)
        return self._sanitize(profile)

    def add_sample(self, speaker_id: str, audio_bytes: bytes) -> dict:
        if speaker_id not in self._cache:
            raise KeyError(f"Speaker {speaker_id} not enrolled")

        y, sr = self.audio_processor.load_audio_from_bytes(audio_bytes)
        _, aggregated = self.audio_processor.process_audio(y)
        feature_vector = self.audio_processor.get_feature_vector(aggregated, target_length=64)

        profile = self._cache[speaker_id]
        profile["feature_vectors"].append(feature_vector.tolist())
        profile["num_samples"] = len(profile["feature_vectors"])

        vectors = np.array(profile["feature_vectors"])
        profile["feature_vector_mean"] = np.mean(vectors, axis=0).tolist()
        profile["feature_vector_std"] = np.std(vectors, axis=0).tolist()

        if profile["num_samples"] > 10:
            profile["feature_vectors"] = profile["feature_vectors"][-10:]

        self._save(speaker_id, profile)
        logger.info("Added sample for speaker %s (total: %d)", speaker_id, profile["num_samples"])
        return self._sanitize(profile)

    def get_profile(self, speaker_id: str) -> dict | None:
        profile = self._cache.get(speaker_id)
        return self._sanitize(profile) if profile else None

    def list_speakers(self) -> list[str]:
        return list(self._cache.keys())

    def delete(self, speaker_id: str) -> bool:
        if speaker_id in self._cache:
            del self._cache[speaker_id]
            path = self.enrollment_dir / f"{speaker_id}.json"
            path.unlink(missing_ok=True)
            logger.info("Deleted enrollment for %s", speaker_id)
            return True
        return False

    def _save(self, speaker_id: str, profile: dict):
        path = self.enrollment_dir / f"{speaker_id}.json"
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)

    def _sanitize(self, profile: dict) -> dict:
        return {
            "speaker_id": profile["speaker_id"],
            "label": profile["label"],
            "enrolled_at": profile["enrolled_at"],
            "num_samples": profile["num_samples"],
        }


class CrossSessionConsistency:
    """Compares ongoing call features against enrolled voiceprints."""

    def __init__(self, enrollment_manager: SpeakerEnrollment | None = None):
        self.enrollment = enrollment_manager or SpeakerEnrollment()
        self.audio_processor = AudioProcessor()

    def verify_speaker(
        self,
        speaker_id: str,
        audio_bytes: bytes,
    ) -> dict:
        profile = self.enrollment._cache.get(speaker_id)
        if not profile:
            return {
                "verified": False,
                "reason": "speaker_not_enrolled",
                "similarity_score": 0.0,
            }

        y, sr = self.audio_processor.load_audio_from_bytes(audio_bytes)
        _, aggregated = self.audio_processor.process_audio(y)
        feature_vector = self.audio_processor.get_feature_vector(aggregated, target_length=64)

        enrolled_mean = np.array(profile["feature_vector_mean"])
        enrolled_std = np.array(profile["feature_vector_std"])
        enrolled_std = np.maximum(enrolled_std, 1e-6)

        cosine_sim = self._cosine_similarity(feature_vector, enrolled_mean)
        z_score = np.mean(np.abs((feature_vector - enrolled_mean) / enrolled_std))
        euclidean_dist = float(np.linalg.norm(feature_vector - enrolled_mean))
        mahalanobis = self._modified_mahalanobis(feature_vector, enrolled_mean, enrolled_std)

        combined_score = (
            0.35 * cosine_sim
            + 0.25 * max(0, 1.0 - z_score / 3.0)
            + 0.20 * max(0, 1.0 - euclidean_dist / 5.0)
            + 0.20 * max(0, 1.0 - mahalanobis / 3.0)
        )

        threshold = 0.55
        verified = combined_score >= threshold

        anomaly_flags = []
        if cosine_sim < 0.5:
            anomaly_flags.append("low_spectral_similarity")
        if z_score > 3.0:
            anomaly_flags.append("high_feature_deviation")
        if euclidean_dist > 4.0:
            anomaly_flags.append("large_spectral_distance")

        return {
            "verified": verified,
            "similarity_score": round(float(combined_score), 4),
            "cosine_similarity": round(float(cosine_sim), 4),
            "feature_deviation_zscore": round(float(z_score), 4),
            "euclidean_distance": round(euclidean_dist, 4),
            "mahalanobis_distance": round(float(mahalanobis), 4),
            "anomaly_flags": anomaly_flags,
            "enrolled_samples": profile["num_samples"],
            "threshold": threshold,
        }

    def check_cross_session_anomaly(
        self,
        speaker_id: str,
        audio_bytes: bytes,
        historical_call_ids: list[str] | None = None,
    ) -> dict:
        verification = self.verify_speaker(speaker_id, audio_bytes)

        result = {
            "speaker_id": speaker_id,
            "verification": verification,
            "historical_call_ids": historical_call_ids or [],
            "cross_session_assessment": "consistent",
        }

        if not verification["verified"]:
            result["cross_session_assessment"] = "anomaly_detected"
            result["alert"] = {
                "level": "HIGH" if verification["similarity_score"] < 0.3 else "MEDIUM",
                "message": (
                    f"Voice mismatch detected for speaker {speaker_id}. "
                    f"Similarity: {verification['similarity_score']:.2%}. "
                    "Recommend secondary verification."
                ),
            }

        return result

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _modified_mahalanobis(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        z = (x - mean) / std
        return float(np.sqrt(np.sum(z ** 2) / len(z)))
