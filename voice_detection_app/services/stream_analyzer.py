import numpy as np

from voice_detection_app.models.detector import VoiceDetector
from voice_detection_app.services.audio_processor import AudioProcessor
from voice_detection_app.services.risk_scorer import RiskScorer


class StreamAnalyzer:
    """Analyzes audio streams in segments for real-time detection."""

    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.detector = VoiceDetector()
        self.risk_scorer = RiskScorer()

    def analyze_audio(self, y: np.ndarray, context: dict | None = None) -> dict:
        _, aggregated = self.audio_processor.process_audio(y)
        feature_vector = self.audio_processor.get_feature_vector(aggregated, target_length=64)
        prediction = self.detector.predict(feature_vector)
        risk = self.risk_scorer.compute_risk_score(
            prediction["synthetic_probability"], context
        )
        return {
            "prediction": prediction,
            "risk": risk,
            "features": aggregated,
        }

    def analyze_segments(self, y: np.ndarray, context: dict | None = None) -> dict:
        sr = self.audio_processor.sr
        segment_samples = int(self.audio_processor.segment_duration * sr)
        segment_scores = []
        segment_details = []

        for start in range(0, len(y), segment_samples):
            segment = y[start : start + segment_samples]
            if len(segment) < segment_samples // 2:
                break

            _, agg = self.audio_processor.process_audio(segment)
            fv = self.audio_processor.get_feature_vector(agg, target_length=64)
            pred = self.detector.predict(fv)
            segment_scores.append(pred["synthetic_probability"])
            segment_details.append({
                "start_sec": round(start / sr, 2),
                "synthetic_probability": round(pred["synthetic_probability"], 4),
                "is_synthetic": pred["is_synthetic"],
            })

        streaming_risk = self.risk_scorer.compute_streaming_risk(segment_scores, context)

        return {
            "streaming_risk": streaming_risk,
            "segments": segment_details,
            "total_segments": len(segment_scores),
        }
