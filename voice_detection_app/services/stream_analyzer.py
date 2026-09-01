import numpy as np

from voice_detection_app.models.detector import VoiceDetector
from voice_detection_app.services.audio_processor import AudioProcessor
from voice_detection_app.services.forensic_analyzer import analyze_forensic, hybrid_score
from voice_detection_app.services.risk_scorer import RiskScorer


class StreamAnalyzer:
    """Analyzes audio streams in segments for real-time detection."""

    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.detector = VoiceDetector()
        self.risk_scorer = RiskScorer()

    def analyze_audio(self, y: np.ndarray, context: dict | None = None) -> dict:
        feature_vector, metadata = self.audio_processor.process_audio(y)
        prediction = self.detector.predict(feature_vector)
        ml_prob = float(prediction["synthetic_probability"])

        try:
            forensic = analyze_forensic(y)
            forensic_score = float(forensic["forensic_score"])
        except:
            forensic = {"forensic_score": ml_prob, "factors": {}, "dominant_clues": [], "human_similarity": (1-ml_prob)*100, "ai_similarity": ml_prob*100}
            forensic_score = ml_prob

        hybrid = hybrid_score(ml_prob, forensic_score)
        risk = self.risk_scorer.compute_risk_score(hybrid["final_synthetic_prob"], context)

        return {
            "prediction": {"synthetic_probability": hybrid["final_synthetic_prob"], "genuine_probability": hybrid["final_genuine_prob"], "is_synthetic": hybrid["is_synthetic"]},
            "risk": risk,
            "ml_probability": ml_prob,
            "forensic_score": forensic_score,
            "forensic": forensic,
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

            feature_vector, _ = self.audio_processor.process_audio(segment)
            pred = self.detector.predict(feature_vector)
            ml_prob = float(pred["synthetic_probability"])

            try:
                foren = analyze_forensic(segment, sr)
                fs = float(foren["forensic_score"])
            except:
                foren = {"factors": {}, "dominant_clues": []}
                fs = ml_prob

            hyb = hybrid_score(ml_prob, fs)
            final_prob = float(hyb["final_synthetic_prob"])

            segment_scores.append(final_prob)
            segment_details.append({
                "start_sec": round(start / sr, 2),
                "synthetic_probability": round(final_prob, 4),
                "ml_probability": round(ml_prob, 4),
                "forensic_score": round(fs, 4),
                "is_synthetic": hyb["is_synthetic"],
                "dominant_clues": foren.get("dominant_clues", []),
            })

        streaming_risk = self.risk_scorer.compute_streaming_risk(segment_scores, context)

        return {
            "streaming_risk": streaming_risk,
            "segments": segment_details,
            "total_segments": len(segment_scores),
        }
