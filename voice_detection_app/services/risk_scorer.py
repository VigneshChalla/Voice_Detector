from voice_detection_app.config import settings


class RiskScorer:
    """Computes a dynamic impersonation risk score with contextual enrichment."""

    def __init__(self):
        self.high_threshold = settings.risk.high_risk_threshold
        self.medium_threshold = settings.risk.medium_risk_threshold
        self.low_threshold = settings.risk.low_risk_threshold
        self.context_multipliers = settings.risk.context_multipliers

    def compute_risk_score(
        self,
        synthetic_probability: float,
        context: dict | None = None,
    ) -> dict:
        context = context or {}
        call_type = context.get("call_type", "regular_call")
        multiplier = self.context_multipliers.get(call_type, 1.0)

        base_score = synthetic_probability * multiplier
        risk_score = min(base_score, 1.0)

        if risk_score >= self.high_threshold:
            risk_level = "HIGH"
            recommendation = (
                "Immediate secondary verification required. "
                "Recommend callback to verified number or in-person confirmation."
            )
        elif risk_score >= self.medium_threshold:
            risk_level = "MEDIUM"
            recommendation = (
                "Additional verification recommended. "
                "Consider callback or multi-factor authentication."
            )
        elif risk_score >= self.low_threshold:
            risk_level = "LOW"
            recommendation = "Voice appears likely genuine. Standard monitoring applies."
        else:
            risk_level = "MINIMAL"
            recommendation = "No anomalies detected. Proceed with normal workflow."

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "synthetic_probability": round(synthetic_probability, 4),
            "context_multiplier": multiplier,
            "call_type": call_type,
            "recommendation": recommendation,
            "requires_secondary_verification": risk_level in ("HIGH", "MEDIUM"),
            "should_block_transaction": risk_level == "HIGH",
        }

    def compute_streaming_risk(
        self,
        segment_scores: list[float],
        context: dict | None = None,
    ) -> dict:
        if not segment_scores:
            return self.compute_risk_score(0.0, context)

        weights = [1.0 + 0.1 * i for i in range(len(segment_scores))]
        weighted_avg = sum(s * w for s, w in zip(segment_scores, weights)) / sum(weights)
        trend = segment_scores[-1] - segment_scores[0] if len(segment_scores) > 1 else 0.0

        adjusted = weighted_avg * (1.0 + 0.1 * max(trend, 0))
        adjusted = min(adjusted, 1.0)

        result = self.compute_risk_score(adjusted, context)
        result["segment_count"] = len(segment_scores)
        result["trend"] = round(trend, 4)
        result["trend_direction"] = "increasing" if trend > 0.05 else "decreasing" if trend < -0.05 else "stable"
        result["weighted_average"] = round(weighted_avg, 4)

        return result
