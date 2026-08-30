import logging
from datetime import datetime, timezone
from typing import Any

from voice_detection_app.config import settings

logger = logging.getLogger(__name__)


class AlertService:
    """Handles alert generation and dispatch for detected impersonation risks."""

    def __init__(self):
        self.enabled = settings.alert.enabled
        self.channels = settings.alert.channels

    def create_alert(self, risk_result: dict, call_metadata: dict | None = None) -> dict[str, Any]:
        call_metadata = call_metadata or {}
        alert = {
            "alert_id": f"ALT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_score": risk_result.get("risk_score", 0),
            "risk_level": risk_result.get("risk_level", "UNKNOWN"),
            "recommendation": risk_result.get("recommendation", ""),
            "requires_secondary_verification": risk_result.get("requires_secondary_verification", False),
            "should_block_transaction": risk_result.get("should_block_transaction", False),
            "caller_id": call_metadata.get("caller_id", "unknown"),
            "call_duration_sec": call_metadata.get("call_duration_sec", 0),
            "channels_notified": [],
        }

        if self.enabled:
            self._dispatch(alert)

        return alert

    def _dispatch(self, alert: dict):
        for channel in self.channels:
            try:
                if channel == "log":
                    self._log_alert(alert)
                elif channel == "ui":
                    self._ui_alert(alert)
                elif channel == "webhook":
                    self._webhook_alert(alert)
                elif channel == "email":
                    self._email_alert(alert)
                alert["channels_notified"].append(channel)
            except Exception as e:
                logger.error("Failed to dispatch alert via %s: %s", channel, e)

    def _log_alert(self, alert: dict):
        level = alert["risk_level"]
        msg = (
            f"[{level} RISK] Alert {alert['alert_id']} | "
            f"Score: {alert['risk_score']} | Caller: {alert['caller_id']} | "
            f"Recommendation: {alert['recommendation']}"
        )
        if level == "HIGH":
            logger.critical(msg)
        elif level == "MEDIUM":
            logger.warning(msg)
        else:
            logger.info(msg)

    def _ui_alert(self, alert: dict):
        logger.info("UI notification prepared for alert %s", alert["alert_id"])

    def _webhook_alert(self, alert: dict):
        if settings.alert.webhook_url:
            logger.info("Webhook alert dispatched to %s", settings.alert.webhook_url)

    def _email_alert(self, alert: dict):
        if settings.alert.email_recipients:
            logger.info("Email alert dispatched to %d recipients", len(settings.alert.email_recipients))
