"""WebSocket endpoint for real-time audio streaming and live detection."""
import asyncio
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voice_detection_app.models.detector import VoiceDetector
from voice_detection_app.services.alert_service import AlertService
from voice_detection_app.services.audio_processor import AudioProcessor
from voice_detection_app.services.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)

ws_router = APIRouter()


class AudioStreamBuffer:
    """Accumulates raw audio chunks and yields fixed-size segments."""

    def __init__(self, sample_rate: int = 16000, segment_sec: float = 3.0):
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_sec * sample_rate)
        self.buffer: list[np.ndarray] = []
        self.buffer_size = 0

    def add_chunk(self, audio_data: np.ndarray):
        self.buffer.append(audio_data)
        self.buffer_size += len(audio_data)

    def get_segments(self) -> list[np.ndarray]:
        if self.buffer_size < self.segment_samples:
            return []

        full = np.concatenate(self.buffer)
        segments = []
        for start in range(0, len(full), self.segment_samples):
            end = start + self.segment_samples
            if end > len(full):
                break
            segments.append(full[start:end])

        consumed = len(segments) * self.segment_samples
        remaining = full[consumed:]
        self.buffer = [remaining] if len(remaining) > 0 else []
        self.buffer_size = len(remaining)
        return segments

    def flush(self) -> np.ndarray | None:
        if self.buffer_size == 0:
            return None
        full = np.concatenate(self.buffer)
        self.buffer = []
        self.buffer_size = 0
        return full


class LiveCallSession:
    """Manages a single live call WebSocket session."""

    def __init__(self, websocket: WebSocket, caller_id: str = "", call_type: str = "regular_call"):
        self.websocket = websocket
        self.caller_id = caller_id
        self.call_type = call_type
        self.audio_processor = AudioProcessor()
        self.detector = VoiceDetector()
        self.risk_scorer = RiskScorer()
        self.alert_service = AlertService()
        self.buffer = AudioStreamBuffer(sample_rate=self.audio_processor.sr)
        self.segment_scores: list[float] = []
        self.start_time = time.time()
        self.is_active = False

    async def run(self):
        self.is_active = True
        await self.websocket.accept()
        await self._send_json({
            "type": "connected",
            "caller_id": self.caller_id,
            "call_type": self.call_type,
            "message": "Audio stream connected. Send binary audio chunks.",
        })

        try:
            while self.is_active:
                message = await self.websocket.receive()

                if message["type"] == "websocket.receive":
                    if "text" in message:
                        await self._handle_text(message["text"])
                    elif "bytes" in message:
                        await self._handle_audio(message["bytes"])

        except WebSocketDisconnect:
            logger.info("Client disconnected: %s", self.caller_id)
        except Exception as e:
            logger.error("WebSocket error: %s", e)
        finally:
            await self._finalize()

    async def _handle_text(self, text: str):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", "")

        if msg_type == "end_stream":
            await self._finalize()
        elif msg_type == "update_context":
            self.call_type = data.get("call_type", self.call_type)
            self.caller_id = data.get("caller_id", self.caller_id)
            await self._send_json({"type": "context_updated", "call_type": self.call_type})

    async def _handle_audio(self, raw_bytes: bytes):
        if len(raw_bytes) < 2:
            return

        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer.add_chunk(audio)

        segments = self.buffer.get_segments()
        for segment in segments:
            await self._analyze_segment(segment)

    async def _analyze_segment(self, segment: np.ndarray):
        _, aggregated = self.audio_processor.process_audio(segment)
        feature_vector = self.audio_processor.get_feature_vector(aggregated, target_length=64)
        prediction = self.detector.predict(feature_vector)

        score = prediction["synthetic_probability"]
        self.segment_scores.append(score)

        risk = self.risk_scorer.compute_streaming_risk(self.segment_scores, {
            "caller_id": self.caller_id,
            "call_type": self.call_type,
        })

        elapsed = time.time() - self.start_time

        response = {
            "type": "segment_result",
            "segment_index": len(self.segment_scores),
            "elapsed_sec": round(elapsed, 2),
            "synthetic_probability": round(score, 4),
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "trend": risk.get("trend_direction", "stable"),
            "recommendation": risk["recommendation"],
            "requires_secondary_verification": risk["requires_secondary_verification"],
        }

        await self._send_json(response)

        if risk["risk_level"] in ("HIGH", "MEDIUM"):
            alert = self.alert_service.create_alert(risk, {
                "caller_id": self.caller_id,
                "call_duration_sec": round(elapsed, 2),
            })
            await self._send_json({"type": "alert", "alert": alert})

    async def _finalize(self):
        self.is_active = False
        remaining = self.buffer.flush()

        final_score = 0.0
        if remaining is not None and len(remaining) > 0:
            _, aggregated = self.audio_processor.process_audio(remaining)
            fv = self.audio_processor.get_feature_vector(aggregated, target_length=64)
            pred = self.detector.predict(fv)
            final_score = pred["synthetic_probability"]
            self.segment_scores.append(final_score)

        if self.segment_scores:
            final_risk = self.risk_scorer.compute_streaming_risk(self.segment_scores, {
                "caller_id": self.caller_id,
                "call_type": self.call_type,
            })
        else:
            final_risk = self.risk_scorer.compute_risk_score(0.0)

        elapsed = time.time() - self.start_time

        summary = {
            "type": "stream_ended",
            "caller_id": self.caller_id,
            "total_segments": len(self.segment_scores),
            "total_duration_sec": round(elapsed, 2),
            "final_risk_score": final_risk["risk_score"],
            "final_risk_level": final_risk["risk_level"],
            "final_recommendation": final_risk["recommendation"],
            "segment_scores": [round(s, 4) for s in self.segment_scores],
        }

        await self._send_json(summary)
        await self.websocket.close()

    async def _send_json(self, data: dict):
        try:
            await self.websocket.send_text(json.dumps(data))
        except Exception:
            pass


@ws_router.websocket("/ws/detect")
async def websocket_detect(
    websocket: WebSocket,
    caller_id: str = "",
    call_type: str = "regular_call",
):
    session = LiveCallSession(websocket, caller_id, call_type)
    await session.run()
