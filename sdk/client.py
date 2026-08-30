"""
Voice Detection Python SDK
Client library for integrating with the Voice Cloning Detection API.

Installation:
    pip install voice-detection-sdk

Usage:
    from sdk import VoiceDetectionClient

    client = VoiceDetectionClient(
        base_url="http://localhost:8000",
        api_key="your_api_key"
    )

    # Detect voice from file
    result = client.detect("audio.wav")
    print(result["risk_level"])

    # Stream detection
    with client.stream_detection(caller_id="user123") as stream:
        for chunk in audio_chunks:
            stream.send(chunk)
            result = stream.get_latest_result()
            if result and result["risk_level"] == "HIGH":
                print("ALERT:", result["recommendation"])
"""
import io
import json
import logging
from pathlib import Path
from typing import Any

import requests
import websocket

logger = logging.getLogger(__name__)


class VoiceDetectionError(Exception):
    """Base exception for voice detection SDK."""
    pass


class AuthenticationError(VoiceDetectionError):
    pass


class RateLimitError(VoiceDetectionError):
    pass


class DetectionError(VoiceDetectionError):
    pass


class VoiceDetectionClient:
    """HTTP client for the Voice Cloning Detection API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)

        try:
            resp = self.session.request(method, url, **kwargs)
        except requests.ConnectionError:
            raise VoiceDetectionError(f"Cannot connect to {self.base_url}")
        except requests.Timeout:
            raise VoiceDetectionError("Request timed out")

        if resp.status_code == 401:
            raise AuthenticationError("Invalid or missing API key")
        if resp.status_code == 429:
            raise RateLimitError(f"Rate limited: {resp.json().get('detail', '')}")
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text)
            raise DetectionError(f"API error {resp.status_code}: {detail}")

        return resp.json()

    def health(self) -> dict:
        """Check API health status."""
        return self._request("GET", "/api/v1/health")

    def detect(
        self,
        audio_path: str | Path | bytes,
        caller_id: str = "",
        call_type: str = "regular_call",
    ) -> dict:
        """Detect voice cloning from an audio file or bytes."""
        if isinstance(audio_path, (str, Path)):
            path = Path(audio_path)
            if not path.exists():
                raise DetectionError(f"File not found: {audio_path}")
            files = {"file": (path.name, open(path, "rb"), "audio/wav")}
        elif isinstance(audio_path, bytes):
            files = {"file": ("audio.wav", io.BytesIO(audio_path), "audio/wav")}
        else:
            raise DetectionError("audio_path must be file path or bytes")

        params = {"caller_id": caller_id, "call_type": call_type}

        try:
            return self._request("POST", "/api/v1/detect", files=files, params=params)
        finally:
            if isinstance(audio_path, (str, Path)):
                files["file"][1].close()

    def detect_stream(
        self,
        audio_path: str | Path | bytes,
        caller_id: str = "",
        call_type: str = "regular_call",
    ) -> dict:
        """Analyze audio in segments for streaming detection."""
        if isinstance(audio_path, (str, Path)):
            path = Path(audio_path)
            files = {"file": (path.name, open(path, "rb"), "audio/wav")}
        elif isinstance(audio_path, bytes):
            files = {"file": ("audio.wav", io.BytesIO(audio_path), "audio/wav")}
        else:
            raise DetectionError("audio_path must be file path or bytes")

        params = {"caller_id": caller_id, "call_type": call_type}

        try:
            return self._request("POST", "/api/v1/detect/stream", files=files, params=params)
        finally:
            if isinstance(audio_path, (str, Path)):
                files["file"][1].close()

    def compute_risk(
        self,
        synthetic_probability: float,
        call_type: str = "regular_call",
    ) -> dict:
        """Compute risk score from a synthetic probability."""
        return self._request(
            "POST",
            "/api/v1/risk/score",
            params={"synthetic_probability": synthetic_probability, "call_type": call_type},
        )

    def enroll_speaker(
        self,
        audio_path: str | Path | bytes,
        speaker_id: str,
        label: str = "",
    ) -> dict:
        """Enroll a speaker voiceprint."""
        if isinstance(audio_path, (str, Path)):
            path = Path(audio_path)
            files = {"file": (path.name, open(path, "rb"), "audio/wav")}
        elif isinstance(audio_path, bytes):
            files = {"file": ("audio.wav", io.BytesIO(audio_path), "audio/wav")}
        else:
            raise DetectionError("audio_path must be file path or bytes")

        params = {"speaker_id": speaker_id, "label": label}

        try:
            return self._request("POST", "/api/v1/speaker/enroll", files=files, params=params)
        finally:
            if isinstance(audio_path, (str, Path)):
                files["file"][1].close()

    def verify_speaker(
        self,
        audio_path: str | Path | bytes,
        speaker_id: str,
    ) -> dict:
        """Verify a speaker against their enrolled voiceprint."""
        if isinstance(audio_path, (str, Path)):
            path = Path(audio_path)
            files = {"file": (path.name, open(path, "rb"), "audio/wav")}
        elif isinstance(audio_path, bytes):
            files = {"file": ("audio.wav", io.BytesIO(audio_path), "audio/wav")}
        else:
            raise DetectionError("audio_path must be file path or bytes")

        params = {"speaker_id": speaker_id}

        try:
            return self._request("POST", "/api/v1/speaker/verify", files=files, params=params)
        finally:
            if isinstance(audio_path, (str, Path)):
                files["file"][1].close()

    def cross_session_check(
        self,
        audio_path: str | Path | bytes,
        speaker_id: str,
    ) -> dict:
        """Run cross-session anomaly detection."""
        if isinstance(audio_path, (str, Path)):
            path = Path(audio_path)
            files = {"file": (path.name, open(path, "rb"), "audio/wav")}
        elif isinstance(audio_path, bytes):
            files = {"file": ("audio.wav", io.BytesIO(audio_path), "audio/wav")}
        else:
            raise DetectionError("audio_path must be file path or bytes")

        params = {"speaker_id": speaker_id}

        try:
            return self._request("POST", "/api/v1/speaker/cross-session", files=files, params=params)
        finally:
            if isinstance(audio_path, (str, Path)):
                files["file"][1].close()

    def list_speakers(self) -> list[str]:
        """List enrolled speakers."""
        result = self._request("GET", "/api/v1/speaker/list")
        return result.get("speakers", [])

    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker enrollment."""
        self._request("DELETE", f"/api/v1/speaker/{speaker_id}")
        return True

    def edge_status(self) -> dict:
        """Check edge inference status."""
        return self._request("GET", "/api/v1/edge/status")

    def open_stream(self, caller_id: str = "", call_type: str = "regular_call") -> "DetectionStream":
        """Open a WebSocket connection for real-time streaming detection."""
        return DetectionStream(self.base_url, self.api_key, caller_id, call_type)


class DetectionStream:
    """WebSocket client for real-time streaming detection."""

    def __init__(self, base_url: str, api_key: str, caller_id: str, call_type: str):
        ws_url = base_url.replace("http", "ws")
        self.ws_url = f"{ws_url}/ws/detect?caller_id={caller_id}&call_type={call_type}"
        self.api_key = api_key
        self._ws = None
        self._latest_result = None
        self._results = []

    def connect(self):
        """Connect to the WebSocket."""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        self._ws = websocket.WebSocketApp(
            self.ws_url,
            header=headers,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            self._latest_result = data
            if data.get("type") != "connected":
                self._results.append(data)
        except json.JSONDecodeError:
            pass

    def _on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket closed")

    def send_audio(self, audio_bytes: bytes):
        """Send audio chunk to the stream."""
        if self._ws:
            self._ws.send(audio_bytes, opcode=websocket.ABNF.OPCODE_BINARY)

    def send_end(self):
        """Signal end of audio stream."""
        if self._ws:
            self._ws.send(json.dumps({"type": "end_stream"}))

    def get_latest_result(self) -> dict | None:
        """Get the most recent detection result."""
        return self._latest_result

    def get_all_results(self) -> list[dict]:
        """Get all results from this session."""
        return self._results

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        if self._ws:
            self.send_end()
            self._ws.close()
