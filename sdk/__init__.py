# Voice Cloning Detection SDK
# Client library for the Voice Cloning Detection API

from sdk.client import (
    VoiceDetectionClient,
    DetectionStream,
    VoiceDetectionError,
    AuthenticationError,
    RateLimitError,
    DetectionError,
)

__all__ = [
    "VoiceDetectionClient",
    "DetectionStream",
    "VoiceDetectionError",
    "AuthenticationError",
    "RateLimitError",
    "DetectionError",
]
