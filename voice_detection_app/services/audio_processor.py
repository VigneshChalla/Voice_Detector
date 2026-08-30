import io
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

from voice_detection_app.config import settings

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = "ffmpeg"


class AudioProcessor:
    """Extracts acoustic and spectral features from audio for voice authenticity analysis."""

    def __init__(self):
        self.sr = settings.audio.sample_rate
        self.n_mfcc = settings.audio.n_mfcc
        self.n_fft = settings.audio.n_fft
        self.hop_length = settings.audio.hop_length
        self.max_duration = settings.audio.max_duration_sec
        self.segment_duration = settings.audio.segment_duration_sec

    def load_audio(self, file_path: str | Path) -> tuple[np.ndarray, int]:
        y, sr = librosa.load(str(file_path), sr=self.sr, duration=self.max_duration)
        return y, sr

    def load_audio_from_bytes(self, audio_bytes: bytes, content_type: str = "") -> tuple[np.ndarray, int]:
        ext = ".wav"
        if audio_bytes[:4] == b'fLaC':
            ext = ".flac"
        elif audio_bytes[:4] == b'OggS':
            ext = ".ogg"
        elif audio_bytes[:4] == b'ID3\x03' or audio_bytes[:2] == b'\xff\xfb':
            ext = ".mp3"
        elif len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp':
            ext = ".m4a"
        elif audio_bytes[:4] == b'caff':
            ext = ".m4a"
        elif content_type:
            if "m4a" in content_type or "aac" in content_type:
                ext = ".m4a"
            elif "mp3" in content_type:
                ext = ".mp3"
            elif "ogg" in content_type:
                ext = ".ogg"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in.flush()
            tmp_in_path = tmp_in.name

        try:
            if ext != ".wav":
                tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_out_path = tmp_out.name
                tmp_out.close()
                subprocess.run(
                    [FFMPEG_PATH, "-y", "-i", tmp_in_path, "-ar", str(self.sr), "-ac", "1", tmp_out_path],
                    capture_output=True, check=True, timeout=30,
                )
                y, sr = librosa.load(tmp_out_path, sr=self.sr, duration=self.max_duration)
                os.unlink(tmp_out_path)
            else:
                y, sr = librosa.load(tmp_in_path, sr=self.sr, duration=self.max_duration)
        finally:
            os.unlink(tmp_in_path)

        return y, sr

    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
        )

    def extract_spectral_features(self, y: np.ndarray) -> dict[str, float]:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        return {
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spectral_rolloff_std": float(np.std(spectral_rolloff)),
            "spectral_contrast_mean": float(np.mean(spectral_contrast)),
            "zero_crossing_rate_mean": float(np.mean(zero_crossing)),
            "rms_energy_mean": float(np.mean(rms)),
            "rms_energy_std": float(np.std(rms)),
        }

    def extract_prosody_features(self, y: np.ndarray) -> dict[str, float]:
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) == 0:
            pitch_values = np.array([0.0])

        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)

        return {
            "pitch_mean": float(np.mean(pitch_values)),
            "pitch_std": float(np.std(pitch_values)),
            "pitch_range": float(np.ptp(pitch_values)),
            "tempo": float(tempo) if np.isscalar(tempo) else float(tempo[0]),
            "onset_strength_mean": float(np.mean(onset_env)),
            "onset_strength_std": float(np.std(onset_env)),
        }

    def extract_phase_features(self, y: np.ndarray) -> dict[str, float]:
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        phase = np.angle(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        phase_diff = np.diff(phase, axis=1)

        return {
            "phase_diff_mean": float(np.mean(np.abs(phase_diff))),
            "phase_diff_std": float(np.std(phase_diff)),
            "phase_entropy": float(
                -np.sum(
                    (np.abs(phase) / (np.sum(np.abs(phase)) + 1e-10))
                    * np.log2(np.abs(phase) / (np.sum(np.abs(phase)) + 1e-10) + 1e-10)
                )
            ),
            "stft_energy_mean": float(np.mean(stft)),
            "stft_energy_std": float(np.std(stft)),
        }

    def extract_all_features(self, y: np.ndarray) -> dict[str, float]:
        features = {}
        features.update(self.extract_spectral_features(y))
        features.update(self.extract_prosody_features(y))
        features.update(self.extract_phase_features(y))
        return features

    def compute_segment_features(self, y: np.ndarray) -> list[dict[str, float]]:
        sr = self.sr
        segment_samples = int(self.segment_duration * sr)
        segments = []

        for start in range(0, len(y), segment_samples):
            segment = y[start : start + segment_samples]
            if len(segment) < segment_samples // 2:
                break
            features = self.extract_all_features(segment)
            features["segment_start_sec"] = start / sr
            segments.append(features)

        return segments

    def aggregate_features(self, segments: list[dict[str, float]]) -> dict[str, float]:
        if not segments:
            return {}

        feature_keys = [k for k in segments[0] if k != "segment_start_sec"]
        aggregated = {}
        for key in feature_keys:
            values = [s[key] for s in segments]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_max"] = float(np.max(values))
            aggregated[f"{key}_min"] = float(np.min(values))

        aggregated["num_segments"] = len(segments)
        aggregated["total_duration_sec"] = segments[-1].get("segment_start_sec", 0) + self.segment_duration
        return aggregated

    def process_audio(self, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        aggregated = self.aggregate_features(self.compute_segment_features(y))
        mfcc = self.extract_mfcc(y)
        mfcc_stats = {
            "mfcc_mean": float(np.mean(mfcc)),
            "mfcc_std": float(np.std(mfcc)),
        }
        aggregated.update(mfcc_stats)
        return mfcc, aggregated

    def get_feature_vector(self, aggregated: dict[str, float], target_length: int = 64) -> np.ndarray:
        keys = sorted(aggregated.keys())
        values = [aggregated[k] for k in keys]

        if len(values) < target_length:
            values.extend([0.0] * (target_length - len(values)))
        elif len(values) > target_length:
            values = values[:target_length]

        return np.array(values, dtype=np.float32)
