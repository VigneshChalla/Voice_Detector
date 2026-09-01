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
    """Extracts 128-dim acoustic features for voice authenticity analysis.
    
    Features match the training pipeline exactly (extract_v2_features.py).
    """

    def __init__(self):
        self.sr = settings.audio.sample_rate
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

    def extract_features_128(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract 128-dim feature vector matching the training pipeline."""
        if sr is None:
            sr = self.sr
        feats = {}

        # === F0 via pyin ===
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=60, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) < 10:
            f0_clean = np.array([120.0])
        feats["f0_mean"] = float(np.mean(f0_clean))
        feats["f0_std"] = float(np.std(f0_clean))
        feats["f0_cv"] = float(np.std(f0_clean) / (np.mean(f0_clean) + 1e-6))
        feats["f0_range"] = float(np.ptp(f0_clean))
        feats["f0_median"] = float(np.median(f0_clean))
        feats["f0_q25"] = float(np.percentile(f0_clean, 25))
        feats["f0_q75"] = float(np.percentile(f0_clean, 75))
        feats["voiced_fraction"] = float(np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0.5

        # Jitter
        if len(f0_clean) > 5:
            periods = 1.0 / (f0_clean + 1e-6)
            abs_diff = np.abs(np.diff(periods))
            feats["jitter_percent"] = float(np.mean(abs_diff) / np.mean(periods) * 100)
        else:
            feats["jitter_percent"] = 0.0

        # RMS
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if len(rms) > 5:
            abs_diff_amp = np.abs(np.diff(rms))
            feats["shimmer_percent"] = float(np.mean(abs_diff_amp) / (np.mean(rms) + 1e-6) * 100)
        else:
            feats["shimmer_percent"] = 0.0

        # HNR
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_mean = np.mean(S, axis=1)
        if np.max(S_mean) > 0:
            harmonic_e = np.sum(S_mean[:len(S_mean)//2]**2)
            noise_e = np.sum((S_mean[:len(S_mean)//2] - np.convolve(S_mean[:len(S_mean)//2], np.ones(5)/5, mode='same'))**2)
            feats["hnr_db"] = float(10 * np.log10((harmonic_e + 1e-10) / (noise_e + 1e-10)))
        else:
            feats["hnr_db"] = 0.0

        # Spectral features
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        scon = librosa.feature.spectral_contrast(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        feats["sc_mean"] = float(np.mean(sc)); feats["sc_std"] = float(np.std(sc))
        feats["sc_cv"] = float(np.std(sc) / (np.mean(sc) + 1e-6))
        feats["sb_mean"] = float(np.mean(sb)); feats["sb_std"] = float(np.std(sb))
        feats["sb_cv"] = float(np.std(sb) / (np.mean(sb) + 1e-6))
        feats["sro_mean"] = float(np.mean(sro)); feats["sro_std"] = float(np.std(sro))
        feats["scon_mean"] = float(np.mean(scon)); feats["scon_std"] = float(np.std(scon))
        feats["flatness_mean"] = float(np.mean(flatness)); feats["flatness_std"] = float(np.std(flatness))
        feats["flatness_cv"] = float(np.std(flatness) / (np.mean(flatness) + 1e-6))
        feats["zcr_mean"] = float(np.mean(zcr)); feats["zcr_std"] = float(np.std(zcr))

        # Spectral entropy
        S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
        spec_entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)
        feats["spec_entropy_mean"] = float(np.mean(spec_entropy))
        feats["spec_entropy_std"] = float(np.std(spec_entropy))

        # Spectral flux
        S_diff = np.diff(S, axis=1)
        spectral_flux = np.sqrt(np.mean(S_diff**2, axis=0))
        feats["spec_flux_mean"] = float(np.mean(spectral_flux))
        feats["spec_flux_std"] = float(np.std(spectral_flux))

        # Crest factor
        spec_rms = np.sqrt(np.mean(S**2, axis=0))
        spec_peak = np.max(S, axis=0)
        feats["crest_factor_mean"] = float(np.mean(spec_peak / (spec_rms + 1e-10)))

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        feats["mfcc_mean"] = float(np.mean(mfcc)); feats["mfcc_std"] = float(np.std(mfcc))
        feats["mfcc_delta_mean"] = float(np.mean(np.abs(mfcc_delta)))
        feats["mfcc_delta_std"] = float(np.std(mfcc_delta))
        feats["mfcc_delta2_mean"] = float(np.mean(np.abs(mfcc_delta2)))
        feats["mfcc_delta2_std"] = float(np.std(mfcc_delta2))

        # Formants
        try:
            order = 2 + sr // 1000
            a = librosa.lpc(y, order=order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = sorted(angles * (sr / (2 * np.pi)))
            freqs = [f for f in freqs if 90 < f < sr/2 - 90][:4]
            if len(freqs) >= 2:
                feats["formant1_mean"] = float(freqs[0])
                feats["formant2_mean"] = float(freqs[1]) if len(freqs) > 1 else float(freqs[0])
                feats["formant_spread"] = float(np.std(freqs))
                feats["formant_range"] = float(np.ptp(freqs))
            else:
                feats["formant1_mean"] = 500.0; feats["formant2_mean"] = 1500.0
                feats["formant_spread"] = 100.0; feats["formant_range"] = 200.0
        except:
            feats["formant1_mean"] = 500.0; feats["formant2_mean"] = 1500.0
            feats["formant_spread"] = 100.0; feats["formant_range"] = 200.0

        # Harmonic prominence
        S_half = S[:S.shape[0]//2, :]
        f0_est = feats["f0_mean"]
        if f0_est > 60:
            harm_mask = np.zeros(S_half.shape[0], dtype=bool)
            for i in range(S_half.shape[0]):
                freq = i * sr / (2 * S_half.shape[0])
                for h in range(1, 10):
                    if abs(freq - h * f0_est) < 40:
                        harm_mask[i] = True; break
            S_harm = S_half[harm_mask]; S_noise = S_half[~harm_mask]
            harm_e = np.sum(S_harm**2); noise_e = np.sum(S_noise**2)
            feats["harmonic_prominence"] = float(harm_e / (harm_e + noise_e + 1e-10))
        else:
            feats["harmonic_prominence"] = 0.3

        # Energy
        feats["rms_mean"] = float(np.mean(rms)); feats["rms_std"] = float(np.std(rms))
        feats["rms_cv"] = float(np.std(rms) / (np.mean(rms) + 1e-6))

        # Phase
        phase = np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
        pd = np.diff(phase, axis=1)
        feats["phase_diff_mean"] = float(np.mean(np.abs(pd)))
        feats["phase_diff_std"] = float(np.std(pd))
        pd_abs = np.abs(pd)
        pd_norm = pd_abs / (np.sum(pd_abs, axis=0, keepdims=True) + 1e-10)
        feats["phase_entropy"] = float(-np.mean(np.sum(pd_norm * np.log2(pd_norm + 1e-10), axis=0)))

        # Silence
        threshold = np.mean(rms) * 0.1
        silent = rms < threshold
        feats["silence_ratio"] = float(np.mean(silent))
        transitions = np.diff(silent.astype(int))
        breath_count = float(np.sum(transitions == -1))
        feats["breath_rate"] = float(breath_count / (len(rms) * 512 / sr + 1e-6))

        # Modulation
        if len(rms) > 64:
            rms_fft = np.abs(np.fft.rfft(rms - np.mean(rms)))
            feats["mod_spec_peak"] = float(np.max(rms_fft[1:]))
            feats["mod_spec_mean"] = float(np.mean(rms_fft[1:]))
            feats["mod_spec_ratio"] = float(np.max(rms_fft[1:]) / (np.mean(rms_fft[1:]) + 1e-10))
        else:
            feats["mod_spec_peak"] = 0.0; feats["mod_spec_mean"] = 0.0; feats["mod_spec_ratio"] = 1.0

        # Cepstral
        try:
            cepstrum = np.fft.irfft(np.log(S_mean + 1e-10))
            peak_idx = np.argmax(np.abs(cepstrum[20:])) + 20
            feats["cpp"] = float(np.abs(cepstrum[peak_idx]) / (np.mean(np.abs(cepstrum)) + 1e-10))
        except:
            feats["cpp"] = 1.0

        # Peak regularity
        if len(rms) > 20:
            peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0, wait=10)
            if len(peaks) > 2:
                intervals = np.diff(peaks)
                feats["peak_regularity"] = float(1.0 / (np.std(intervals) / (np.mean(intervals) + 1e-6) + 1e-6))
            else:
                feats["peak_regularity"] = 1.0
        else:
            feats["peak_regularity"] = 1.0

        # Segment-level aggregation
        segment_samples = int(self.segment_duration * sr)
        segments = []
        for start in range(0, len(y), segment_samples):
            seg = y[start:start + segment_samples]
            if len(seg) < segment_samples // 2:
                break
            seg_feats = {}
            sc_s = librosa.feature.spectral_centroid(y=seg, sr=sr)
            seg_feats["sc_m"] = float(np.mean(sc_s)); seg_feats["sc_s"] = float(np.std(sc_s))
            sb_s = librosa.feature.spectral_bandwidth(y=seg, sr=sr)
            seg_feats["sb_m"] = float(np.mean(sb_s)); seg_feats["sb_s"] = float(np.std(sb_s))
            rms_s = librosa.feature.rms(y=seg)
            seg_feats["rms_m"] = float(np.mean(rms_s)); seg_feats["rms_s"] = float(np.std(rms_s))
            zcr_s = librosa.feature.zero_crossing_rate(seg)
            seg_feats["zcr_m"] = float(np.mean(zcr_s))
            pitches, mags = librosa.piptrack(y=seg, sr=sr)
            pv = pitches[mags > np.median(mags)]
            if len(pv) > 5:
                pv = pv[(pv > 60) & (pv < 500)]
            if len(pv) < 5:
                pv = np.array([120.0])
            seg_feats["pitch_m"] = float(np.mean(pv)); seg_feats["pitch_s"] = float(np.std(pv))
            segments.append(seg_feats)

        if segments:
            seg_keys = list(segments[0].keys())
            for k in seg_keys:
                vals = [s[k] for s in segments]
                feats[f"seg_{k}_mean"] = float(np.mean(vals))
                feats[f"seg_{k}_std"] = float(np.std(vals))

        # Pad/truncate to 128
        keys = sorted(feats.keys())
        values = [feats[k] for k in keys]
        if len(values) < 128:
            values.extend([0.0] * (128 - len(values)))
        elif len(values) > 128:
            values = values[:128]

        return np.array(values, dtype=np.float32)

    def process_audio(self, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Extract 128-dim feature vector and metadata."""
        feature_vector = self.extract_features_128(y)
        metadata = {
            "duration_sec": len(y) / self.sr,
            "sample_rate": self.sr,
            "feature_dim": len(feature_vector),
        }
        return feature_vector, metadata
