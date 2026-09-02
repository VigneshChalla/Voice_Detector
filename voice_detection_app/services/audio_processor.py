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
    """Extracts 192-dim Tier-2 forensic-grade features with StandardScaler normalization."""

    def __init__(self):
        self.sr = settings.audio.sample_rate
        self.max_duration = settings.audio.max_duration_sec
        self.segment_duration = settings.audio.segment_duration_sec
        # Load scaler for normalization (computed on train set)
        self.scaler_mean = None
        self.scaler_std = None
        scaler_path = Path(settings.model.scaler_path) if hasattr(settings.model, 'scaler_path') else Path("voice_detection_app/scaler_v3.npz")
        if scaler_path.exists():
            try:
                data = np.load(str(scaler_path))
                self.scaler_mean = data["mean"].astype(np.float32)
                self.scaler_std = data["std"].astype(np.float32)
            except:
                pass
        # fallback to v2 scaler if v3 not found
        if self.scaler_mean is None:
            alt = Path("voice_detection_app/scaler_v3.npz")
            if alt.exists():
                try:
                    data = np.load(str(alt))
                    self.scaler_mean = data["mean"].astype(np.float32)
                    self.scaler_std = data["std"].astype(np.float32)
                except:
                    pass

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

    def extract_features_192(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract 192-dim Tier-2 feature vector matching train_v3 pipeline."""
        if sr is None:
            sr = self.sr
        feats = {}
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
        feats["f0_min"] = float(np.min(f0_clean))
        feats["f0_max"] = float(np.max(f0_clean))
        feats["voiced_fraction"] = float(np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0.5
        if len(f0_clean) > 10:
            df0 = np.diff(f0_clean)
            feats["f0_delta_mean"] = float(np.mean(np.abs(df0)))
            feats["f0_delta_std"] = float(np.std(df0))
            feats["f0_slope"] = float(np.polyfit(np.arange(len(f0_clean)), f0_clean, 1)[0])
        else:
            feats["f0_delta_mean"] = 0.0; feats["f0_delta_std"] = 0.0; feats["f0_slope"] = 0.0
        if len(f0_clean) > 5:
            periods = 1.0 / (f0_clean + 1e-6)
            abs_diff = np.abs(np.diff(periods))
            feats["jitter_percent"] = float(np.mean(abs_diff) / np.mean(periods) * 100)
            if len(periods) > 5:
                rap = np.mean([abs(periods[i+1] - (periods[i]+periods[i+1]+periods[i+2])/3) for i in range(len(periods)-2)]) / np.mean(periods) * 100
                feats["jitter_rap"] = float(rap)
            else:
                feats["jitter_rap"] = feats["jitter_percent"]
        else:
            feats["jitter_percent"] = 0.0; feats["jitter_rap"] = 0.0
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if len(rms) > 5:
            abs_diff_amp = np.abs(np.diff(rms))
            feats["shimmer_percent"] = float(np.mean(abs_diff_amp) / (np.mean(rms) + 1e-6) * 100)
            drms = np.diff(rms)
            feats["rms_delta_mean"] = float(np.mean(np.abs(drms)))
            feats["rms_delta_std"] = float(np.std(drms))
        else:
            feats["shimmer_percent"] = 0.0; feats["rms_delta_mean"] = 0.0; feats["rms_delta_std"] = 0.0
        feats["rms_mean"] = float(np.mean(rms)); feats["rms_std"] = float(np.std(rms))
        feats["rms_cv"] = float(np.std(rms) / (np.mean(rms) + 1e-6))
        feats["rms_skew"] = float(((rms - np.mean(rms))**3).mean() / (np.std(rms)**3 + 1e-9))
        feats["rms_kurt"] = float(((rms - np.mean(rms))**4).mean() / (np.std(rms)**4 + 1e-9))
        threshold = np.mean(rms) * 0.12
        silent = rms < threshold
        feats["silence_ratio"] = float(np.mean(silent))
        sil_len = 0; pauses = []
        for v in silent:
            if v:
                sil_len += 1
            else:
                if sil_len > 0:
                    pauses.append(sil_len * 512 / sr)
                    sil_len = 0
        if sil_len > 0:
            pauses.append(sil_len * 512 / sr)
        if pauses:
            feats["pause_count"] = float(len(pauses))
            feats["pause_mean_dur"] = float(np.mean(pauses))
            feats["pause_max_dur"] = float(np.max(pauses))
            feats["pause_rate"] = float(len(pauses) / (len(y)/sr + 1e-6))
        else:
            feats["pause_count"] = 0.0; feats["pause_mean_dur"] = 0.0; feats["pause_max_dur"] = 0.0; feats["pause_rate"] = 0.0
        transitions = np.diff(silent.astype(int))
        breath_count = float(np.sum(transitions == -1))
        feats["breath_rate"] = float(breath_count / (len(rms) * 512 / sr + 1e-6))
        S_full = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        hf_idx = np.where((freqs >= 2000) & (freqs <= 4000))[0]
        if len(pauses) > 0 and len(hf_idx) > 0:
            hf_energy_sil = np.mean(S_full[hf_idx][:, silent]) if np.any(silent) else 0.0
            hf_energy_voiced = np.mean(S_full[hf_idx][:, ~silent]) if np.any(~silent) else 1.0
            feats["breath_hf_ratio"] = float(hf_energy_sil / (hf_energy_voiced + 1e-9))
        else:
            feats["breath_hf_ratio"] = 0.0
        S = S_full
        S_mean = np.mean(S, axis=1)
        S_db = librosa.amplitude_to_db(S_mean + 1e-9)
        try:
            tilt = np.polyfit(freqs[:len(S_db)], S_db, 1)[0]
            feats["spectral_tilt"] = float(tilt)
        except:
            feats["spectral_tilt"] = 0.0
        total_energy = np.sum(S_mean) + 1e-9
        centroid = np.sum(freqs[:len(S_mean)] * S_mean) / total_energy
        spread = np.sqrt(np.sum(((freqs[:len(S_mean)] - centroid)**2) * S_mean) / total_energy)
        skew = np.sum(((freqs[:len(S_mean)] - centroid)**3) * S_mean) / (total_energy * (spread**3 + 1e-9))
        kurt = np.sum(((freqs[:len(S_mean)] - centroid)**4) * S_mean) / (total_energy * (spread**4 + 1e-9))
        feats["spectral_skewness"] = float(skew)
        feats["spectral_kurtosis"] = float(kurt)
        if np.max(S_mean) > 0:
            harmonic_e = np.sum(S_mean[:len(S_mean)//2]**2)
            resid = S_mean[:len(S_mean)//2] - np.convolve(S_mean[:len(S_mean)//2], np.ones(5)/5, mode='same')
            noise_e = np.sum(resid**2)
            feats["hnr_db"] = float(10 * np.log10((harmonic_e + 1e-10) / (noise_e + 1e-10)))
            low_idx = len(S_mean)//4
            h_low = np.sum(S_mean[:low_idx]**2)
            n_low = np.sum((S_mean[:low_idx] - np.convolve(S_mean[:low_idx], np.ones(3)/3, mode='same'))**2)
            feats["hnr_low_db"] = float(10*np.log10((h_low+1e-10)/(n_low+1e-10)))
        else:
            feats["hnr_db"] = 0.0; feats["hnr_low_db"] = 0.0
        lf_idx = np.where(freqs < 500)[0]
        hf_idx2 = np.where(freqs >= 4000)[0]
        mf_idx = np.where((freqs >= 500) & (freqs < 2000))[0]
        feats["lf_energy_ratio"] = float(np.sum(S_mean[lf_idx]) / (total_energy + 1e-9))
        feats["hf_energy_ratio"] = float(np.sum(S_mean[hf_idx2]) / (total_energy + 1e-9))
        feats["mf_energy_ratio"] = float(np.sum(S_mean[mf_idx]) / (total_energy + 1e-9))
        k_vals = np.arange(1, len(S_mean))
        feats["spectral_decrease"] = float(np.sum((S_mean[k_vals] - S_mean[0]) / k_vals) / (len(k_vals)+1e-9))
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        scon = librosa.feature.spectral_contrast(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats["sc_mean"] = float(np.mean(sc)); feats["sc_std"] = float(np.std(sc))
        feats["sc_cv"] = float(np.std(sc) / (np.mean(sc) + 1e-6))
        feats["sc_skew"] = float(((sc - np.mean(sc))**3).mean() / (np.std(sc)**3 + 1e-9))
        feats["sb_mean"] = float(np.mean(sb)); feats["sb_std"] = float(np.std(sb))
        feats["sb_cv"] = float(np.std(sb) / (np.mean(sb) + 1e-6))
        feats["sro_mean"] = float(np.mean(sro)); feats["sro_std"] = float(np.std(sro))
        feats["scon_mean"] = float(np.mean(scon)); feats["scon_std"] = float(np.std(scon))
        for lo, hi, name in [(0,1000,"sb_0_1k"), (1000,2000,"sb_1_2k"), (2000,4000,"sb_2_4k"), (4000,8000,"sb_4_8k")]:
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            if len(idx)>0:
                e = np.sum(S_mean[idx]) + 1e-9
                c = np.sum(freqs[idx] * S_mean[idx]) / e
                feats[f"subcent_{name}"] = float(c)
            else:
                feats[f"subcent_{name}"] = 0.0
        feats["flatness_mean"] = float(np.mean(flatness)); feats["flatness_std"] = float(np.std(flatness))
        feats["flatness_cv"] = float(np.std(flatness) / (np.mean(flatness) + 1e-6))
        feats["flatness_skew"] = float(((flatness - np.mean(flatness))**3).mean() / (np.std(flatness)**3 + 1e-9))
        feats["zcr_mean"] = float(np.mean(zcr)); feats["zcr_std"] = float(np.std(zcr))
        feats["zcr_cv"] = float(np.std(zcr) / (np.mean(zcr) + 1e-6))
        if len(zcr)>5:
            feats["zcr_delta_mean"] = float(np.mean(np.abs(np.diff(zcr))))
        else:
            feats["zcr_delta_mean"] = 0.0
        S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
        spec_entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)
        feats["spec_entropy_mean"] = float(np.mean(spec_entropy))
        feats["spec_entropy_std"] = float(np.std(spec_entropy))
        S_diff = np.diff(S, axis=1)
        spectral_flux = np.sqrt(np.mean(S_diff**2, axis=0))
        feats["spec_flux_mean"] = float(np.mean(spectral_flux))
        feats["spec_flux_std"] = float(np.std(spectral_flux))
        feats["spec_flux_skew"] = float(((spectral_flux - np.mean(spectral_flux))**3).mean() / (np.std(spectral_flux)**3 + 1e-9))
        spec_rms = np.sqrt(np.mean(S**2, axis=0))
        spec_peak = np.max(S, axis=0)
        feats["crest_factor_mean"] = float(np.mean(spec_peak / (spec_rms + 1e-10)))
        feats["crest_factor_std"] = float(np.std(spec_peak / (spec_rms + 1e-10)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        feats["mfcc_mean"] = float(np.mean(mfcc)); feats["mfcc_std"] = float(np.std(mfcc))
        feats["mfcc_skew"] = float(((mfcc - np.mean(mfcc))**3).mean() / (np.std(mfcc)**3 + 1e-9))
        feats["mfcc_kurt"] = float(((mfcc - np.mean(mfcc))**4).mean() / (np.std(mfcc)**4 + 1e-9))
        feats["mfcc_delta_mean"] = float(np.mean(np.abs(mfcc_delta)))
        feats["mfcc_delta_std"] = float(np.std(mfcc_delta))
        feats["mfcc_delta2_mean"] = float(np.mean(np.abs(mfcc_delta2)))
        feats["mfcc_delta2_std"] = float(np.std(mfcc_delta2))
        for i in range(3):
            feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
            feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        try:
            order = 2 + sr // 1000
            a = librosa.lpc(y, order=order)
            roots = np.roots(a)
            roots = [r for r in roots if np.imag(r) >= 0]
            angs = np.arctan2(np.imag(roots), np.real(roots))
            freqs_f = sorted([a * sr / (2*np.pi) for a in angs])
            bw = [-np.log(abs(r)) * sr / np.pi for r in roots]
            freqs_f = [(f,b) for f,b in zip(freqs_f, bw) if 90 < f < sr/2 - 90]
            freqs_f = sorted(freqs_f)[:4]
            if len(freqs_f) >= 2:
                f_vals = [f for f,b in freqs_f]; b_vals = [b for f,b in freqs_f]
                feats["formant1_mean"] = float(f_vals[0])
                feats["formant2_mean"] = float(f_vals[1]) if len(f_vals)>1 else float(f_vals[0])
                feats["formant_spread"] = float(np.std(f_vals))
                feats["formant_range"] = float(np.ptp(f_vals))
                feats["formant1_bw"] = float(b_vals[0])
                feats["formant_bw_mean"] = float(np.mean(b_vals))
            else:
                feats["formant1_mean"] = 500.0; feats["formant2_mean"] = 1500.0
                feats["formant_spread"] = 100.0; feats["formant_range"] = 200.0
                feats["formant1_bw"] = 100.0; feats["formant_bw_mean"] = 100.0
        except:
            feats["formant1_mean"] = 500.0; feats["formant2_mean"] = 1500.0
            feats["formant_spread"] = 100.0; feats["formant_range"] = 200.0
            feats["formant1_bw"] = 100.0; feats["formant_bw_mean"] = 100.0
        S_half = S[:S.shape[0]//2, :]
        freqs_half = freqs[:S_half.shape[0]]
        f0_est = feats["f0_mean"]
        if f0_est > 60:
            harm_mask = np.zeros(S_half.shape[0], dtype=bool)
            for i, freq in enumerate(freqs_half):
                for h in range(1, 12):
                    if abs(freq - h * f0_est) < 35:
                        harm_mask[i] = True; break
            S_harm = S_half[harm_mask]; S_noise = S_half[~harm_mask]
            harm_e = np.sum(S_harm**2); noise_e = np.sum(S_noise**2)
            feats["harmonic_prominence"] = float(harm_e / (harm_e + noise_e + 1e-10))
            feats["inharmonicity"] = float(noise_e / (harm_e + noise_e + 1e-10))
        else:
            feats["harmonic_prominence"] = 0.3; feats["inharmonicity"] = 0.7
        phase = np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
        pd = np.diff(phase, axis=1)
        feats["phase_diff_mean"] = float(np.mean(np.abs(pd)))
        feats["phase_diff_std"] = float(np.std(pd))
        gd = -np.diff(np.unwrap(phase, axis=0), axis=0)
        feats["group_delay_mean"] = float(np.mean(np.abs(gd)))
        feats["group_delay_std"] = float(np.std(gd))
        pd_abs = np.abs(pd)
        pd_norm = pd_abs / (np.sum(pd_abs, axis=0, keepdims=True) + 1e-10)
        feats["phase_entropy"] = float(-np.mean(np.sum(pd_norm * np.log2(pd_norm + 1e-10), axis=0)))
        try:
            Y = np.fft.rfft(y[:4096] if len(y)>=4096 else y)
            mag = np.abs(Y); ph = np.angle(Y)
            bis_vals = []
            for k in range(5, min(50, len(Y)-10), 7):
                f1=k; f2=k+3; f3=f1+f2
                if f3 < len(Y):
                    b = mag[f1]*mag[f2]*mag[f3]*np.cos(ph[f1]+ph[f2]-ph[f3])
                    norm = (mag[f1]*mag[f2]*mag[f3] + 1e-9)
                    bis_vals.append(abs(b)/norm)
            feats["bicoherence_mean"] = float(np.mean(bis_vals)) if bis_vals else 0.0
            feats["bicoherence_std"] = float(np.std(bis_vals)) if bis_vals else 0.0
        except:
            feats["bicoherence_mean"] = 0.0; feats["bicoherence_std"] = 0.0
        if len(rms) > 64:
            rms_fft = np.abs(np.fft.rfft(rms - np.mean(rms)))
            feats["mod_spec_peak"] = float(np.max(rms_fft[1:]))
            feats["mod_spec_mean"] = float(np.mean(rms_fft[1:]))
            feats["mod_spec_ratio"] = float(np.max(rms_fft[1:]) / (np.mean(rms_fft[1:]) + 1e-10))
            feats["mod_spec_entropy"] = float(-np.sum((rms_fft[1:]/ (np.sum(rms_fft[1:])+1e-9) ) * np.log2(rms_fft[1:]/ (np.sum(rms_fft[1:])+1e-9) + 1e-10)))
        else:
            feats["mod_spec_peak"] = 0.0; feats["mod_spec_mean"] = 0.0; feats["mod_spec_ratio"] = 1.0; feats["mod_spec_entropy"] = 0.0
        try:
            cepstrum = np.fft.irfft(np.log(S_mean + 1e-10))
            peak_idx = np.argmax(np.abs(cepstrum[20:])) + 20
            feats["cpp"] = float(np.abs(cepstrum[peak_idx]) / (np.mean(np.abs(cepstrum)) + 1e-10))
        except:
            feats["cpp"] = 1.0
        if len(rms) > 20:
            peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0, wait=10)
            if len(peaks) > 2:
                intervals = np.diff(peaks)
                feats["peak_regularity"] = float(1.0 / (np.std(intervals) / (np.mean(intervals) + 1e-6) + 1e-6))
            else:
                feats["peak_regularity"] = 1.0
        else:
            feats["peak_regularity"] = 1.0
        segment_duration = 3.0
        segment_samples = int(segment_duration * sr)
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
                feats[f"seg_{k}_max"] = float(np.max(vals))
                feats[f"seg_{k}_min"] = float(np.min(vals))
        keys = sorted(feats.keys())
        values = [feats[k] for k in keys]
        if len(values) < 192:
            values.extend([0.0] * (192 - len(values)))
        elif len(values) > 192:
            values = values[:192]
        raw_vec = np.array(values, dtype=np.float32)
        # Normalize with scaler if available
        if self.scaler_mean is not None and self.scaler_std is not None:
            # ensure length matches
            if len(self.scaler_mean) == len(raw_vec):
                raw_vec = (raw_vec - self.scaler_mean) / (self.scaler_std + 1e-9)
                raw_vec = np.clip(raw_vec, -10, 10)
        return raw_vec

    def process_audio(self, y: np.ndarray) -> tuple[np.ndarray, dict]:
        vec = self.extract_features_192(y)
        metadata = {"duration_sec": len(y)/self.sr, "sample_rate": self.sr, "feature_dim": len(vec)}
        return vec, metadata
