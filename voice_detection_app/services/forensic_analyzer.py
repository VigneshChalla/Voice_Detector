"""Forensic v3 - 15-factor Tier-2 analysis calibrated on 100 files (50 genuine / 50 synthetic)."""
import numpy as np
import librosa

# Calibrated from find_best_v3.py (50 genuine Libra+WaveFake vs 50 WaveFake synthetic)
CALIBRATION = {
    "harmonic_prominence": {"genuine_mean": 0.45, "synthetic_mean": 0.24, "higher_is_genuine": True},
    "scon_std":            {"genuine_mean": 9.29, "synthetic_mean": 12.06, "higher_is_genuine": False},
    "formant2_mean":       {"genuine_mean": 1121.19, "synthetic_mean": 833.81, "higher_is_genuine": True},
    "f0_q75":              {"genuine_mean": 221.04, "synthetic_mean": 287.45, "higher_is_genuine": False},
    "pause_mean_dur":      {"genuine_mean": 0.23, "synthetic_mean": 0.13, "higher_is_genuine": True},
    "f0_mean":             {"genuine_mean": 192.70, "synthetic_mean": 244.38, "higher_is_genuine": False},
    "mfcc_mean":           {"genuine_mean": -8.15, "synthetic_mean": -9.93, "higher_is_genuine": True},
    "mfcc_3_mean":         {"genuine_mean": 12.29, "synthetic_mean": -3.18, "higher_is_genuine": True},
    "scon_mean":           {"genuine_mean": 22.02, "synthetic_mean": 24.41, "higher_is_genuine": False},
    "silence_ratio":       {"genuine_mean": 0.14, "synthetic_mean": 0.09, "higher_is_genuine": True},
    "formant_range":       {"genuine_mean": 2712.91, "synthetic_mean": 2372.21, "higher_is_genuine": True},
    "sb_cv":               {"genuine_mean": 0.28, "synthetic_mean": 0.32, "higher_is_genuine": False},
    "mfcc_skew":           {"genuine_mean": -5.25, "synthetic_mean": -4.62, "higher_is_genuine": False},
    "mfcc_kurt":           {"genuine_mean": 44.71, "synthetic_mean": 38.56, "higher_is_genuine": True},
    "subcent_sb_0_1k":     {"genuine_mean": 471.05, "synthetic_mean": 525.73, "higher_is_genuine": False},
}

# Weights proportional to separation score
_seps = {
    "harmonic_prominence": 0.79, "scon_std": 0.71, "formant2_mean": 0.69, "f0_q75": 0.68, "pause_mean_dur": 0.66,
    "f0_mean": 0.65, "mfcc_mean": 0.62, "mfcc_3_mean": 0.60, "scon_mean": 0.60, "silence_ratio": 0.58,
    "formant_range": 0.57, "sb_cv": 0.56, "mfcc_skew": 0.62, "mfcc_kurt": 0.60, "subcent_sb_0_1k": 0.53,
}
_total = sum(_seps.values())
FACTOR_WEIGHTS = {k: v/_total for k,v in _seps.items()}

def _score_factor(value: float, cal: dict) -> float:
    g = cal["genuine_mean"]; s = cal["synthetic_mean"]
    midpoint = (g + s) / 2.0
    scale = abs(s - g) / 4.0 + 1e-6
    logit = (midpoint - value) / scale if cal["higher_is_genuine"] else (value - midpoint) / scale
    logit = float(np.clip(logit, -10, 10))
    prob = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(prob, 0.02, 0.98))

def extract_forensic_metrics(y: np.ndarray, sr: int = 16000) -> dict:
    metrics = {}
    # F0
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=60, fmax=500, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 10:
        f0_clean = np.array([120.0])
    metrics["f0_mean"] = float(np.mean(f0_clean))
    metrics["f0_q75"] = float(np.percentile(f0_clean, 75))
    metrics["f0_std"] = float(np.std(f0_clean))
    # Formants
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
            f_vals = [f for f,b in freqs_f]
            metrics["formant2_mean"] = float(sorted(f_vals)[1] if len(f_vals)>1 else f_vals[0])
            metrics["formant_range"] = float(np.ptp(f_vals))
        else:
            metrics["formant2_mean"] = 1000.0; metrics["formant_range"] = 2500.0
    except:
        metrics["formant2_mean"] = 1000.0; metrics["formant_range"] = 2500.0
    # Spectral contrast
    scon = librosa.feature.spectral_contrast(y=y, sr=sr)
    metrics["scon_mean"] = float(np.mean(scon)); metrics["scon_std"] = float(np.std(scon))
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    metrics["sb_cv"] = float(np.std(sb) / (np.mean(sb) + 1e-6))
    # Harmonic prominence
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_mean = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    S_half = S[:S.shape[0]//2, :]
    freqs_half = freqs[:S_half.shape[0]]
    f0_est = metrics["f0_mean"]
    if f0_est > 60:
        harm_mask = np.zeros(S_half.shape[0], dtype=bool)
        for i, freq in enumerate(freqs_half):
            for h in range(1, 12):
                if abs(freq - h * f0_est) < 35:
                    harm_mask[i] = True; break
        S_harm = S_half[harm_mask]; S_noise = S_half[~harm_mask]
        harm_e = np.sum(S_harm**2); noise_e = np.sum(S_noise**2)
        metrics["harmonic_prominence"] = float(harm_e / (harm_e + noise_e + 1e-10))
    else:
        metrics["harmonic_prominence"] = 0.3
    # MFCC 40
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    metrics["mfcc_mean"] = float(np.mean(mfcc))
    metrics["mfcc_skew"] = float(((mfcc - np.mean(mfcc))**3).mean() / (np.std(mfcc)**3 + 1e-9))
    metrics["mfcc_kurt"] = float(((mfcc - np.mean(mfcc))**4).mean() / (np.std(mfcc)**4 + 1e-9))
    metrics["mfcc_3_mean"] = float(np.mean(mfcc[2]))
    # Subcent 0-1k
    total_energy = np.sum(S_mean) + 1e-9
    idx = np.where((freqs >= 0) & (freqs < 1000))[0]
    if len(idx)>0:
        e = np.sum(S_mean[idx]) + 1e-9
        c = np.sum(freqs[idx] * S_mean[idx]) / e
        metrics["subcent_sb_0_1k"] = float(c)
    else:
        metrics["subcent_sb_0_1k"] = 500.0
    # Pause / silence
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    threshold = np.mean(rms) * 0.12
    silent = rms < threshold
    metrics["silence_ratio"] = float(np.mean(silent))
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
    metrics["pause_mean_dur"] = float(np.mean(pauses)) if pauses else 0.0
    # Phase
    phase = np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
    pd = np.diff(phase, axis=1)
    metrics["phase_diff_std"] = float(np.std(np.abs(pd)))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    metrics["zcr_mean"] = float(np.mean(zcr))
    return metrics

INTERPRETATIONS = {
    "harmonic_prominence": lambda v,s: "Strong harmonic structure (human-like)" if s < 0.4 else "Weak harmonic structure (AI-like)" if s > 0.6 else "Moderate harmonic structure",
    "scon_std": lambda v,s: "Stable spectral contrast (human-like)" if s < 0.4 else "Unstable spectral contrast (AI-like)" if s > 0.6 else "Moderate spectral contrast",
    "formant2_mean": lambda v,s: "Wide vocal tract resonance (human-like)" if s < 0.4 else "Narrow resonance (AI-like)" if s > 0.6 else "Moderate resonance",
    "f0_q75": lambda v,s: "Natural pitch ceiling (human-like)" if s < 0.4 else "High pitch ceiling (AI-like)" if s > 0.6 else "Moderate pitch ceiling",
    "pause_mean_dur": lambda v,s: "Natural pauses (human-like)" if s < 0.4 else "Short/unnatural pauses (AI-like)" if s > 0.6 else "Moderate pauses",
    "f0_mean": lambda v,s: "Natural pitch center (human-like)" if s < 0.4 else "High pitch center (AI-like)" if s > 0.6 else "Moderate pitch center",
    "mfcc_mean": lambda v,s: "Natural spectral envelope (human-like)" if s < 0.4 else "Atypical envelope (AI-like)" if s > 0.6 else "Moderate envelope",
    "mfcc_3_mean": lambda v,s: "Natural cepstral structure (human-like)" if s < 0.4 else "Atypical cepstral (AI-like)" if s > 0.6 else "Moderate cepstral",
    "scon_mean": lambda v,s: "Low spectral contrast (human-like)" if s < 0.4 else "High contrast (AI-like)" if s > 0.6 else "Moderate contrast",
    "silence_ratio": lambda v,s: "Natural pause patterns (human-like)" if s < 0.4 else "Few pauses (AI-like)" if s > 0.6 else "Moderate pauses",
    "formant_range": lambda v,s: "Wide formant range (human-like)" if s < 0.4 else "Narrow range (AI-like)" if s > 0.6 else "Moderate range",
    "sb_cv": lambda v,s: "Stable bandwidth (human-like)" if s < 0.4 else "Unstable bandwidth (AI-like)" if s > 0.6 else "Moderate bandwidth",
    "mfcc_skew": lambda v,s: "Natural cepstral skew (human-like)" if s < 0.4 else "Skewed cepstral (AI-like)" if s > 0.6 else "Moderate skew",
    "mfcc_kurt": lambda v,s: "Natural cepstral kurtosis (human-like)" if s < 0.4 else "Flat kurtosis (AI-like)" if s > 0.6 else "Moderate kurtosis",
    "subcent_sb_0_1k": lambda v,s: "Natural low-freq centroid (human-like)" if s < 0.4 else "High low-freq centroid (AI-like)" if s > 0.6 else "Moderate centroid",
}

def analyze_forensic(y: np.ndarray, sr: int = 16000) -> dict:
    raw = extract_forensic_metrics(y, sr)
    factors = {}
    weighted_sum = 0.0; weight_total = 0.0
    for name, weight in FACTOR_WEIGHTS.items():
        if name not in raw:
            continue
        val = raw[name]; cal = CALIBRATION[name]
        syn_score = _score_factor(val, cal)
        human_score = 1.0 - syn_score
        interp = INTERPRETATIONS.get(name, lambda v,s: "AI-like" if s>0.6 else "Human-like" if s<0.4 else "Inconclusive")(val, syn_score)
        status = "AI" if syn_score > 0.6 else "HUMAN" if syn_score < 0.4 else "UNCLEAR"
        factors[name] = {
            "raw_value": round(float(val), 4),
            "synthetic_score": round(syn_score, 4),
            "human_score": round(human_score, 4),
            "synthetic_percent": round(syn_score * 100, 1),
            "human_percent": round(human_score * 100, 1),
            "weight": round(weight, 4),
            "interpretation": interp,
            "status": status,
        }
        weighted_sum += syn_score * weight; weight_total += weight
    forensic_score = weighted_sum / (weight_total + 1e-9)
    forensic_score = float(np.clip(forensic_score, 0.02, 0.98))
    # Dominant clues
    sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]["synthetic_score"] - 0.5), reverse=True)
    dominant = []
    for fname, fdata in sorted_factors[:3]:
        dominant.append({"factor": fname, "status": fdata["status"], "confidence": round(abs(fdata["synthetic_score"] - 0.5) * 200, 1), "interpretation": fdata["interpretation"]})
    return {
        "forensic_score": round(forensic_score, 4),
        "forensic_synthetic_percent": round(forensic_score * 100, 1),
        "forensic_human_percent": round((1 - forensic_score) * 100, 1),
        "human_similarity": round((1 - forensic_score) * 100, 1),
        "ai_similarity": round(forensic_score * 100, 1),
        "factors": factors,
        "dominant_clues": dominant,
        "raw_metrics": {k: round(float(v), 4) for k, v in raw.items()},
        "is_synthetic_forensic": forensic_score > 0.5,
    }

def hybrid_score(ml_prob: float, forensic_score: float, ml_weight: float = 0.70) -> dict:
    """ML dominates (AUC 0.996); forensic only adjusts when ML uncertain."""
    forensic_weight = 1.0 - ml_weight
    combined = ml_prob * ml_weight + forensic_score * forensic_weight
    # Agreement boost only when both confident
    if (ml_prob > 0.7 and forensic_score > 0.65) or (ml_prob < 0.3 and forensic_score < 0.35):
        if combined > 0.5:
            combined = min(0.98, combined + 0.05)
        else:
            combined = max(0.02, combined - 0.05)
    # Forensic override ONLY when ML is uncertain (0.15-0.85), not when ML is extreme
    if forensic_score > 0.80 and 0.15 < ml_prob < 0.55:
        combined = max(combined, 0.60)
    if forensic_score < 0.20 and 0.45 < ml_prob < 0.85:
        combined = min(combined, 0.40)
    combined = float(np.clip(combined, 0.02, 0.98))
    return {
        "final_synthetic_prob": round(combined, 4),
        "final_genuine_prob": round(1 - combined, 4),
        "final_synthetic_percent": round(combined * 100, 1),
        "final_human_percent": round((1 - combined) * 100, 1),
        "is_synthetic": combined > 0.5,
        "ml_contribution": round(ml_prob * ml_weight / (combined + 1e-9) * 100, 1) if combined > 0 else 50,
        "forensic_contribution": round(forensic_score * forensic_weight / (combined + 1e-9) * 100, 1) if combined > 0 else 50,
        "agreement": "AGREE" if (ml_prob > 0.5) == (forensic_score > 0.5) else "DISAGREE",
        "confidence": round(abs(combined - 0.5) * 200, 1),
    }
