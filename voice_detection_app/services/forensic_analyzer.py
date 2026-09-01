"""Forensic voice analysis v2 - multi-factor AI detection with empirically-calibrated features.

Calibrated from find_best_features.py analysis of 40 LibriSpeech + 40 WaveFake
(11,666 files total, top 10 features by separation score).
"""
import numpy as np
import librosa


# Empirical calibration from find_best_features.py (11,666 files)
# separation = |genuine_mean - synthetic_mean| / (genuine_std + synthetic_std)
CALIBRATION = {
    # scon_std: synthetic=12.060+-0.311, genuine=8.556+-3.668, sep=1.76
    "scon_std":           {"genuine_mean": 8.556,  "synthetic_mean": 12.060, "higher_is_genuine": False},
    # f0_mean: synthetic=246.940+-11.048, genuine=180.335+-65.522, sep=1.74
    "f0_mean":            {"genuine_mean": 180.335, "synthetic_mean": 246.940, "higher_is_genuine": False},
    # formant1_mean: genuine=1781.480+-266.812, synthetic=1529.925+-28.925, sep=1.70
    "formant1_mean":      {"genuine_mean": 1781.480, "synthetic_mean": 1529.925, "higher_is_genuine": True},
    # mfcc_mean: genuine=-14.219+-5.451, synthetic=-19.497+-0.887, sep=1.67
    "mfcc_mean":          {"genuine_mean": -14.219, "synthetic_mean": -19.497, "higher_is_genuine": True},
    # harmonic_prominence: genuine=0.310+-0.118, synthetic=0.194+-0.028, sep=1.59
    "harmonic_prominence":{"genuine_mean": 0.310,  "synthetic_mean": 0.194,  "higher_is_genuine": True},
    # scon_mean: genuine=21.188+-3.701, synthetic=24.448+-0.489, sep=1.56
    "scon_mean":          {"genuine_mean": 21.188, "synthetic_mean": 24.448, "higher_is_genuine": False},
    # silence_ratio: genuine=0.162+-0.097, synthetic=0.079+-0.014, sep=1.50
    "silence_ratio":      {"genuine_mean": 0.162,  "synthetic_mean": 0.079,  "higher_is_genuine": True},
    # sb_cv: genuine=0.264+-0.065, synthetic=0.320+-0.015, sep=1.40
    "sb_cv":              {"genuine_mean": 0.264,  "synthetic_mean": 0.320,  "higher_is_genuine": False},
    # f0_std: genuine=46.526+-19.935, synthetic=63.652+-4.712, sep=1.39
    "f0_std":             {"genuine_mean": 46.526, "synthetic_mean": 63.652, "higher_is_genuine": False},
    # formant_range: genuine=2799.053+-527.092, synthetic=2380.602+-81.394, sep=1.38
    "formant_range":      {"genuine_mean": 2799.053, "synthetic_mean": 2380.602, "higher_is_genuine": True},
}

# Weights proportional to separation score
TOTAL_SEP = sum(v["sep"] for v in [
    {"sep": 1.76}, {"sep": 1.74}, {"sep": 1.70}, {"sep": 1.67}, {"sep": 1.59},
    {"sep": 1.56}, {"sep": 1.50}, {"sep": 1.40}, {"sep": 1.39}, {"sep": 1.38},
])
FACTOR_WEIGHTS = {
    "scon_std":            1.76 / TOTAL_SEP,
    "f0_mean":             1.74 / TOTAL_SEP,
    "formant1_mean":       1.70 / TOTAL_SEP,
    "mfcc_mean":           1.67 / TOTAL_SEP,
    "harmonic_prominence": 1.59 / TOTAL_SEP,
    "scon_mean":           1.56 / TOTAL_SEP,
    "silence_ratio":       1.50 / TOTAL_SEP,
    "sb_cv":               1.40 / TOTAL_SEP,
    "f0_std":              1.39 / TOTAL_SEP,
    "formant_range":       1.38 / TOTAL_SEP,
}


def _score_factor(value: float, cal: dict) -> float:
    """Map raw metric -> synthetic probability 0..1 via sigmoid centered at midpoint."""
    g = cal["genuine_mean"]
    s = cal["synthetic_mean"]
    midpoint = (g + s) / 2.0
    scale = abs(s - g) / 4.0 + 1e-6
    if cal["higher_is_genuine"]:
        logit = (midpoint - value) / scale
    else:
        logit = (value - midpoint) / scale
    logit = float(np.clip(logit, -10, 10))
    prob = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(prob, 0.02, 0.98))


def extract_forensic_metrics(y: np.ndarray, sr: int = 16000) -> dict:
    """Extract raw forensic metrics from audio."""
    metrics = {}

    # === F0 via pyin (more accurate than piptrack) ===
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=60, fmax=500, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 10:
        f0_clean = np.array([120.0])
    metrics["f0_mean"] = float(np.mean(f0_clean))
    metrics["f0_std"] = float(np.std(f0_clean))
    metrics["f0_cv"] = float(np.std(f0_clean) / (np.mean(f0_clean) + 1e-6))
    metrics["f0_range"] = float(np.ptp(f0_clean))

    # === Spectral centroid ===
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    metrics["sc_mean"] = float(np.mean(sc))
    metrics["sc_std"] = float(np.std(sc))
    metrics["sc_cv"] = float(np.std(sc) / (np.mean(sc) + 1e-6))

    # === Spectral bandwidth ===
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    metrics["sb_mean"] = float(np.mean(sb))
    metrics["sb_std"] = float(np.std(sb))
    metrics["sb_cv"] = float(np.std(sb) / (np.mean(sb) + 1e-6))

    # === Spectral contrast ===
    scon = librosa.feature.spectral_contrast(y=y, sr=sr)
    metrics["scon_mean"] = float(np.mean(scon))
    metrics["scon_std"] = float(np.std(scon))

    # === Formants via LPC ===
    try:
        order = 2 + sr // 1000
        a = librosa.lpc(y, order=order)
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = sorted(angles * (sr / (2 * np.pi)))
        freqs = [f for f in freqs if 90 < f < sr / 2 - 90][:4]
        if len(freqs) >= 2:
            metrics["formant1_mean"] = float(freqs[0])
            metrics["formant_range"] = float(np.ptp(freqs))
            metrics["formant_spread"] = float(np.std(freqs))
        else:
            metrics["formant1_mean"] = 1500.0
            metrics["formant_range"] = 2500.0
            metrics["formant_spread"] = 100.0
    except:
        metrics["formant1_mean"] = 1500.0
        metrics["formant_range"] = 2500.0
        metrics["formant_spread"] = 100.0

    # === MFCC ===
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    metrics["mfcc_mean"] = float(np.mean(mfcc))
    metrics["mfcc_std"] = float(np.std(mfcc))
    mfcc_delta = librosa.feature.delta(mfcc)
    metrics["mfcc_delta_std"] = float(np.std(mfcc_delta))

    # === Harmonic prominence ===
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_half = S[:S.shape[0] // 2, :]
    f0_est = metrics["f0_mean"]
    if f0_est > 60:
        harm_mask = np.zeros(S_half.shape[0], dtype=bool)
        for i in range(S_half.shape[0]):
            freq = i * sr / (2 * S_half.shape[0])
            for h in range(1, 10):
                if abs(freq - h * f0_est) < 40:
                    harm_mask[i] = True
                    break
        S_harm = S_half[harm_mask]
        S_noise = S_half[~harm_mask]
        harm_e = float(np.sum(S_harm ** 2))
        noise_e = float(np.sum(S_noise ** 2))
        metrics["harmonic_prominence"] = harm_e / (harm_e + noise_e + 1e-10)
    else:
        metrics["harmonic_prominence"] = 0.3

    # === Silence ratio ===
    rms = librosa.feature.rms(y=y)[0]
    metrics["rms_mean"] = float(np.mean(rms))
    metrics["rms_cv"] = float(np.std(rms) / (np.mean(rms) + 1e-6))
    threshold = np.mean(rms) * 0.1
    silent = rms < threshold
    metrics["silence_ratio"] = float(np.mean(silent))

    # === Phase ===
    phase = np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
    pd = np.diff(phase, axis=1)
    metrics["pd_std"] = float(np.std(np.abs(pd)))

    # === ZCR ===
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    metrics["zcr_mean"] = float(np.mean(zcr))

    # === Spectral rolloff ===
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    metrics["rolloff_std"] = float(np.std(rolloff))

    return metrics


INTERPRETATIONS = {
    "scon_std":            lambda v, s: "Unstable spectral contrast (AI-like)" if s > 0.6 else "Stable spectral contrast (human-like)" if s < 0.4 else "Moderate spectral contrast",
    "f0_mean":             lambda v, s: "High pitch center (AI-like)" if s > 0.6 else "Natural pitch center (human-like)" if s < 0.4 else "Moderate pitch center",
    "formant1_mean":       lambda v, s: "Narrow formant resonance (AI-like)" if s > 0.6 else "Wide formant resonance (human-like)" if s < 0.4 else "Moderate formant resonance",
    "mfcc_mean":           lambda v, s: "Atypical spectral envelope (AI-like)" if s > 0.6 else "Natural spectral envelope (human-like)" if s < 0.4 else "Moderate spectral envelope",
    "harmonic_prominence": lambda v, s: "Weak harmonic structure (AI-like)" if s > 0.6 else "Strong harmonic structure (human-like)" if s < 0.4 else "Moderate harmonic structure",
    "scon_mean":           lambda v, s: "High spectral contrast (AI-like)" if s > 0.6 else "Low spectral contrast (human-like)" if s < 0.4 else "Moderate spectral contrast",
    "silence_ratio":       lambda v, s: "Few natural pauses (AI-like)" if s > 0.6 else "Natural pause patterns (human-like)" if s < 0.4 else "Moderate pause patterns",
    "sb_cv":               lambda v, s: "Unstable bandwidth (AI-like)" if s > 0.6 else "Stable bandwidth (human-like)" if s < 0.4 else "Moderate bandwidth",
    "f0_std":              lambda v, s: "High pitch jitter (AI-like)" if s > 0.6 else "Natural pitch variation (human-like)" if s < 0.4 else "Moderate pitch variation",
    "formant_range":       lambda v, s: "Narrow formant range (AI-like)" if s > 0.6 else "Wide formant range (human-like)" if s < 0.4 else "Moderate formant range",
}


def analyze_forensic(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Return detailed forensic breakdown with empirically-calibrated factors.
    """
    raw = extract_forensic_metrics(y, sr)
    factors = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for name, weight in FACTOR_WEIGHTS.items():
        if name not in raw:
            continue
        val = raw[name]
        cal = CALIBRATION[name]
        syn_score = _score_factor(val, cal)
        human_score = 1.0 - syn_score
        interp = INTERPRETATIONS.get(name, lambda v, s: "AI-like" if s > 0.6 else "Human-like" if s < 0.4 else "Inconclusive")(val, syn_score)
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
        weighted_sum += syn_score * weight
        weight_total += weight

    forensic_score = weighted_sum / (weight_total + 1e-9)
    forensic_score = float(np.clip(forensic_score, 0.02, 0.98))

    human_similarity = (1 - forensic_score) * 100
    ai_similarity = forensic_score * 100

    # Dominant clues: top 3 factors where score is most extreme
    sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]["synthetic_score"] - 0.5), reverse=True)
    dominant = []
    for fname, fdata in sorted_factors[:3]:
        dominant.append({
            "factor": fname,
            "status": fdata["status"],
            "confidence": round(abs(fdata["synthetic_score"] - 0.5) * 200, 1),
            "interpretation": fdata["interpretation"],
        })

    return {
        "forensic_score": round(forensic_score, 4),
        "forensic_synthetic_percent": round(forensic_score * 100, 1),
        "forensic_human_percent": round((1 - forensic_score) * 100, 1),
        "human_similarity": round(human_similarity, 1),
        "ai_similarity": round(ai_similarity, 1),
        "factors": factors,
        "dominant_clues": dominant,
        "raw_metrics": {k: round(float(v), 4) for k, v in raw.items()},
        "is_synthetic_forensic": forensic_score > 0.5,
    }


def hybrid_score(ml_prob: float, forensic_score: float, ml_weight: float = 0.55) -> dict:
    """
    Combine ML and forensic scores.
    ML weight 0.55, forensic 0.45 - forensic corrects ML bias on unseen TTS.
    """
    forensic_weight = 1.0 - ml_weight
    combined = ml_prob * ml_weight + forensic_score * forensic_weight
    # Agreement boost
    if (ml_prob > 0.6 and forensic_score > 0.6) or (ml_prob < 0.4 and forensic_score < 0.4):
        if combined > 0.5:
            combined = min(0.98, combined + 0.08)
        else:
            combined = max(0.02, combined - 0.08)
    # Forensic override for strong signals
    if forensic_score > 0.75 and ml_prob < 0.4:
        combined = max(combined, 0.62)
    if forensic_score < 0.25 and ml_prob > 0.6:
        combined = min(combined, 0.38)

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
