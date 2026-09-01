"""Forensic voice analysis - multi-factor AI detection beyond ML.

Each factor independently scores synthetic likelihood 0..1 based on
calibrated distributions from LibriSpeech (genuine) vs WaveFake (synthetic).
"""
import numpy as np
import librosa


# Calibrated thresholds from 40 genuine (LibriSpeech) vs 40 synthetic (WaveFake) - 10 sec each
# Values chosen at midpoint between means, with sigmoid slope to map to 0..1
CALIBRATION = {
    # Lower pitch_cv = more synthetic (AI has stable pitch)
    "pitch_cv":        {"genuine_mean": 0.3157, "synthetic_mean": 0.2906, "higher_is_genuine": True},
    # Higher spectral centroid CV = more synthetic
    "sc_cv":           {"genuine_mean": 0.5291, "synthetic_mean": 0.6243, "higher_is_genuine": False},
    # Lower RMS CV = more synthetic (AI has flat energy)
    "rms_cv":          {"genuine_mean": 0.8116, "synthetic_mean": 0.6286, "higher_is_genuine": True},
    # Higher spectral bandwidth std = more synthetic
    "sb_std":          {"genuine_mean": 425.9,  "synthetic_mean": 509.9,  "higher_is_genuine": False},
    # Lower mfcc delta std = more genuine?? Actually genuine 6.72 synthetic 7.42 => higher synthetic
    "mfcc_delta_std":  {"genuine_mean": 6.72,   "synthetic_mean": 7.42,   "higher_is_genuine": False},
    # Higher phase diff std = more synthetic (very slight)
    "pd_std":          {"genuine_mean": 2.5547, "synthetic_mean": 2.5594, "higher_is_genuine": False},
    # Pitch range: genuine slightly higher? but weak
    "pitch_range":     {"genuine_mean": 346.87, "synthetic_mean": 346.18, "higher_is_genuine": True},
    # Spectral centroid std: synthetic higher
    "sc_std":          {"genuine_mean": 1156.3, "synthetic_mean": 1292.9, "higher_is_genuine": False},
}

# Weights per factor (based on separation score)
FACTOR_WEIGHTS = {
    "pitch_cv": 0.22,
    "rms_cv": 0.18,
    "sb_std": 0.17,
    "sc_cv": 0.12,
    "pd_std": 0.10,
    "mfcc_delta_std": 0.09,
    "sc_std": 0.07,
    "pitch_range": 0.05,
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

    # Pitch
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pv = pitches[mags > np.median(mags)]
    if len(pv) < 10:
        pv = pitches.flatten()
    pv = pv[(pv > 60) & (pv < 500)]
    if len(pv) < 10:
        pv = np.array([120.0])
    pitch_mean = float(np.mean(pv))
    pitch_std = float(np.std(pv))
    metrics["pitch_cv"] = pitch_std / (pitch_mean + 1e-6)
    metrics["pitch_range"] = float(np.ptp(pv))
    metrics["pitch_mean"] = pitch_mean
    metrics["pitch_std"] = pitch_std

    # Spectral centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    metrics["sc_mean"] = float(np.mean(sc))
    metrics["sc_std"] = float(np.std(sc))
    metrics["sc_cv"] = float(np.std(sc) / (np.mean(sc) + 1e-6))

    # Spectral bandwidth
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    metrics["sb_std"] = float(np.std(sb))
    metrics["sb_mean"] = float(np.mean(sb))

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    metrics["rms_cv"] = float(np.std(rms) / (np.mean(rms) + 1e-6))
    metrics["rms_std"] = float(np.std(rms))
    metrics["rms_mean"] = float(np.mean(rms))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    metrics["zcr_std"] = float(np.std(zcr))
    metrics["zcr_mean"] = float(np.mean(zcr))

    # Phase
    phase = np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
    pd = np.diff(phase, axis=1)
    metrics["pd_std"] = float(np.std(np.abs(pd)))
    metrics["pd_mean"] = float(np.mean(np.abs(pd)))

    # MFCC delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    metrics["mfcc_delta_std"] = float(np.std(delta))
    metrics["mfcc_mean"] = float(np.mean(mfcc))

    # Additional: spectral rolloff std, spectral flux
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    metrics["rolloff_std"] = float(np.std(rolloff))
    # Spectral flux = mean squared diff of successive magnitude spectra
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    flux = np.mean(np.sqrt(np.mean(np.diff(S, axis=1)**2, axis=0)))
    metrics["spectral_flux"] = float(flux)

    # Jitter-like: pitch period variation
    if len(pv) > 20:
        pitch_diff = np.abs(np.diff(pv))
        metrics["pitch_jitter"] = float(np.mean(pitch_diff) / (pitch_mean + 1e-6))
    else:
        metrics["pitch_jitter"] = 0.02

    return metrics


def analyze_forensic(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Return detailed forensic breakdown:
      - factors: per-factor {raw_value, synthetic_score, human_score, weight, interpretation, status}
      - forensic_score: weighted synthetic probability 0..1
      - human_similarity, ai_similarity %
      - dominant_clues: list of top clues
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

        # Interpretation
        if name == "pitch_cv":
            interp = "Voice pitch is very stable (AI-like)" if syn_score > 0.6 else "Natural pitch variation (human-like)" if syn_score < 0.4 else "Moderate pitch variation"
        elif name == "rms_cv":
            interp = "Flat loudness (AI-like)" if syn_score > 0.6 else "Natural energy dynamics (human-like)" if syn_score < 0.4 else "Moderate energy variation"
        elif name == "sb_std":
            interp = "Wide spectral spread variation (AI-like)" if syn_score > 0.6 else "Consistent spectral spread (human-like)" if syn_score < 0.4 else "Moderate spectral variation"
        elif name == "sc_cv":
            interp = "Unstable brightness (AI-like)" if syn_score > 0.6 else "Stable brightness (human-like)" if syn_score < 0.4 else "Moderate brightness variation"
        elif name == "mfcc_delta_std":
            interp = "High articulation change (AI-like)" if syn_score > 0.6 else "Smooth articulation (human-like)" if syn_score < 0.4 else "Moderate articulation"
        elif name == "pd_std":
            interp = "Irregular phase (AI-like)" if syn_score > 0.6 else "Coherent phase (human-like)" if syn_score < 0.4 else "Moderate phase coherence"
        elif name == "pitch_range":
            interp = "Narrow pitch range (AI-like)" if syn_score > 0.6 else "Wide pitch range (human-like)" if syn_score < 0.4 else "Moderate pitch range"
        elif name == "sc_std":
            interp = "Large brightness swings (AI-like)" if syn_score > 0.6 else "Stable brightness (human-like)" if syn_score < 0.4 else "Moderate brightness swings"
        else:
            interp = "AI-like" if syn_score > 0.6 else "Human-like" if syn_score < 0.4 else "Inconclusive"

        status = "AI" if syn_score > 0.6 else "HUMAN" if syn_score < 0.4 else "UNCLEAR"
        factors[name] = {
            "raw_value": round(float(val), 4),
            "synthetic_score": round(syn_score, 4),
            "human_score": round(human_score, 4),
            "synthetic_percent": round(syn_score * 100, 1),
            "human_percent": round(human_score * 100, 1),
            "weight": weight,
            "interpretation": interp,
            "status": status,
        }
        weighted_sum += syn_score * weight
        weight_total += weight

    forensic_score = weighted_sum / (weight_total + 1e-9)
    forensic_score = float(np.clip(forensic_score, 0.02, 0.98))

    # Additional overall metrics for display
    human_similarity = (1 - forensic_score) * 100
    ai_similarity = forensic_score * 100

    # Dominant clues: top 3 factors where score is most extreme
    sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]["synthetic_score"] - 0.5), reverse=True)
    dominant = []
    for fname, fdata in sorted_factors[:3]:
        dominant.append({
            "factor": fname,
            "status": fdata["status"],
            "confidence": round(abs(fdata["synthetic_score"] - 0.5) * 200, 1),  # 0..100
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
    If both agree, confidence high. If disagree, forensic acts as tie-breaker.
    """
    forensic_weight = 1.0 - ml_weight
    combined = ml_prob * ml_weight + forensic_score * forensic_weight
    # Confidence boosting: if both >0.6 or both <0.4, boost confidence
    if (ml_prob > 0.6 and forensic_score > 0.6) or (ml_prob < 0.4 and forensic_score < 0.4):
        # Agreement -> push further from 0.5
        if combined > 0.5:
            combined = min(0.98, combined + 0.08)
        else:
            combined = max(0.02, combined - 0.08)
    # Strong forensic AI signal should not be ignored even if ML says human
    if forensic_score > 0.75 and ml_prob < 0.4:
        combined = max(combined, 0.62)  # force to at least 62% AI if forensic very sure
    if forensic_score < 0.25 and ml_prob > 0.6:
        combined = min(combined, 0.38)  # force to at most 38% if forensic very sure human

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
