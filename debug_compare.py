"""Compare feature vectors from training pipeline vs inference pipeline on same audio."""
import sys
import numpy as np
import librosa
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TEST_FILE = list(Path("data/librispeech-genuine").glob("*.wav"))[0]
print(f"Testing on: {TEST_FILE}")

y, sr = librosa.load(str(TEST_FILE), sr=16000, duration=30.0)

# === TRAINING PIPELINE (train_diverse.py) ===
segment_duration = 3.0
segment_samples = int(segment_duration * sr)
segments_train = []

for start in range(0, len(y), segment_samples):
    segment = y[start:start + segment_samples]
    if len(segment) < segment_samples // 2:
        break
    feats = {}
    feats["spectral_centroid_mean"] = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
    feats["spectral_centroid_std"] = float(np.std(librosa.feature.spectral_centroid(y=segment, sr=sr)))
    feats["spectral_bandwidth_mean"] = float(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)))
    feats["spectral_bandwidth_std"] = float(np.std(librosa.feature.spectral_bandwidth(y=segment, sr=sr)))
    feats["spectral_rolloff_mean"] = float(np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr)))
    feats["spectral_rolloff_std"] = float(np.std(librosa.feature.spectral_rolloff(y=segment, sr=sr)))
    feats["spectral_contrast_mean"] = float(np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr)))
    feats["zero_crossing_rate_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(segment)))
    feats["rms_energy_mean"] = float(np.mean(librosa.feature.rms(y=segment)))
    feats["rms_energy_std"] = float(np.std(librosa.feature.rms(y=segment)))
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_values) == 0:
        pitch_values = np.array([0.0])
    feats["pitch_mean"] = float(np.mean(pitch_values))
    feats["pitch_std"] = float(np.std(pitch_values))
    feats["pitch_range"] = float(np.ptp(pitch_values))
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    feats["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    feats["onset_strength_mean"] = float(np.mean(onset_env))
    feats["onset_strength_std"] = float(np.std(onset_env))
    phase = np.angle(librosa.stft(segment, n_fft=2048, hop_length=512))
    phase_diff = np.diff(phase, axis=1)
    feats["phase_diff_mean"] = float(np.mean(np.abs(phase_diff)))
    feats["phase_diff_std"] = float(np.std(phase_diff))
    feats["phase_entropy"] = float(-np.sum(
        (np.abs(phase) / (np.sum(np.abs(phase)) + 1e-10))
        * np.log2(np.abs(phase) / (np.sum(np.abs(phase)) + 1e-10) + 1e-10)
    ))
    stft_mag = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
    feats["stft_energy_mean"] = float(np.mean(stft_mag))
    feats["stft_energy_std"] = float(np.std(stft_mag))
    feats["segment_start_sec"] = start / sr
    segments_train.append(feats)

# Aggregate training
feature_keys = [k for k in segments_train[0] if k != "segment_start_sec"]
agg_train = {}
for key in feature_keys:
    values = [s[key] for s in segments_train]
    agg_train[f"{key}_mean"] = float(np.mean(values))
    agg_train[f"{key}_std"] = float(np.std(values))
    agg_train[f"{key}_max"] = float(np.max(values))
    agg_train[f"{key}_min"] = float(np.min(values))
agg_train["num_segments"] = len(segments_train)
agg_train["total_duration_sec"] = segments_train[-1].get("segment_start_sec", 0) + segment_duration
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
agg_train["mfcc_mean"] = float(np.mean(mfcc))
agg_train["mfcc_std"] = float(np.std(mfcc))

keys_train = sorted(agg_train.keys())
vals_train = [agg_train[k] for k in keys_train]
if len(vals_train) < 64:
    vals_train.extend([0.0] * (64 - len(vals_train)))
fv_train = np.array(vals_train[:64], dtype=np.float32)

# === INFERENCE PIPELINE (audio_processor.py) ===
from voice_detection_app.services.audio_processor import AudioProcessor
ap = AudioProcessor()
y2, sr2 = ap.load_audio(TEST_FILE)
_, agg_inf = ap.process_audio(y2)
fv_inf = ap.get_feature_vector(agg_inf, target_length=64)

# === COMPARE ===
print(f"\nTraining keys ({len(keys_train)}): {keys_train[:10]}...")
print(f"Inference keys: {sorted(agg_inf.keys())[:10]}...")
print(f"\nTraining fv shape: {fv_train.shape}")
print(f"Inference fv shape: {fv_inf.shape}")

print(f"\nTraining fv[:5]: {fv_train[:5]}")
print(f"Inference fv[:5]: {fv_inf[:5]}")

print(f"\nMax abs diff: {np.max(np.abs(fv_train - fv_inf)):.6f}")
print(f"Mean abs diff: {np.mean(np.abs(fv_train - fv_inf)):.6f}")

# Check which features differ
for i, (k, v1, v2) in enumerate(zip(keys_train[:64], fv_train[:64], fv_inf[:64])):
    diff = abs(v1 - v2)
    if diff > 0.01:
        print(f"  DIFF [{i}] {k}: train={v1:.4f} infer={v2:.4f} diff={diff:.4f}")
