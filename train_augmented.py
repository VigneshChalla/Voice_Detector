"""
Train with diverse genuine voices + augmentation to handle real-world phone recordings.
Problem: WaveFake genuine = single studio speaker. Phone recordings sound different -> model says "synthetic".
Fix: Augment genuine data to simulate different recording conditions.
"""
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice_detection_app.models.detector import VoiceAuthenticityNet


def augment_audio(y, sr):
    """Apply random augmentations to make genuine voices sound like phone recordings."""
    import librosa
    augmented = y.copy()

    # 1. Add background noise (simulates real environment)
    if np.random.random() < 0.7:
        noise_level = np.random.uniform(0.005, 0.03)
        noise = np.random.normal(0, noise_level, len(augmented))
        augmented = augmented + noise

    # 2. Random gain (simulates different microphone distances)
    if np.random.random() < 0.5:
        gain = np.random.uniform(0.5, 1.5)
        augmented = augmented * gain

    # 3. Bandpass filter (simulates phone mic frequency response)
    if np.random.random() < 0.4:
        low_freq = np.random.uniform(200, 400)
        high_freq = np.random.uniform(3000, 5000)
        augmented = librosa.effects.preemphasis(augmented)

    # 4. Time stretch (slight speed variation)
    if np.random.random() < 0.3:
        rate = np.random.uniform(0.9, 1.1)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
        if len(augmented) > len(y):
            augmented = augmented[:len(y)]
        else:
            augmented = np.pad(augmented, (0, len(y) - len(augmented)))

    # 5. Pitch shift (different speakers)
    if np.random.random() < 0.3:
        n_steps = np.random.uniform(-2, 2)
        augmented = librosa.effects.pitch_shift(y=augmented, sr=sr, n_steps=n_steps)

    # 6. Clipping (simulates cheap mic distortion)
    if np.random.random() < 0.2:
        clip_thresh = np.random.uniform(0.6, 0.9)
        augmented = np.clip(augmented, -clip_thresh, clip_thresh)

    # 7. Add reverb (simulates room acoustics)
    if np.random.random() < 0.3:
        reverb_len = np.random.randint(sr // 10, sr // 3)
        reverb = np.zeros(len(augmented) + reverb_len)
        reverb[:len(augmented)] = augmented
        impulse = np.random.uniform(0.01, 0.05, reverb_len)
        impulse[0] = 1.0
        augmented = np.convolve(augmented, impulse, mode='full')[:len(augmented)]

    # Normalize
    max_val = np.max(np.abs(augmented)) + 1e-8
    augmented = augmented / max_val * 0.9

    return augmented.astype(np.float32)


def extract_feature_vector(audio_path, augment=False):
    """Extract 64-dim feature vector matching AudioProcessor pipeline (3s segments)."""
    import librosa
    try:
        y, sr = librosa.load(str(audio_path), sr=16000, duration=30.0)
        if len(y) < sr:
            return None

        # Apply augmentation to genuine voices to simulate real-world conditions
        if augment:
            y = augment_audio(y, sr)

        segment_duration = 3.0
        segment_samples = int(segment_duration * sr)
        segments = []

        for start in range(0, len(y), segment_samples):
            segment = y[start:start + segment_samples]
            if len(segment) < segment_samples // 2:
                break

            feats = {}
            sc = librosa.feature.spectral_centroid(y=segment, sr=sr)
            sb = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
            sro = librosa.feature.spectral_rolloff(y=segment, sr=sr)
            scon = librosa.feature.spectral_contrast(y=segment, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(segment)
            rms = librosa.feature.rms(y=segment)

            feats["spectral_centroid_mean"] = float(np.mean(sc))
            feats["spectral_centroid_std"] = float(np.std(sc))
            feats["spectral_bandwidth_mean"] = float(np.mean(sb))
            feats["spectral_bandwidth_std"] = float(np.std(sb))
            feats["spectral_rolloff_mean"] = float(np.mean(sro))
            feats["spectral_rolloff_std"] = float(np.std(sro))
            feats["spectral_contrast_mean"] = float(np.mean(scon))
            feats["zero_crossing_rate_mean"] = float(np.mean(zcr))
            feats["rms_energy_mean"] = float(np.mean(rms))
            feats["rms_energy_std"] = float(np.std(rms))

            pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            if len(pitch_values) == 0:
                pitch_values = np.array([0.0])
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

            feats["pitch_mean"] = float(np.mean(pitch_values))
            feats["pitch_std"] = float(np.std(pitch_values))
            feats["pitch_range"] = float(np.ptp(pitch_values))
            feats["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            feats["onset_strength_mean"] = float(np.mean(onset_env))
            feats["onset_strength_std"] = float(np.std(onset_env))

            stft_mag = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
            phase = np.angle(librosa.stft(segment, n_fft=2048, hop_length=512))
            phase_diff = np.diff(phase, axis=1)

            feats["phase_diff_mean"] = float(np.mean(np.abs(phase_diff)))
            feats["phase_diff_std"] = float(np.std(phase_diff))
            feats["phase_entropy"] = float(-np.sum(
                (np.abs(phase) / (np.sum(np.abs(phase)) + 1e-10))
                * np.log2(np.abs(phase) / (np.sum(np.abs(phase)) + 1e-10) + 1e-10)
            ))
            feats["stft_energy_mean"] = float(np.mean(stft_mag))
            feats["stft_energy_std"] = float(np.std(stft_mag))

            feats["segment_start_sec"] = start / sr
            segments.append(feats)

        if not segments:
            return None

        feature_keys = [k for k in segments[0] if k != "segment_start_sec"]
        aggregated = {}
        for key in feature_keys:
            values = [s[key] for s in segments]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_max"] = float(np.max(values))
            aggregated[f"{key}_min"] = float(np.min(values))

        aggregated["num_segments"] = len(segments)
        aggregated["total_duration_sec"] = segments[-1].get("segment_start_sec", 0) + segment_duration

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        aggregated["mfcc_mean"] = float(np.mean(mfcc))
        aggregated["mfcc_std"] = float(np.std(mfcc))

        keys = sorted(aggregated.keys())
        values = [aggregated[k] for k in keys]

        if len(values) < 64:
            values.extend([0.0] * (64 - len(values)))
        elif len(values) > 64:
            values = values[:64]

        return np.array(values, dtype=np.float32)

    except Exception:
        return None


def _extract_one_augmented(args):
    """Extract features with augmentation (multiple augmented versions per genuine file)."""
    path_str, label, augment_count = args
    results = []

    # Original features (always)
    fv = extract_feature_vector(Path(path_str), augment=False)
    if fv is not None:
        results.append((fv, label))

    # Augmented features (only for genuine)
    if label == 0 and augment_count > 0:
        for _ in range(augment_count):
            fv = extract_feature_vector(Path(path_str), augment=True)
            if fv is not None:
                results.append((fv, label))

    return results


def load_wavefake_files(data_dir):
    genuine_dir = data_dir / "wavefake" / "genuine"
    synthetic_dir = data_dir / "wavefake" / "synthetic"

    files_labels = []
    for p in genuine_dir.glob("*.wav"):
        files_labels.append((str(p), 0))
    for p in synthetic_dir.glob("*.wav"):
        files_labels.append((str(p), 1))

    logger.info("Found %d files (%d genuine, %d synthetic)",
                len(files_labels),
                sum(1 for _, l in files_labels if l == 0),
                sum(1 for _, l in files_labels if l == 1))
    return files_labels


def extract_all_features(files_labels, cache_path, augment_genuine=3, num_workers=4):
    """Extract features with augmentation for genuine voices."""
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        logger.info("Loaded %d samples (G:%d S:%d)", len(X), int(np.sum(y==0)), int(np.sum(y==1)))
        return X, y

    # Prepare args: augment genuine 3x, synthetic 0x
    tasks = []
    for path, label in files_labels:
        aug_count = augment_genuine if label == 0 else 0
        tasks.append((path, label, aug_count))

    logger.info("Extracting features with augmentation (genuine: %dx augmented)...", augment_genuine)
    start_time = time.time()

    all_results = []
    with Pool(processes=num_workers) as pool:
        for i, results in enumerate(pool.imap_unordered(_extract_one_augmented, tasks, chunksize=20)):
            all_results.extend(results)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                logger.info("  %d/%d files -> %d features (%.0fs)",
                            i + 1, len(tasks), len(all_results), elapsed)

    elapsed = time.time() - start_time
    logger.info("Done in %.0fs -> %d total features", elapsed, len(all_results))

    X_list, y_list = [], []
    for fv, label in all_results:
        if fv is not None:
            X_list.append(fv)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, y=y)
    logger.info("Cached to %s", cache_path)

    return X, y


def train_model(X, y, epochs=300, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Balance to 1:1
    genuine_idx = np.where(y == 0)[0]
    synthetic_idx = np.where(y == 1)[0]
    n_genuine = len(genuine_idx)
    n_synthetic = len(synthetic_idx)
    logger.info("Before balancing: %d genuine, %d synthetic", n_genuine, n_synthetic)

    np.random.seed(42)
    if n_genuine > n_synthetic:
        genuine_balanced = np.random.choice(genuine_idx, size=n_synthetic, replace=False)
        balanced_idx = np.concatenate([genuine_balanced, synthetic_idx])
    else:
        synthetic_balanced = np.random.choice(synthetic_idx, size=n_genuine, replace=False)
        balanced_idx = np.concatenate([genuine_idx, synthetic_balanced])

    np.random.shuffle(balanced_idx)
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    logger.info("After balancing: %d total (1:1)", len(X_balanced))

    X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    logger.info("Train: %d (G:%d S:%d) | Val: %d (G:%d S:%d)",
                len(X_train), int(np.sum(y_train==0)), int(np.sum(y_train==1)),
                len(X_val), int(np.sum(y_val==0)), int(np.sum(y_val==1)))

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                            batch_size=batch_size)

    model = VoiceAuthenticityNet(input_size=64, hidden_sizes=[512, 256, 128, 64], dropout=0.3).to(device)
    logger.info("Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_combined = 0.0
    patience = 0
    save_path = Path("voice_detection_app/trained_model.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Training for %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            correct += ((torch.sigmoid(out) > 0.5).float() == by).sum().item()
            total += by.size(0)
        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx).squeeze(-1)
                v_loss += criterion(out, by).item()
                probs = torch.sigmoid(out)
                v_correct += ((probs > 0.5).float() == by).sum().item()
                v_total += by.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(by.cpu().numpy())

        all_probs_np = np.array(all_probs)
        all_labels_np = np.array(all_labels)
        try:
            auc = roc_auc_score(all_labels_np, all_probs_np)
        except:
            auc = 0.0

        tl = t_loss / max(len(train_loader), 1)
        ta = correct / max(total, 1)
        va = v_correct / max(v_total, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("Epoch %3d/%d | T L:%.4f A:%.4f | V A:%.4f AUC:%.4f",
                        epoch+1, epochs, tl, ta, va, auc)

        combined = va * 0.5 + auc * 0.5
        if combined > best_combined:
            best_combined = combined
            patience = 0
            torch.save(model.state_dict(), str(save_path))
        else:
            patience += 1
            if patience >= 40:
                logger.info("Early stop at epoch %d", epoch+1)
                break

    total_time = time.time() - start_time
    logger.info("Done in %.0fs | Best combined: %.4f", total_time, best_combined)

    # Final eval
    model.load_state_dict(torch.load(str(save_path), map_location=device, weights_only=True))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx).squeeze(-1)
            probs = torch.sigmoid(out)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(by.cpu().numpy())

    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    binary_preds = (all_probs_np > 0.5).astype(int)

    try:
        auc = roc_auc_score(all_labels_np, all_probs_np)
        logger.info("Val AUC: %.4f", auc)
    except:
        pass

    print("\nClassification Report:")
    print(classification_report(all_labels_np, binary_preds, target_names=["Genuine", "Synthetic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels_np, binary_preds))
    logger.info("Genuine mean: %.4f | Synthetic mean: %.4f",
                all_probs_np[all_labels_np == 0].mean(), all_probs_np[all_labels_np == 1].mean())

    # Export ONNX
    import onnx
    import onnxruntime as ort
    model_cpu = VoiceAuthenticityNet(input_size=64, hidden_sizes=[512, 256, 128, 64], dropout=0.0)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    dummy = torch.randn(1, 64)
    onnx_path = Path("voice_detection_app/trained_model.onnx")
    torch.onnx.export(model_cpu, dummy, str(onnx_path), opset_version=14,
                      input_names=["features"], output_names=["synthetic_prob"],
                      dynamic_axes={"features": {0: "batch"}, "synthetic_prob": {0: "batch"}},
                      dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    logger.info("ONNX: %s (%.2f MB)", onnx_path, os.path.getsize(onnx_path) / 1e6)

    return model


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Training with augmented genuine voices")
    logger.info("Key: Augment genuine 3x to simulate phone/street/home recordings")
    logger.info("=" * 60)

    data_dir = Path("data")
    cache_path = data_dir / "wavefake_features_v3.npz"

    files_labels = load_wavefake_files(data_dir)
    if not files_labels:
        logger.error("No data!")
        sys.exit(1)

    X, y = extract_all_features(files_labels, cache_path, augment_genuine=3, num_workers=4)
    logger.info("Dataset: %d | G:%d S:%d | Features: %d",
                len(X), int(np.sum(y==0)), int(np.sum(y==1)), X.shape[1])

    model = train_model(X, y, epochs=300, batch_size=64, lr=0.001)
