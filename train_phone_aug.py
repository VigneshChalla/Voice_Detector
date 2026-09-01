"""
Train with phone-like augmentation on BOTH genuine and synthetic.
Key insight: augment BOTH classes equally so model ignores recording quality.
"""
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


def phone_augment(y, sr):
    """Simulate phone recording conditions - applied to BOTH classes."""
    import random
    out = y.copy()

    # 1. Add background noise (cafeteria/office/home)
    if random.random() < 0.8:
        noise_level = random.uniform(0.005, 0.04)
        noise = np.random.normal(0, noise_level, len(out))
        out = out + noise

    # 2. Bandpass filter (phone mic response: 300Hz-3400Hz typical)
    if random.random() < 0.6:
        from scipy.signal import butter, sosfilt
        low = random.uniform(200, 400)
        high = random.uniform(3000, 4500)
        sos = butter(4, [low, high], btype='band', fs=sr, output='sos')
        out = sosfilt(sos, out)

    # 3. Random gain (distance from mic)
    if random.random() < 0.5:
        gain = random.uniform(0.4, 1.8)
        out = out * gain

    # 4. Light reverb (room reflection)
    if random.random() < 0.4:
        reverb_len = random.randint(sr // 10, sr // 4)
        decay = np.random.uniform(0.01, 0.06, reverb_len)
        decay[0] = 1.0
        reverb_tail = np.convolve(out, decay, mode='full')[:len(out)]
        out = out * 0.7 + reverb_tail * 0.3

    # 5. Soft clipping (phone ADC)
    if random.random() < 0.3:
        threshold = random.uniform(0.7, 0.95)
        out = np.where(np.abs(out) > threshold, np.tanh(out / threshold) * threshold, out)

    # Normalize
    peak = np.max(np.abs(out)) + 1e-8
    out = out / peak * 0.85
    return out.astype(np.float32)


def extract_feature_vector(audio_path, augment=False):
    """Extract 64-dim feature vector matching AudioProcessor pipeline."""
    import librosa
    try:
        y, sr = librosa.load(str(audio_path), sr=16000, duration=30.0)
        if len(y) < sr:
            return None

        if augment:
            y = phone_augment(y, sr)

        segment_duration = 3.0
        segment_samples = int(segment_duration * sr)
        segments = []

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


def _extract_one(args):
    path_str, label, augment_count = args
    results = []
    fv = extract_feature_vector(Path(path_str), augment=False)
    if fv is not None:
        results.append((fv, label))
    for _ in range(augment_count):
        fv = extract_feature_vector(Path(path_str), augment=True)
        if fv is not None:
            results.append((fv, label))
    return results


def load_all_files(data_dir):
    files_labels = []
    for p in (data_dir / "wavefake" / "genuine").glob("*.wav"):
        files_labels.append((str(p), 0))
    for p in (data_dir / "librispeech-genuine").glob("*.wav"):
        files_labels.append((str(p), 0))
    for p in (data_dir / "wavefake" / "synthetic").glob("*.wav"):
        files_labels.append((str(p), 1))
    n_g = sum(1 for _, l in files_labels if l == 0)
    n_s = sum(1 for _, l in files_labels if l == 1)
    logger.info("Files: %d genuine (WF:%d + LS:%d) | %d synthetic",
                n_g, len(list((data_dir / "wavefake" / "genuine").glob("*.wav"))),
                len(list((data_dir / "librispeech-genuine").glob("*.wav"))), n_s)
    return files_labels


def extract_all_features(files_labels, cache_path, augment_each=2, num_workers=4):
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        logger.info("Loaded %d samples (G:%d S:%d)", len(X), int(np.sum(y==0)), int(np.sum(y==1)))
        return X, y

    tasks = [(path, label, augment_each) for path, label in files_labels]

    logger.info("Extracting features (original + %d augmented per file for BOTH classes)...", augment_each)
    start_time = time.time()

    all_results = []
    with Pool(processes=num_workers) as pool:
        for i, results in enumerate(pool.imap_unordered(_extract_one, tasks, chunksize=20)):
            all_results.extend(results)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                logger.info("  %d/%d files -> %d features (%.0fs)", i+1, len(tasks), len(all_results), elapsed)

    elapsed = time.time() - start_time
    logger.info("Done in %.0fs -> %d features", elapsed, len(all_results))

    X = np.array([fv for fv, _ in all_results], dtype=np.float32)
    y = np.array([label for _, label in all_results], dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, y=y)
    logger.info("Cached %d features to %s", len(X), cache_path)
    return X, y


def train_model(X, y, epochs=300, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    genuine_idx = np.where(y == 0)[0]
    synthetic_idx = np.where(y == 1)[0]
    logger.info("Before balance: %d genuine, %d synthetic", len(genuine_idx), len(synthetic_idx))

    np.random.seed(42)
    min_count = min(len(genuine_idx), len(synthetic_idx))
    genuine_balanced = np.random.choice(genuine_idx, size=min_count, replace=False)
    synthetic_balanced = np.random.choice(synthetic_idx, size=min_count, replace=False)
    balanced_idx = np.concatenate([genuine_balanced, synthetic_balanced])
    np.random.shuffle(balanced_idx)

    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    logger.info("After balance: %d (1:1)", len(X_balanced))

    X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    logger.info("Train: %d (G:%d S:%d) | Val: %d (G:%d S:%d)",
                len(X_train), int(np.sum(y_train==0)), int(np.sum(y_train==1)),
                len(X_val), int(np.sum(y_val==0)), int(np.sum(y_val==1)))

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size)

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
        v_correct, v_total = 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx).squeeze(-1)
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

    # ONNX export
    import onnx
    model_cpu = VoiceAuthenticityNet(input_size=64, hidden_sizes=[512, 256, 128, 64], dropout=0.0)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    onnx_path = Path("voice_detection_app/trained_model.onnx")
    torch.onnx.export(model_cpu, torch.randn(1, 64), str(onnx_path), opset_version=14,
                      input_names=["features"], output_names=["synthetic_prob"],
                      dynamic_axes={"features": {0: "batch"}, "synthetic_prob": {0: "batch"}},
                      dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    logger.info("ONNX: %s (%.2f MB)", onnx_path, os.path.getsize(onnx_path) / 1e6)
    return model


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Training with PHONE augmentation on BOTH classes")
    logger.info("=" * 60)

    data_dir = Path("data")
    cache_path = data_dir / "features_phone_aug_v1.npz"

    files_labels = load_all_files(data_dir)
    X, y = extract_all_features(files_labels, cache_path, augment_each=2, num_workers=4)
    logger.info("Dataset: %d | G:%d S:%d", len(X), int(np.sum(y==0)), int(np.sum(y==1)))

    model = train_model(X, y, epochs=300, batch_size=64, lr=0.001)
