"""
Parallel Feature Extraction + Training for Voice Cloning Detection.
Reads WAV files from data/wavefake/, extracts features in parallel, trains VoiceAuthenticityNet.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice_detection_app.models.detector import VoiceAuthenticityNet


# =============================================================================
# Feature Extraction (copied exactly from train_local.py extract_feature_vector)
# =============================================================================
def extract_feature_vector(audio_path_or_signal, sr: int = 16000, is_signal: bool = False):
    import librosa
    try:
        if is_signal:
            y = audio_path_or_signal
        else:
            y, sr = librosa.load(str(audio_path_or_signal), sr=sr, duration=30.0)

        if len(y) < sr:
            return None

        segment_duration = 2.0
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


# =============================================================================
# Multiprocessing worker
# =============================================================================
def _extract_one(args):
    path_str, label = args
    fv = extract_feature_vector(Path(path_str), sr=16000, is_signal=False)
    return (fv, label)


# =============================================================================
# Parallel feature extraction with caching
# =============================================================================
def load_wavefake_files(data_dir: Path):
    genuine_dir = data_dir / "wavefake" / "genuine"
    synthetic_dir = data_dir / "wavefake" / "synthetic"

    files_labels = []
    for p in genuine_dir.glob("*.wav"):
        files_labels.append((str(p), 0))
    for p in synthetic_dir.glob("*.wav"):
        files_labels.append((str(p), 1))

    logger.info("Found %d WAV files (%d genuine, %d synthetic)",
                len(files_labels),
                sum(1 for _, l in files_labels if l == 0),
                sum(1 for _, l in files_labels if l == 1))
    return files_labels


def extract_features_parallel(files_labels, cache_path: Path, num_workers: int = 4, chunksize: int = 50):
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        logger.info("Loaded %d samples, feature dim=%d", len(X), X.shape[1])
        return X, y

    logger.info("Extracting features from %d files using %d workers...", len(files_labels), num_workers)
    start_time = time.time()

    results = []
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_extract_one, files_labels, chunksize=chunksize)):
            results.append(result)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                logger.info("  Processed %d/%d files (%.1fs elapsed)", i + 1, len(files_labels), elapsed)

    elapsed = time.time() - start_time
    logger.info("Feature extraction complete in %.1f seconds", elapsed)

    X_list, y_list = [], []
    for fv, label in results:
        if fv is not None:
            X_list.append(fv)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, y=y)
    logger.info("Cached %d samples to %s", len(X), cache_path)

    return X, y


# =============================================================================
# Training
# =============================================================================
def train_model(X, y, epochs=300, batch_size=64, lr=0.001, hidden_sizes=None, dropout=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info("Train: %d | Val: %d", len(X_train), len(X_val))

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    if hidden_sizes is None:
        hidden_sizes = [512, 256, 128, 64]

    model = VoiceAuthenticityNet(input_size=64, hidden_sizes=hidden_sizes, dropout=dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %s", f"{num_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs // 6, T_mult=2)

    save_dir = Path("voice_detection_app")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "trained_model.pth"

    best_val_loss = float("inf")
    best_accuracy = 0.0
    patience = 0
    max_patience = 30
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    logger.info("Training for %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X).squeeze(-1)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            predicted = (torch.sigmoid(output) > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X).squeeze(-1)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                probs = torch.sigmoid(output)
                predicted = (probs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        train_acc = correct / total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f | LR: %.6f",
                epoch + 1, epochs, avg_train_loss, train_acc, avg_val_loss, val_acc, lr_now,
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_accuracy = val_acc
            patience = 0
            torch.save(model.state_dict(), str(save_path))
        else:
            patience += 1
            if patience >= max_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    total_time = time.time() - start_time
    logger.info("Training complete in %.1fs | Best val loss: %.4f | Best accuracy: %.4f",
                total_time, best_val_loss, best_accuracy)

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    binary_preds = (all_probs_np > 0.5).astype(int)

    try:
        auc_score = roc_auc_score(all_labels_np, all_probs_np)
        logger.info("AUC-ROC: %.4f", auc_score)
    except Exception:
        auc_score = 0.0
        logger.warning("AUC-ROC could not be computed")

    logger.info("\nClassification Report:")
    print(classification_report(all_labels_np, binary_preds, target_names=["Genuine", "Synthetic"]))

    logger.info("Confusion Matrix:")
    print(confusion_matrix(all_labels_np, binary_preds))

    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    # ONNX export
    try:
        export_onnx(model, save_dir)
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)

    return model, history, auc_score


def export_onnx(model, save_dir):
    import onnx
    import onnxruntime as ort

    model_cpu = VoiceAuthenticityNet(input_size=64, hidden_sizes=[512, 256, 128, 64], dropout=0.0)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    dummy = torch.randn(1, 64)
    onnx_path = save_dir / "trained_model.onnx"

    torch.onnx.export(
        model_cpu, dummy, str(onnx_path),
        opset_version=14,
        input_names=["features"],
        output_names=["synthetic_prob"],
        dynamic_axes={"features": {0: "batch"}, "synthetic_prob": {0: "batch"}},
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(onnx_path) / 1e6
    logger.info("ONNX exported: %s (%.2f MB)", onnx_path, size_mb)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, 64).astype(np.float32)
    result = sess.run(None, {"features": test_input})
    logger.info("ONNX inference test: output=%.4f", result[0].flatten()[0])


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("Voice Cloning Detection - Parallel Training")
    logger.info("=" * 60)

    data_dir = Path("data")
    cache_path = data_dir / "wavefake_features.npz"

    # Load file list
    files_labels = load_wavefake_files(data_dir)
    if not files_labels:
        logger.error("No WAV files found in data/wavefake/")
        sys.exit(1)

    # Extract features (parallel, with caching)
    X, y = extract_features_parallel(files_labels, cache_path, num_workers=4, chunksize=50)

    logger.info("Dataset: %d samples | Genuine: %d | Synthetic: %d | Features: %d",
                len(X), int(np.sum(y == 0)), int(np.sum(y == 1)), X.shape[1])

    # Train
    model, history, auc_score = train_model(
        X, y,
        epochs=300,
        batch_size=64,
        lr=0.001,
        hidden_sizes=[512, 256, 128, 64],
        dropout=0.3,
    )

    # Verify ONNX
    onnx_path = Path("voice_detection_app/trained_model.onnx")
    pth_path = Path("voice_detection_app/trained_model.pth")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Model: %s (%.2f KB)", pth_path, pth_path.stat().st_size / 1e3 if pth_path.exists() else 0)
    if onnx_path.exists():
        logger.info("ONNX: %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    else:
        logger.warning("ONNX file not found")


if __name__ == "__main__":
    main()
