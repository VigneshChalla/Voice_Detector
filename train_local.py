"""
Voice Cloning Detection - Local Training Script
Trains on WaveFake + synthetic data (no external downloads required).

Usage:
    python train_local.py                    # Train with synthetic data (fast, no downloads)
    python train_local.py --use-wavefake     # Download WaveFake + train (needs internet)
    python train_local.py --epochs 100       # Custom epochs
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice_detection_app.config import settings
from voice_detection_app.models.detector import VoiceAuthenticityNet


# =============================================================================
# Feature Extraction
# =============================================================================
def extract_features_from_audio(audio_path: Path) -> np.ndarray | None:
    """Extract feature vector from a single audio file."""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=16000, duration=30.0)
        if len(y) < 16000:
            return None

        features = {}

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        features["mfcc_mean"] = float(np.mean(mfcc))
        features["mfcc_std"] = float(np.std(mfcc))

        # Spectral
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr)
        scon = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        features["spectral_centroid_mean"] = float(np.mean(sc))
        features["spectral_centroid_std"] = float(np.std(sc))
        features["spectral_bandwidth_mean"] = float(np.mean(sb))
        features["spectral_bandwidth_std"] = float(np.std(sb))
        features["spectral_rolloff_mean"] = float(np.mean(sr_))
        features["spectral_rolloff_std"] = float(np.std(sr_))
        features["spectral_contrast_mean"] = float(np.mean(scon))
        features["zero_crossing_rate_mean"] = float(np.mean(zcr))
        features["rms_energy_mean"] = float(np.mean(rms))
        features["rms_energy_std"] = float(np.std(rms))

        # Prosody
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) == 0:
            pitch_values = np.array([0.0])
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        features["pitch_mean"] = float(np.mean(pitch_values))
        features["pitch_std"] = float(np.std(pitch_values))
        features["pitch_range"] = float(np.ptp(pitch_values))
        features["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        features["onset_strength_mean"] = float(np.mean(onset_env))
        features["onset_strength_std"] = float(np.std(onset_env))

        # Phase
        stft_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        phase = np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
        phase_diff = np.diff(phase, axis=1)
        features["phase_diff_mean"] = float(np.mean(np.abs(phase_diff)))
        features["phase_diff_std"] = float(np.std(phase_diff))
        features["stft_energy_mean"] = float(np.mean(stft_mag))
        features["stft_energy_std"] = float(np.std(stft_mag))

        # Build vector
        keys = sorted(features.keys())
        values = np.array([features[k] for k in keys], dtype=np.float32)

        # Pad/truncate to 64
        if len(values) < 64:
            values = np.pad(values, (0, 64 - len(values)))
        elif len(values) > 64:
            values = values[:64]

        return values
    except Exception as e:
        return None


def extract_features_parallel(audio_paths: list[Path], labels: list[int], max_workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from multiple audio files with progress."""
    from tqdm import tqdm

    X_list = []
    y_list = []

    for path, label in tqdm(zip(audio_paths, labels), total=len(audio_paths), desc="Extracting"):
        fv = extract_features_from_audio(path)
        if fv is not None:
            X_list.append(fv)
            y_list.append(label)

    if not X_list:
        return np.array([]), np.array([])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# =============================================================================
# Synthetic Data Generation (fast bootstrap)
# =============================================================================
def generate_synthetic_audio(sr: int = 16000, duration: float = 5.0) -> tuple[np.ndarray, int]:
    """Generate a single genuine-sounding speech-like audio signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    f0 = np.random.uniform(85, 250)
    signal = np.zeros(n_samples)

    # Rich harmonic structure (varied across time)
    for h in range(1, 15):
        amp = (0.5 / h) * np.random.uniform(0.7, 1.3)
        freq_mod = 1.0 + 0.002 * np.sin(2 * np.pi * np.random.uniform(2, 6) * t)
        signal += amp * np.sin(2 * np.pi * f0 * h * t * freq_mod)

    # Natural vibrato and jitter
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * np.random.uniform(4, 7) * t)
    signal *= vibrato

    # Formant-like resonances with time-varying characteristics
    for _ in range(np.random.randint(3, 6)):
        f_center = np.random.uniform(200, 4000)
        bw = np.random.uniform(50, 200)
        resonance = np.sin(2 * np.pi * f_center * t)
        envelope = 0.2 * np.exp(-bw * np.abs(np.sin(2 * np.pi * np.random.uniform(0.5, 3) * t)))
        signal += envelope * resonance

    # Natural prosody with varied dynamics
    segments = np.random.randint(4, 8)
    seg_len = n_samples // segments
    for i in range(segments):
        start = i * seg_len
        end = min(start + seg_len, n_samples)
        seg_amp = np.random.uniform(0.3, 1.0)
        attack = min(sr // 10, end - start)
        signal[start:start + attack] *= np.linspace(0, 1, attack)
        signal[start + attack:end] *= seg_amp

    # Breathing and natural pauses
    num_pauses = np.random.randint(1, 5)
    for _ in range(num_pauses):
        start = np.random.randint(0, max(1, n_samples - sr))
        length = np.random.randint(sr // 8, sr // 2)
        end = min(start + length, n_samples)
        signal[start:end] *= np.linspace(1, 0, end - start)

    # Natural background noise at varying levels
    noise_level = np.random.uniform(0.002, 0.01)
    signal += np.random.normal(0, noise_level, n_samples)

    # Slight clipping / saturation (natural recording artifact)
    clip_thresh = np.random.uniform(0.85, 0.95)
    mask = np.abs(signal) > clip_thresh
    signal[mask] = np.sign(signal[mask]) * (clip_thresh + 0.05 * np.tanh((np.abs(signal[mask]) - clip_thresh) * 20))

    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.85

    return signal.astype(np.float32), sr


def generate_synthetic_clone_audio(sr: int = 16000, duration: float = 5.0) -> tuple[np.ndarray, int]:
    """Generate synthetic/clone audio with strong vocoder artifacts."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    f0 = np.random.uniform(100, 220)
    signal = np.zeros(n_samples)

    # Fewer, more rigid harmonics
    for h in range(1, 8):
        amp = 0.4 / h
        signal += amp * np.sin(2 * np.pi * f0 * h * t)

    # Rigid prosody (robotic flatness)
    signal *= 0.6 + 0.05 * np.sin(2 * np.pi * 3.0 * t)

    # Vocoder artifacts
    vocoder_freq = np.random.uniform(5000, 8000)
    signal += 0.12 * np.sin(2 * np.pi * vocoder_freq * t)

    # Periodic buzzing
    buzz_freq = np.random.uniform(100, 300)
    signal += 0.06 * np.sign(np.sin(2 * np.pi * buzz_freq * t))

    # Quantization artifacts (step-like)
    quant_levels = np.random.choice([32, 64, 128])
    signal = np.round(signal * quant_levels) / quant_levels

    # Phase discontinuities (concatenation artifacts)
    n_breaks = np.random.randint(5, 15)
    for _ in range(n_breaks):
        pos = np.random.randint(sr // 4, max(sr // 4 + 1, n_samples - sr // 4))
        width = np.random.randint(5, 50)
        signal[pos:pos + width] += np.random.uniform(-0.4, 0.4, width)

    # Spectral gating artifacts (typical of noise reduction in TTS)
    noise = np.random.normal(0, 0.003, n_samples)
    gate = np.abs(signal) > np.percentile(np.abs(signal), 30)
    signal[~gate] *= 0.01

    # Stationary noise (too clean = synthetic)
    signal += np.random.normal(0, 0.001, n_samples)

    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.7

    return signal.astype(np.float32), sr


def extract_feature_vector(audio_path_or_signal, sr: int = 16000, is_signal: bool = False) -> np.ndarray | None:
    """Extract feature vector matching AudioProcessor.get_feature_vector() output."""
    import librosa
    try:
        if is_signal:
            y = audio_path_or_signal
        else:
            y, sr = librosa.load(str(audio_path_or_signal), sr=sr, duration=30.0)

        if len(y) < sr:
            return None

        # Segment features (matching AudioProcessor.compute_segment_features)
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

        # Aggregate (matching AudioProcessor.aggregate_features)
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

        # Add MFCC stats (matching AudioProcessor.process_audio)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        aggregated["mfcc_mean"] = float(np.mean(mfcc))
        aggregated["mfcc_std"] = float(np.std(mfcc))

        # Sort keys and build vector (matching AudioProcessor.get_feature_vector)
        keys = sorted(aggregated.keys())
        values = [aggregated[k] for k in keys]

        if len(values) < 64:
            values.extend([0.0] * (64 - len(values)))
        elif len(values) > 64:
            values = values[:64]

        return np.array(values, dtype=np.float32)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR in extract_feature_vector: {e}")
        return None


def generate_synthetic_data(num_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data by creating synthetic audio signals and extracting features."""
    from tqdm import tqdm

    logger.info("Generating %d audio signals and extracting features...", num_samples)
    n = num_samples // 2

    X_list = []
    y_list = []

    for i in tqdm(range(n), desc="Genuine audio"):
        duration = np.random.uniform(3.0, 8.0)
        signal, sr = generate_synthetic_audio(sr=16000, duration=duration)
        fv = extract_feature_vector(signal, sr=sr, is_signal=True)
        if fv is not None:
            X_list.append(fv)
            y_list.append(0)

    for i in tqdm(range(n), desc="Synthetic audio"):
        duration = np.random.uniform(3.0, 8.0)
        signal, sr = generate_synthetic_clone_audio(sr=16000, duration=duration)
        fv = extract_feature_vector(signal, sr=sr, is_signal=True)
        if fv is not None:
            X_list.append(fv)
            y_list.append(1)

    if not X_list:
        return np.array([]), np.array([])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    perm = np.random.permutation(len(X_list))
    return X[perm], y[perm]


# =============================================================================
# WaveFake Download
# =============================================================================
def download_wavefake(data_dir: Path) -> list[tuple[Path, int]]:
    """Download WaveFake from HuggingFace and return (path, label) pairs."""
    wavefake_dir = data_dir / "wavefake"
    wavefake_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(wavefake_dir.rglob("*.wav"))
    if len(existing) >= 100:
        logger.info("WaveFake already downloaded: %d files", len(existing))
        return [(p, 1) for p in existing[:5000]]

    try:
        from datasets import load_dataset
        import soundfile as sf

        logger.info("Downloading WaveFake from HuggingFace...")
        ds = load_dataset("ajaykarthick/wavefake-audio", split="train", streaming=True)

        count = 0
        max_files = 5000
        for i, sample in enumerate(ds):
            if count >= max_files:
                break
            try:
                audio = sample["audio"]
                audio_data = audio["array"]
                sr = audio["sampling_rate"]
                out_path = wavefake_dir / f"wavefake_{i:05d}.wav"
                sf.write(str(out_path), audio_data, sr)
                count += 1
                if count % 500 == 0:
                    logger.info("  Downloaded %d/%d files...", count, max_files)
            except Exception:
                continue

        logger.info("Downloaded %d WaveFake files", count)
        return [(p, 1) for p in wavefake_dir.rglob("*.wav")]

    except ImportError:
        logger.warning("datasets library not installed. Run: pip install datasets")
        logger.warning("Generating synthetic data instead.")
        return []
    except Exception as e:
        logger.warning("WaveFake download failed: %s", e)
        return []


def download_asvspoof(data_dir: Path) -> list[tuple[Path, int]]:
    """Download ASVspoof 2021 from Zenodo and return (path, label) pairs."""
    asvspoof_dir = data_dir / "asvspoof"
    asvspoof_dir.mkdir(parents=True, exist_ok=True)

    existing = list(asvspoof_dir.rglob("*.flac"))
    if len(existing) >= 100:
        logger.info("ASVspoof already downloaded: %d files", len(existing))
        # Determine labels from path
        pairs = []
        for p in existing:
            path_str = str(p).lower()
            label = 0 if "bonafide" in path_str else 1
            pairs.append((p, label))
        return pairs[:8000]

    try:
        import subprocess
        import zipfile

        urls = {
            "train": "https://zenodo.org/record/4837280/files/ASVspoof2021Train.zip",
            "dev": "https://zenodo.org/record/4837280/files/ASVspoof2021Dev.zip",
        }

        for name, url in urls.items():
            zip_path = asvspoof_dir / f"{name}.zip"
            extract_dir = asvspoof_dir / f"ASVspoof2021{name.capitalize()}"

            if extract_dir.exists() and any(extract_dir.rglob("*.flac")):
                logger.info("ASVspoof %s already extracted", name)
                continue

            if not zip_path.exists():
                logger.info("Downloading ASVspoof %s (~1-2 GB)...", name)
                subprocess.run(["wget", "-q", "--show-progress", url, "-O", str(zip_path)], check=True)

            logger.info("Extracting ASVspoof %s...", name)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(asvspoof_dir)
            os.remove(zip_path)

        # Download labels
        label_url = "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"
        label_tar = asvspoof_dir / "LA-keys-full.tar.gz"
        if not label_tar.exists() and not (asvspoof_dir / "LA-keys-full").exists():
            logger.info("Downloading labels...")
            subprocess.run(["wget", "-q", "--show-progress", label_url, "-O", str(label_tar)], check=True)
            import tarfile
            with tarfile.open(label_tar, "r:gz") as t:
                t.extractall(asvspoof_dir)
            os.remove(label_tar)

        # Collect files
        pairs = []
        for p in asvspoof_dir.rglob("*.flac"):
            path_str = str(p).lower()
            label = 0 if "bonafide" in path_str else 1
            pairs.append((p, label))

        logger.info("Found %d ASVspoof files", len(pairs))
        return pairs[:8000]

    except Exception as e:
        logger.warning("ASVspoof download failed: %s", e)
        return []


# =============================================================================
# Training
# =============================================================================
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 0.001,
    hidden_sizes: list[int] | None = None,
    dropout: float = 0.3,
    save_dir: str = "voice_detection_app",
):
    """Train the voice authenticity model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info("Train: %d samples | Val: %d samples", len(X_train), len(X_val))

    # Data loaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]

    model = VoiceAuthenticityNet(
        input_size=64,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{num_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_loss = float("inf")
    best_accuracy = 0.0
    patience = 0
    max_patience = 15
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    save_path = Path(save_dir) / "trained_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training for %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(epochs):
        # Train
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

        # Validate
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
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f | LR: %.6f | Time: %.0fs",
                epoch + 1, epochs, avg_train_loss, train_acc, avg_val_loss, val_acc, lr_now, elapsed,
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
    logger.info("Training complete in %.1f seconds", total_time)
    logger.info("Best val loss: %.4f | Best accuracy: %.4f", best_val_loss, best_accuracy)
    logger.info("Model saved to: %s", save_path)

    # Final evaluation
    logger.info("\n%s", "=" * 60)
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

    logger.info("\nClassification Report:")
    print(classification_report(all_labels_np, binary_preds, target_names=["Genuine", "Synthetic"]))

    logger.info("Confusion Matrix:")
    print(confusion_matrix(all_labels_np, binary_preds))

    # Save training history
    history_path = Path(save_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to: %s", history_path)

    # Export to ONNX
    try:
        export_onnx(model, save_dir)
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)

    return model, history


def export_onnx(model: nn.Module, save_dir: str = "voice_detection_app"):
    """Export model to ONNX format."""
    import onnx
    import onnxruntime as ort

    model_cpu = VoiceAuthenticityNet(
        input_size=64,
        hidden_sizes=[512, 256, 128, 64],
        dropout=0.0,
    )
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    dummy = torch.randn(1, 64)
    onnx_path = Path(save_dir) / "trained_model.onnx"

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
    logger.info("ONNX model exported: %s (%.2f MB)", onnx_path, size_mb)

    # Test ONNX inference
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, 64).astype(np.float32)
    result = sess.run(None, {"features": test_input})
    logger.info("ONNX inference test: output=%.4f", result[0].flatten()[0])


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Voice Cloning Detection Model")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs (default: 150)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--hidden", type=str, default="512,256,128,64", help="Hidden sizes (default: 512,256,128,64)")
    parser.add_argument("--use-wavefake", action="store_true", help="Download and use WaveFake dataset")
    parser.add_argument("--use-asvspoof", action="store_true", help="Download and use ASVspoof 2021 dataset")
    parser.add_argument("--synthetic-only", action="store_true", help="Train on synthetic data only (fastest)")
    parser.add_argument("--save-dir", type=str, default="voice_detection_app", help="Model save directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    hidden = [int(x.strip()) for x in args.hidden.split(",")]
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Voice Cloning Detection - Local Training")
    logger.info("=" * 60)

    X_all = None
    y_all = None

    # Try real datasets if requested
    if args.use_wavefake:
        logger.info("\n--- Loading WaveFake Dataset ---")
        wavefake_pairs = download_wavefake(data_dir)
        if wavefake_pairs:
            paths, labels = zip(*wavefake_pairs)
            X_wf, y_wf = extract_features_parallel(list(paths), list(labels))
            if len(X_wf) > 0:
                X_all = X_wf
                y_all = y_wf
                logger.info("WaveFake: %d samples", len(X_wf))

    if args.use_asvspoof:
        logger.info("\n--- Loading ASVspoof 2021 Dataset ---")
        asvspoof_pairs = download_asvspoof(data_dir)
        if asvspoof_pairs:
            paths, labels = zip(*asvspoof_pairs)
            X_as, y_as = extract_features_parallel(list(paths), list(labels))
            if len(X_as) > 0:
                if X_all is not None:
                    X_all = np.vstack([X_all, X_as])
                    y_all = np.concatenate([y_all, y_as])
                else:
                    X_all = X_as
                    y_all = y_as
                logger.info("ASVspoof: %d samples", len(X_as))

    # Fall back to synthetic data
    if X_all is None or len(X_all) == 0 or args.synthetic_only:
        logger.info("\n--- Generating Synthetic Training Data ---")
        X_synth, y_synth = generate_synthetic_data(2000)
        if X_all is not None:
            X_all = np.vstack([X_all, X_synth])
            y_all = np.concatenate([y_all, y_synth])
        else:
            X_all = X_synth
            y_all = y_synth

    logger.info("\n--- Final Dataset ---")
    logger.info("Total samples: %d", len(X_all))
    logger.info("  Genuine (0): %d", int(np.sum(y_all == 0)))
    logger.info("  Synthetic (1): %d", int(np.sum(y_all == 1)))
    logger.info("  Feature dimension: %d", X_all.shape[1])

    # Train
    logger.info("\n--- Starting Training ---")
    model, history = train_model(
        X_all, y_all,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_sizes=hidden,
        dropout=args.dropout,
        save_dir=args.save_dir,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Files created:")
    logger.info("  %s/trained_model.pth", args.save_dir)
    logger.info("  %s/trained_model.onnx", args.save_dir)
    logger.info("  %s/training_history.json", args.save_dir)
    logger.info("\nRun the API server:")
    logger.info("  python -m voice_detection_app.app")


if __name__ == "__main__":
    main()
