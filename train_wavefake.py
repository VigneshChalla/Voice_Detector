"""Full pipeline: download WaveFake, parallel feature extraction, train, export ONNX."""
import urllib.request
import json
import io
import os
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_FILE = PROJECT_ROOT / "data" / "wavefake_features.npz"


def download_all_shards(data_dir, max_files_per_class=6000):
    """Download all shards, balancing genuine/synthetic."""
    genuine_dir = data_dir / "genuine"
    synthetic_dir = data_dir / "synthetic"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    existing_g = len(list(genuine_dir.glob("*.wav")))
    existing_s = len(list(synthetic_dir.glob("*.wav")))
    print(f"Existing: {existing_g} genuine, {existing_s} synthetic")

    if existing_g >= max_files_per_class and existing_s >= max_files_per_class:
        print("Already have enough data")
        return

    repo = "ajaykarthick/wavefake-audio"
    api_url = f"https://huggingface.co/api/datasets/{repo}/parquet"
    print("Fetching parquet URLs...")
    req = urllib.request.Request(api_url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        urls_data = json.loads(resp.read().decode())
        urls = urls_data["default"]["train"]

    print(f"Found {len(urls)} shards")
    g_count = existing_g
    s_count = existing_s

    for shard_idx, url in enumerate(urls):
        if g_count >= max_files_per_class and s_count >= max_files_per_class:
            break

        print(f"Shard {shard_idx+1}/{len(urls)}...", end=" ", flush=True)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                parquet_bytes = resp.read()

            table = pq.read_table(io.BytesIO(parquet_bytes))
            df = table.to_pandas()

            saved_g, saved_s = 0, 0
            for _, row in df.iterrows():
                try:
                    audio_bytes = row["audio"]["bytes"]
                    label_str = row["real_or_fake"]
                    is_genuine = (label_str == "R")

                    if is_genuine and g_count >= max_files_per_class:
                        continue
                    if not is_genuine and s_count >= max_files_per_class:
                        continue

                    samples, sr = sf.read(io.BytesIO(audio_bytes))
                    if isinstance(samples, np.ndarray) and samples.ndim > 1:
                        samples = samples.mean(axis=1)

                    if is_genuine:
                        out_path = genuine_dir / f"wf_{g_count:05d}.wav"
                        sf.write(str(out_path), samples.astype(np.float32), sr)
                        g_count += 1
                        saved_g += 1
                    else:
                        out_path = synthetic_dir / f"wf_{s_count:05d}.wav"
                        sf.write(str(out_path), samples.astype(np.float32), sr)
                        s_count += 1
                        saved_s += 1
                except Exception:
                    continue

            print(f"+{saved_g}g +{saved_s}s (total: {g_count}g {s_count}s)")
        except Exception as e:
            print(f"error: {e}")
            continue

    print(f"\nDownload complete: {g_count} genuine, {s_count} synthetic")


def extract_one_file(args):
    """Extract features from a single audio file (for multiprocessing)."""
    audio_path, label = args
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=16000, duration=30.0)
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

            feats["sc_mean"] = float(np.mean(sc))
            feats["sc_std"] = float(np.std(sc))
            feats["sb_mean"] = float(np.mean(sb))
            feats["sb_std"] = float(np.std(sb))
            feats["sro_mean"] = float(np.mean(sro))
            feats["sro_std"] = float(np.std(sro))
            feats["scon_mean"] = float(np.mean(scon))
            feats["zcr_mean"] = float(np.mean(zcr))
            feats["rms_mean"] = float(np.mean(rms))
            feats["rms_std"] = float(np.std(rms))

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
            feats["onset_mean"] = float(np.mean(onset_env))
            feats["onset_std"] = float(np.std(onset_env))

            stft_mag = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
            phase = np.angle(librosa.stft(segment, n_fft=2048, hop_length=512))
            phase_diff = np.diff(phase, axis=1)

            feats["pd_mean"] = float(np.mean(np.abs(phase_diff)))
            feats["pd_std"] = float(np.std(phase_diff))
            feats["energy_mean"] = float(np.mean(stft_mag))
            feats["energy_std"] = float(np.std(stft_mag))

            feats["seg_start"] = start / sr
            segments.append(feats)

        if not segments:
            return None

        feature_keys = [k for k in segments[0] if k != "seg_start"]
        aggregated = {}
        for key in feature_keys:
            values = [s[key] for s in segments]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_max"] = float(np.max(values))
            aggregated[f"{key}_min"] = float(np.min(values))

        aggregated["num_seg"] = len(segments)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        aggregated["mfcc_mean"] = float(np.mean(mfcc))
        aggregated["mfcc_std"] = float(np.std(mfcc))

        keys = sorted(aggregated.keys())
        values = [aggregated[k] for k in keys]

        if len(values) < 64:
            values.extend([0.0] * (64 - len(values)))
        elif len(values) > 64:
            values = values[:64]

        return (np.array(values, dtype=np.float32), label)
    except Exception:
        return None


def extract_features_parallel(data_dir, max_per_class=6000, workers=4):
    """Extract features in parallel and cache."""
    if CACHE_FILE.exists():
        print(f"Loading cached features from {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        return data["X"], data["y"]

    genuine_dir = data_dir / "genuine"
    synthetic_dir = data_dir / "synthetic"

    genuine_files = sorted(genuine_dir.glob("*.wav"))[:max_per_class]
    synthetic_files = sorted(synthetic_dir.glob("*.wav"))[:max_per_class]

    tasks = [(str(f), 0) for f in genuine_files] + [(str(f), 1) for f in synthetic_files]
    print(f"Extracting features from {len(tasks)} files with {workers} workers...")

    start = time.time()
    X_list = []
    y_list = []

    with mp.Pool(workers) as pool:
        results = pool.imap(extract_one_file, tasks, chunksize=20)
        for i, result in enumerate(results):
            if result is not None:
                X_list.append(result[0])
                y_list.append(result[1])
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate
                print(f"  {i+1}/{len(tasks)} ({rate:.1f} files/s, ETA: {eta/60:.1f}min)")

    elapsed = time.time() - start
    print(f"Feature extraction complete: {len(X_list)} samples in {elapsed/60:.1f} minutes")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(CACHE_FILE), X=X, y=y)
    print(f"Features cached to {CACHE_FILE}")

    return X, y


def train(X, y, epochs=300, batch_size=64, lr=0.001, save_dir="voice_detection_app"):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from voice_detection_app.models.detector import VoiceAuthenticityNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"  Train: Genuine={int(np.sum(y_train==0))} Synthetic={int(np.sum(y_train==1))}")
    print(f"  Val:   Genuine={int(np.sum(y_val==0))} Synthetic={int(np.sum(y_val==1))}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                            batch_size=batch_size)

    model = VoiceAuthenticityNet(input_size=64, hidden_sizes=[512, 256, 128, 64], dropout=0.3).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_val_loss = float("inf")
    best_acc = 0.0
    patience = 0
    start_time = time.time()
    save_path = Path(save_dir) / "trained_model.pth"

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

        tl = t_loss / len(train_loader)
        vl = v_loss / len(val_loader)
        ta = correct / total
        va = v_correct / v_total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train L:{tl:.4f} A:{ta:.4f} | Val L:{vl:.4f} A:{va:.4f} | {time.time()-start_time:.0f}s")

        if vl < best_val_loss:
            best_val_loss = vl
            best_acc = va
            patience = 0
            torch.save(model.state_dict(), str(save_path))
        else:
            patience += 1
            if patience >= 30:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining done in {(time.time()-start_time)/60:.1f} min | Best val acc: {best_acc:.4f}")

    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    binary_preds = (all_probs_np > 0.5).astype(int)
    try:
        auc = roc_auc_score(all_labels_np, all_probs_np)
        print(f"AUC-ROC: {auc:.4f}")
    except Exception:
        pass
    print(classification_report(all_labels_np, binary_preds, target_names=["Genuine", "Synthetic"]))
    print(confusion_matrix(all_labels_np, binary_preds))

    # Export ONNX
    import onnx
    import onnxruntime as ort
    model_cpu = VoiceAuthenticityNet(input_size=64, hidden_sizes=[512, 256, 128, 64], dropout=0.0)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    dummy = torch.randn(1, 64)
    onnx_path = Path(save_dir) / "trained_model.onnx"
    torch.onnx.export(model_cpu, dummy, str(onnx_path), opset_version=14,
                      input_names=["features"], output_names=["synthetic_prob"],
                      dynamic_axes={"features": {0: "batch"}, "synthetic_prob": {0: "batch"}},
                      dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data" / "wavefake"

    print("=" * 60)
    print("VOICE CLONE DETECTION - WaveFake Training")
    print("=" * 60)

    print("\n--- Step 1: Download ---")
    download_all_shards(data_dir, max_files_per_class=6000)

    print("\n--- Step 2: Feature Extraction (parallel) ---")
    X, y = extract_features_parallel(data_dir, max_per_class=6000, workers=4)
    print(f"Dataset: {len(X)} samples (Genuine={int(np.sum(y==0))}, Synthetic={int(np.sum(y==1))})")

    print("\n--- Step 3: Training ---")
    train(X, y, epochs=300, batch_size=64, lr=0.001, save_dir="voice_detection_app")

    print("\nDone!")
