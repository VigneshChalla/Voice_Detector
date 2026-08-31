"""Download WaveFake dataset directly from parquet shards and train."""
import urllib.request
import json
import io
import time
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

data_dir = Path("data/wavefake")
data_dir.mkdir(parents=True, exist_ok=True)

existing_wavs = list(data_dir.rglob("*.wav"))
print(f"Existing WAV files: {len(existing_wavs)}")

if len(existing_wavs) >= 1000:
    print("Already have enough data, skipping download")
else:
    repo = "ajaykarthick/wavefake-audio"
    api_url = f"https://huggingface.co/api/datasets/{repo}/parquet"
    
    print("Fetching parquet URLs...")
    req = urllib.request.Request(api_url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        urls = data["default"]["train"]
    
    print(f"Found {len(urls)} shards")
    
    total_saved = 0
    max_files = 8000
    max_shards = 20
    
    for shard_idx, url in enumerate(urls[:max_shards]):
        if total_saved >= max_files:
            break
        
        print(f"\nShard {shard_idx+1}/{max_shards}...", end=" ", flush=True)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                parquet_bytes = resp.read()
            
            buf = io.BytesIO(parquet_bytes)
            table = pq.read_table(buf)
            df = table.to_pandas()
            
            saved_this_shard = 0
            for _, row in df.iterrows():
                if total_saved >= max_files:
                    break
                try:
                    audio_data = row["audio"]["bytes"]
                    label_str = row["real_or_fake"]
                    label = 0 if label_str == "R" else 1
                    
                    samples, sr = sf.read(io.BytesIO(audio_data))
                    if isinstance(samples, np.ndarray) and samples.ndim > 1:
                        samples = samples.mean(axis=1)
                    
                    out_dir = data_dir / ("genuine" if label == 0 else "synthetic")
                    out_dir.mkdir(exist_ok=True)
                    out_path = out_dir / f"wf_{total_saved:05d}.wav"
                    sf.write(str(out_path), samples.astype(np.float32), sr)
                    total_saved += 1
                    saved_this_shard += 1
                except Exception:
                    continue
            
            print(f"saved {saved_this_shard} (total: {total_saved})")
            
        except Exception as e:
            print(f"error: {e}")
            continue
    
    print(f"\nDownload complete: {total_saved} files")

# Count files
genuine = list((data_dir / "genuine").rglob("*.wav")) if (data_dir / "genuine").exists() else []
synthetic = list((data_dir / "synthetic").rglob("*.wav")) if (data_dir / "synthetic").exists() else []
print(f"Genuine: {len(genuine)}, Synthetic: {len(synthetic)}")
