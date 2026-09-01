"""Debug: compare features from training vs inference on same audio file."""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_detection_app.models.detector import VoiceAuthenticityNet, VoiceDetector
from voice_detection_app.services.audio_processor import AudioProcessor

# Test files
GENUINE_FILE = Path("data/librispeech-genuine/1272_128104_0000.wav")
SYNTHETIC_FILE = Path("data/wavefake/synthetic/bark_00001.wav")

if not GENUINE_FILE.exists():
    # Find any available file
    genuines = list(Path("data/librispeech-genuine").glob("*.wav"))[:1]
    if genuines:
        GENUINE_FILE = genuines[0]
    else:
        print("No genuine files found!")
        sys.exit(1)

synthetics = list(Path("data/wavefake/synthetic").glob("*.wav"))[:1]
if synthetics:
    SYNTHETIC_FILE = synthetics[0]

print(f"Genuine: {GENUINE_FILE}")
print(f"Synthetic: {SYNTHETIC_FILE}")

# Method 1: Inference pipeline (AudioProcessor)
ap = AudioProcessor()

for name, path in [("GENUINE", GENUINE_FILE), ("SYNTHETIC", SYNTHETIC_FILE)]:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    y, sr = ap.load_audio(path)
    print(f"Audio length: {len(y)/sr:.1f}s")

    _, aggregated = ap.process_audio(y)
    fv = ap.get_feature_vector(aggregated, target_length=64)

    print(f"Feature vector shape: {fv.shape}")
    print(f"Feature vector min: {fv.min():.4f}")
    print(f"Feature vector max: {fv.max():.4f}")
    print(f"Feature vector mean: {fv.mean():.4f}")
    print(f"Feature vector std: {fv.std():.4f}")
    print(f"Non-zero features: {np.count_nonzero(fv)}/64")
    print(f"First 10 values: {fv[:10]}")

    # Run model
    detector = VoiceDetector()
    detector.load_model()
    result = detector.predict(fv)
    print(f"Model prediction: {result}")

    # Check model weights
    weights = detector.model.network[0].weight.data
    print(f"Layer 0 weight mean: {weights.mean():.6f}")
    print(f"Layer 0 weight std: {weights.std():.6f}")
    print(f"Layer 0 bias mean: {detector.model.network[0].bias.data.mean():.6f}")
