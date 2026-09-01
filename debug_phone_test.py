import numpy as np, sys
from pathlib import Path
sys.path.insert(0, '.')
from train_phone_aug import phone_augment
from voice_detection_app.models.detector import VoiceDetector
from voice_detection_app.services.audio_processor import AudioProcessor
import librosa, tempfile, soundfile as sf, os

genuines = list(Path('data/librispeech-genuine').glob('*.wav'))[:5]
synthetics = list(Path('data/wavefake/synthetic').glob('*.wav'))[:5]
det = VoiceDetector()
det.load_model()
ap = AudioProcessor()

print('=== CLEAN ===')
for p in genuines:
    y,sr = ap.load_audio(p)
    _, agg = ap.process_audio(y)
    fv = ap.get_feature_vector(agg)
    r = det.predict(fv)
    print(f'G clean {p.name[:22]} -> {r["synthetic_probability"]:.4f} {"SYN" if r["is_synthetic"] else "GEN"}')
for p in synthetics:
    y,sr = ap.load_audio(p)
    _, agg = ap.process_audio(y)
    fv = ap.get_feature_vector(agg)
    r = det.predict(fv)
    print(f'S clean {p.name[:22]} -> {r["synthetic_probability"]:.4f} {"SYN" if r["is_synthetic"] else "GEN"}')

print('\n=== PHONE-AUGMENTED (simulating phone) ===')
for p in genuines:
    y,sr = librosa.load(str(p), sr=16000, duration=30.0)
    y_aug = phone_augment(y, sr)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
        sf.write(tf.name, y_aug, sr)
        tfp = tf.name
    y2, _ = ap.load_audio(Path(tfp))
    _, agg = ap.process_audio(y2)
    fv = ap.get_feature_vector(agg)
    r = det.predict(fv)
    os.unlink(tfp)
    print(f'G phone {p.name[:22]} -> {r["synthetic_probability"]:.4f} {"SYN" if r["is_synthetic"] else "GEN"}')
for p in synthetics:
    y,sr = librosa.load(str(p), sr=16000, duration=30.0)
    y_aug = phone_augment(y, sr)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
        sf.write(tf.name, y_aug, sr)
        tfp = tf.name
    y2, _ = ap.load_audio(Path(tfp))
    _, agg = ap.process_audio(y2)
    fv = ap.get_feature_vector(agg)
    r = det.predict(fv)
    os.unlink(tfp)
    print(f'S phone {p.name[:22]} -> {r["synthetic_probability"]:.4f} {"SYN" if r["is_synthetic"] else "GEN"}')
