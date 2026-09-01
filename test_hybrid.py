"""Test hybrid detector on real files."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
import librosa
from voice_detection_app.models.detector import VoiceDetector
from voice_detection_app.services.audio_processor import AudioProcessor
from voice_detection_app.services.forensic_analyzer import analyze_forensic, hybrid_score

det = VoiceDetector()
det.load_model()
ap = AudioProcessor()

def test_file(path):
    y,sr = ap.load_audio(path)
    _, agg = ap.process_audio(y)
    fv = ap.get_feature_vector(agg)
    ml = det.predict(fv)
    ml_prob = ml["synthetic_probability"]
    forensic = analyze_forensic(y,sr)
    hybrid = hybrid_score(ml_prob, forensic["forensic_score"])
    return ml_prob, forensic["forensic_score"], hybrid["final_synthetic_prob"], forensic

genuines = list(Path('data/librispeech-genuine').glob('*.wav'))[:3] + list(Path('data/wavefake/genuine').glob('*.wav'))[:2]
synthetics = list(Path('data/wavefake/synthetic').glob('*.wav'))[:5]

print("=== GENUINE (should be HUMAN ~0% AI) ===")
for p in genuines:
    ml, f, final, forensic = test_file(p)
    print(f"\n{p.name[:30]}")
    print(f"  ML: {ml*100:.1f}% AI | Forensic: {f*100:.1f}% AI | FINAL: {final*100:.1f}% AI -> {'SYN' if final>0.5 else 'HUMAN'}")
    print(f"  Human similarity: {forensic['human_similarity']:.1f}% | AI similarity: {forensic['ai_similarity']:.1f}%")
    for k, v in list(forensic['factors'].items())[:3]:
        print(f"    {k}: raw {v['raw_value']} -> {v['synthetic_percent']}% AI ({v['status']}) - {v['interpretation']}")

print("\n=== SYNTHETIC (should be AI ~100%) ===")
for p in synthetics:
    ml, f, final, forensic = test_file(p)
    print(f"\n{p.name[:30]}")
    print(f"  ML: {ml*100:.1f}% AI | Forensic: {f*100:.1f}% AI | FINAL: {final*100:.1f}% AI -> {'SYN' if final>0.5 else 'HUMAN'}")
    print(f"  Human similarity: {forensic['human_similarity']:.1f}% | AI similarity: {forensic['ai_similarity']:.1f}%")
    for k, v in list(forensic['factors'].items())[:3]:
        print(f"    {k}: raw {v['raw_value']} -> {v['synthetic_percent']}% AI ({v['status']})")
