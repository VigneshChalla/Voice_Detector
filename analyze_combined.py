import librosa, numpy as np
from pathlib import Path

def extract_raw_metrics(path):
    y,sr = librosa.load(str(path), sr=16000, duration=10.0)
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pv = pitches[mags > np.median(mags)]
    if len(pv)==0: pv=np.array([100])
    pv = pv[(pv>60)&(pv<500)]
    if len(pv)==0: pv=np.array([100])
    pitch_mean=np.mean(pv); pitch_std=np.std(pv); pitch_cv=pitch_std/(pitch_mean+1e-6); pitch_range=np.ptp(pv)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sc_mean=np.mean(sc); sc_std=np.std(sc); sc_cv=sc_std/(sc_mean+1e-6)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sb_mean=np.mean(sb); sb_std=np.std(sb)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean=np.mean(zcr); zcr_std=np.std(zcr)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean=np.mean(rms); rms_std=np.std(rms); rms_cv=rms_std/(rms_mean+1e-6)
    phase=np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
    pd=np.diff(phase,axis=1)
    pd_mean=np.mean(np.abs(pd)); pd_std=np.std(pd)
    mfcc=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta=librosa.feature.delta(mfcc)
    mfcc_delta_std=np.std(mfcc_delta)
    return dict(pitch_cv=pitch_cv, pitch_range=pitch_range, sc_cv=sc_cv, sc_std=sc_std, zcr_std=zcr_std, rms_cv=rms_cv, pd_std=pd_std, mfcc_delta_std=mfcc_delta_std, sb_std=sb_std)

genuine_libri = list(Path('data/librispeech-genuine').glob('*.wav'))[:30]
genuine_wavefake = list(Path('data/wavefake/genuine').glob('*.wav'))[:30]
genuine_combined = genuine_libri + genuine_wavefake
synthetics = list(Path('data/wavefake/synthetic').glob('*.wav'))[:40]

metrics=['pitch_cv','pitch_range','sc_cv','sc_std','zcr_std','rms_cv','pd_std','mfcc_delta_std','sb_std']
print('Metric               | LibriSpeech          | WaveFake genuine     | Combined genuine     | Synthetic            | sep_combined')
print('-'*130)
for m in metrics:
    gl=[extract_raw_metrics(p)[m] for p in genuine_libri]
    gw=[extract_raw_metrics(p)[m] for p in genuine_wavefake]
    gc=[extract_raw_metrics(p)[m] for p in genuine_combined]
    s=[extract_raw_metrics(p)[m] for p in synthetics]
    def stats(arr):
        return np.mean(arr), np.std(arr), np.median(arr)
    glm, gls,_ = stats(gl); gwm,gws,_=stats(gw); gcm,gcs,_=stats(gc); sm,ss,_=stats(s)
    sep = abs(gcm-sm)/ ((gcs+ss)/2 +1e-9)
    print(f'{m:20s} | {glm:6.3f}+-{gls:.3f} | {gwm:6.3f}+-{gws:.3f} | {gcm:6.3f}+-{gcs:.3f} | {sm:6.3f}+-{ss:.3f} | {sep:.2f}')
