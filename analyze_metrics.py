import librosa, numpy as np
from pathlib import Path

def extract_raw_metrics(path):
    y,sr = librosa.load(str(path), sr=16000, duration=10.0)
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pv = pitches[mags > np.median(mags)]
    if len(pv)==0: pv=np.array([100])
    pv = pv[(pv>50)&(pv<500)]
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
    stft=np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    phase=np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
    pd=np.diff(phase,axis=1)
    pd_mean=np.mean(np.abs(pd)); pd_std=np.std(pd)
    mfcc=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta=librosa.feature.delta(mfcc)
    mfcc_delta_std=np.std(mfcc_delta)
    return dict(pitch_cv=pitch_cv, pitch_range=pitch_range, sc_cv=sc_cv, sc_std=sc_std, zcr_std=zcr_std, rms_cv=rms_cv, pd_std=pd_std, mfcc_delta_std=mfcc_delta_std, sb_std=sb_std)

genuines = list(Path('data/librispeech-genuine').glob('*.wav'))[:40]
synthetics = list(Path('data/wavefake/synthetic').glob('*.wav'))[:40]

metrics=['pitch_cv','pitch_range','sc_cv','sc_std','zcr_std','rms_cv','pd_std','mfcc_delta_std','sb_std']
print(f'Metric               | Genuine mean +- std (median)        | Synthetic mean +- std (median)      | sep')
print('-'*110)
for m in metrics:
    gvals=[extract_raw_metrics(p)[m] for p in genuines]
    svals=[extract_raw_metrics(p)[m] for p in synthetics]
    gm=np.mean(gvals); gs=np.std(gvals); gmed=np.median(gvals)
    sm=np.mean(svals); ss=np.std(svals); smed=np.median(svals)
    sep = abs(gm-sm)/ ((gs+ss)/2 +1e-9)
    print(f'{m:20s} | {gm:7.4f} +- {gs:.4f} ({gmed:.4f}) | {sm:7.4f} +- {ss:.4f} ({smed:.4f}) | {sep:.2f}')
