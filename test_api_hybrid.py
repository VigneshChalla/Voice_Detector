import requests
from pathlib import Path
url='http://127.0.0.1:8000/api/v1/detect'
headers={'X-API-Key':'vd_dev_key_2024'}
for name, p in [('GENUINE Libri', list(Path('data/librispeech-genuine').glob('*.wav'))[0]), ('SYNTHETIC', list(Path('data/wavefake/synthetic').glob('*.wav'))[0])]:
    with open(p,'rb') as f:
        files={'file': (p.name, f, 'audio/wav')}
        r=requests.post(url, headers=headers, files=files, data={'caller_id':'test','call_type':'regular_call'}, timeout=30)
        j=r.json()
        print(name)
        print(" Final:", round(j['synthetic_probability']*100,1), "% AI | ML", round(j['ml_probability']*100,1), "| Forensic", round(j['forensic_score']*100,1), "|", j['risk_level'], "|", j['agreement'])
        print(" Human", j['human_similarity'], "AI", j['ai_similarity'])
        print(" Summary:", j['analysis_summary'][:130])
        print(" Factors:", list(j['forensic_factors'].keys())[:3])
        print()
