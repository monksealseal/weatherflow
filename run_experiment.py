import json
import httpx
import time
from pathlib import Path
p = Path(__file__).resolve().parent
payload_file = p / 'experiment_payload.json'
with open(payload_file) as f:
    payload = json.load(f)
print('Posting experiment to http://127.0.0.1:8000/api/experiments')
start = time.time()
try:
    r = httpx.post('http://127.0.0.1:8000/api/experiments', json=payload, timeout=1200.0)
    print('Status', r.status_code)
    if r.status_code == 200:
        data = r.json()
        # pretty-print summary fields
        print('Experiment ID:', data.get('experiment_id'))
        print('Execution duration (s):', data.get('execution', {}).get('duration_seconds'))
        print('Metrics train sample:', data.get('metrics', {}).get('train', [])[:2])
        print('Prediction keys:', list(data.get('prediction', {}).keys())[:5])
    else:
        print('Response text:\n', r.text)
except Exception as e:
    print('Request failed:', e)
print('Elapsed', time.time()-start)
