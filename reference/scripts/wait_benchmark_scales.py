"""Launch the bounded scale pilot only after the queued comparison finishes successfully."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+20*3600
print('Waiting for vol2-broad-smoke-e006.service to finish successfully.',flush=True)
while True:
    raw=subprocess.check_output(['systemctl','--user','show','vol2-broad-smoke-e006.service',
                                 '-p','LoadState','-p','ActiveState','-p','Result','-p','ExecMainStatus'],text=True)
    fields=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    if fields.get('LoadState')=='not-found' and fields.get('ActiveState')=='inactive':
        break  # Successful transient units may be collected; validate durable artifacts below.
    if fields.get('LoadState')!='loaded':raise RuntimeError('Comparison service missing: '+raw)
    if fields.get('ActiveState') in ('inactive','failed'):
        if fields.get('Result')!='success' or fields.get('ExecMainStatus')!='0':
            raise RuntimeError('Comparison did not succeed: '+raw)
        break
    if fields.get('ActiveState') not in ('active','activating','deactivating'):
        raise RuntimeError('Unexpected comparison state: '+raw)
    if time.monotonic()>deadline:raise TimeoutError('Comparison wait expired; no scale job launched')
    time.sleep(30)
report=ROOT/'reports/broad-smoke-e006'
events=[json.loads(s) for s in (report/'events.jsonl').read_text().splitlines()]
if not events or events[-1].get('event')!='completed':raise RuntimeError('Comparison lacks terminal completion record')
for model in ('bonsai2','source','student'):
    config=json.loads((report/model/'config.json').read_text())
    scores=json.loads((report/model/'summary.json').read_text())
    if config.get('state')!='completed' or any(scores[f]['total']!=32 for f in ('gsm8k','humaneval','ifeval')):
        raise RuntimeError('Incomplete comparison output: '+model)
selection=json.loads((report/'training-selection.json').read_text())
checkpoint=Path(selection['selected_packed'])
if selection.get('state')!='completed' or not checkpoint.is_dir():raise RuntimeError('Invalid model selection')
python=str(ROOT/'.venv/bin/python')
os.execv(python,[python,'-u',str(ROOT/'scripts/train_scale_recovery.py'),'--checkpoint',str(checkpoint),
                '--output',str(ROOT/'runs/e007-scale-recovery')])
