"""Wait on the actual cache service, then launch training after it relinquishes GPUs."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
start=time.time()
while True:
    status=subprocess.check_output(['systemctl','--user','show','vol2-kd-cache','-p','ActiveState','-p','Result','-p','ExecMainStatus'],text=True)
    fields=dict(line.split('=',1) for line in status.splitlines() if '=' in line)
    if fields.get('ActiveState') in ('inactive','failed'):
        if fields.get('Result')!='success' or fields.get('ExecMainStatus')!='0':raise RuntimeError('Teacher cache service failed: '+status)
        meta=json.loads((ROOT/'data/v3-kd/manifest.json').read_text())
        if meta.get('state')!='completed':raise RuntimeError('Cache service stopped before completion')
        break
    if fields.get('ActiveState') not in ('active','activating','deactivating'):raise RuntimeError('Unknown cache service state: '+status)
    if time.time()-start>4*3600:raise RuntimeError('Cache wait exceeded four hours')
    print('Waiting for live teacher-cache service to complete.',flush=True);time.sleep(30)
os.execv(str(ROOT/'.venv/bin/python'),[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/train_kd.py')])
