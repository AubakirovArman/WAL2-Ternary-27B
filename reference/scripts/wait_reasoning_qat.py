"""Serialize E008 behind completed E007 and completed reasoning collection."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def wait_terminal(service,deadline):
    print(json.dumps({'event':'waiting','service':service}),flush=True)
    while True:
        raw=subprocess.check_output(['systemctl','--user','show',service,'-p','LoadState','-p','ActiveState','-p','Result','-p','ExecMainStatus'],text=True)
        fields=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        if fields.get('LoadState')=='not-found' and fields.get('ActiveState')=='inactive':return
        if fields.get('LoadState')!='loaded':raise RuntimeError('Unknown service: '+raw)
        if fields.get('ActiveState') in ('inactive','failed'):
            if fields.get('Result')!='success' or fields.get('ExecMainStatus')!='0':raise RuntimeError('Prerequisite failed: '+raw)
            return
        if fields.get('ActiveState') not in ('active','activating','deactivating'):raise RuntimeError('Unexpected service state: '+raw)
        if time.monotonic()>deadline:raise TimeoutError('Prerequisite still running; no E008 launch')
        time.sleep(30)

def main():
    deadline=time.monotonic()+20*3600
    wait_terminal('vol2-e007-scale-recovery.service',deadline)
    metrics=json.loads((ROOT/'runs/e007-scale-recovery/metrics.json').read_text())
    if metrics.get('state')!='completed' or not Path(metrics['selected_packed']).is_dir():raise RuntimeError('E007 lacks durable complete result')
    wait_terminal('vol2-reasoning-diverse.service',deadline)
    folder=ROOT/'data/reasoning-diverse-v1';meta=json.loads((folder/'manifest.json').read_text())
    if meta.get('state')!='completed':raise RuntimeError('Reasoning collection incomplete')
    prepared=ROOT/'data/reasoning-diverse-v1-tokenized';python=str(ROOT/'.venv/bin/python')
    if not prepared.exists():
        subprocess.run([python,str(ROOT/'scripts/prepare_reasoning_diverse.py'),str(folder),'--output',str(prepared)],check=True,cwd=ROOT,timeout=600)
    pm=json.loads((prepared/'manifest.json').read_text())
    if pm.get('state')!='completed' or pm['source_accepted_sha256']!=hashlib.sha256((folder/'accepted.jsonl').read_bytes()).hexdigest():
        raise RuntimeError('Prepared data is incomplete or stale')
    print(json.dumps({'event':'starting_reasoning_qat','parent':'preserved E006 latent; independent of E007 scale branch'}),flush=True)
    os.execv(python,[python,'-u',str(ROOT/'scripts/train_reasoning_qat.py'),'--data',str(prepared),'--output',str(ROOT/'runs/e008-reasoning-qat')])

if __name__=='__main__':main()
