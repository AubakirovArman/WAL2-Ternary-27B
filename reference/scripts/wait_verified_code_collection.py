"""Queue code collection after the current arithmetic collector exits successfully."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    deadline=time.monotonic()+4*3600
    while True:
        output=subprocess.check_output(['systemctl','--user','show','vol2-reasoning-verified-v2.service','-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True)
        state=dict(line.split('=',1) for line in output.splitlines())
        if state['MainPID']=='0' and state['ActiveState'] not in ('active','activating','deactivating'):
            if state['ExecMainStatus']!='0':raise RuntimeError('Arithmetic collector failed; inspect before starting next collection')
            manifest=json.loads((ROOT/'data/reasoning-verified-v2/manifest.json').read_text())
            if manifest['state']!='completed':raise RuntimeError('Arithmetic collection not complete')
            break
        if time.monotonic()>deadline:raise TimeoutError('Waiting for arithmetic collection exceeded four hours')
        time.sleep(15)
    print('Arithmetic collector completed; starting sequential code collection',flush=True)
    subprocess.run(['/usr/bin/python3','-u',str(ROOT/'scripts/collect_reasoning_code.py'),'--output',str(ROOT/'data/reasoning-code-verified-v1'),'--count','128','--deadline-hours','3'],cwd=ROOT,check=True)

if __name__=='__main__':main()
