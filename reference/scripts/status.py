"""Read-only progress summary; does not initialize CUDA or contact the teacher."""
import json,subprocess,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]

def complete_rows(path):
    if not path.exists():return []
    text=path.read_text()
    lines=text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    return [json.loads(line) for line in lines]

def current_status():
    print('Services (active can mean waiting; command identifies the current worker):')
    services=['vol2-broad-smoke-e006','vol2-e007-scale-recovery','vol2-e008-reasoning-qat',
              'vol2-recovery-benchmarks','vol2-reasoning-diverse-v2','vol2-reasoning-verified-v2',
              'vol2-reasoning-code-verified-v1','vol2-e009-mixed-reasoning','vol2-e009-benchmarks',
              'vol2-e010-ce-continuation','vol2-e010-reasoning-continuation','vol2-e011-low-lr-reasoning',
              'vol2-reasoning-diverse-v3','vol2-fresh-reasoning-data']
    for service in services:
        result=subprocess.run(['systemctl','--user','show',service+'.service','-p','ActiveState','-p','MainPID'],
                              capture_output=True,text=True,timeout=5)
        fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        pid=fields.get('MainPID','0');command=''
        if pid!='0':
            command=subprocess.run(['ps','-p',pid,'-o','args='],capture_output=True,text=True,timeout=5).stdout.strip()
        print(' ',service,fields.get('ActiveState','unavailable'),'PID='+pid,command)
        if pid!='0':
            children=subprocess.run(['ps','--ppid',pid,'-o','pid=,args='],capture_output=True,text=True,timeout=5).stdout.strip()
            if children:print('    child:',children)
    print('\nMatched core benchmark:')
    base=ROOT/'reports/broad-smoke-e006'
    for model in ('bonsai2','source','student'):
        folder=base/model;rows=complete_rows(folder/'responses.jsonl')
        config=json.loads((folder/'config.json').read_text()) if (folder/'config.json').exists() else {}
        print(' ',model,config.get('state','not started'),f'{len(rows)}/96 responses')
        if config.get('state')=='completed' and (folder/'summary.json').exists():
            scores=json.loads((folder/'summary.json').read_text())
            print('   ',{k:f"{v['correct']}/{v['total']}" for k,v in scores.items()})
    print('\nRecovery generation benchmarks:')
    for folder in sorted([*(ROOT/'reports/recovery-smoke').glob('*/'),*(ROOT/'reports/e009-smoke').glob('*/')]):
        config_path=folder/'config.json'
        if not config_path.exists():continue
        config=json.loads(config_path.read_text())
        rows=complete_rows(folder/'responses.jsonl')
        thinking=[r for r in rows if r.get('thinking')]
        state=config.get('state','unknown')
        if state!='completed' and (folder.parent/'INTERRUPTED_FOR_TRAINING.json').exists():
            state='interrupted for continued training'
        print(' ',folder.name,state,f'{len(rows)}/96 responses',
              'completed thinking:',f"{sum(bool(r.get('thinking_completed')) for r in thinking)}/{len(thinking)}")
        if config.get('state')=='completed' and (folder/'summary.json').exists():
            print('   scored:',json.loads((folder/'summary.json').read_text()))
    print('\nRecovery training:')
    for name in ('e007-scale-recovery','e008-reasoning-qat','e009-mixed-reasoning','e010-ce-continuation','e010-reasoning-continuation','e011-low-lr-reasoning'):
        folder=ROOT/'runs'/name;path=folder/'metrics.json'
        if not path.exists():
            print(' ',name,'no validation metrics yet')
            metrics={}
        else:
            metrics=json.loads(path.read_text())
            state='interrupted' if (folder/'INTERRUPTED.json').exists() else metrics.get('state')
            print(' ',name,state,'selected:',metrics.get('selected_packed','pending'))
        if metrics.get('selected_validation'):print('   validation:',metrics['selected_validation'])
        if metrics.get('selected_reasoning'):print('   thinking validation:',metrics['selected_reasoning'])
        checks=metrics.get('checks') or metrics.get('pilots') or []
        if checks:print('   latest validation check:',checks[-1])
        if metrics.get('reasoning_candidate_latent'):
            print('   candidate awaiting generation evaluation:',metrics['reasoning_candidate_latent'])
        if metrics.get('checks'):
            print('   Closing-token counts use gold prefixes; they do not measure free-generation success.')
        logs=[folder/'training.jsonl',*sorted(folder.glob('lr-*/training.jsonl'))]
        for log in logs:
            rows=complete_rows(log)
            if rows:print('   ',log.parent.name,'latest step:',rows[-1])
    print('\nReasoning collection:')
    for name,target in [('reasoning-diverse-v1',160),('reasoning-diverse-v2',1152),('reasoning-diverse-v3',1728),('reasoning-verified-v2',256),('reasoning-code-verified-v1',128)]:
        folder=ROOT/'data'/name;events=complete_rows(folder/'events.jsonl');accepted=complete_rows(folder/'accepted.jsonl')
        print(' ',name,f'{len(events)}/{target} processed; accepted:',dict(Counter(r['split'] for r in accepted)))
    print('\nThese counters/CE values do not establish Bonsai quality parity.')

if '--current' in sys.argv:
    current_status()
    raise SystemExit(0)

data=ROOT/'data/teacher_pairs.jsonl'
rows=[json.loads(s) for s in data.read_text().splitlines()] if data.exists() else []
print('Teacher pairs:',len(rows),dict(Counter(r['split'] for r in rows)))
new_data=ROOT/'data/v2/teacher_new.jsonl'
if new_data.exists():
    text=new_data.read_text();complete=text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    print('Teacher V2 accepted:',len(complete),'/ 5000')
kd_manifest=ROOT/'data/v3-kd/manifest.json'
if kd_manifest.exists():
    m=json.loads(kd_manifest.read_text())
    print('Teacher probability cache:',m.get('state'),m.get('cached',len(m.get('examples',{}))),'/',m.get('count_target'))
for run in sorted((ROOT/'runs').iterdir()):
    if not run.is_dir():continue
    print('\n'+run.name)
    metrics=run/'metrics.json'
    if metrics.exists():
        m=json.loads(metrics.read_text())
        print('  state:',m.get('state','unknown'))
        print('  source CE:',m.get('source_validation',{}).get('answer_ce'))
        print('  initial ternary CE:',m.get('ternary_initial_validation',{}).get('answer_ce'))
        print('  latest validation:',(m.get('validation') or [None])[-1])
        print('  best checkpoint:',m.get('best_checkpoint','not finalized'))
        if m.get('selected_lr') is not None:print('  selected LR:',m['selected_lr'])
        if m.get('selected_validation'):print('  selected validation:',m['selected_validation'])
        if m.get('recommended_packed'):print('  recommended checkpoint:',m['recommended_packed'])
    training=run/'training.jsonl'
    if training.exists():print('  latest step:',json.loads(training.read_text().splitlines()[-1]))
    for child in sorted(run.iterdir()):
        if not child.is_dir() or not (child/'metrics.json').exists():continue
        c=json.loads((child/'metrics.json').read_text())
        print('  stage:',child.name,'state:',c.get('state'),'best CE:',c.get('best_validation_ce'))
        if c.get('validation'):print('    latest validation:',c['validation'][-1])
        train=child/'training.jsonl'
        if train.exists():
            text=train.read_text();lines=text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
            if lines:print('    latest step:',json.loads(lines[-1]))
events=ROOT/'logs/events.jsonl'
if events.exists():
    print('\nRecent events:')
    for line in events.read_text().splitlines()[-5:]:print(line)
