"""Independent native teacher-forced CE per category for E018 and all retained E019."""
import concurrent.futures as cf
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

from run_release_benchmarks import ROOT, BACK, BIN, env_for, write
from ternary_core import GPU_IDS

OUT=ROOT/'reports/e019-category-ce'
MODELS={
 'e018':ROOT/'reports/e018-final-aime10/e018-final.gguf',
 'step2560':ROOT/'reports/e019-best-math68/e019-best.gguf',
 'step3584':ROOT/'reports/e019-step3584-math68/e019-best.gguf',
 'step4096':ROOT/'reports/e019-step4096-math68/e019-best.gguf',
}


def evaluate(name,gpu,rows):
    expected=set(range(len(rows)));seen=set();cases=[]
    with (OUT/(name+'.log')).open('w') as log,(OUT/(name+'.jsonl')).open('w') as output:
        proc=subprocess.Popen([str(BIN),str(MODELS[name]),str(OUT/'inputs.txt'),str(BACK),'1','32768','512'],
                              env=env_for(GPU_IDS[gpu-6]),stdout=subprocess.PIPE,stderr=log,text=True)
        try:
            for line in proc.stdout:
                result=json.loads(line);assert result['kind']=='ce'
                index=result['id'];assert index in expected and index not in seen
                row=rows[index];assert result['tokens']==sum(x!=-100 for x in row['labels'])
                seen.add(index);cases.append(result);output.write(line);output.flush()
                write(OUT/(name+'-status.json'),dict(state='running',completed=len(cases),total=len(rows),updated=time.time()))
            if proc.wait()!=0:raise RuntimeError(name+' native CE failed')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
    assert seen==expected
    groups={};families={}
    for case in cases:
        row=rows[case['id']];ce=case['nll']/case['tokens']
        for dest,key in ((groups,row['category']),(families,row['category']+'/'+row['topic'])):
            group=dest.setdefault(key,dict(nll=0.,tokens=0,example_ce_sum=0.,examples=0))
            group['nll']+=case['nll'];group['tokens']+=case['tokens'];group['example_ce_sum']+=ce;group['examples']+=1
    def finish(dest):
        return {key:dict(ce=g['nll']/g['tokens'],example_mean_ce=g['example_ce_sum']/g['examples'],tokens=g['tokens'],examples=g['examples']) for key,g in dest.items()}
    result=dict(aggregate_ce=sum(c['nll'] for c in cases)/sum(c['tokens'] for c in cases),by_category=finish(groups),by_family=finish(families),
                scope='Native PQ2 gold-prefix CE. Does not measure standalone reasoning accuracy; numerical implementation differs from training reference.')
    write(OUT/(name+'-summary.json'),result)
    write(OUT/(name+'-status.json'),dict(state='completed',completed=len(cases),total=len(rows),updated=time.time()))
    return result


def main():
    OUT.mkdir(exist_ok=False)
    lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=(ROOT/'data/e019-reasoning-pilot-v1/validation.jsonl').read_bytes()
    manifest=json.loads((ROOT/'data/e019-reasoning-pilot-v1/manifest.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==manifest['files']['validation']['sha256']
    rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)==256
    lines=[]
    for i,row in enumerate(rows):
        ids=row['input_ids'];p=row['prompt_tokens']
        assert row['labels']==[-100]*p+ids[p:]
        lines.append(f'E {i} {p} {len(ids)} '+' '.join(map(str,ids)))
    (OUT/'inputs.txt').write_text('\n'.join(lines)+'\n')
    write(OUT/'protocol.json',dict(validation_sha256=manifest['files']['validation']['sha256'],models={k:str(v) for k,v in MODELS.items()},
                                 cases=256,context=32768,batch_tokens=512,gpus=[6,7],scope='No weight updates; same gold input IDs and full target masks for all four models'))
    result={}
    for first,second in (('e018','step2560'),('step3584','step4096')):
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(evaluate,first,6,rows);b=pool.submit(evaluate,second,7,rows)
            result[first]=a.result();result[second]=b.result()
    write(OUT/'summary.json',result)
    write(OUT/'status.json',dict(state='completed',models=4,cases_per_model=256,updated=time.time()))
    print('Категориальные CE четырёх моделей проверены.',flush=True)


if __name__=='__main__':main()
