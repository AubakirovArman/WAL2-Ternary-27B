"""E022 paired experiment supervisor; no allocations outside physicalGPU2/6."""
import asyncio,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from ternary_core import ROOT
from train_v2 import write_json

RUN=ROOT/'runs/e022-joint-scales';PY=ROOT/'.venv/bin/python'
def status(stage,**kw):
    value=dict(state='running',stage=stage,updated=time.time(),**kw)
    write_json(RUN/'status.json',value)
    with (RUN/'journal.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(time.strftime('%H:%M:%S')+' | '+stage,flush=True)
def cpuenv():return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',OMP_NUM_THREADS='8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def export(cfg,arm,step):
    folder=RUN/'evaluation'/f'{arm}-{step:03d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.export.json').exists():return folder
    latent=RUN/arm/f'checkpoint-{step:03d}'
    with (folder/'export.log').open('w') as log:
        if not (folder/'packed/manifest.json').exists():
            if (folder/'packed').exists():shutil.rmtree(folder/'packed')
            subprocess.run([str(PY),str(ROOT/'scripts/export_latent_cpu.py'),str(latent),cfg['packed_template'],str(folder/'packed')],env=cpuenv(),stdout=log,stderr=log,check=True)
        subprocess.run([str(PY),str(ROOT/'scripts/export_prism_pq2.py'),str(folder/'packed'),str(folder/'model.gguf'),'--name',f'Vol2 E022 {arm} step{step}'],env=cpuenv(),stdout=log,stderr=log,check=True)
    return folder
def evaluate(cfg,folder,label):
    import run_e020_generation as gen
    # Only this supervisor's imported module instance is configured; scripts
    # on disk and all older experiment settings remain unchanged.
    gen.native.GPU_IDS=cfg['gpu_uuids']
    tasks=json.loads(Path(cfg['tasks_file']).read_text())
    if (folder/'completed.json').exists():return
    handles=[]
    for name in ['gpu2.lock','gpu67.lock']:
        h=(ROOT/'runs'/name).open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
    for gpu in cfg['gpu_uuids']:
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<25*1024:raise RuntimeError('Insufficient free memory for native evaluation; do not stop services')
    done={json.loads(l)['id'] for l in (folder/'scores.jsonl').open()} if (folder/'scores.jsonl').exists() else set()
    asyncio.run(gen.evaluate(str(folder/'model.gguf'),label,folder,[t for t in tasks if t['id'] not in done]))
    gen.summarize(folder)
    scores=[json.loads(l) for l in (folder/'scores.jsonl').open()]
    assert len(scores)==136 and {r['id'] for r in scores}=={t['id'] for t in tasks}
    write_json(folder/'completed.json',dict(model=label,total=136,tasks_sha256=cfg['tasks_sha256']))
    for h in handles:h.close()
def comparison(cfg):
    parent=[json.loads(l) for l in (ROOT/'reports/e020-ab/B-1024/scores.jsonl').open()]
    summaries={};paired={}
    for label,rows in [('parent',parent)]+[(f'{a}-{s:03d}',[json.loads(l) for l in (RUN/'evaluation'/f'{a}-{s:03d}/scores.jsonl').open()]) for a in ['fixed','joint'] for s in cfg['checkpoints']]:
        summaries[label]=dict(correct=sum(r['correct'] for r in rows),total=len(rows),truncated=sum(r['truncated'] for r in rows),families={})
        for family in sorted({r['family'] for r in rows}):
            own=[r for r in rows if r['family']==family]
            summaries[label]['families'][family]=dict(correct=sum(r['correct'] for r in own),total=len(own),truncated=sum(r['truncated'] for r in own))
        if label!='parent':
            original={r['id']:r for r in parent}
            paired[label]=dict(gained=sum(r['correct'] and not original[r['id']]['correct'] for r in rows),lost=sum(not r['correct'] and original[r['id']]['correct'] for r in rows))
    write_json(RUN/'comparison.json',dict(summary=summaries,paired_vs_parent=paired,
                scope='136 exposed development tasks, answer-only checks; no automatic model promotion'))
def main():
    lock=(RUN/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((RUN/'config.json').read_text())
    for file,key in [(RUN/'train.json','train_sha256'),(RUN/'validation.json','validation_sha256'),(Path(cfg['cache_manifest']),'cache_manifest_sha256'),(Path(cfg['tasks_file']),'tasks_sha256')]:
        assert sha(file)==cfg[key],f'Frozen input changed: {file}'
    # Train both arms before native evaluation so each arm's GPU residency
    # stays continuous and neither branch receives additional training data.
    for arm in ['fixed','joint']:
        if (RUN/'STOP').exists():raise RuntimeError('User STOP requested')
        folder=RUN/arm;folder.mkdir(exist_ok=True)
        if not (folder/'training-completed.json').exists():
            status(f'Обучение {"A: фиксированные масштабы" if arm=="fixed" else "B: веса и масштабы"}',arm=arm,phase_folder=str(folder))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(cfg['gpu_uuids']),PYTHONPATH=str(ROOT/'tools/fla-probe-packages'),PYTORCH_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='8')
            with (folder/'trainer.log').open('a') as log:subprocess.run([str(PY),'-u',str(ROOT/'scripts/e022_gpu_train.py'),arm],env=env,stdout=log,stderr=log,check=True)
    initial=[json.loads((RUN/a/'initial-validation.json').read_text()) for a in ['fixed','joint']]
    assert initial[0]['tokens']==initial[1]['tokens']
    if abs(initial[0]['ce']-initial[1]['ce'])>1e-5:
        raise RuntimeError('Zero-gain joint arm differs from parent validation; paired comparison invalid')
    for arm in ['fixed','joint']:
        for step in cfg['checkpoints']:
            if (RUN/'STOP').exists():raise RuntimeError('User STOP requested')
            status(f'{arm}, шаг {step}: экспорт тернарных весов',arm=arm,step=step)
            folder=export(cfg,arm,step)
            status(f'{arm}, шаг {step}: проверка генерацией 136 задач',arm=arm,step=step,phase_folder=str(folder))
            evaluate(cfg,folder,f'{arm}-{step:03d}')
    comparison(cfg)
    write_json(RUN/'status.json',dict(state='completed',stage='Обе ветки обучены; четыре модели проверены на 136 задачах',updated=time.time(),comparison=str(RUN/'comparison.json')))
if __name__=='__main__':
    try:main()
    except Exception as error:
        old=json.loads((RUN/'status.json').read_text());old.update(state='stopped' if (RUN/'STOP').exists() else 'failed',error=repr(error),updated=time.time())
        write_json(RUN/'status.json',old);raise
