"""Sequential GPU6/7 OPD/QAD phases, fresh deployed policy for every update."""
import asyncio,concurrent.futures as cf,fcntl,hashlib,json,os,queue,shutil,subprocess,threading,time
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,GPU_IDS
from train_v2 import write_json
from run_release_benchmarks import BACK,env_for
from benchmark_tasks import final_text
from e021_objective import grouped_advantages

RUN=ROOT/'runs/e021-opd';PY=ROOT/'.venv/bin/python';BIN=ROOT/'tools/bonsai-runtime/prism-opd-rollout'
def status(stage,**kw):
    value=dict(state='running',stage=stage,updated=time.time(),**kw);write_json(RUN/'status.json',value)
    with (RUN/'journal.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(time.strftime('%H:%M:%S')+' | '+stage,flush=True)
def cpuenv():return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',OMP_NUM_THREADS='8')
def gpuphase(kind,folder,parent=None,step=0):
    if (RUN/'STOP').exists():raise RuntimeError('User STOP requested')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(GPU_IDS),PYTHONPATH='' if kind=='teacher' else str(ROOT/'tools/fla-probe-packages'),PYTORCH_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='8')
    cmd=[str(PY),'-u',str(ROOT/'scripts/e021_gpu_phase.py'),kind,str(folder),'--step',str(step)]
    if parent is not None:cmd+=['--parent',str(parent)]
    with (folder/(kind+'.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=log,check=True)
def export(latent,folder,name):
    cfg=json.loads((RUN/'config.json').read_text())
    if (folder/'model.export.json').is_file() and (folder/'model.gguf').is_file():return folder/'model.gguf'
    if (folder/'packed').exists() and not (folder/'packed/manifest.json').is_file():
        (folder/'packed').rename(folder/f'packed-interrupted-{time.time_ns()}')
    if (folder/'model.gguf').exists():(folder/'model.gguf').rename(folder/f'model-interrupted-{time.time_ns()}.gguf')
    with (folder/'export.log').open('w') as log:
        if not (folder/'packed/manifest.json').exists():subprocess.run([str(PY),str(ROOT/'scripts/export_latent_cpu.py'),str(latent),cfg['packed_template'],str(folder/'packed')],env=cpuenv(),stdout=log,stderr=log,check=True)
        subprocess.run([str(PY),str(ROOT/'scripts/export_prism_pq2.py'),str(folder/'packed'),str(folder/'model.gguf'),'--name',name],env=cpuenv(),stdout=log,stderr=log,check=True)
    # Exact trit roundtrip validation is performed by both existing exporters.
    if not (folder/'model.export.json').is_file():raise RuntimeError('Native export not confirmed')
    return folder/'model.gguf'
def native_rollouts(model,folder,questions,budget,slots,step=0):
    jobs=[]
    for question in questions:
        for g in range(4):jobs.append(dict(id=len(jobs),task=question['task'],prompt_ids=question['prompt_ids'],seed=20261002+step*1000+len(jobs)))
    (folder/'inputs.txt').write_text(''.join(f'{j["id"]} {j["seed"]} {len(j["prompt_ids"])} '+' '.join(map(str,j['prompt_ids']))+'\n' for j in jobs))
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);started=time.monotonic()
    lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Two independent native workers, each bound to one authorized physical UUID.
    messages=queue.Queue();processes=[];mutex=threading.Lock()
    def worker(index):
        own=[j for j in jobs if (j['id']//4)%2==index]
        inp=folder/f'inputs-gpu{6+index}.txt'
        inp.write_text(''.join(f'{j["id"]} {j["seed"]} {len(j["prompt_ids"])} '+' '.join(map(str,j['prompt_ids']))+'\n' for j in own))
        try:
            with (folder/f'native-gpu{6+index}.log').open('w') as log:
                proc=subprocess.Popen([str(BIN),str(model),str(inp),str(BACK),str(budget),'8192',str(slots)],env=env_for(GPU_IDS[index]),stdout=subprocess.PIPE,stderr=log,text=True)
                with mutex:processes.append(proc)
                try:
                    for line in proc.stdout:messages.put(('row',line))
                    if proc.wait()!=0:raise RuntimeError(f'GPU{6+index} native sampler failed; see its native log')
                finally:
                    if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
        except Exception as e:messages.put(('error',repr(e)))
        finally:messages.put(('end',index))
    pool=cf.ThreadPoolExecutor(max_workers=2);futures=[pool.submit(worker,i) for i in range(2)]
    try:
        done=0;completed=0;tokens=0
        with (folder/'native.jsonl').open('w') as out,(folder/'rollouts-unscored.jsonl').open('w') as unscored:
            while done<2:
                kind,value=messages.get()
                if kind=='end':done+=1;continue
                if kind=='error':raise RuntimeError(value)
                out.write(value);out.flush();r=json.loads(value);j=jobs[r['id']]
                assert len(r['tokens'])==len(r['behavior_logp']) and len(r['tokens'])>0
                text=tok.decode(r['tokens'],skip_special_tokens=False);answer,closed=final_text(text,True)
                row=dict(j,**{k:v for k,v in r.items() if k!='id'},raw_response=text,answer=answer,thinking_completed=closed,generated_tokens=len(r['tokens']))
                unscored.write(json.dumps(row,ensure_ascii=False)+'\n');unscored.flush()
                completed+=1;tokens+=len(r['tokens']);elapsed=time.monotonic()-started
                write_json(folder/'phase-status.json',dict(stage='Генерация ответов студента на GPU6/7',completed=completed,total=len(jobs),seconds=elapsed,tokens_per_second=tokens/max(elapsed,1),updated=time.time()))
    finally:
        with mutex:
            for proc in processes:
                if proc.poll() is None:proc.terminate()
        pool.shutdown(wait=True)
    lock.close()
    rows=sorted([json.loads(l) for l in (folder/'rollouts-unscored.jsonl').open()],key=lambda r:r['id'])
    if len(rows)!=len(jobs):raise RuntimeError('Missing rollouts')
    from run_e020_generation import score_one
    def score(row):
        return score_one(row['task'],dict(row,model='e021-rollout',finish_reason=row['finish_reason']))
    with cf.ThreadPoolExecutor(max_workers=8) as pool:scores=list(pool.map(score,rows))
    for start in range(0,len(rows),4):
        rewards=[float(s['correct'] and not s['truncated']) for s in scores[start:start+4]]
        adv=grouped_advantages(rewards)
        for i in range(4):rows[start+i].update(reward=rewards[i],task_advantage=adv[i],score=scores[start+i])
    (folder/'rollouts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    seconds=time.monotonic()-started
    write_json(folder/'rollout-summary.json',dict(samples=len(rows),correct=sum(r['reward'] for r in rows),truncated=sum(r['finish_reason']=='length' for r in rows),tokens=sum(len(r['tokens']) for r in rows),seconds=seconds,tokens_per_second=sum(len(r['tokens']) for r in rows)/seconds,model=str(model),model_sha256=hashlib.file_digest(model.open('rb'),'sha256').hexdigest()))
def review(model,label):
    from run_e020_generation import evaluate,summarize
    folder=RUN/'evaluation'/label
    if (folder/'completed.json').is_file():return
    folder.mkdir(parents=True,exist_ok=True)
    tasks=json.loads((ROOT/'reports/e020-ab/tasks.json').read_text())
    lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    done={json.loads(l)['id'] for l in (folder/'scores.jsonl').open()} if (folder/'scores.jsonl').exists() else set()
    asyncio.run(evaluate(str(model),label,folder,[t for t in tasks if t['id'] not in done]));summarize(folder);lock.close()
    actual={json.loads(l)['id'] for l in (folder/'scores.jsonl').open()}
    if actual!={t['id'] for t in tasks}:raise RuntimeError('Incomplete matched136question generation review')
    write_json(folder/'completed.json',dict(total=136,model=str(model),tasks_sha256=hashlib.sha256((ROOT/'reports/e020-ab/tasks.json').read_bytes()).hexdigest()))
def latest(arm):
    base=RUN/arm;path=base/'working-latent';backup=base/'working-latent.previous';partial=base/'working-latent.writing'
    if not path.exists() and backup.exists():backup.rename(path)
    if partial.exists():partial.rename(base/f'interrupted-save-{time.time_ns()}')
    if not path.exists():return 0
    meta=json.loads((path/'metadata.json').read_text());assert meta['arm']==arm
    assert (path/'manifest.json').is_file() and (path/'training_state.pt').is_file()
    if backup.exists():shutil.rmtree(backup)
    return meta['step']
def retain(arm,step):
    saved=RUN/arm/f'checkpoint-{step:03d}'
    if saved.exists() and not (saved/'metadata.json').is_file():saved.rename(saved.with_name(saved.name+f'-interrupted-{time.time_ns()}'))
    if not saved.exists():shutil.copytree(RUN/arm/'working-latent',saved,copy_function=shutil.copy2)
def main():
    pipeline_lock=(RUN/'pipeline.lock').open('a');fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((RUN/'config.json').read_text());raw=(RUN/'questions.jsonl').read_bytes()
    os.environ['OPENBLAS_NUM_THREADS']='1'
    assert hashlib.sha256(raw).hexdigest()==cfg['questions_sha256']
    assert hashlib.sha256((ROOT/'reports/e020-ab/tasks.json').read_bytes()).hexdigest()==cfg['evaluation_tasks_sha256']
    questions=[json.loads(l) for l in raw.splitlines()]
    for arm in ('opd','qad'):(RUN/arm).mkdir(exist_ok=True)
    # The bounded sampler smoke must have completed before allocating the trainer.
    smoke=[json.loads(l) for l in (RUN/'smoke.jsonl').open()]
    assert len(smoke)==4 and all(len(r['tokens'])==len(r['behavior_logp'])==128 for r in smoke)
    preflight=RUN/'preflight';preflight.mkdir(exist_ok=True)
    rows=[dict(r,prompt_ids=questions[0]['prompt_ids']) for r in smoke]
    (preflight/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    status('Предварительная сверка вероятностей движка и тренера',update=0,total_updates=cfg['updates'],phase_folder=str(preflight))
    if not (preflight/'parity.json').exists():gpuphase('parity',preflight,Path(cfg['parent_latent']))
    assert json.loads((preflight/'parity.json').read_text())['passed']
    finished=latest('opd');parent=RUN/'opd/working-latent' if finished else Path(cfg['parent_latent'])
    model=Path(cfg['parent_gguf'])
    if finished:
        folder=RUN/'opd'/f'update-{finished:03d}'
        model=export(parent,folder,f'Vol2 E021 OPD step{finished}')
        if finished in cfg['retained_updates']:retain('opd',finished)
    for step in range(finished+1,cfg['updates']+1):
        if (RUN/'STOP').exists():raise RuntimeError('User STOP requested')
        folder=RUN/'opd'/f'update-{step:03d}';folder.mkdir(exist_ok=True)
        subset=questions[(step-1)*8:step*8]
        status(f'OPD {step}/{cfg["updates"]}: генерация 32 ответов',arm='opd',update=step,total_updates=cfg['updates'],phase_folder=str(folder))
        if not (folder/'rollout-summary.json').exists():native_rollouts(model,folder,subset,cfg['max_new_tokens'],cfg['rollout_slots'],step)
        else:
            summary=json.loads((folder/'rollout-summary.json').read_text())
            assert summary['model_sha256']==hashlib.file_digest(model.open('rb'),'sha256').hexdigest()
        status(f'OPD {step}: локальный учитель оценивает токены',arm='opd',update=step,total_updates=cfg['updates'],phase_folder=str(folder))
        teacher_rows=[json.loads(l) for l in (folder/'teacher.jsonl').open()] if (folder/'teacher.jsonl').exists() else []
        if len(teacher_rows)!=32:
            if (folder/'teacher.jsonl').exists():(folder/'teacher.jsonl').rename(folder/f'teacher-interrupted-{time.time_ns()}.jsonl')
            gpuphase('teacher',folder)
        status(f'OPD {step}: сверка вероятностей и обучение',arm='opd',update=step,total_updates=cfg['updates'],phase_folder=str(folder))
        gpuphase('opd',folder,parent,step)
        parent=RUN/'opd/working-latent'
        status(f'OPD {step}: экспорт новой тернарной политики',arm='opd',update=step,total_updates=cfg['updates'],phase_folder=str(folder))
        model=export(parent,folder,f'Vol2 E021 OPD step{step}')
        if step in cfg['retained_updates']:
            retain('opd',step)
        # Remove only this experiment's superseded transient deployment builds.
        if step>1 and step-1 not in cfg['retained_updates']:
            old=RUN/'opd'/f'update-{step-1:03d}'
            if (old/'packed').exists():shutil.rmtree(old/'packed')
            (old/'model.gguf').unlink(missing_ok=True)
    status('OPD завершён: проверка 136 задач генерацией',arm='opd',update=cfg['updates'],total_updates=cfg['updates'])
    review(model,'opd-016')
    finished=latest('qad');parent=RUN/'qad/working-latent' if finished else Path(cfg['parent_latent'])
    if finished in cfg['retained_updates']:retain('qad',finished)
    for step in range(finished+1,cfg['updates']+1):
        folder=RUN/'qad'/f'update-{step:03d}';folder.mkdir(exist_ok=True)
        rows=[q['reference'] for q in questions[(step-1)*8:step*8] for _ in range(4)]
        write_json(folder/'qad-rows.json',rows)
        status(f'Контроль QAD {step}/{cfg["updates"]}: обучение',arm='qad',update=step,total_updates=cfg['updates'],phase_folder=str(folder))
        gpuphase('qad',folder,parent,step);parent=RUN/'qad/working-latent'
        if step in cfg['retained_updates']:
            retain('qad',step)
    folder=RUN/'qad/update-016';model=export(parent,folder,'Vol2 E021 paired QAD step16')
    status('Контроль QAD завершён: проверка 136 задач генерацией',arm='qad',update=16,total_updates=16)
    review(model,'qad-016')
    summaries={}
    for label in ('opd-016','qad-016'):
        summaries[label]=json.loads((RUN/'evaluation'/label/'summary.json').read_text())
    summaries['parent_B1024']=json.loads((ROOT/'reports/e020-ab/B-1024/summary.json').read_text())
    write_json(RUN/'comparison.json',summaries)
    write_json(RUN/'status.json',dict(state='completed',stage='OPD и контроль QAD завершены; результаты 136 задач сохранены',update=16,total_updates=16,updated=time.time(),comparison=str(RUN/'comparison.json')))
if __name__=='__main__':
    try:main()
    except Exception as error:
        old=json.loads((RUN/'status.json').read_text())
        old.update(state='stopped' if (RUN/'STOP').exists() else 'failed',error=repr(error),updated=time.time())
        write_json(RUN/'status.json',old)
        with (RUN/'journal.jsonl').open('a') as f:f.write(json.dumps(old,ensure_ascii=False)+'\n')
        raise
