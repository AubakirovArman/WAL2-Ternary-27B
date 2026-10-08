"""Frozen math + code/instruction/logic controls; native PQ2 batched GPU6/7."""
import argparse,asyncio,collections,concurrent.futures as cf,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
import httpx
from transformers import AutoTokenizer
import run_e019_d as native
from benchmark_tasks import final_text
from collect_e020_data import api
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/e020-ab';DATA=ROOT/'data/e020-natural-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def append(path,obj):
    with path.open('a') as stream:stream.write(json.dumps(obj,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'tasks.json').exists():return
    cfg=json.loads((DATA/'config.json').read_text());assert sha(DATA/'control-tasks.jsonl')==cfg['control_sha256']
    tok=AutoTokenizer.from_pretrained(native.SOURCE,local_files_only=True)
    tasks=json.loads((ROOT/'reports/e019-d/comparison-tasks.json').read_text())
    for task in tasks:task['category']='math';task['scope']='Exposed E019 development math; unchanged protocol'
    for line in (DATA/'control-tasks.jsonl').open():
        task=json.loads(line);task['family']='e020_'+task['category']+'_control'
        prompt=tok.apply_chat_template([dict(role='user',content=task['instruction'])],tokenize=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='medium')
        ids=tok.encode(prompt,add_special_tokens=False)
        task.update(input_ids=ids,prompt_tokens=len(ids),scope='Frozen disjoint fresh development prompts; not formal unseen benchmark')
        tasks.append(task)
    assert len(tasks)==136 and len({t['id'] for t in tasks})==136
    # No evaluation prompt may be an exact dataset candidate, including unused/rejected.
    pool={json.loads(l)['sha256'] for l in (DATA/'tasks.jsonl').open()}
    assert not (pool&{hashlib.sha256(t['instruction'].encode()).hexdigest() for t in tasks})
    write(OUT/'tasks.json',tasks)
    write(OUT/'protocol.json',dict(tasks=136,old_math=68,fresh_math=20,code=16,instructions=16,logic=16,
                                  tasks_sha256=sha(OUT/'tasks.json'),training_control_exclusion=True,
                                  decoding='Same original E019-D greedy temp0,repeat1,thinking medium,8192 output tokens,ctx32768',
                                  official_scope='Answer-only metrics; independent proof review separate; development controls, no withheld official test claim'))
def summarize(folder):
    p=folder/'scores.jsonl';rows=[json.loads(l) for l in p.open()] if p.exists() else []
    groups=collections.defaultdict(list)
    for r in rows:groups[r['model'],r['family']].append(r)
    report={}
    for (model,family),items in groups.items():
        good=sum(r['correct'] for r in items)
        report.setdefault(model,{})[family]=dict(correct=good,total=len(items),accuracy=good/len(items),
                                                truncated=sum(r['truncated'] for r in items),verification_errors=sum(bool(r['verification_error']) for r in items),
                                                generated_tokens=sum(r['generated_tokens'] for r in items))
    write(folder/'summary.json',report)
def score_one(task,response):
    proc=subprocess.run([str(ROOT/'.venv-release/bin/python'),str(ROOT/'scripts/score_e020_response.py')],
                        input=json.dumps(dict(task=task,response=response)),text=True,capture_output=True,
                        timeout=35,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'))
    if proc.returncode:raise RuntimeError('CPU scorer process failed: '+proc.stderr[-500:])
    result=json.loads(proc.stdout)
    return dict(model=response['model'],id=task['id'],family=task['family'],**result,truncated=response['finish_reason']=='length',
                generated_tokens=response['generated_tokens'],thinking_completed=response['thinking_completed'])
async def evaluate(model,label,folder,tasks):
    native.OUT=folder;tok=AutoTokenizer.from_pretrained(native.SOURCE,local_files_only=True)
    ports=(18806,18807);servers=[native.start_server(model,gpu,port,12) for gpu,port in zip(native.GPU_IDS,ports)]
    done=0;tokens=0;start=time.monotonic()
    async with httpx.AsyncClient(timeout=1800,limits=httpx.Limits(max_connections=40),trust_env=False) as client:
        try:
            await asyncio.gather(*(native.healthy(client,p,s) for p,s in zip(ports,servers)))
            sem=[asyncio.Semaphore(12),asyncio.Semaphore(12)];score_sem=asyncio.Semaphore(8)
            async def job(task,i):
                nonlocal done,tokens
                async with sem[i%2]:raw=await native.completion(client,ports[i%2],task['input_ids'],8192)
                text=tok.decode(raw['tokens'],skip_special_tokens=False);answer,closed=final_text(text,True)
                row=dict(model=label,id=task['id'],family=task['family'],raw_response=text,answer=answer,thinking_completed=closed,
                         finish_reason='length' if raw['stop_type']=='limit' else 'eos',generated_tokens=len(raw['tokens']),raw=raw)
                append(folder/'responses.jsonl',row)
                async with score_sem:score=await asyncio.to_thread(score_one,task,row)
                append(folder/'scores.jsonl',score);summarize(folder);done+=1;tokens+=len(raw['tokens']);elapsed=time.monotonic()-start
                write(folder/'status.json',dict(state='running',model=label,completed=done,total=len(tasks),tokens_per_second=tokens/max(elapsed,1),eta_seconds=(len(tasks)-done)*elapsed/max(done,1)))
                if done%8==0 or done==len(tasks):print(f'{label}: проверено {done}/{len(tasks)}; {tokens/max(elapsed,1):.0f} ток/с',flush=True)
            await asyncio.gather(*(job(t,i) for i,t in enumerate(tasks)))
        finally:native.stop_servers()
    write(folder/'status.json',dict(state='responses_complete',model=label,completed=len(tasks),total=len(tasks),seconds=time.monotonic()-start))
def baseline():
    freeze();tasks=json.loads((OUT/'tasks.json').read_text());folder=OUT/'baseline';folder.mkdir(exist_ok=True)
    if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text()).get('state')=='completed':return
    # Reuse fixed original math scores and answers for both references, unchanged.
    if not (folder/'scores.jsonl').exists():
        for line in (ROOT/'reports/e019-d/comparison-scores.jsonl').open():
            r=json.loads(line)
            if r['model'] in ('e018','teacher'):append(folder/'scores.jsonl',dict(r,thinking_completed=r['closed']))
        for line in (ROOT/'reports/e019-d/comparison-responses.jsonl').open():
            r=json.loads(line)
            if r['model'] in ('e018','teacher'):append(folder/'responses.jsonl',r)
    done={(r['model'],r['id']) for r in (json.loads(l) for l in (folder/'scores.jsonl').open())}
    def teacher_job(task):
        raw=api(task['instruction']);choice=raw['choices'][0];msg=choice['message'];reason=msg.get('reasoning_content') or ''
        row=dict(model='teacher',id=task['id'],family=task['family'],raw_response=msg,answer=msg.get('content') or '',thinking_completed=bool(reason.strip()) and choice['finish_reason']=='stop',
                 finish_reason='length' if choice['finish_reason']=='length' else 'eos',generated_tokens=raw.get('usage',{}).get('completion_tokens',0),raw=raw)
        return row,score_one(task,row)
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with cf.ThreadPoolExecutor(max_workers=20) as pool:
        futures=[pool.submit(teacher_job,t) for t in tasks if ('teacher',t['id']) not in done]
        for f in cf.as_completed(futures):
            row,score=f.result();append(folder/'responses.jsonl',row);append(folder/'scores.jsonl',score);summarize(folder)
    lock.close()
    gpu_lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for gpu in native.GPU_IDS:
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<100*1024:raise RuntimeError('GPU6/7 occupied; do not disturb')
    model=json.loads((ROOT/'reports/e019-d/config.json').read_text())['models']['e018']
    asyncio.run(evaluate(model,'e018',folder,[t for t in tasks if ('e018',t['id']) not in done]));summarize(folder)
    # Report per-task teacher/student difficulty bands on fresh math, never train on these.
    scores={(r['model'],r['id']):r for r in (json.loads(l) for l in (folder/'scores.jsonl').open())}
    bands=[]
    for t in tasks:
        if t['family']!='e020_math_control':continue
        a=scores['e018',t['id']]['correct'];b=scores['teacher',t['id']]['correct']
        bands.append(dict(id=t['id'],e018_correct=a,teacher_correct=b,band='student_solved' if a else 'teacher_only' if b else 'neither_solved'))
    write(folder/'difficulty-bands.json',bands)
    assert len(scores)==272
    write(folder/'status.json',dict(state='completed',models=['e018','teacher'],completed=272,total=272))
def candidate(branch,step):
    freeze();label=f'{branch}-{step}';folder=OUT/label
    if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text()).get('state')=='completed':return
    folder.mkdir(exist_ok=True);latent=ROOT/f'runs/e020-ab-{branch.lower()}/candidates/step-{step:04d}';assert json.loads((latent/'metadata.json').read_text())['step']==step
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8');packed=folder/'packed';gguf=folder/'model.gguf'
    if not (packed/'manifest.json').exists():subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/export_latent_cpu.py'),str(latent),str(ROOT/'reports/e018-final-aime10/direct-packed'),str(packed)],env=cpu_env,check=True)
    if not (folder/'model.export.json').exists():subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/export_prism_pq2.py'),str(packed),str(gguf),'--name',f'Vol2 E020 {label}'],env=cpu_env,check=True)
    tasks=json.loads((OUT/'tasks.json').read_text());done={r['id'] for r in (json.loads(l) for l in (folder/'scores.jsonl').open())} if (folder/'scores.jsonl').exists() else set()
    lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for gpu in native.GPU_IDS:
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<100*1024:raise RuntimeError('GPU6/7 occupied; existing jobs left unchanged')
    asyncio.run(evaluate(str(gguf),label,folder,[t for t in tasks if t['id'] not in done]));summarize(folder)
    baseline_scores={(r['model'],r['id']):r for r in (json.loads(l) for l in (OUT/'baseline/scores.jsonl').open())}
    own={r['id']:r for r in (json.loads(l) for l in (folder/'scores.jsonl').open())};pairs=[]
    for t in tasks:
        a=own[t['id']];b=baseline_scores['e018',t['id']]
        pairs.append(dict(id=t['id'],family=t['family'],correct=a['correct'],e018_correct=b['correct'],gained=a['correct'] and not b['correct'],lost=b['correct'] and not a['correct']))
    write(folder/'paired.json',pairs)
    # Newly correct math finals are flagged for separate reasoning review, not automatically promoted.
    responses={r['id']:r for r in (json.loads(l) for l in (folder/'responses.jsonl').open())}
    write(folder/'new-correct-math-review.json',[dict(task=t,response=responses[t['id']],proof_review='pending') for t in tasks if t['category']=='math' and own[t['id']]['correct'] and not baseline_scores['e018',t['id']]['correct']])
    assert len(own)==136
    write(folder/'status.json',dict(state='completed',model=label,completed=136,total=136))
def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');p.add_argument('--baseline',action='store_true');p.add_argument('--branch',choices=['A','B']);p.add_argument('--step',type=int,choices=[512,1024]);a=p.parse_args()
    if a.freeze_only:freeze()
    elif a.baseline:baseline()
    elif a.branch and a.step:candidate(a.branch,a.step)
    else:raise ValueError('Choose frozen baseline or explicit candidate')
if __name__=='__main__':main()
