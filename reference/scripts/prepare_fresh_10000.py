"""Sequential teacher collection; frozen disjoint task families; no training launch."""
import argparse,collections,fcntl,hashlib,json,os,random,sys,time,urllib.request
from pathlib import Path
from transformers import AutoTokenizer
from reasoning_data import encode,SOURCE
from prepare_fresh_5000 import exclusions,norm,write,append
from fresh10000_tasks import TARGET,HOLDOUT,make_task,verify
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/fresh-10000-v2'
URL='http://127.0.0.1:8000/v1';MODEL='Qwen/Qwen3.8-27B-FP8'
def sha(b):return hashlib.sha256(b).hexdigest()
def load(path):
    if not path.exists():return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        offset=raw.rfind(b'\n')+1
        path.with_name(path.name+f'.partial-{time.time_ns()}').write_bytes(raw[offset:])
        with path.open('r+b') as f:f.truncate(offset)
        raw=raw[:offset]
    return [json.loads(l) for l in raw.splitlines()]
def schedule():
    seen,_,files,_=exclusions()
    for p in sorted((ROOT/'data').glob('**/accepted.jsonl')):
        raw=p.read_bytes();files[str(p.relative_to(ROOT))]=sha(raw)
        for l in raw.splitlines():
            r=json.loads(l)
            if r.get('instruction'):seen.add(norm(r['instruction']))
    tasks=[];counts=collections.Counter()
    for i in range(max(TARGET.values())*3):
        for split,targets in [('validation',HOLDOUT),('train',TARGET)]:
            for category,target in targets.items():
                if i>=target*3:continue
                r=make_task(category,i,split);key=norm(r['instruction'])
                if key in seen:continue
                seen.add(key);tasks.append(r);counts[(split,category)]+=1
    for split,targets in [('validation',HOLDOUT),('train',TARGET)]:
        for c,n in targets.items():assert counts[(split,c)]>=n*1.5,(split,c,counts[(split,c)])
    OUT.mkdir(exist_ok=False)
    write(OUT/'tasks.json',tasks)
    scripts={}
    for name in ['prepare_fresh_10000.py','fresh10000_tasks.py','reasoning_data.py','code_sandbox.py','code_sandbox_child.py','verified_code_tasks.py']:
        raw=(ROOT/'scripts'/name).read_bytes();(OUT/name).write_bytes(raw);scripts[name]=sha(raw)
    write(OUT/'config.json',dict(target=TARGET,holdout=HOLDOUT,model=MODEL,endpoint=URL,max_length=2048,concurrency=1,seed=20260922,tasks_sha256=sha((OUT/'tasks.json').read_bytes()),scripts=scripts,excluded_files=files,verification='Final numeric/JSON answers checked; final code passes 30 sandbox fixtures. Reasoning and prose explanations not verified. Finite synthetic families; not a broad real-world corpus.',split_policy='math and structured-data families disjoint; code ordered operation triples disjoint, shared primitive operations; heldout never training',training_started=False))
    return tasks

def run(limit=None):
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads((OUT/'tasks.json').read_text()) if OUT.exists() else schedule()
    cfg=json.loads((OUT/'config.json').read_text())
    assert sha((OUT/'tasks.json').read_bytes())==cfg['tasks_sha256']
    for name,h in cfg['scripts'].items():assert sha((ROOT/'scripts'/name).read_bytes())==h,('collector code changed',name)
    if (OUT/'manifest.json').exists():print('Набор уже завершён.',flush=True);return
    with urllib.request.urlopen(URL+'/models',timeout=30) as f:models=json.load(f)
    assert MODEL in {r['id'] for r in models['data']},'Teacher model mismatch'
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    commits=load(OUT/'committed.jsonl');events=load(OUT/'events.jsonl');done={r['row']['id'] for r in commits};attempted={r['id'] for r in events}|done
    assert len(done)==len(commits)
    counts=collections.Counter((x['row']['split'],x['row']['category']) for x in commits)
    nstart=len(commits);started=time.monotonic();calls=0;tokens=0;api_seconds=0
    def status(state='collecting',**extra):
        elapsed=time.monotonic()-started
        write(OUT/'status.json',dict(state=state,accepted=sum(counts.values()),train=sum(counts[('train',c)] for c in TARGET),validation=sum(counts[('validation',c)] for c in TARGET),target_train=10000,target_validation=500,by_split={s:{c:counts[(s,c)] for c in TARGET} for s in ('train','validation')},attempted=len(attempted),segment_seconds=elapsed,segment_accepted=sum(counts.values())-nstart,segment_completion_tokens=tokens,segment_api_seconds=api_seconds,completion_tokens_per_api_second=tokens/api_seconds if api_seconds else None,updated=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra))
    status()
    for task in tasks:
        key=(task['split'],task['category']);quota=(TARGET if task['split']=='train' else HOLDOUT)[task['category']]
        if task['id'] in attempted or counts[key]>=quota:continue
        if limit is not None and calls>=limit:status('paused_after_pilot');return
        body=dict(model=MODEL,messages=[dict(role='user',content=task['instruction'])],temperature=0,max_tokens=2048,reasoning_effort='medium',chat_template_kwargs={'enable_thinking':True})
        req=urllib.request.Request(URL+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        start=time.monotonic()
        with urllib.request.urlopen(req,timeout=240) as f:raw=json.load(f)
        seconds=time.monotonic()-start;api_seconds+=seconds;tokens+=raw.get('usage',{}).get('completion_tokens',0);calls+=1
        append(OUT/'requests.jsonl',dict(id=task['id'],request=body,response=raw,seconds=seconds))
        try:
            choice=raw['choices'][0];msg=choice['message']
            if choice['finish_reason']!='stop':raise ValueError('unfinished_response')
            row=dict(task,reasoning=msg.get('reasoning_content') or '',response=msg.get('content') or '')
            if not verify(task,row['response']):raise ValueError('final_verification_failed')
            row['verification']='final_'+task['validator']+'_checked_reasoning_unverified'
            encoding=dict(row,verification='teacher_answer_and_reasoning_unverified')
            item=encode(encoding,tok,2048,allow_validation=True,allow_teacher_only=True)
            if item is None:raise ValueError('sequence_over_2048_not_truncated')
            item.update(split=row['split'],category=row['category'],topic=row['topic'],language=row['language'],verification=row['verification'])
            append(OUT/'committed.jsonl',dict(row=row,encoded=item));counts[key]+=1
            event=dict(id=task['id'],accepted=True)
        except (ValueError,ZeroDivisionError) as e:event=dict(id=task['id'],accepted=False,reason=str(e))
        append(OUT/'events.jsonl',event);attempted.add(task['id']);status()
        if event['accepted'] and sum(counts.values())%25==0:print(f"Принято {sum(counts.values())}/10500; обучение {sum(counts[('train',c)] for c in TARGET)}/10000; контроль {sum(counts[('validation',c)] for c in TARGET)}/500",flush=True)
        time.sleep(1)
    if any(counts[(s,c)]!=n for s,q in [('train',TARGET),('validation',HOLDOUT)] for c,n in q.items()):
        status('insufficient_candidates');raise SystemExit(78)
    commits=load(OUT/'committed.jsonl');random.Random(20260922).shuffle(commits);files={}
    assert len({x['row']['sha256'] for x in commits})==10500
    for split in ('train','validation'):
        rows=[x['encoded'] for x in commits if x['row']['split']==split]
        raw=''.join(json.dumps(r)+'\n' for r in rows).encode();(OUT/f'{split}.jsonl').write_bytes(raw)
        files[split]=dict(count=len(rows),sha256=sha(raw),tokens=sum(len(r['input_ids']) for r in rows))
    raw=''.join(json.dumps(x['row'],ensure_ascii=False)+'\n' for x in commits).encode();(OUT/'accepted.jsonl').write_bytes(raw)
    write(OUT/'manifest.json',dict(state='completed',files=files,max_length=2048,truncated=False,training_started=False,config_sha256=sha((OUT/'config.json').read_bytes()),accepted_sha256=sha(raw),verification=cfg['verification'],split_policy=cfg['split_policy']))
    status('completed');print('Готово: 10000 обучающих и 500 контрольных примеров. Обучение не запускалось.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args()
    try:run(a.limit)
    except Exception as e:
        if OUT.exists():
            old=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
            write(OUT/'status.json',dict(old,state='failed',error=str(e)))
        raise
