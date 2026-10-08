"""Collect a frozen4096+256pilot; certified calculation facts, repair contexts.

Resumable append-only commit log; original benchmarks never enter training.
"""
import argparse
import collections
import concurrent.futures as cf
import fcntl
import hashlib
import json
import multiprocessing
import os
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from e019_math_tasks import make_math
from e018_tasks import verify_task, repeat_reason
from code_sandbox import run

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e019-reasoning-pilot-v1'
MODEL='Qwen/Qwen3.8-27B-FP8'
CONTROL=('<think>','</think>','<|im_start|>','<|im_end|>','<|endoftext|>')


def write(name,value):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)


def append(name,value):
    with (OUT/name).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def load(name):
    p=OUT/name
    if not p.exists():return []
    raw=p.read_bytes()
    if raw and not raw.endswith(b'\n'):
        cut=raw.rfind(b'\n')+1;(OUT/(name+'.partial-'+str(time.time_ns()))).write_bytes(raw[cut:]);p.write_bytes(raw[:cut]);raw=raw[:cut]
    return [json.loads(l) for l in raw.splitlines()]


def request(task):
    mathematical=task['category'] in ('math','math_repair')
    if mathematical:
        prompt='Write a concise, correct explanation for each already checked mathematical fact below. These facts are exact references, NOT guesses. Preserve their order and do not introduce new numerical claims or formulas outside the supplied facts. Return ONLY JSON with keys steps and final: steps is an array of {id,explanation}, exactly one entry per fact in the supplied order; each explanation is one short natural-language sentence of ten to three hundred characters, without digits, equations or control tokens. final must be EXACTLY the supplied final-answer string, using a/b for a fraction, never LaTeX. Explain why each step is useful, in English. Do not print the fact again in the explanation: the verified fact is inserted automatically. If an unfinished student draft is present, recompute from the checked facts; never copy its defects.\nPROBLEM:\n'+task['instruction']+'\nEXACT CALCULATION CERTIFICATE:\n'+json.dumps(task['facts'],ensure_ascii=False)+'\nEXACT FINAL STRING:\n'+task['gold']
        thinking=False
    else:
        prompt=task['instruction']+'\nKeep the reasoning finite, concise and check the result. Close the reasoning and produce the required final answer.';thinking=True
    body=dict(model=MODEL,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=8192,reasoning_effort='medium',chat_template_kwargs={'enable_thinking':thinking})
    start=time.monotonic();errors=[]
    for attempt in range(3):
        try:
            req=urllib.request.Request(os.environ.get('VOL2_TEACHER_URL','http://127.0.0.1:8000/v1/chat/completions'),data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=480) as f:raw=json.load(f)
            return dict(raw=raw,seconds=time.monotonic()-start,retries=errors)
        except Exception as e:
            errors.append(str(e))
            if attempt<2:time.sleep(2**(attempt+1))
    return dict(error=errors[-1],seconds=time.monotonic()-start,retries=errors)


def verified_function(task,answer):
    match=re.fullmatch(r'\s*```(?:python|py)\s*\n(.*?)```\s*',answer,re.S)
    if not match:return False
    program='import copy,json\nns={}\nexec('+repr(match[1])+',ns)\nf=ns["solve"]\n'
    for case in task['fixtures']:
        program+=f"x={case['input']!r}\ny=copy.deepcopy(x)\nassert json.dumps(f(x),sort_keys=True)==json.dumps({case['expected']!r},sort_keys=True)\nassert x==y\n"
    return run(program)['passed']


def validate(task,result):
    if 'error' in result:return dict(accepted=False,reason='api_error')
    try:
        choice=result['raw']['choices'][0]
        if choice['finish_reason']!='stop':return dict(accepted=False,reason='unfinished')
        msg=choice['message'];answer=msg.get('content') or ''
        if task['category'] in ('math','math_repair'):
            reference=make_math(task['family_index'],task['serial'],task['split'],task['repair'])
            if task['facts']!=reference['facts'] or task['gold']!=reference['gold'] or task['base_instruction']!=reference['base_instruction']:raise ValueError('Calculation certificate changed')
            text=answer.strip()
            if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
            obj=json.loads(text)
            if set(obj)!= {'steps','final'} or not isinstance(obj['steps'],list):raise ValueError('Invalid math schema')
            if obj['final']!=task['gold']:return dict(accepted=False,reason='wrong_final_string')
            if [s.get('id') for s in obj['steps']]!=[f['id'] for f in task['facts']]:return dict(accepted=False,reason='missing_or_reordered_certificate_steps')
            reasoning=[]
            if task['repair']:reasoning.append('I will recompute from the definitions and a finite checked derivation, without relying on the unfinished draft.')
            for step,fact in zip(obj['steps'],task['facts']):
                explanation=step.get('explanation')
                if set(step)!={'id','explanation'} or not isinstance(explanation,str) or not 10<=len(explanation)<=300:raise ValueError('Invalid explanation')
                if any(c in explanation for c in CONTROL) or re.search(r'[0-9=<>*^\\]',explanation):return dict(accepted=False,reason='new_formula_outside_certificate')
                reasoning.append(explanation.strip()+'\nChecked fact: '+fact['claim'])
            reason='\n\n'.join(reasoning);answer='\\boxed{'+task['gold']+'}'
            verification='exact_answer_and_listed_calculation_certificate_checked_teacher_prose_not_formally_verified'
        else:
            reason=msg.get('reasoning_content') or ''
            if not reason.strip() or not answer.strip():return dict(accepted=False,reason='empty_reasoning_or_answer')
            if repeat_reason(reason+'\n'+answer):return dict(accepted=False,reason='repetition')
            passed=verified_function(task,answer) if task['category']=='code' else verify_task(task,answer)
            if not passed:return dict(accepted=False,reason='final_verification_failed')
            verification='final_'+task['category']+'_checked_reasoning_unverified'
        if any(marker in reason or marker in answer for marker in CONTROL):return dict(accepted=False,reason='unexpected_control_delimiter')
        return dict(accepted=True,reasoning=reason,response=answer,verification=verification)
    except Exception as e:return dict(accepted=False,reason='verification_exception',detail=str(e)[:300])


def key(task):return (task['split'],task['category'],task.get('family_index') if task['category'] in ('math','math_repair') else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',type=int,default=0);args=parser.parse_args()
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((OUT/'config.json').read_text());raw=(OUT/'tasks.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==cfg['tasks_sha256']
    for name,h in cfg['task_code'].items():assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==h,('Task code changed',name)
    if (OUT/'manifest.json').exists():print('Набор уже завершён.',flush=True);return
    tasks=[json.loads(l) for l in raw.splitlines()];byid={t['id']:t for t in tasks};del raw
    commits=load('committed.jsonl');events=load('events.jsonl');done={r['row']['id'] for r in commits};assert len(done)==len(commits)
    attempted=done|{r['id'] for r in events};counts=collections.Counter(key(r['row']) for r in commits);rejected=collections.Counter(r['reason'] for r in events if not r['accepted'])
    quota={}
    for f in range(16):quota['train','math',f]=112;quota['train','math_repair',f]=16
    for f in range(16,20):quota['validation','math',f]=32
    for split,qs in [('train',cfg['target']),('validation',cfg['holdout'])]:
        for cat,n in qs.items():
            if cat not in ('math','math_repair'):quota[split,cat,None]=n
    queues={k:collections.deque(t for t in tasks if key(t)==k and t['id'] not in attempted) for k in quota};prefixes={}
    pending=collections.Counter();network={};validating={};start=time.monotonic();initial=len(commits);tokens=0;submitted=0;keys=list(quota);cursor=0
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    encoder=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/e018_encode_server.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
    assert json.loads(encoder.stdout.readline()).get('ready')
    def update(state='collecting',**extra):
        train=sum(v for (s,c,f),v in counts.items() if s=='train');val=sum(v for (s,c,f),v in counts.items() if s=='validation');elapsed=time.monotonic()-start;added=train+val-initial
        write('status.json',dict(state=state,train=train,target_train=4096,validation=val,target_validation=256,counts={s:{c:sum(v for (ss,cc,ff),v in counts.items() if ss==s and cc==c) for c in set(cfg['target'])|set(cfg['holdout'])} for s in ('train','validation')},math_family_counts={str(k):v for k,v in counts.items() if k[1] in ('math','math_repair')},rejected=dict(rejected),attempted=len(attempted),network_active=len(network),validation_active=len(validating),concurrency=50,updated=time.time(),segment_seconds=elapsed,segment_accepted=added,segment_completion_tokens=tokens,tokens_per_second=tokens/max(elapsed,1),eta_seconds=(4352-train-val)*elapsed/added if added else None,**extra))
    def refresh_prefixes():
        path=OUT/'student/prefixes.jsonl'
        raw=path.read_bytes() if path.exists() else b''
        # The student process is the writer; never truncate its in-flight record.
        if raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
        for r in (json.loads(line) for line in raw.splitlines()):
            assert r['id'] in byid and byid[r['id']]['repair'] and byid[r['id']]['split']=='train'
            prefixes[r['id']]=r
    def available(k):
        q=queues[k]
        if k[1]!='math_repair':return q.popleft() if q else None
        for _ in range(len(q)):
            t=q.popleft()
            if t['id'] in prefixes:
                t=dict(t);r=prefixes[t['id']];draft=r['draft']
                for marker in CONTROL:draft=draft.replace(marker,'['+marker.strip('<>|')+']')
                t['instruction']=t['base_instruction']+'\n\nA low-bit student produced the following possibly incorrect or unfinished draft. Recompute a correct finite solution from the definitions. The draft is context, not a reference.\nBEGIN STUDENT DRAFT\n'+draft+'\nEND STUDENT DRAFT'
                t['sha256']=hashlib.sha256(t['instruction'].encode()).hexdigest();t['student_prefix_tokens']=len(r['prefix_tokens']);t['student_prefix_sha256']=hashlib.sha256(json.dumps(r['prefix_tokens']).encode()).hexdigest();return t
            q.append(t)
        return None
    update();last_prefix_refresh=0;api_errors=0
    try:
        with cf.ThreadPoolExecutor(max_workers=50) as net,cf.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as validators:
            while True:
                if time.monotonic()-last_prefix_refresh>5:refresh_prefixes();last_prefix_refresh=time.monotonic()
                while len(network)<50 and len(validating)<100 and (not args.pilot or submitted<args.pilot):
                    task=None
                    for _ in keys:
                        k=keys[cursor%len(keys)];cursor+=1
                        if counts[k]+pending[k]<quota[k]:task=available(k)
                        if task is not None:break
                    if task is None:break
                    pending[key(task)]+=1;network[net.submit(request,task)]=task;submitted+=1
                if not network and not validating:
                    if args.pilot or all(counts[k]>=v for k,v in quota.items()):break
                    missing=[k for k,v in quota.items() if counts[k]<v]
                    ps=OUT/'student/status.json';student=json.loads(ps.read_text()) if ps.exists() else {}
                    if all(k[1]=='math_repair' for k in missing) and student.get('state') not in ('completed','failed'):
                        update('waiting_student_prefixes');time.sleep(5);continue
                    break
                ready,_=cf.wait(list(network)+list(validating),timeout=5,return_when=cf.FIRST_COMPLETED)
                for future in ready:
                    if future in network:
                        task=network.pop(future);result=future.result();tokens+=result.get('raw',{}).get('usage',{}).get('completion_tokens',0);api_errors=api_errors+1 if 'error' in result else 0
                        if api_errors>=10:raise RuntimeError('10consecutive API errors after retries')
                        append('requests.jsonl',dict(id=task['id'],result=result));validating[validators.submit(validate,task,result)]=(task,result)
                    else:
                        task,result=validating.pop(future);verdict=future.result();pending[key(task)]-=1;attempted.add(task['id'])
                        if verdict['accepted']:
                            row=dict(task,reasoning=verdict['reasoning'],response=verdict['response'],verification=verdict['verification'])
                            encoder.stdin.write(json.dumps(row)+'\n');encoder.stdin.flush();encoded=json.loads(encoder.stdout.readline())
                            if encoded.get('item') is None:verdict=dict(accepted=False,reason='encoding_or_length',detail=encoded.get('error'))
                            else:
                                item=encoded['item'];item.update(split=row['split'],category=row['category'],topic=row.get('family',row.get('topic',row['category'])),verification=row['verification'],repair=row['repair']);commit=dict(row=row,encoded=item);append('committed.jsonl',commit);commits.append(commit);counts[key(row)]+=1
                        event=dict(id=task['id'],accepted=verdict['accepted'],reason=verdict.get('reason','accepted'),detail=verdict.get('detail'),category=task['category'],split=task['split'],time=time.time());append('events.jsonl',event)
                        if not verdict['accepted']:rejected[event['reason']]+=1
                update()
                if len(commits) and len(commits)%32==0:print(f"Подготовлено {len(commits)}/4352; скорость {tokens/max(time.monotonic()-start,1):.0f} ток/с",flush=True)
        if args.pilot:update('pilot_completed');return
        missing={str(k):v-counts[k] for k,v in quota.items() if counts[k]<v}
        if missing:update('insufficient_candidates',missing=missing);raise SystemExit(78)
        random.Random(1904096).shuffle(commits);files={}
        for split in ('train','validation'):
            h=hashlib.sha256();n=0;ntokens=0
            with (OUT/(split+'.jsonl')).open('wb') as stream:
                for commit in commits:
                    if commit['row']['split']!=split:continue
                    data=(json.dumps(commit['encoded'])+'\n').encode();stream.write(data);h.update(data);n+=1;ntokens+=len(commit['encoded']['input_ids'])
            files[split]=dict(count=n,tokens=ntokens,sha256=h.hexdigest())
        with (OUT/'accepted.jsonl').open('w') as stream:
            for commit in commits:stream.write(json.dumps(commit['row'],ensure_ascii=False)+'\n')
        write('manifest.json',dict(state='completed',files=files,max_length=12288,max_output_tokens=8192,truncated=False,training_started=False,verification=cfg['verification'],split_policy=cfg['split_policy'],repair_scope=cfg['repair_scope'],config_sha256=hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest()))
        update('completed');print('Готово4096обучающих+256контрольных. Обучение ещё не запускалось.',flush=True)
    except Exception as e:update('failed',error=str(e));raise
    finally:encoder.stdin.close();encoder.wait(timeout=30)


if __name__=='__main__':main()
