"""Resumable50-request teacher collection; verified accepted rows, no training."""
import argparse,collections,concurrent.futures as cf,fcntl,hashlib,json,multiprocessing,os,random,subprocess,time,urllib.request
from pathlib import Path
from e018_tasks import TARGET,HOLDOUT,verify_task,repeat_reason
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/e018-25000';MODEL='Qwen/Qwen3.8-27B-FP8'
def write(name,x):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def append(name,x):
    with (OUT/name).open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def load(name):
    p=OUT/name
    if not p.exists():return []
    raw=p.read_bytes()
    if raw and not raw.endswith(b'\n'):
        cut=raw.rfind(b'\n')+1;(OUT/(name+f'.partial-{time.time_ns()}')).write_bytes(raw[cut:])
        with p.open('r+b') as f:f.truncate(cut)
        raw=raw[:cut]
    return [json.loads(l) for l in raw.splitlines()]
def request(t):
    body=dict(model=MODEL,messages=[dict(role='user',content=t['instruction'])],temperature=0,max_tokens=8192,reasoning_effort='medium',chat_template_kwargs={'enable_thinking':True})
    start=time.monotonic();errors=[]
    for attempt in range(3):
        try:
            req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=360) as f:raw=json.load(f)
            return dict(raw=raw,seconds=time.monotonic()-start,retries=errors)
        except Exception as e:
            errors.append(str(e))
            if attempt<2:time.sleep(2**(attempt+1))
    return dict(error=errors[-1],retries=errors,seconds=time.monotonic()-start)
def validate(t,res):
    if 'error' in res:return dict(accepted=False,reason='api_error')
    try:
        ch=res['raw']['choices'][0];msg=ch['message']
        if ch['finish_reason']!='stop':return dict(accepted=False,reason='unfinished')
        reason=msg.get('reasoning_content') or '';answer=msg.get('content') or ''
        if not reason.strip() or not answer.strip():return dict(accepted=False,reason='empty_reasoning_or_answer')
        if repeat_reason(reason+'\n'+answer):return dict(accepted=False,reason='repetition')
        if not verify_task(t,answer):return dict(accepted=False,reason='verification_failed')
        return dict(accepted=True,reasoning=reason,response=answer)
    except Exception as e:return dict(accepted=False,reason='verification_exception',detail=str(e)[:250])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=int,default=0);a=ap.parse_args()
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((OUT/'config.json').read_text());raw=(OUT/'tasks.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==cfg['tasks_sha256']
    for name,h in cfg.get('collector_code',{}).items():assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==h,('Code changed',name)
    if (OUT/'manifest.json').exists():print('Набор уже завершён.',flush=True);return
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=15) as f:assert MODEL in [x['id'] for x in json.load(f)['data']]
    tasks=[json.loads(l) for l in raw.splitlines()];del raw
    commits=load('committed.jsonl');events=load('events.jsonl');done={x['row']['id'] for x in commits};assert len(done)==len(commits)
    attempted=done|{x['id'] for x in events};counts=collections.Counter((x['row']['split'],x['row']['category']) for x in commits)
    rejected=collections.Counter(x['reason'] for x in events if not x['accepted']);start=time.monotonic();initial=len(commits);tokens=0;submitted=0;pending_counts=collections.Counter();network={};validating={};quota={(s,c):n for s,d in [('train',TARGET),('validation',HOLDOUT)] for c,n in d.items()}
    queues={k:collections.deque(t for t in tasks if (t['split'],t['category'])==k and t['id'] not in attempted) for k in quota};del tasks;last_accepted=time.time();last_event=time.time()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    encoder=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/e018_encode_server.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
    assert json.loads(encoder.stdout.readline()).get('ready')
    def status(state='collecting',**extra):
        train=sum(counts['train',c] for c in TARGET);val=sum(counts['validation',c] for c in TARGET);elapsed=time.monotonic()-start
        write('status.json',dict(state=state,train=train,target_train=25000,validation=val,target_validation=1500,counts={s:{c:counts[s,c] for c in TARGET} for s in ['train','validation']},rejected=rejected,attempted=len(attempted),network_active=len(network),validation_active=len(validating),concurrency=50,updated=time.time(),last_accepted=last_accepted,last_event=last_event,segment_seconds=elapsed,segment_accepted=train+val-initial,segment_completion_tokens=tokens,tokens_per_second=tokens/max(elapsed,1),**extra))
    status();keys=list(quota);cursor=0;consecutive_api_errors=0
    try:
        with cf.ThreadPoolExecutor(max_workers=50) as net,cf.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as validators:
            while True:
                # At most50 API requests; bounded validation backlog and no quota overshoot.
                while len(network)<50 and len(validating)<100 and (not a.pilot or submitted<a.pilot):
                    pick=None
                    for _ in keys:
                        key=keys[cursor%len(keys)];cursor+=1
                        if queues[key] and counts[key]+pending_counts[key]<quota[key]:pick=queues[key].popleft();break
                    if pick is None:break
                    key=(pick['split'],pick['category']);pending_counts[key]+=1;network[net.submit(request,pick)]=pick;submitted+=1
                if not network and not validating:break
                ready,_=cf.wait(list(network)+list(validating),timeout=5,return_when=cf.FIRST_COMPLETED)
                for future in ready:
                    if future in network:
                        t=network.pop(future);res=future.result();tokens+=res.get('raw',{}).get('usage',{}).get('completion_tokens',0)
                        consecutive_api_errors=consecutive_api_errors+1 if 'error' in res else 0
                        if consecutive_api_errors>=10:
                            status('failed',error='10 consecutive API failures after retries; automatic restart will retry uncommitted tasks')
                            raise RuntimeError('Teacher API unavailable')
                        append('requests.jsonl',dict(id=t['id'],result=res));validating[validators.submit(validate,t,res)]=(t,res);last_event=time.time()
                    else:
                        t,res=validating.pop(future);verdict=future.result();key=(t['split'],t['category']);pending_counts[key]-=1;attempted.add(t['id'])
                        if verdict['accepted']:
                            row=dict(t,reasoning=verdict['reasoning'],response=verdict['response'],verification='final_'+t['validator']+'_checked_reasoning_unverified')
                            encoder.stdin.write(json.dumps(row)+'\n');encoder.stdin.flush();encoded=json.loads(encoder.stdout.readline())
                            if encoded.get('item') is None:verdict=dict(accepted=False,reason='encoding_or_length',detail=encoded.get('error','sequence_over12288'))
                            else:
                                item=encoded['item'];item.update(split=t['split'],category=t['category'],topic=t['topic'],verification=row['verification'])
                                commit=dict(row=row,encoded=item);append('committed.jsonl',commit);commits.append(commit);counts[key]+=1;last_accepted=time.time()
                        event=dict(id=t['id'],accepted=verdict['accepted'],reason=verdict.get('reason','accepted'),category=t['category'],split=t['split'],time=time.time())
                        if not verdict['accepted']:rejected[event['reason']]+=1
                        append('events.jsonl',event);last_event=time.time()
                        if len(attempted)%50==0:print(f"Готово обучение {sum(counts['train',c] for c in TARGET)}/25000; контроль {sum(counts['validation',c] for c in TARGET)}/1500; отклонено {sum(rejected.values())}",flush=True)
                status()
        if a.pilot:status('pilot_completed');return
        if any(counts[k]<v for k,v in quota.items()):status('insufficient_candidates',missing={str(k):v-counts[k] for k,v in quota.items() if counts[k]<v});raise SystemExit(78)
        random.Random(18025000).shuffle(commits);files={}
        for split in ['train','validation']:
            path=OUT/(split+'.jsonl');h=hashlib.sha256();total=0;n=0
            with path.open('wb') as f:
                for x in commits:
                    if x['row']['split']!=split:continue
                    b=(json.dumps(x['encoded'])+'\n').encode();f.write(b);h.update(b);total+=len(x['encoded']['input_ids']);n+=1
            files[split]=dict(count=n,tokens=total,sha256=h.hexdigest())
        with (OUT/'accepted.jsonl').open('w') as f:
            for x in commits:f.write(json.dumps(x['row'],ensure_ascii=False)+'\n')
        write('manifest.json',dict(state='completed',files=files,max_length=12288,max_output_tokens=8192,truncated=False,training_started=False,config_sha256=hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),verification=cfg['verification_scope'],split_policy=cfg['split_policy']))
        status('completed');print('Готово:25000 обучающих и1500 контрольных. Обучение не запускалось.',flush=True)
    except Exception as e:status('failed',error=str(e));raise
    finally:
        encoder.stdin.close();encoder.wait(timeout=30)
if __name__=='__main__':main()
