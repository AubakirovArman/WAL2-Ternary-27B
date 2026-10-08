"""100 measured responses: same 25 tasks at concurrency 1/2/5/10. No server changes."""
import concurrent.futures as cf
import argparse,fcntl,json,os,statistics,subprocess,threading,time,urllib.request
from pathlib import Path
from fresh10000_tasks import make_task,verify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/teacher-concurrency-20260925'
MODEL='Qwen/Qwen3.8-27B-FP8'
def write(name,obj):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def append(name,obj):
    with (OUT/name).open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush()
def call(task,c,max_tokens):
    body=dict(model=MODEL,messages=[dict(role='user',content=task['instruction'])],temperature=0,max_tokens=max_tokens,reasoning_effort='medium',chat_template_kwargs={'enable_thinking':True})
    start=time.monotonic()
    try:
        req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=300) as f:raw=json.load(f)
        return dict(id=task['id'],concurrency=c,max_tokens=max_tokens,seconds=time.monotonic()-start,response=raw)
    except Exception as e:return dict(id=task['id'],concurrency=c,max_tokens=max_tokens,seconds=time.monotonic()-start,error=str(e))
def monitor(stop,active):
    while not stop.is_set():
        try:
            r=subprocess.run(['nvidia-smi','--id=3,4,5','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            append('gpu.jsonl',dict(time=time.time(),concurrency=active[0],csv=r.stdout.strip()))
        except Exception as e:append('gpu.jsonl',dict(error=str(e)))
        stop.wait(5)
def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--extra20',action='store_true');ap.add_argument('--bulk100',action='store_true');ap.add_argument('--sweep50',action='store_true');ap.add_argument('--max-tokens',type=int,default=2048,choices=[2048,4096,8192]);args=ap.parse_args()
    assert sum([args.extra20,args.bulk100,args.sweep50])<=1
    if args.bulk100:OUT=OUT/'bulk100'
    if args.sweep50:OUT=ROOT/'reports/teacher-concurrency-to50-20260925'
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=args.extra20)
    tasks=[make_task(cat,75000+i,'train') for cat,n in [('math',8),('code',6),('instructions',6),('grounded',5)] for i in range(n)]
    if args.bulk100:tasks=[make_task(cat,85000+i,'train') for cat,n in [('math',32),('code',24),('instructions',24),('grounded',20)] for i in range(n)]
    if args.sweep50:tasks=[make_task(cat,95000+i,'train') for cat,n in [('math',32),('code',24),('instructions',24),('grounded',20)] for i in range(n)]
    # Interleave categories; identical order and prompts in each arm.
    tasks.sort(key=lambda r:(int(r['id'].split('-')[-1]),r['category']))
    if args.extra20:
        assert json.loads((OUT/'tasks.json').read_text())==json.loads(json.dumps(tasks))
        assert json.loads((OUT/'status.json').read_text())['state']=='completed'
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(concurrencies=[1,2,5,10,20],total_responses=sum(r['requests'] for r in json.loads((OUT/'results.json').read_text()))+25,extension_reason='Additional concurrency20 and user-requested output budgets4096/8192.');protocol.setdefault('extensions',[]).append(dict(concurrency=20,max_tokens=args.max_tokens));write('protocol.json',protocol)
    else:
        write('tasks.json',tasks)
        write('protocol.json',dict(model=MODEL,endpoint='http://127.0.0.1:8000/v1',concurrencies=[1,2,5,10],tasks_per_arm=25,total_responses=100,max_tokens=2048,thinking='medium',temperature=0,scope='Same 25 fresh synthetic prompts repeated four times. Old task families; not a quality pilot for the broader E018 dataset. Later arms can benefit from server prefix caching. No server configuration changes. No training.'))
        if args.bulk100:write('protocol.json',dict(model=MODEL,concurrencies=[20],tasks_per_arm=100,total_responses=100,max_tokens=args.max_tokens,thinking='medium',temperature=0,scope='100 previously unused synthetic prompts; rolling concurrency20; includes end-of-run drain. Old task families, not representative proof for broader E018. No model/server changes.'))
        if args.sweep50:write('protocol.json',dict(model=MODEL,concurrencies=[10,20,30,40,50],execution_order=[30,10,50,20,40],tasks_per_arm=100,total_responses=500,max_tokens=args.max_tokens,thinking='medium',temperature=0,scope='Identical 100 prompts, rolling queue; fixed non-monotonic concurrency order. Prefix caching is not disabled; later arms may benefit. Wall time includes drain. Existing API only, no server changes. Old synthetic task families, not broad E018 quality evaluation.'))
    active=[0];stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,active),daemon=True);thread.start()
    results=json.loads((OUT/'results.json').read_text()) if args.extra20 else []
    if args.extra20:assert not any(r['concurrency']==20 and r.get('max_tokens',2048)==args.max_tokens for r in results)
    try:
        for c in ([30,10,50,20,40] if args.sweep50 else [20] if args.extra20 or args.bulk100 else [1,2,5,10]):
            active[0]=c;rows=[];start=time.monotonic()
            write('status.json',dict(state='running',concurrency=c,completed_arm=0,completed_total=sum(r['requests'] for r in results),results=results))
            with cf.ThreadPoolExecutor(max_workers=c) as pool:
                for f in cf.as_completed([pool.submit(call,t,c,args.max_tokens) for t in tasks]):
                    r=f.result();append('responses.jsonl',r);rows.append(r)
                    write('status.json',dict(state='running',concurrency=c,completed_arm=len(rows),completed_total=sum(x['requests'] for x in results)+len(rows),results=results))
                    print(f'Параллельно {c}: готово {len(rows)}/{len(tasks)}',flush=True)
            wall=time.monotonic()-start;active[0]=0
            tokens=sum(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in rows)
            checked=[];vstart=time.monotonic();lookup={t['id']:t for t in tasks}
            for r in rows:
                choice=r.get('response',{}).get('choices',[{}])[0];message=choice.get('message',{});complete=choice.get('finish_reason')=='stop' and bool(message.get('content'))
                try:valid=bool(complete and verify(lookup[r['id']],message.get('content','')))
                except Exception:valid=False
                checked.append(dict(id=r['id'],complete=complete,verified=valid,finish=choice.get('finish_reason')))
            verification_seconds=time.monotonic()-vstart
            summary=dict(concurrency=c,max_tokens=args.max_tokens,requests=len(rows),wall_seconds=wall,completion_tokens=tokens,tokens_per_second=tokens/wall,mean_request_seconds=statistics.mean(r['seconds'] for r in rows),median_request_seconds=statistics.median(r['seconds'] for r in rows),completed=sum(x['complete'] for x in checked),verified=sum(x['verified'] for x in checked),errors=sum('error' in r for r in rows),verification_seconds=verification_seconds,checks=checked)
            summary['p95_request_seconds']=sorted(r['seconds'] for r in rows)[int(.95*(len(rows)-1))]
            summary['hours_for_25000_verified_same_mix']=25000*(wall+verification_seconds)/max(summary['verified'],1)/3600
            results.append(summary);write('results.json',results);print(json.dumps({k:v for k,v in summary.items() if k!='checks'},ensure_ascii=False),flush=True)
        write('status.json',dict(state='completed',completed_total=sum(r['requests'] for r in results),results=results))
    finally:stop.set();thread.join(timeout=12)
if __name__=='__main__':main()
