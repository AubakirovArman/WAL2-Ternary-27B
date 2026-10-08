"""One user-authorized 100-concurrent arm on the frozen sweep tasks."""
import concurrent.futures as cf
import fcntl,json,statistics,threading,time,urllib.request
from pathlib import Path
import benchmark_teacher_concurrency as bench
from fresh10000_tasks import verify

def main():
    root=bench.ROOT;out=root/'reports/teacher-concurrency-100-20260925'
    lock=(root/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(exist_ok=False);bench.OUT=out
    tasks=json.loads((root/'reports/teacher-concurrency-to50-20260925/tasks.json').read_text());assert len(tasks)==100
    bench.write('tasks.json',tasks)
    bench.write('protocol.json',dict(concurrency=100,max_tokens=8192,model=bench.MODEL,thinking='medium',temperature=0,scope='Same100tasks as10..50sweep; single arm, existing API only. Cache may be warm. No server changes.'))
    with urllib.request.urlopen('http://127.0.0.1:8000/workers',timeout=15) as f:bench.write('workers-before.json',json.load(f))
    stop=threading.Event();active=[100];monitor=threading.Thread(target=bench.monitor,args=(stop,active),daemon=True);monitor.start()
    def status(state,n,**extra):bench.write('status.json',dict(state=state,completed=n,total=100,**extra))
    rows=[];status('running',0);start=time.monotonic()
    try:
        with cf.ThreadPoolExecutor(max_workers=100) as pool:
            for future in cf.as_completed([pool.submit(bench.call,t,100,8192) for t in tasks]):
                row=future.result();rows.append(row);bench.append('responses.jsonl',row);status('running',len(rows))
                print(f'Готово {len(rows)}/100',flush=True)
        wall=time.monotonic()-start;active[0]=0;vstart=time.monotonic();lookup={t['id']:t for t in tasks};checks=[]
        for r in rows:
            ch=r.get('response',{}).get('choices',[{}])[0];message=ch.get('message',{});complete=ch.get('finish_reason')=='stop' and bool(message.get('content'))
            try:valid=bool(complete and verify(lookup[r['id']],message.get('content','')))
            except Exception:valid=False
            checks.append(dict(id=r['id'],complete=complete,verified=valid,finish=ch.get('finish_reason')))
        verification=time.monotonic()-vstart;tokens=sum(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in rows)
        result=dict(concurrency=100,max_tokens=8192,requests=100,wall_seconds=wall,completion_tokens=tokens,tokens_per_second=tokens/wall,mean_request_seconds=statistics.mean(r['seconds'] for r in rows),median_request_seconds=statistics.median(r['seconds'] for r in rows),p95_request_seconds=sorted(r['seconds'] for r in rows)[94],completed=sum(c['complete'] for c in checks),verified=sum(c['verified'] for c in checks),errors=sum('error' in r for r in rows),truncated=sum(c['finish']=='length' for c in checks),verification_seconds=verification,checks=checks)
        result['hours_for_25000_verified_same_mix']=25000*(wall+verification)/max(result['verified'],1)/3600
        bench.write('result.json',result);status('completed',100,result={k:v for k,v in result.items() if k!='checks'})
        print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False),flush=True)
    except Exception as e:status('failed',len(rows),error=str(e));raise
    finally:stop.set();monitor.join(timeout=12)
if __name__=='__main__':main()
