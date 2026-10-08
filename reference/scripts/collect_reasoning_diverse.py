"""One-request-at-a-time teacher traces on unused training instructions."""
import argparse,fcntl,hashlib,json,random,time,urllib.request
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'data/reasoning-diverse-v1')
    parser.add_argument('--count',type=int,default=160)
    parser.add_argument('--validation-stride',type=int,default=5)
    parser.add_argument('--seed',type=int,default=2026091920)
    parser.add_argument('--id-prefix',default='reasoning-diverse')
    parser.add_argument('--exclude-schedule',type=Path)
    parser.add_argument('--deadline-hours',type=float,default=1.5)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.count<2 or args.validation_stride<2 or args.deadline_hours<=0:raise ValueError('Invalid collection bounds')
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=args.output
    if args.resume:
        if not out.is_dir() or (out/'manifest.json').exists():raise ValueError('Resume requires an unfinished existing collection')
    else:out.mkdir(exist_ok=False)
    cache=json.loads((ROOT/'data/v3-kd/manifest.json').read_text())
    corpus=Path(cache['corpus']).read_bytes()
    if hashlib.sha256(corpus).hexdigest()!=cache['corpus_sha256']:raise ValueError('Corpus changed')
    used=set(map(str,cache['order_ids']));excluded=set()
    if args.exclude_schedule:
        excluded={str(r['source_id']) for r in json.loads(args.exclude_schedule.read_text())}
    used.update(excluded);groups=defaultdict(list)
    for line in corpus.splitlines():
        row=json.loads(line)
        if row['split']=='train' and str(row['id']) not in used:groups[(row['language'],row['topic'])].append(row)
    rng=random.Random(args.seed);keys=sorted(groups);rng.shuffle(keys)
    for values in groups.values():rng.shuffle(values)
    chosen=[]
    while len(chosen)<args.count:
        before=len(chosen)
        for key in keys:
            if groups[key]:chosen.append(groups[key].pop())
            if len(chosen)==args.count:break
        if len(chosen)==before:raise ValueError('Not enough eligible examples')
    tasks=[]
    for i,row in enumerate(chosen):
        tasks.append({'id':f'{args.id_prefix}-{i:04d}','source_id':row['id'],
            'split':'validation' if i%args.validation_stride==args.validation_stride-1 else 'train','instruction':row['instruction'],
            'language':row['language'],'topic':row['topic'],'sha256':hashlib.sha256(row['instruction'].encode()).hexdigest()})
    if len({r['sha256'] for r in tasks})!=len(tasks):raise ValueError('Duplicate instructions')
    if args.resume:
        if json.loads((out/'tasks.json').read_text())!=tasks:raise ValueError('Resume schedule mismatch')
    else:
        (out/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2))
        (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    config={'source_corpus_sha256':cache['corpus_sha256'],'source_cache_excluded_count':len(cache['order_ids']),
            'prior_schedule_excluded_count':len(excluded),'seed':args.seed,'deadline_hours':args.deadline_hours,
            'excluded_schedule_sha256':hashlib.sha256(args.exclude_schedule.read_bytes()).hexdigest() if args.exclude_schedule else None,
            'requested_splits':dict(Counter(r['split'] for r in tasks)),
            'languages':dict(Counter(r['language'] for r in tasks)),'topic_language_groups':len({(r['language'],r['topic']) for r in tasks}),
            'validation_caveat':'New teacher traces on formerly training-designated instructions; not an independent final benchmark',
            'verification':'Teacher answers and reasoning are not independently verified','max_new_tokens':2048}
    done=[];existing=[]
    if args.resume:
        if json.loads((out/'config.json').read_text())!=config:raise ValueError('Resume config mismatch')
        done=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
        existing=[json.loads(x) for x in (out/'accepted.jsonl').read_text().splitlines()]
        if len(done)>len(tasks) or [r['id'] for r in done]!=[r['id'] for r in tasks[:len(done)]]:raise ValueError('Invalid completed prefix')
        if [r['id'] for r in existing]!=[r['id'] for r in done if r.get('accepted')]:raise ValueError('Accepted/event mismatch; manual recovery required')
        (out/'collector-resume.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps({'event':'resumed','completed':len(done),'remaining':len(tasks)-len(done)}),flush=True)
    else:(out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    start=time.monotonic();accepted=Counter(r['split'] for r in existing);failures=0
    for row in tasks[len(done):]:
        if time.monotonic()-start>args.deadline_hours*3600:raise TimeoutError('Collection time bound exceeded')
        body={'model':'Qwen/Qwen3.8-27B-FP8','messages':[{'role':'user','content':row['instruction']}],
              'temperature':0,'max_tokens':2048,'reasoning_effort':'medium','chat_template_kwargs':{'enable_thinking':True}}
        req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=180) as response:raw=json.load(response)
            with (out/'requests.jsonl').open('a') as f:f.write(json.dumps({'id':row['id'],'request':body,'response':raw},ensure_ascii=False)+'\n')
            choice=raw['choices'][0];msg=choice['message'];reason=msg.get('reasoning_content') or '';answer=msg.get('content') or ''
            valid=choice['finish_reason']=='stop' and bool(reason.strip()) and bool(answer.strip())
            valid=valid and not any(s in reason+answer for s in ('<think>','</think>','<|im_start|>','<|im_end|>','<|endoftext|>'))
            event={'id':row['id'],'split':row['split'],'accepted':valid,'finish_reason':choice['finish_reason'],'usage':raw.get('usage')}
            if valid:
                saved={**row,'reasoning':reason,'response':answer,'enable_thinking':True,'source':out.name,
                       'verification':'teacher_answer_and_reasoning_unverified'}
                with (out/'accepted.jsonl').open('a') as f:f.write(json.dumps(saved,ensure_ascii=False)+'\n')
                accepted[row['split']]+=1
            failures=0
        except Exception as error:
            failures+=1;event={'id':row['id'],'accepted':False,'error':str(error)}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        if failures>=3:raise RuntimeError('Three consecutive API failures')
        time.sleep(1)
    report={**config,'state':'completed','attempted':len(tasks),'accepted_splits':dict(accepted),'seconds':time.monotonic()-start,'resumed_from':len(done)}
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
