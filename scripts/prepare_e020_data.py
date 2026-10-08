"""New exact-reference pool plus frozen generation controls. No benchmark training."""
import collections,hashlib,json,random,re,time
from pathlib import Path
from e020_math_tasks import make_math,FAMILIES
from e018_tasks import synthetic_code,instruction_task,logic_task
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e020-natural-v1'
TARGET={'math':640,'code':256,'instructions':64,'logic':64}
HOLDOUT={'math':128,'code':64,'instructions':32,'logic':32}
def norm(text):return ' '.join(re.findall(r'\w+',text.casefold()))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    if (OUT/'config.json').exists():print('Пул уже зафиксирован.');return
    OUT.mkdir(exist_ok=True)
    seen=set();benchsets=[];sources=[]
    for p in sorted((ROOT/'data').glob('**/accepted.jsonl')):
        sources.append(dict(path=str(p),sha256=sha(p)))
        for line in p.open():
            row=json.loads(line)
            if row.get('instruction'):seen.add(norm(row['instruction']))
    for p in [ROOT/'data/release-benchmarks-v1/tasks-with-references.json',ROOT/'reports/e019-d/comparison-tasks.json']:
        sources.append(dict(path=str(p),sha256=sha(p)))
        for row in json.loads(p.read_text()):
            words=norm(row['instruction']).split();seen.add(' '.join(words))
            benchsets.append({' '.join(words[i:i+8]) for i in range(len(words)-7)})
    index=collections.defaultdict(set)
    for i,gs in enumerate(benchsets):
        for g in gs:index[g].add(i)
    rejected=collections.Counter();tasks=[];control=[]
    def add(t,dest):
        t=dict(t);t['id']=t['id'].replace('e019-','e020-').replace('e018-','e020-')
        t['enable_thinking']=True;t['repair']=False
        n=norm(t['instruction'])
        if n in seen:rejected['duplicate']+=1;return
        words=n.split();gs={' '.join(words[i:i+8]) for i in range(len(words)-7)}
        hits=collections.Counter(j for g in gs for j in index.get(g,()))
        if any(h>=3 and h/max(1,min(len(gs),len(benchsets[j])))>=.55 for j,h in hits.items()):rejected['benchmark_overlap']+=1;return
        seen.add(n);dest.append(t)
    for f in range(20):
        split='train' if f<16 else 'validation'
        for serial in range(420000,420000+(400 if split=='train' else 300)):
            t=make_math(f,serial,split);t['topic']=t['family'];add(t,tasks)
    for serial in range(600000,600300):
        t=make_math(19,serial,'validation');t['topic']=t['family'];add(t,tasks)
    # Full family holdout for math and code; original instruction-topic groups.
    for serial in range(430000,432400):
        t=synthetic_code(serial);t['category']='code';add(t,tasks)
    for serial in range(440000,442000):add(instruction_task(serial),tasks)
    for serial in range(450000,451000):
        t=logic_task(serial);t['split']='validation' if t['topic']=='logic-network-size-9' else 'train';add(t,tasks)
    # Additional controls are excluded before any teacher text collection.
    # Math includes familiar held-out parameters from all 20 families, not training rows.
    for f in range(20):
        for offset in range(1000):
            before=len(control)
            t=make_math(f,470000+1000*f+offset,'train' if f<16 else 'validation');t.update(reference_split=t['split'],split='control',topic=t['family']);add(t,control)
            if len(control)>before:break
        else:raise RuntimeError('No disjoint math control for family '+str(f))
    for cat,fn,start in [('code',synthetic_code,480000),('instructions',instruction_task,490000),('logic',logic_task,500000)]:
        before=len(control)
        for i in range(1000):
            t=fn(start+i);t.update(category=cat,split='control');add(t,control)
            if len(control)==before+16:break
    assert len(control)==68
    counts=collections.Counter((t['split'],t['category']) for t in tasks)
    for split,quota in [('train',TARGET),('validation',HOLDOUT)]:
        for cat,count in quota.items():assert counts[split,cat]>=count*2,(split,cat,counts[split,cat])
    random.Random(2001024).shuffle(tasks)
    for name,rows in [('tasks.jsonl',tasks),('control-tasks.jsonl',control)]:
        (OUT/name).write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in rows))
    cfg=dict(state='prepared',target=TARGET,holdout=HOLDOUT,total_train=1024,total_validation=256,
             concurrency=50,max_output_tokens=8192,max_sequence_tokens=12288,
             teacher='Qwen/Qwen3.8-27B-FP8',tasks_sha256=sha(OUT/'tasks.jsonl'),control_sha256=sha(OUT/'control-tasks.jsonl'),
             candidate_counts={str(k):v for k,v in counts.items()},exclusions=dict(rejected),exclusion_sources=sources,
             math_families=FAMILIES,math_train_quota_per_family=40,math_validation_quota_per_family=32,
             split_policy='Entire math families0..15 train vs16..19 val; code-family and instruction-topic holdouts; development controls disjoint exact prompts, semantic/base-pretraining independence not proven',
             verification='Exact final/math reference regeneration + separate teacher key-reasoning audit and exact rational arithmetic checks where extracted; not formal proof. Code sandbox tests; instruction/logic exact constraints. No supplied certificate or gold in solver prompt.',created=time.time())
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    print(json.dumps(dict(pool=len(tasks),controls=len(control),counts=cfg['candidate_counts'],exclusions=dict(rejected)),ensure_ascii=False))
if __name__=='__main__':main()
