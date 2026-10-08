"""Freeze4096+256pilot candidate pool with exact math references. CPU only."""
import collections
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path

from e019_math_tasks import make_math, FAMILIES
from e018_tasks import synthetic_code, instruction_task, logic_task

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e019-reasoning-pilot-v1'
TARGET={'math':1792,'math_repair':256,'code':1024,'instructions':512,'logic':512}
HOLDOUT={'math':128,'code':64,'instructions':32,'logic':32}


def norm(text):return ' '.join(re.findall(r'\w+',text.casefold()))


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'config.json').exists(),'Pool already frozen'
    seen=set();benchsets=[];sources=[]
    for p in sorted((ROOT/'data').glob('**/accepted.jsonl')):
        raw=p.read_bytes();sources.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
        for l in raw.splitlines():
            r=json.loads(l)
            if r.get('instruction'):seen.add(norm(r['instruction']))
    for p in [ROOT/'data/release-benchmarks-v1/tasks-with-references.json',ROOT/'reports/e019-d/comparison-tasks.json']:
        raw=p.read_bytes();sources.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
        for r in json.loads(raw):
            text=r.get('instruction','');seen.add(norm(text));words=norm(text).split();benchsets.append({' '.join(words[i:i+8]) for i in range(len(words)-7)})
            ref=r.get('reference',{})
            if isinstance(ref,dict):
                for k in ('question','problem'):
                    if isinstance(ref.get(k),str):seen.add(norm(ref[k]))
    index=collections.defaultdict(set)
    for i,gs in enumerate(benchsets):
        for g in gs:index[g].add(i)
    rejected=collections.Counter();counts=collections.Counter();tasks=[]
    def add(t):
        n=norm(t['instruction'])
        if n in seen:rejected['exact_duplicate']+=1;return
        words=n.split();gs={' '.join(words[i:i+8]) for i in range(len(words)-7)};hits=collections.Counter(j for g in gs for j in index.get(g,()))
        if any(h>=3 and h/max(1,min(len(gs),len(benchsets[j])))>=.55 for j,h in hits.items()):rejected['benchmark_8gram_overlap']+=1;return
        seen.add(n);counts[t['split'],t['category']]+=1;tasks.append(t)
    for family in range(20):
        split='train' if family<16 else 'validation'
        for i in range(384 if split=='train' else 96):add(make_math(family,310000+i,split))
        if split=='train':
            for i in range(160):add(make_math(family,330000+i,split,repair=True))
    for i in range(320000,326000):
        t=synthetic_code(i);t.update(id='e019-'+t['id'],category='code',family_index=i%24,source='New seeded code compositions; existing sandbox tests',repair=False);add(t)
    for i in range(340000,344000):
        t=instruction_task(i);t.update(id='e019-'+t['id'],repair=False,source='New seeded constrained instruction tasks');add(t)
    for i in range(350000,354000):
        t=logic_task(i);t.update(id='e019-'+t['id'],repair=False,source='New seeded directed-graph tasks')
        t['split']='validation' if t['topic']=='logic-network-size-9' else 'train'
        add(t)
    for split,quota in [('train',TARGET),('validation',HOLDOUT)]:
        for cat,target in quota.items():assert counts[split,cat]>=target*1.5,(split,cat,counts[split,cat],target)
    # Save mathematics family allocation separately to avoid one-family domination.
    math_counts=collections.Counter((t['split'],t['category'],t.get('family_index')) for t in tasks if t['category'] in ('math','math_repair'))
    assert all(math_counts['train','math',f]>=140 and math_counts['train','math_repair',f]>=40 for f in range(16)),math_counts
    assert all(math_counts['validation','math',f]>=48 for f in range(16,20)),math_counts
    random.Random(1904096).shuffle(tasks)
    path=OUT/'tasks.jsonl'
    with path.open('w') as stream:
        for t in tasks:stream.write(json.dumps(t,ensure_ascii=False)+'\n')
    (OUT/'code').mkdir(exist_ok=True)
    hashes={}
    for name in ('e019_math_tasks.py','prepare_e019_pilot.py','e018_tasks.py','code_sandbox.py','code_sandbox_child.py'):
        raw=(ROOT/'scripts'/name).read_bytes();(OUT/'code'/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    cfg=dict(state='prepared',target=TARGET,holdout=HOLDOUT,total_train=4096,total_validation=256,concurrency=50,max_output_tokens=8192,max_sequence_tokens=12288,teacher='Qwen/Qwen3.8-27B-FP8',tasks_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),task_code=hashes,candidate_counts={s:{c:counts[s,c] for c in set(TARGET)|set(HOLDOUT)} for s in ('train','validation')},math_families=FAMILIES,math_quota='train112normal+16repair per each0..15; validation32 per each16..19',exclusions=dict(rejected),exclusion_files=sources,verification='Math: exact computed answer and supplied calculation/algorithm certificate reproduced from frozen generator; teacher explains each fact in bounded prose, which is not formally verified. Code: all sandbox fixtures and mutation check; instruction constraints and graph JSON exact. No benchmark answer appears in training.',repair_scope='256math examples conditioned on E018768-token prefixes from new train-only tasks; defective draft is prompt context masked from loss. Verified target restarts a finite derivation; this is repair-SFT, not full on-policy KD.',split_policy='Entire math task families0..15 vs16..19; code original family holdouts; instruction topics; graph groups. Not proof of semantic or base-pretraining independence.',collection_only=True,created=time.time())
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));(OUT/'status.json').write_text(json.dumps(dict(state='prepared',train=0,target_train=4096,validation=0,target_validation=256)))
    print(json.dumps(dict(candidate_counts=cfg['candidate_counts'],exclusions=cfg['exclusions']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
