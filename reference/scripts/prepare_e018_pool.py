"""Freeze broad-source train-only candidate pool, excluding local benchmark text."""
import collections,hashlib,json,random,re,sys,time
from pathlib import Path
import pyarrow.parquet as pq
from e018_tasks import *
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/e018-25000'

def main():
    assert not (OUT/'tasks.jsonl').exists(),'Pool already frozen'
    seen=set();benchmarks=[];excluded_files={}
    def strings(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in ('instruction','problem','question','prompt') and isinstance(v,str):yield v
                elif isinstance(v,(dict,list)):yield from strings(v)
        elif isinstance(obj,list):
            for v in obj:yield from strings(v)
    for p in sorted(set(list((ROOT/'data').glob('**/accepted.jsonl'))+list((ROOT/'reports').glob('**/tasks.json'))+[ROOT/'data/release-benchmarks-v1/tasks-with-references.json'])):
        raw=p.read_bytes();excluded_files[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
        obj=json.loads(raw) if p.suffix=='.json' else [json.loads(l) for l in raw.splitlines()]
        for s in strings(obj):
            z=norm(s);seen.add(z)
            if '/reports/' in str(p) or 'release-benchmarks' in str(p):benchmarks.append(z.split())
    index=collections.defaultdict(set);benchsets=[]
    for words in benchmarks:
        grams={' '.join(words[i:i+8]) for i in range(len(words)-7)}
        j=len(benchsets);benchsets.append(grams)
        for g in grams:index[g].add(j)
    counts=collections.Counter();rejections=collections.Counter();tasks=[]
    def add(t,raw=None):
        z=norm(raw or t['instruction']);key=norm(t['instruction'])
        if z in seen or key in seen:rejections['exact_duplicate']+=1;return
        words=z.split();grams={' '.join(words[i:i+8]) for i in range(len(words)-7)};hits=collections.Counter(j for g in grams for j in index.get(g,()))
        if any(h>=3 and h/max(1,min(len(grams),len(benchsets[j])))>=.55 for j,h in hits.items()):rejections['benchmark_8gram_overlap']+=1;return
        if len(t['instruction'])>18000:rejections['long_prompt']+=1;return
        seen.add(z);seen.add(key);counts[(t['split'],t['category'])]+=1;tasks.append(t)
    table=pq.read_table(OUT/'sources/numina.parquet',columns=['source','problem','solution']).to_pylist();random.Random(1801).shuffle(table)
    math_sources=collections.Counter()
    for j,x in enumerate(table):
        if x['source'] not in ('cn_k12','orca_math','olympiads'):continue
        if '\\boxed{' not in x['solution'] or len(x['solution'])>22000:continue
        if re.search(r'\b(prove|proof|show that)\b',x['problem'],re.I):continue
        if math_sources[x['source']]>=13000:continue
        t=row('math',digest(x['problem'])[:20],x['problem']+'\nGive a clear, finite solution. Check your result and end with the final answer in \\boxed{...}.','math',group='numina-'+norm(x['problem']),gold=x['solution'],source='NuminaMath-CoT/train-shard0/'+x['source'])
        before=len(tasks);add(t,x['problem']);math_sources[x['source']]+=len(tasks)>before
    for j,line in enumerate((OUT/'sources/apps.jsonl').open()):
        x=json.loads(line)
        try:io=json.loads(x.get('input_output') or '{}')
        except ValueError:continue
        ins=io.get('inputs',[]);outs=io.get('outputs',[]);fn=io.get('fn_name')
        if not ins or len(ins)!=len(outs) or len(ins)>100 or len(json.dumps(io))>100000:continue
        if fn and not all(isinstance(v,list) for v in ins):continue
        if not fn and not all(isinstance(v,str) for v in ins+outs):continue
        if x.get('difficulty')=='competition' and len(x['question'])>12000:continue
        inst=x['question']+'\n'+(x.get('starter_code') or '')+'\nWrite a complete Python3 solution using only the standard library. Final answer: exactly one fenced Python code block. '+('Preserve the specified function signature.' if fn else 'Read from standard input and write to standard output.')
        source_id=x.get('problem_id',x.get('id',j))
        add(row('code_apps',source_id,inst,'apps',group='apps-'+str(x.get('url') or source_id),tests=list(zip(ins,outs)),fn_name=fn,source='APPS/train',source_id=source_id),x['question'])
    sq=pq.read_table(OUT/'sources/squad.parquet').to_pylist();random.Random(1802).shuffle(sq);titles=collections.Counter()
    for x in sq:
        if titles[x['title']]>=60 or len(x['context'])>10000:continue
        inst='Use only the passage below to answer the question. In your final answer return only JSON with keys "answer" (a minimal verbatim answer span from the passage) and "evidence" (a verbatim passage excerpt that contains the answer and supports it).\nPassage:\n'+x['context']+'\nQuestion: '+x['question']
        before=len(tasks);add(row('grounded',x['id'],inst,'grounded',group='squad-title-'+x['title'],context=x['context'],answers=x['answers']['text'],source='SQuAD/train'),x['question']);titles[x['title']]+=len(tasks)>before
    for i in range(24000):add(synthetic_code(i))
    for i in range(24000):add(instruction_task(i))
    for i in range(12000):
        t=logic_task(i);t['split']=split_for('graph-'+norm(t['instruction']));add(t)
    for split,quotas in [('train',TARGET),('validation',HOLDOUT)]:
        for cat,n in quotas.items():assert counts[split,cat]>=n*1.5,(split,cat,counts[split,cat],n)
    random.Random(18025000).shuffle(tasks)
    with (OUT/'tasks.jsonl').open('w') as f:
        for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+'\n')
    cfg=dict(state='prepared',target=TARGET,holdout=HOLDOUT,total_train=sum(TARGET.values()),total_validation=sum(HOLDOUT.values()),concurrency=50,max_output_tokens=8192,max_sequence_tokens=12288,model='Qwen/Qwen3.8-27B-FP8',counts={s:{c:counts[s,c] for c in TARGET} for s in ('train','validation')},tasks_sha256=hashlib.sha256((OUT/'tasks.jsonl').read_bytes()).hexdigest(),exclusions=rejections,excluded_files=excluded_files,split_policy='SQuAD by article title; synthetic code holds out4algorithm families; instructions by topic; other tasks by stable source/problem hash. Not a guarantee of semantic independence.',verification_scope='Final answers, source tests, format constraints; reasoning/prose not fully verified. Numeric/reference sources may contain errors. No benchmark items deliberately included. Conservative exact/8gram screening, no proof of zero semantic contamination.',sources=[json.loads(p.read_text()) for p in sorted((OUT/'sources').glob('*.provenance.json'))])
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print(json.dumps({k:cfg[k] for k in ['counts','exclusions','total_train','total_validation']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
