"""Freeze full text benchmarks, official reference data and split provenance."""
import ast,collections,csv,gzip,hashlib,io,json,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datasets import Dataset
from evalplus.data import get_human_eval_plus,get_mbpp_plus
from evalplus.data.utils import CACHE_DIR
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/release-benchmarks-v1';OUT.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
provenance=[]
def fetch(repo,rev,name):
 url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}'
 path=OUT/'sources'/repo.replace('/','--')/name;path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=120).read())
 raw=path.read_bytes();provenance.append(dict(url=url,sha256=sha(raw),path=str(path.relative_to(OUT))));return raw
jobs=[]
def add(family,key,instruction,reference):jobs.append(dict(id=len(jobs),family=family,source_key=str(key),instruction=instruction,reference=reference,thinking=True,reasoning_effort='medium',max_new_tokens=8192))
# Preserve exact public datasets locally; refs never fed to generation.
for family,get in [('humaneval_plus',get_human_eval_plus),('mbpp_plus',get_mbpp_plus)]:
 data=get();raw=[x for x in Path(CACHE_DIR).glob('*.jsonl') if ('HumanEvalPlus' if family=='humaneval_plus' else 'MbppPlus') in x.name]
 for p in raw:
  dst=OUT/'sources'/p.name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(p.read_bytes());provenance.append(dict(source='evalplus0.3.1 official downloader',file=p.name,sha256=sha(p.read_bytes())))
 for key,r in data.items():
  add(family,key,'Implement the following Python function. Return the complete function and any required imports in one Python code block.\n\n'+r['prompt'],{'task_id':key})
for i,r in enumerate(map(json.loads,(ROOT/'data/evaluation-v1/ifeval-input.jsonl').read_text().splitlines())):add('ifeval',r['key'],r['prompt'],r)
for i,r in enumerate(map(json.loads,(ROOT/'data/evaluation-v1/gsm8k-test.jsonl').read_text().splitlines())):add('gsm8k',i,r['question']+'\n\nSolve the problem. End your final answer with #### followed by the numeric result.',r)
raw=fetch('HuggingFaceH4/MATH-500','6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be','test.jsonl')
for i,r in enumerate(map(json.loads,raw.splitlines())):add('math500',i,r['problem']+'\n\nSolve the problem and put your final answer inside \\boxed{}.',r)
for part in ['I','II']:
 raw=fetch('opencompass/AIME2025','a6ad95f611d72cf628a80b58bd0432ef6638f958',f'aime2025-{part}.jsonl')
 for i,r in enumerate(map(json.loads,raw.splitlines())):add('aime25',f'{part}-{i}',r['question']+'\n\nSolve the problem and put your final integer answer inside \\boxed{}.',r)
raw=fetch('TAUR-Lab/MuSR','7c365b439a222150f317764d4f16ae6c96d7d94a','all.csv')
for i,r in enumerate(csv.DictReader(io.StringIO(raw.decode()))):
 choices=ast.literal_eval(r['choices']);instruction=r['narrative']+'\n\n'+r['question']+'\n'+'\n'.join(f'{chr(65+j)}. {s}' for j,s in enumerate(choices))+'\nEnd your final answer with ANSWER: followed by the single option letter.'
 add('musr',i,instruction,{'answer_index':int(r['answer_index'])})
repo='edinburgh-dawg/mmlu-redux-2.0';rev='372ea425445d51e1ba1188c56e5e893f8138621f'
meta=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}'));names=sorted(x['rfilename'] for x in meta['siblings'] if x['rfilename'].endswith('.arrow'));assert len(names)==57
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(lambda name:fetch(repo,rev,name),names))
skipped=[]
for name in names:
 path=OUT/'sources'/repo.replace('/','--')/name
 for i,r in enumerate(Dataset.from_file(str(path))):
  key=name.split('/')[0]+f'-{i}'
  if r['error_type']=='ok':answer=r['answer']
  elif r['error_type']=='wrong_groundtruth' and str(r.get('correct_answer')).strip() in ['0','1','2','3']:answer=int(r['correct_answer'])
  else:skipped.append(dict(key=key,error_type=r['error_type'],correct_answer=r.get('correct_answer')));continue
  instruction=r['question']+'\n'+'\n'.join(f'{chr(65+j)}. {s}' for j,s in enumerate(r['choices']))+'\nEnd your final answer with ANSWER: followed by the single option letter.'
  add('mmlu_redux',key,instruction,{'answer_index':int(answer),'subject':name.split('/')[0],'error_type':r['error_type']})
# Interleave families so progress is informative and early failure does not hide all domains.
by=collections.defaultdict(list)
for r in jobs:by[r['family']].append(r)
ordered=[]
while any(by.values()):
 for family in sorted(by):
  if by[family]:ordered.append(by[family].pop(0))
for i,r in enumerate(ordered):r['id']=i
raw=json.dumps(ordered,ensure_ascii=False,indent=2).encode();(OUT/'tasks-with-references.json').write_bytes(raw)
for name in ['gsm8k-test.jsonl','ifeval-input.jsonl']:
 p=ROOT/'data/evaluation-v1'/name;provenance.append(dict(source=str(p),sha256=sha(p.read_bytes())))
report=dict(state='prepared',counts=dict(collections.Counter(r['family'] for r in ordered)),total=len(ordered),tasks_sha256=sha(raw),sources=provenance,mmlu_policy='All 5700 rows considered; use ok, correct wrong_groundtruth integer0..3; discard unclear/multiple/no-correct-answer rather than score faulty labels.',mmlu_excluded=skipped,thinking=True,max_new_tokens=8192,context=32768,training_use=False,scope='Eight text benchmarks from Bonsai suite. Not exact reproduction of unpublished Prism evaluation configuration. Vision/BFCL/LiveCodeBench/IFBench/AIME26 not included.')
(OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['sources','mmlu_excluded']},indent=2))
