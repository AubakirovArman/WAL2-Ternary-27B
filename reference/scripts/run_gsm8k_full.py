"""One complete GSM8K test split, paired E012/E013 native PQ2 evaluation.
Two independent processes on physical GPUs6/7. No model updates or teacher calls.
"""
import argparse,hashlib,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,guard
from benchmark_tasks import final_text
from score_broad_benchmark import numeric
from compare_benchmarks import compare,wilson
OUT=ROOT/'reports/gsm8k-full-v1'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def readlines(path):
 if not path.exists():return []
 text=path.read_text()
 if text and not text.endswith('\n'):raise ValueError('Partial JSONL record: '+str(path))
 return [json.loads(l) for l in text.splitlines()]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');a=parser.parse_args();guard()
 if OUT.exists() and not a.resume:raise FileExistsError(OUT)
 OUT.mkdir(exist_ok=True)
 spec=json.loads((ROOT/'data/evaluation-v1/PLAN.json').read_text())['sets']['gsm8k']
 raw=(ROOT/'data/evaluation-v1'/spec['file']).read_bytes();assert sha(raw)==spec['sha256']
 data=[json.loads(l) for l in raw.splitlines()];assert len(data)==1319
 tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);jobs=[]
 for i,r in enumerate(data):
  instruction=r['question']+'\n\nSolve the problem. End your final answer with #### followed by the numeric result.'
  prompt=tok.apply_chat_template([{'role':'user','content':instruction}],tokenize=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='medium')
  ids=tok.encode(prompt,add_special_tokens=False);assert len(ids)+2048<=8192
  jobs.append(dict(id=i,family='gsm8k',source_index=i,source_key=i,instruction=instruction,prompt=prompt,input_ids=ids,thinking=True,reasoning_effort='medium',max_new_tokens=2048))
 taskraw=json.dumps(jobs,ensure_ascii=False,indent=2).encode()
 if (OUT/'tasks.json').exists():assert (OUT/'tasks.json').read_bytes()==taskraw
 else:(OUT/'tasks.json').write_bytes(taskraw)
 # Exact normalized overlap against the fresh training set, both bare and suffixed prompts.
 import re
 norm=lambda s:' '.join(re.findall(r'\w+',s.casefold()))
 train={norm(r['instruction']) for r in readlines(ROOT/'data/fresh-5000-v1/accepted.jsonl')}
 overlap=[i for i,r in enumerate(data) if norm(r['question']) in train or norm(jobs[i]['instruction']) in train]
 assert not overlap,overlap
 write(OUT/'protocol.json',dict(benchmark='GSM8K full official test',count=1319,dataset_sha256=sha(raw),tasks_sha256=sha(taskraw),decoding='greedy',thinking='medium',max_new_tokens=2048,context_limit=8192,grader='same strict #### numeric parser as prior smoke; no judge, no retries',fresh_train_exact_normalized_overlap=overlap,prior_smoke_indices=spec['smoke_indices'],note='Includes the prior32 smoke tasks; additionally report remaining1287 separately. No claim about base model pretraining contamination.'))
 devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
 tool=ROOT/'tools/bonsai-runtime';backend=tool/'llama-prism-b10709-9a9394a'
 def evaluate(label,device):
  out=OUT/label;out.mkdir(exist_ok=a.resume)
  model=ROOT/f'models/vol2-{label}-prism/vol2-{label}-pq2.gguf'
  cfg=dict(model=label+'-native-pq2',checkpoint=str(model),subset='full',suite='gsm8k',decoding='greedy',max_new_tokens=2048,context_limit=8192,reasoning_effort='medium',tasks_sha256=sha(taskraw),dataset_sha256=sha(raw),physical_gpu=6 if label=='e012' else 7,state='running')
  if (out/'config.json').exists():
   old=json.loads((out/'config.json').read_text())
   for key in cfg:
    if key!='state':assert cfg[key]==old[key],key
   if old['state']=='completed':return
  (out/'tasks.json').write_bytes(taskraw);write(out/'config.json',cfg)
  prior=readlines(out/'responses.jsonl');assert [r['id'] for r in prior]==list(range(len(prior)))
  lines=[]
  for job in jobs[len(prior):]:
   ids=job['input_ids'];base=f"{job['id']} {len(ids)} {len(ids)} "+' '.join(map(str,ids))
   lines.extend(['T '+base+' '+job['prompt'].encode().hex(),'G '+base])
  inp=out/f'inputs-from-{len(prior)}.txt';inp.write_text('\n'.join(lines)+'\n')
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
  env['LD_LIBRARY_PATH']=':'.join(map(str,[backend,tool/'cuda-deps/nvidia/cuda_runtime/lib',tool/'cuda-deps/nvidia/cublas/lib']))
  completed=list(prior)
  def status(state='running',error=None):
   write(out/'status.json',dict(state=state,completed=len(completed),total=1319,correct=sum(r['correct'] for r in completed),truncated=sum(r['finish_reason']=='length' for r in completed),error=error,updated=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
  status();start=time.monotonic()
  try:
   if len(prior)<1319:
    with (out/'native.log').open('a') as err,(out/'native.jsonl').open('a') as native,(out/'responses.jsonl').open('a') as responses:
     with subprocess.Popen([str(tool/'prism-native-eval'),str(model),str(inp),str(backend),'2048','8192','512'],env=env,stdout=subprocess.PIPE,stderr=err,text=True) as process:
      for line in process.stdout:
       r=json.loads(line);job=jobs[len(completed)]
       assert r['kind']=='generation' and r['id']==job['id']
       native.write(line);native.flush()
       text=tok.decode(r['tokens'],skip_special_tokens=False);answer,closed=final_text(text,True)
       actual=numeric(answer);expected=numeric(data[job['id']]['answer']);assert expected is not None
       row={**job,'raw_response':text,'answer':answer,'thinking_completed':closed,'finish_reason':r['finish_reason'],'generated_tokens':len(r['tokens']),'seconds':r['seconds'],'correct':actual is not None and actual==expected,'predicted':str(actual),'expected':str(expected)}
       responses.write(json.dumps(row,ensure_ascii=False)+'\n');responses.flush();os.fsync(responses.fileno());completed.append(row);status()
       if len(completed)%25==0:print(f'{label}: {len(completed)}/1319, верно {sum(x["correct"] for x in completed)}',flush=True)
      if process.wait()!=0:raise RuntimeError('Native generation failed')
   assert len(completed)==1319
   scores=[dict(id=r['id'],family='gsm8k',source_index=r['source_index'],correct=r['correct'],truncated=r['finish_reason']=='length',thinking_completed=r['thinking_completed'],predicted=r['predicted'],expected=r['expected']) for r in completed]
   def summary(rows):
    n=len(rows);c=sum(r['correct'] for r in rows)
    return dict(correct=c,total=n,accuracy=c/n,wilson95=wilson(c,n),truncated=sum(r['truncated'] for r in rows))
   write(out/'scores.json',scores);write(out/'summary.json',{'gsm8k':summary(scores)})
   write(out/'previously_unrun1287.json',summary([r for r in scores if r['id'] not in set(spec['smoke_indices'])]))
   cfg.update(state='completed',current_segment_seconds=time.monotonic()-start);write(out/'config.json',cfg);status('completed')
  except Exception as e:
   cfg.update(state='failed',error=str(e));write(out/'config.json',cfg);status('failed',str(e));raise
 with ThreadPoolExecutor(max_workers=2) as executor:
  futures=[executor.submit(evaluate,label,device) for label,device in zip(('e012','e013'),devices)]
  for future in futures:future.result()
 comparison=compare(OUT/'e013',OUT/'e012');write(OUT/'comparison.json',comparison)
 summaries={label:json.loads((OUT/label/'summary.json').read_text())['gsm8k'] for label in ('e012','e013')}
 text='''# Полный GSM8K: E012 и E013\n\n1319 задач на каждую модель; greedy, thinking medium, лимит2048, контекст8192. Нативный Prism PQ2, по одной H200 на модель. Это один математический бенчмарк, не общая оценка всех способностей.\n\n| Модель | Правильно | Доля | Достигнут лимит |\n|---|---:|---:|---:|\n'''
 for label,s in summaries.items():text+=f"| {label.upper()} | {s['correct']}/{s['total']} | {100*s['accuracy']:.2f}% | {s['truncated']} |\n"
 text+='\nПарное сравнение и интервалы: comparison.json. Для1287 задач вне прежней smoke-выборки: e012/previously_unrun1287.json и e013/previously_unrun1287.json. Все ошибочные/обрезанные ответы учитываются; лучшие из нескольких попыток не выбирались.\n'
 (OUT/'REPORT_RU.md').write_text(text)
 with (ROOT/'EXPERIMENTS.md').open('a') as f:f.write('\n## GSM8K full завершён\n'+text+'\n')
 print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
