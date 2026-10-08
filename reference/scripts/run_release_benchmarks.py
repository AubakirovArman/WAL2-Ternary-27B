"""Matched E017/Bonsai text benchmark campaign on physical GPUs6/7."""
import argparse,collections,hashlib,json,os,re,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,guard
from benchmark_tasks import final_text
from compare_benchmarks import compare
OUT=ROOT/'reports/release-benchmarks-v1';DATA=ROOT/'data/release-benchmarks-v1'
TOOL=ROOT/'tools/bonsai-runtime';BACK=TOOL/'llama-prism-b10709-9a9394a';BIN=TOOL/'prism-native-eval'
MODELS={'e017':ROOT/'models/vol2-e017-prism/vol2-e017-pq2.gguf','bonsai2':ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf'}
def sha(b):return hashlib.sha256(b).hexdigest()
def write(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def readrows(path):
 if not path.exists():return []
 raw=path.read_bytes()
 if raw and not raw.endswith(b'\n'):
  n=raw.rfind(b'\n')+1;path.with_suffix('.partial-'+str(time.time_ns())).write_bytes(raw[n:])
  with path.open('r+b') as f:f.truncate(n)
  raw=raw[:n]
 return [json.loads(l) for l in raw.splitlines()]
def env_for(device):
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device;env['LD_LIBRARY_PATH']=':'.join(map(str,[BACK,TOOL/'cuda-deps/nvidia/cuda_runtime/lib',TOOL/'cuda-deps/nvidia/cublas/lib']));return env
def input_lines(jobs):
 lines=[]
 for j in jobs:
  ids=j['input_ids'];base=f"{j['id']} {len(ids)} {len(ids)} "+' '.join(map(str,ids));lines.extend(['T '+base+' '+j['prompt'].encode().hex(),'G '+base])
 return '\n'.join(lines)+'\n'
def pilot(tok,device):
 p=OUT/'speed-pilot';p.mkdir(exist_ok=True)
 jobs=json.loads((ROOT/'reports/gsm8k-full-v1/tasks.json').read_text())[:9]
 (p/'inputs.txt').write_text(input_lines(jobs));summary={}
 candidates={'bonsai-pq2':MODELS['bonsai2'],'bonsai-ptq1':ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf'}
 for label,model in candidates.items():
  with (p/(label+'.jsonl')).open('w') as out,(p/(label+'.log')).open('w') as err:
   subprocess.run([str(BIN),str(model),str(p/'inputs.txt'),str(BACK),'256','32768','512'],env=env_for(device),stdout=out,stderr=err,check=True,timeout=1200)
  rows=readrows(p/(label+'.jsonl'));assert len(rows)==9
  # First request warms runtime; same eight subsequent prompts and budget.
  rs=rows[1:];tokens=sum(len(r['tokens']) for r in rs);seconds=sum(r['seconds'] for r in rs)
  summary[label]=dict(model=str(model),tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds)
 chosen=max(summary,key=lambda k:summary[k]['tokens_per_second']);summary['selected']=chosen;summary['scope']='8 same prompts after1 warmup, max256tokens, includes prefill; packing chosen by speed only, not accuracy.'
 write(OUT/'speed-pilot.json',summary);print('SPEED',json.dumps(summary),flush=True);return Path(summary[chosen]['model'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');a=ap.parse_args();guard()
 OUT.mkdir(exist_ok=a.resume)
 tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);manifest=json.loads((DATA/'manifest.json').read_text());raw=(DATA/'tasks-with-references.json').read_bytes();assert sha(raw)==manifest['tasks_sha256']
 refs=json.loads(raw);jobs=[]
 for r in refs:
  j={k:v for k,v in r.items() if k!='reference'};j['source_index']=r['id']
  j['prompt']=tok.apply_chat_template([{'role':'user','content':j['instruction']}],tokenize=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='medium');j['input_ids']=tok.encode(j['prompt'],add_special_tokens=False)
  assert len(j['input_ids'])+8192<=32768,('context overflow',j['family'],j['source_key'])
  jobs.append(j)
 taskraw=json.dumps(jobs,ensure_ascii=False,indent=2).encode()
 if (OUT/'tasks.json').exists():assert (OUT/'tasks.json').read_bytes()==taskraw
 else:(OUT/'tasks.json').write_bytes(taskraw)
 # Conservative exact-prompt exposure audit across collected data. No benchmark item silently removed.
 norm=lambda s:' '.join(re.findall(r'\w+',s.casefold()))
 seen=set()
 for p in (ROOT/'data').glob('**/accepted.jsonl'):
  if 'release-benchmarks' in str(p):continue
  for line in p.read_text().splitlines():
   r=json.loads(line)
   if r.get('split')=='train' and r.get('instruction'):seen.add(norm(r['instruction']))
 overlap=[]
 for r in refs:
  forms=[r['instruction']]+[r['reference'][k] for k in ('question','problem','prompt') if isinstance(r['reference'].get(k),str)]
  if any(norm(s) in seen for s in forms):overlap.append(dict(id=r['id'],family=r['family'],source_key=r['source_key']))
 write(OUT/'exposure-audit.json',dict(exact_matches=overlap,scope='Normalized exact match against local accepted training prompts. Does not rule out semantic or base-pretraining contamination. GSM8K already used to compare prior models.'))
 devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
 if (OUT/'speed-pilot.json').exists():
  speed=json.loads((OUT/'speed-pilot.json').read_text());MODELS['bonsai2']=Path(speed[speed['selected']]['model'])
 else:MODELS['bonsai2']=pilot(tok,devices[1])
 write(OUT/'protocol.json',dict(counts=manifest['counts'],total=len(jobs),tasks_sha256=sha(taskraw),decoding='greedy',thinking='medium',max_new_tokens=8192,context=32768,attempts_per_task=1,model_selection='E017 step10000 selected before this campaign; Bonsai packing selected by speed pilot only',limitations=manifest['scope'],mmlu_policy=manifest['mmlu_policy']))
 def evaluate(label,device):
  out=OUT/label;out.mkdir(exist_ok=a.resume);model=MODELS[label]
  cfg=dict(model=label,checkpoint=str(model),checkpoint_sha256=hashlib.file_digest(model.open('rb'),'sha256').hexdigest(),subset='full-valid',suite='release8-v1',decoding='greedy',max_new_tokens=8192,context_limit=32768,reasoning_effort='medium',tasks_sha256=sha(taskraw),total=len(jobs),state='running')
  if (out/'config.json').exists():
   previous=json.loads((out/'config.json').read_text())
   for key in cfg:
    if key!='state':assert previous[key]==cfg[key],key
  (out/'tasks.json').write_bytes(taskraw);write(out/'config.json',cfg)
  prior=readrows(out/'responses.jsonl');assert [r['id'] for r in prior]==list(range(len(prior)))
  count=len(prior);scorer_env=os.environ.copy();scorer_env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
  scorelog=(out/'scoring.log').open('a');scorer=subprocess.Popen([str(ROOT/'.venv-release/bin/python'),'-u',str(ROOT/'scripts/score_release_benchmarks.py'),str(out),'--follow'],env=scorer_env,stdout=scorelog,stderr=scorelog)
  def status(state='running'):write(out/'status.json',dict(state=state,completed=count,total=len(jobs),updated=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
  status()
  try:
   if count<len(jobs):
    inp=out/f'inputs-from-{count}.txt';inp.write_text(input_lines(jobs[count:]))
    with (out/'native.log').open('a') as err,(out/'native.jsonl').open('a') as native,(out/'responses.jsonl').open('a') as responses:
     with subprocess.Popen([str(BIN),str(model),str(inp),str(BACK),'8192','32768','512'],env=env_for(device),stdout=subprocess.PIPE,stderr=err,text=True) as proc:
      for line in proc.stdout:
       r=json.loads(line);j=jobs[count];assert r['kind']=='generation' and r['id']==j['id']
       native.write(line);native.flush();text=tok.decode(r['tokens'],skip_special_tokens=False);answer,closed=final_text(text,True)
       row={**j,'raw_response':text,'answer':answer,'thinking_completed':closed,'finish_reason':r['finish_reason'],'generated_tokens':len(r['tokens']),'seconds':r['seconds'],'prefill_seconds':r['prefill_seconds']}
       responses.write(json.dumps(row,ensure_ascii=False)+'\n');responses.flush();os.fsync(responses.fileno());count+=1;status()
       if count%25==0:print(f'{label}: {count}/{len(jobs)}',flush=True)
      if proc.wait()!=0:raise RuntimeError('Native generation failed')
   cfg['state']='completed';write(out/'config.json',cfg);status('generation_completed')
   if scorer.wait()!=0:raise RuntimeError('Scoring failed; raw generations retained')
   status('completed')
  except Exception as e:
   # If scoring failed, preserve completed generation config to allow scoring-only repair.
   if count<len(jobs):cfg['state']='failed';write(out/'config.json',cfg)
   write(out/'error.json',dict(error=str(e)));status('failed')
   if scorer.poll() is None:scorer.terminate()
   raise
  finally:scorelog.close()
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(evaluate,label,device) for label,device in zip(('e017','bonsai2'),devices)]
  for f in futures:f.result()
 report=compare(OUT/'e017',OUT/'bonsai2');write(OUT/'comparison.json',report)
 lines=['# E017 и Bonsai2: восемь текстовых бенчмарков','', 'Одинаковые задания; greedy, thinking medium, до8192токенов, одна попытка. Это наш протокол, не точное воспроизведение таблицы Prism.','', '| Тест | E017 | Bonsai2 | Разница, п.п. |','|---|---:|---:|---:|']
 for family,r in report['families'].items():lines.append(f"| {family} | {r['student_correct']}/{r['n']} ({100*r['student_accuracy']:.2f}%) | {r['reference_correct']}/{r['n']} ({100*r['reference_accuracy']:.2f}%) | {r['difference_percentage_points']:+.2f} |")
 lines+=['','Интервалы и парные сравнения: comparison.json. Обрезанные/незавершённые ответы учитываются. MMLU использует все валидные аннотированные вопросы, список исключений в data manifest. MBPP+ — все378 задач, возвращаемых EvalPlus0.3.1/v0.2.0. IFEval prompt-strict, prompt-loose дополнительно в summary.json. Предварительные выводы о готовности к публикации требуют просмотра ошибок и ограничений.']
 (OUT/'REPORT_RU.md').write_text('\n'.join(lines)+'\n');write(OUT/'status.json',dict(state='completed'))
 with (ROOT/'EXPERIMENTS.md').open('a') as f:f.write('\n## Release benchmark8 completed\n'+str(OUT/'REPORT_RU.md')+'\n')
if __name__=='__main__':main()
