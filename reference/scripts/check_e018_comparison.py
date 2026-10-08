"""Frozen E018 step3584 AIME10 + Bonsai PQ2 CE on the exact E018 validation1500."""
import argparse,fcntl,hashlib,json,os,subprocess,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,GPU_IDS
from run_release_benchmarks import input_lines,env_for,BIN,BACK
from benchmark_tasks import final_text
OUT=ROOT/'reports/e018-step3584-check'
GPU0='GPU_UUID_REDACTED'
def write(path,obj):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(path)
def status(name,state,**values):write(OUT/(name+'-status.json'),dict(state=state,updated=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values))
def prepare():
 tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
 jobs=json.loads((OUT/'aime10-tasks.json').read_text());(OUT/'aime10-inputs.txt').write_text(input_lines(jobs))
 raw=(ROOT/'data/e018-25000/validation.jsonl').read_bytes();manifest=json.loads((ROOT/'data/e018-25000/manifest.json').read_text());assert hashlib.sha256(raw).hexdigest()==manifest['files']['validation']['sha256']
 rows=[json.loads(l) for l in raw.splitlines()];assert len(rows)==1500
 mapping=[]
 with (OUT/'bonsai1500-inputs.txt').open('w') as f:
  for i,row in enumerate(rows):
   ids=row['input_ids'];labels=row['labels'];p=next(k for k,v in enumerate(labels) if v!=-100)
   assert labels==[-100]*p+ids[p:] and len(ids)<=12288
   text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
   assert tok.encode(text,add_special_tokens=False)==ids,('roundtrip',i)
   base=f'{i} {p} {len(ids)} '+' '.join(map(str,ids));f.write('T '+base+' '+text.encode().hex()+'\nE '+base+'\n')
   mapping.append(dict(id=i,source_id=row['id'],tokens=len(ids)-p))
 write(OUT/'bonsai1500-mapping.json',mapping)
 write(OUT/'protocol.json',dict(student='E018 step3584, best eligible checkpoint at request time',aime_selection='First 10 AIME25 in frozen previous release order; selected before generation',aime_ids=[j['id'] for j in jobs],aime_prompt_sha256=hashlib.sha256((OUT/'aime10-inputs.txt').read_bytes()).hexdigest(),aime_max_new_tokens=8192,aime_context=32768,aime_decoding='greedy, thinking medium, one attempt',bonsai='Ternary-Bonsai-2-27B-PQ2_0.gguf',validation_sha256=hashlib.sha256(raw).hexdigest(),validation_examples=1500,validation_target_tokens=sum(r['tokens'] for r in mapping),ce_context=12288,ce_chunk=512,ce_scope='Token-weighted teacher-forced CE on exactly the same input IDs and label masks, not task error percentage. Bonsai native vs E018 training PyTorch have different numerical kernels.',native_binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest()))
 status('comparison','prepared');print('Inputs prepared',flush=True)
def native(model,inp,device,max_new,ctx,dest,name,total,handle):
 status(name,'running',completed=0,total=total)
 with (OUT/(dest+'.log')).open('w') as log,(OUT/(dest+'.jsonl')).open('w') as output:
  with subprocess.Popen([str(BIN),str(model),str(OUT/inp),str(BACK),str(max_new),str(ctx),'512'],env=env_for(device),stdout=subprocess.PIPE,stderr=log,text=True) as proc:
   halt=threading.Event();guard_errors=[]
   def monitor():
    while not halt.wait(2):
     try:
      free=int(subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
      usage=subprocess.check_output(['nvidia-smi','-i',device,'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
      own=sum(int(l.split(',')[1]) for l in usage.splitlines() if l.split(',')[0].strip()==str(proc.pid))
      if device==GPU0 and (free<16*1024 or own>24*1024):raise RuntimeError(f'GPU0 memory guard: free={free}MiB own={own}MiB')
     except Exception as e:
      guard_errors.append(str(e));proc.terminate();return
   thread=threading.Thread(target=monitor,daemon=True);thread.start()
   try:
    count=0
    for line in proc.stdout:
     row=json.loads(line);handle(row,count);output.write(line);output.flush();count+=1;status(name,'running',completed=count,total=total)
    if proc.wait()!=0:raise RuntimeError(f'{name}: native failed, see {dest}.log; {guard_errors}')
    assert count==total
   except BaseException:
    if proc.poll() is None:proc.terminate();proc.wait()
    raise
   finally:halt.set();thread.join(timeout=5)
 status(name,'completed',completed=total,total=total)
def aime(device=GPU_IDS[0]):
 tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);jobs=json.loads((OUT/'aime10-tasks.json').read_text());responses=[]
 def handle(r,i):
  j=jobs[i];assert r['kind']=='generation' and r['id']==j['id']
  text=tok.decode(r['tokens'],skip_special_tokens=False);answer,closed=final_text(text,True)
  responses.append(dict(**j,raw_response=text,answer=answer,thinking_completed=closed,finish_reason=r['finish_reason'],generated_tokens=len(r['tokens']),seconds=r['seconds']))
  write(OUT/'aime10-responses.json',responses)
 native(OUT/'e018-step3584.gguf','aime10-inputs.txt',device,8192,32768,'aime10-native','aime10',10,handle)
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
 subprocess.run([str(ROOT/'.venv-release/bin/python'),str(ROOT/'scripts/score_e018_aime10.py')],env=env,check=True)
def bonsai(device=GPU_IDS[1]):
 mapping=json.loads((OUT/'bonsai1500-mapping.json').read_text());results=[]
 def handle(r,i):
  assert r['kind']=='ce' and r['id']==mapping[i]['id'] and r['tokens']==mapping[i]['tokens'];results.append(r)
 native(ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf','bonsai1500-inputs.txt',device,1,12288,'bonsai1500-native','bonsai1500',1500,handle)
 n=sum(r['tokens'] for r in results);ce=sum(r['nll'] for r in results)/n
 snapshot=json.loads((OUT/'training-snapshot.json').read_text());check=next(c for c in snapshot['checks'] if c['step']==3584)
 write(OUT/'bonsai1500-metrics.json',dict(bonsai_ce=ce,tokens=n,examples=1500,e017_baseline=snapshot['baseline_reasoning'],e018_step3584=check['reasoning'],scope='Same IDs/masks; native vs PyTorch arithmetic. CE is not percentage wrong.'))
def run():
 # Respect the training lock. No GPU allocations while waiting.
 status('comparison','waiting_for_gpu',reason='Training owns GPU6/7; no interruption requested')
 lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 for gpu in GPU_IDS:
  free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free<100*1024:raise RuntimeError('GPU not free after training lock release')
 while not (OUT/'e018-step3584.export.json').exists():
  st=subprocess.check_output(['systemctl','--user','show','vol2-e018-step3584-export.service','-p','ActiveState','--value'],text=True).strip()
  if st not in ('active','activating'):raise RuntimeError('CPU export not complete')
  time.sleep(10)
 status('comparison','running')
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(aime),pool.submit(bonsai)]
  for f in futures:f.result()
 status('comparison','completed')
 with (ROOT/'EXPERIMENTS.md').open('a') as f:f.write('\nE018 step3584 AIME10 / Bonsai validation1500 finished; reports/e018-step3584-check.\n')
def run_gpu0():
 lock=(ROOT/'runs/e018-comparison-gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 free=int(subprocess.check_output(['nvidia-smi','-i',GPU0,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 if free<40*1024:raise RuntimeError('Require 40GiB free on shared GPU0 before starting')
 assert (OUT/'e018-step3584.export.json').exists()
 protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(execution_gpu=0,execution_gpu_uuid=GPU0,execution='Sequential AIME10 then Bonsai1500; user explicitly authorized GPU0',memory_guard='Require40GiB free at start; stop only own native child if free<16GiB or own>24GiB');write(OUT/'protocol.json',protocol)
 status('comparison','running',physical_gpu=0,execution='sequential')
 errors=[]
 for name,fn in [('aime10',aime),('bonsai1500',bonsai)]:
  try:fn(GPU0)
  except Exception as e:errors.append(str(e));status(name,'failed',error=str(e))
 if errors:raise RuntimeError('; '.join(errors))
 status('comparison','completed',physical_gpu=0)
 with (ROOT/'EXPERIMENTS.md').open('a') as f:f.write('\nGPU0 comparison completed: reports/e018-step3584-check.\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--gpu0',action='store_true');a=p.parse_args()
 try:prepare() if a.prepare else run_gpu0() if a.gpu0 else run()
 except BaseException as e:status('comparison','failed',error=str(e));raise
