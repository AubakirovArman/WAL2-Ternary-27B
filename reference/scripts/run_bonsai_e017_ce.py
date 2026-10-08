"""Explicitly authorized concurrent Bonsai CE; only GPU6, own-process memory guard."""
import fcntl,hashlib,json,os,subprocess,threading,time
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,load_examples
OUT=ROOT/'reports/bonsai-e017-ce';GPU='GPU_UUID_REDACTED'
def write(name,obj):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def free_mib():
 return int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
def main():
 lock=(ROOT/'runs/bonsai-e017-ce.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 OUT.mkdir(exist_ok=False)
 if free_mib()<18*1024:raise RuntimeError('Need 18GiB free on GPU6 before concurrent loading')
 folder=ROOT/'data/fresh-10000-v2';raw=(folder/'validation.jsonl').read_bytes();dm=json.loads((folder/'manifest.json').read_text())
 assert hashlib.sha256(raw).hexdigest()==dm['files']['validation']['sha256']
 kd=json.loads((ROOT/'data/v3-kd/manifest.json').read_text());assert hashlib.sha256(Path(kd['corpus']).read_bytes()).hexdigest()==kd['corpus_sha256']
 sets={'reasoning500':[json.loads(l) for l in raw.splitlines()], 'old128':load_examples(Path(kd['corpus']),512)['validation'],'verified32':load_examples(ROOT/'data/v2/verified_holdout.jsonl',512)['validation']}
 assert [len(x) for x in sets.values()]==[500,128,32]
 tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);lines=[];mapping=[]
 for name,rows in sets.items():
  for row in rows:
   ids=row['input_ids'];labels=row['labels'];p=next(i for i,v in enumerate(labels) if v!=-100)
   assert labels==[-100]*p+ids[p:] and len(ids)<=2048
   text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False);assert tok.encode(text,add_special_tokens=False)==ids,('token roundtrip',name,row['id'])
   i=len(mapping);prefix=f'{i} {p} {len(ids)} '+' '.join(map(str,ids));lines.extend(['T '+prefix+' '+text.encode().hex(),'E '+prefix])
   mapping.append(dict(id=i,source_id=row['id'],set=name,tokens=len(ids)-p))
 write('mapping.json',mapping);inp=OUT/'inputs.txt';inp.write_text('\n'.join(lines)+'\n')
 tool=ROOT/'tools/bonsai-runtime';backend=tool/'llama-prism-b10709-9a9394a';binary=tool/'prism-native-eval';model=ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
 write('config.json',dict(model=str(model),physical_gpu=6,validation_sha256=dm['files']['validation']['sha256'],corpus_sha256=kd['corpus_sha256'],inputs_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),context=2048,chunk=512,scope='Teacher-forced token-weighted CE on identical IDs/masks, native arithmetic differs from PyTorch. Not answer accuracy. Concurrent with E017; no teacher calls.',memory_guard='Require18GiB before loading; terminate ONLY native test if free<6GiB or own process>10GiB'))
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=GPU;env['CUDA_DEVICE_ORDER']='PCI_BUS_ID';env['LD_LIBRARY_PATH']=':'.join(map(str,[backend,tool/'cuda-deps/nvidia/cuda_runtime/lib',tool/'cuda-deps/nvidia/cublas/lib']))
 rows=[];halt=threading.Event();errors=[];peaks=[]
 def monitor(process):
  while not halt.wait(2):
   try:
    free=free_mib();usage=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    own=sum(int(l.split(',')[1]) for l in usage.splitlines() if l.split(',')[0].strip()==str(process.pid));peaks.append(own)
    if free<6*1024 or own>10*1024:
     errors.append(f'Memory guard: free={free}MiB own={own}MiB');process.terminate();return
   except Exception as e:errors.append(str(e));process.terminate();return
 def status(state):write('status.json',dict(state=state,completed=len(rows),total=len(mapping),error=errors,updated=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 status('running')
 with (OUT/'native.log').open('w') as err,(OUT/'results.jsonl').open('w') as output:
  with subprocess.Popen([str(binary),str(model),str(inp),str(backend),'1','2048','512'],env=env,stdout=subprocess.PIPE,stderr=err,text=True) as process:
   thread=threading.Thread(target=monitor,args=(process,),daemon=True);thread.start()
   try:
    for line in process.stdout:
     r=json.loads(line);expected=mapping[len(rows)];assert r['id']==expected['id'] and r['tokens']==expected['tokens'] and r['kind']=='ce'
     output.write(line);output.flush();rows.append(r);status('running')
     if len(rows)%25==0:print(f'Bonsai CE: {len(rows)}/660',flush=True)
    if process.wait()!=0:raise RuntimeError('Native test failed: '+str(errors))
   finally:halt.set();thread.join(timeout=5)
 assert len(rows)==660
 scores={}
 for name in sets:
  group=[r for r,m in zip(rows,mapping) if m['set']==name];n=sum(r['tokens'] for r in group);scores[name]=dict(examples=len(group),tokens=n,ce=sum(r['nll'] for r in group)/n)
 student=json.loads((ROOT/'runs/e017-fresh10000/metrics.json').read_text())
 latest=student['checks'][-1] if student['checks'] else None
 result=dict(state='completed',bonsai=scores,e017_check_snapshot=latest,e017_baseline=student['baseline_reasoning'],peak_native_mib=max(peaks,default=0),scope='Same reference tokens and masks; CE is not percentage error or task accuracy.')
 write('metrics.json',result);status('completed');print(json.dumps(result),flush=True)
 with (ROOT/'EXPERIMENTS.md').open('a') as f:f.write('\n## Bonsai E017 CE completed\n'+json.dumps(result)+'\n')
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():write('error.json',dict(state='failed',error=str(e)))
  raise
