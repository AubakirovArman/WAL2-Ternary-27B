"""CPU export and fixed AIME10 evaluation of an immutable E018 snapshot, on authorized GPU0."""
import argparse,fcntl,json,os,subprocess
from pathlib import Path
from transformers import AutoTokenizer
import check_e018_comparison as comparison
from benchmark_tasks import final_text
ROOT=comparison.ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--model',type=Path);p.add_argument('--thinking-off',action='store_true');p.add_argument('--sampled',action='store_true');p.add_argument('--gpu',type=int,choices=[0,6,7],default=0);p.add_argument('--gpu-lock-fd',type=int);a=p.parse_args();out=a.folder.resolve();comparison.OUT=out
 if a.sampled:comparison.BIN=ROOT/'tools/bonsai-runtime/prism-sampled-eval'
 cfg=json.loads((out/'snapshot.json').read_text());step=cfg['step'];model=out/f'e018-step{step}.gguf'
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8')
 if a.model:
  model=a.model.resolve();assert model.is_file()
 else:
  comparison.status('comparison','exporting',step=step)
  subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/export_latent_cpu.py'),str(out/'latent-snapshot'),str(ROOT/'runs/e017-fresh10000/selected-ternary'),str(out/'packed')],env=env,check=True)
  subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/export_prism_pq2.py'),str(out/'packed'),str(model),'--name',f'Vol2 E018 step{step}'],env=env,check=True)
 device=comparison.GPU0 if a.gpu==0 else comparison.GPU_IDS[a.gpu-6]
 lock_path=ROOT/('runs/e018-comparison-gpu0.lock' if a.gpu==0 else 'runs/gpu67.lock')
 if a.gpu_lock_fd is None:
  lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 else:
  assert a.gpu in (6,7)
  inherited=os.fstat(a.gpu_lock_fd);expected=lock_path.stat();assert (inherited.st_dev,inherited.st_ino)==(expected.st_dev,expected.st_ino)
  fcntl.flock(a.gpu_lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)

 free=int(subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 if free<40*1024:raise RuntimeError('Require40GiB free on shared GPU0')
 tok=AutoTokenizer.from_pretrained(comparison.SOURCE,local_files_only=True);jobs=json.loads((out/'aime10-tasks.json').read_text());responses=[]
 comparison.status('comparison','running',step=step,physical_gpu=a.gpu)
 def handle(r,i):
  j=jobs[i];assert r['kind']=='generation' and r['id']==j['id']
  text=tok.decode(r['tokens'],skip_special_tokens=False);answer,closed=final_text(text,not a.thinking_off)
  responses.append(dict(**j,raw_response=text,answer=answer,thinking_completed=closed,thinking_enabled=not a.thinking_off,finish_reason=r['finish_reason'],generated_tokens=len(r['tokens']),seconds=r['seconds']))
  comparison.write(out/'aime10-responses.json',responses)
 comparison.native(model,'aime10-inputs.txt',device,8192,32768,'aime10-native','aime10',10,handle)
 subprocess.run([str(ROOT/'.venv-release/bin/python'),str(ROOT/'scripts/score_e018_aime10.py'),str(out)],env=env,check=True)
 comparison.status('comparison','completed',step=step)
 with (ROOT/'EXPERIMENTS.md').open('a') as f:f.write(f'\nE018 step{step} AIME10 complete: '+(out/'aime10-metrics.json').read_text()+'\n')
if __name__=='__main__':
 try:main()
 except BaseException as e:comparison.status('comparison','failed',error=str(e));raise
