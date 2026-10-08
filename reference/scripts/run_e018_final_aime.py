"""Wait for completed, verified final E018 then run original AIME10 protocol on GPU6."""
import argparse,hashlib,json,os,shutil,subprocess,time,fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/e018-final-aime10';RUN=ROOT/'runs/e018-fresh25000'
def status(state,**values):
 p=OUT/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,updated=time.time(),**values),ensure_ascii=False,indent=2));tmp.replace(p)
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'status.json').exists():assert json.loads((OUT/'status.json').read_text())['state']=='waiting_for_training'
 parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path);args=parser.parse_args()
 m=json.loads((RUN/'metrics.json').read_text())
 if args.checkpoint:
  st=subprocess.check_output(['systemctl','--user','show','vol2-e018-training.service','-p','ActiveState','--value'],text=True).strip()
  assert st not in ('active','activating','deactivating'),'Stop trainer before direct checkpoint export'
  latent=args.checkpoint.resolve();meta=json.loads((latent/'metadata.json').read_text())
  assert meta['eligible'];step=meta['step'];packed=OUT/'direct-packed'
  snapshot=dict(step=step,path=str(latent),score=meta['reasoning']['reasoning_and_answer_ce'],final_reload_ce_skipped=True)
 else:
  status('waiting_for_training')
  while True:
   m=json.loads((RUN/'metrics.json').read_text());st=subprocess.check_output(['systemctl','--user','show','vol2-e018-training.service','-p','ActiveState','--value'],text=True).strip()
   if m['state']=='completed' and st not in ('active','activating'):break
   if st not in ('active','activating') and m['state']!='completed':raise RuntimeError('Training stopped without completed verified export')
   time.sleep(15)
  latent=Path(m['selected_latent']);latent=latent if latent.is_absolute() else ROOT/latent
  packed=Path(m['selected_packed']);packed=packed if packed.is_absolute() else ROOT/packed
  meta=json.loads((latent/'metadata.json').read_text());step=meta['step'];snapshot=dict(step=step,path=str(latent),score=m['selected_reasoning']['reasoning_and_answer_ce'])
 (OUT/'snapshot.json').write_text(json.dumps(snapshot,indent=2));(OUT/'training-snapshot.json').write_text(json.dumps(m,indent=2))
 src=ROOT/'reports/e018-step3584-check'
 for name in ('aime10-tasks.json','aime10-inputs.txt','e017-aime10-scores.json'):shutil.copy2(src/name,OUT/name)
 (OUT/'protocol.json').write_text(json.dumps(dict(step=step,physical_gpu=6,thinking=True,reasoning_effort='medium',decoding='greedy',max_new_tokens=8192,context=32768,rotation='original H128 identity',alpha_beta='F32',scope='Saved eligible checkpoint. Direct checkpoint mode skips repeat final CE validation by user request; packing integrity checks retained. Same10 questions as Bonsai comparison.',inputs_sha256=hashlib.sha256((OUT/'aime10-inputs.txt').read_bytes()).hexdigest()),indent=2))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8');model=OUT/'e018-final.gguf';status('exporting',step=step)
 if args.checkpoint:
  subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/export_latent_cpu.py'),str(latent),str(ROOT/'runs/e017-fresh10000/selected-ternary'),str(packed)],env=env,check=True)
 subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/export_prism_pq2.py'),str(packed),str(model),'--name',f'Vol2 E018 final selected step{step}'],env=env,check=True)
 hybrid=ROOT/'reports/e018-final-hybrid-aime10';hybrid.mkdir(exist_ok=False)
 for name in ('snapshot.json','aime10-tasks.json','aime10-inputs.txt','e017-aime10-scores.json'):shutil.copy2(OUT/name,hybrid/name)
 protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(physical_gpu=7,scope='Our402 ternary matrices/scales and H128 metadata; Bonsai449 nonternary parameters. Same F32 dtype, questions and generation settings as control.')
 (hybrid/'protocol.json').write_text(json.dumps(protocol,indent=2));hybrid_model=hybrid/'e018-bonsai-exceptions.gguf'
 status('building_hybrid',step=step)
 subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/build_bonsai_exceptions_hybrid.py'),str(model),str(ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf'),str(hybrid_model)],env=env,check=True)
 lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for gpu in ('GPU_UUID_REDACTED','GPU_UUID_REDACTED'):
  free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=100*1024,'GPU busy'
 status('testing',step=step,mode='parallel originalGPU6 hybridGPU7')
 def evaluate(folder,checkpoint,gpu):
  with (folder/'runner.log').open('w') as log:
   subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/run_e018_aime_snapshot.py'),str(folder),'--model',str(checkpoint),'--gpu',str(gpu),'--gpu-lock-fd',str(lock.fileno())],env=env,pass_fds=(lock.fileno(),),stdout=log,stderr=log,check=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(evaluate,OUT,model,6),pool.submit(evaluate,hybrid,hybrid_model,7)]
  for f in futures:f.result()
 status('completed',step=step,metrics=json.loads((OUT/'aime10-metrics.json').read_text()),hybrid_metrics=json.loads((hybrid/'aime10-metrics.json').read_text()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if OUT.exists():status('failed',error=str(e))
  raise
