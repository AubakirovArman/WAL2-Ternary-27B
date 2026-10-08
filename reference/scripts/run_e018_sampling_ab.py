"""Queue matched sampling tests after current greedy experiments finish; GPU0 only."""
import json,os,subprocess,time,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/e018-step21504-sampling'
def write(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def main():
 OUT.mkdir(exist_ok=False);source=ROOT/'reports/e018-step21504-alpha-ab/bf16';jobs=[]
 for rotation in ('h128','h1024'):
  base=ROOT/('reports/e018-step21504-alpha-ab' if rotation=='h128' else 'reports/e018-step21504-h1024')
  for precision in ('bf16','f32'):
   label=rotation+'-'+precision;out=OUT/label;out.mkdir();model=base/(precision+'.gguf')
   for name in ('snapshot.json','aime10-tasks.json','aime10-inputs.txt','e017-aime10-scores.json'):shutil.copy2(source/name,out/name)
   (out/'protocol.json').write_text(json.dumps(dict(step=21504,rotation=rotation,alpha_beta=precision,temperature=1.0,top_k=20,top_p=.95,seed_base=20260927,per_question_seed='20260927 + frozen task id',thinking=True,max_new_tokens=8192,context=32768,physical_gpu=0,model=str(model),binary_sha256=hashlib.sha256((ROOT/'tools/bonsai-runtime/prism-sampled-eval').read_bytes()).hexdigest(),scope='Same10 fixed questions and weights as greedy; one seeded sample per question, not multi-seed accuracy estimate'),indent=2));jobs.append((label,out,model))
 write('status.json',dict(state='waiting',reason='Waiting for ongoing greedy AB tests'))
 while any(subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip() in ('active','activating') for unit in ('vol2-e018-alpha-ab21504.service','vol2-e018-h1024-21504.service')):time.sleep(15)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',VOL2_SAMPLE_SEED='20260927');results={};errors={}
 for label,out,model in jobs:
  write('status.json',dict(state='running',current=label,finished=list(results),errors=errors))
  try:
   if not model.is_file():raise RuntimeError('Model missing: '+str(model))
   with (out/'runner.log').open('w') as log:
    subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/run_e018_aime_snapshot.py'),str(out),'--model',str(model),'--sampled'],env=env,stdout=log,stderr=log,check=True)
   results[label]=json.loads((out/'aime10-metrics.json').read_text());write('results.json',results)
  except Exception as e:errors[label]=str(e)
 write('status.json',dict(state='failed' if errors else 'completed',finished=list(results),errors=errors))
 if errors:raise RuntimeError(str(errors))
if __name__=='__main__':main()
