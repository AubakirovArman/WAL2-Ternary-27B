"""Matched E013 BF16/native checks on physical GPU6 only; GPU7 reserved by guard."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from ternary_core import guard,SOURCE,ROOT
from inspect_checkpoint import load_packed
from fast_inference import enable
from cached_generation import generate_ids
OUT=ROOT/'reports/e013-native';OUT.mkdir(exist_ok=True)
TOOL=ROOT/'tools/bonsai-runtime';BACK=TOOL/'llama-prism-b10709-9a9394a'
MODEL=ROOT/'models/vol2-e013-prism/vol2-e013-pq2.gguf'
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','native']);a=p.parse_args();guard()
 tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
 prompts=[('Объясни, почему сумма двух нечётных чисел чётная.',True),('Write a Python function that returns the longest consecutive run in a list of integers.',True),('Вычисли 37 * 24 - 19. Ответь только целым числом.',False),('Return only a JSON object with keys city and count, with values Paris and 3.',False),('Напиши функцию Python, которая проверяет, является ли строка палиндромом.',True),('Summarize in one sentence: Bees pollinate flowering plants. This helps plants produce seeds and fruit.',False),('Отсортируй числа [9, -2, 5, 0] по возрастанию. Только JSON-массив.',False),('What does this Python print? print(sum(x*x for x in range(5)))',True)]
 jobs=[]
 for i,(prompt,thinking) in enumerate(prompts):
  ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=thinking)
  if not isinstance(ids,list):ids=ids['input_ids']
  jobs.append(dict(id=i,prompt=prompt,input_ids=ids))
 validation=[json.loads(l) for l in (ROOT/'data/fresh-5000-v1/validation.jsonl').read_text().splitlines()]
 if a.stage=='reference':
  enable();model=load_packed(ROOT/'runs/e013-fresh-5000/selected-ternary',single_device='cuda:0')
  stop={tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')};results=[];logits=[]
  with torch.inference_mode():
   generate_ids(model,jobs[0]['input_ids'],8,stop)
   for j in jobs:
    ids=j['input_ids'];l=model.lm_head(model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False).last_hidden_state[:,-1]).float().cpu().numpy()[0];logits.append(l)
    g=generate_ids(model,ids,128,stop);results.append(dict(j,**g,text=tok.decode(g['token_ids'],skip_special_tokens=False)))
    print('reference',j['id'],len(g['token_ids'])/g['seconds'],flush=True)
  np.save(OUT/'reference-logits.npy',np.stack(logits));(OUT/'reference.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));return
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=env['CUDA_VISIBLE_DEVICES'].split(',')[0]
 env['LD_LIBRARY_PATH']=':'.join(map(str,[BACK,TOOL/'cuda-deps/nvidia/cuda_runtime/lib',TOOL/'cuda-deps/nvidia/cublas/lib']))
 def run(label,lines):
  inp=OUT/(label+'.txt');inp.write_text('\n'.join(lines)+'\n')
  with (OUT/(label+'.jsonl')).open('w') as stdout,(OUT/(label+'.log')).open('w') as stderr:
   subprocess.run([str(TOOL/'prism-native-eval'),str(MODEL),str(inp),str(BACK),'128','2048','512'],env=env,stdout=stdout,stderr=stderr,check=True,timeout=3600)
  return [json.loads(l) for l in (OUT/(label+'.jsonl')).read_text().splitlines()]
 def line(mode,i,ids,p=None):return f'{mode} {i} {p or len(ids)} {len(ids)} '+' '.join(map(str,ids))
 lines=[]
 # Untimed warmup response is discarded. All eight timed prompts identical to reference.
 lines.append(line('G',-1,jobs[0]['input_ids']))
 for j in jobs:
  lines.extend([line('L',j['id'],j['input_ids']),line('G',j['id'],j['input_ids'])])
 rows=run('native-generation',lines);ref=np.load(OUT/'reference-logits.npy');checks=[]
 for r in rows:
  if r['kind']!='logits':continue
  x=torch.tensor(ref[r['id']],dtype=torch.float64);y=torch.tensor(r['logits'],dtype=torch.float64)
  kl=float((x.softmax(0)*(x.log_softmax(0)-y.log_softmax(0))).sum())
  checks.append(dict(id=r['id'],kl=kl,top1_equal=int(x.argmax())==int(y.argmax()),max_abs=float((x-y).abs().max())))
 gens=[dict(r,text=tok.decode(r['tokens'],skip_special_tokens=False)) for r in rows if r['kind']=='generation' and r['id']>=0]
 (OUT/'native-generation-readable.json').write_text(json.dumps(gens,ensure_ascii=False,indent=2))
 (OUT/'logit-check.json').write_text(json.dumps(checks,indent=2));print('logit checks',checks,flush=True)
 if max(r['kl'] for r in checks)>.05:raise ValueError('Large logits discrepancy; inspect export before CE')
 ce_lines=[]
 for i,r in enumerate(validation):
  ids=r['input_ids'];p=r['prompt_tokens'];text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
  assert tok.encode(text,add_special_tokens=False)==ids
  ce_lines.extend([line('T',i,ids,p)+' '+text.encode().hex(),line('E',i,ids,p)])
 ce=run('native-ce120',ce_lines)
 assert len(ce)==120 and sum(r['tokens'] for r in ce)==69641
 score=sum(r['nll'] for r in ce)/69641
 baseline=json.loads((ROOT/'runs/e013-fresh-5000/metrics.json').read_text())['selected_reasoning']['reasoning_and_answer_ce']
 refg=json.loads((OUT/'reference.json').read_text());ref_rate=sum(len(r['token_ids']) for r in refg)/sum(r['seconds'] for r in refg)
 native_rate=sum(len(r['tokens']) for r in gens)/sum(r['seconds'] for r in gens)
 report=dict(state='completed',native_ce=score,reference_ce=baseline,ce_delta=score-baseline,reference_tokens_per_second=ref_rate,native_tokens_per_second=native_rate,speedup=native_rate/ref_rate,logit_checks=checks,native_tokens=sum(len(r['tokens']) for r in gens),reference_tokens=sum(len(r['token_ids']) for r in refg),scope='8 prompts, up to128 greedy tokens; speed includes prefill; full frozen CE120 and tokenizer identity; not full benchmark')
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
