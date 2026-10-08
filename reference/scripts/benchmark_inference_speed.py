"""Paired inference timings and token identity on E012; no weight updates."""
import json,time,torch
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import guard,SOURCE
from inspect_checkpoint import load_packed
from cached_generation import generate_ids
from fast_inference import enable

guard();out=Path('reports/e012-inference-speed');out.mkdir(exist_ok=False)
model=load_packed('runs/e012-fresh-reasoning/selected-ternary')
tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
prompts=['Объясни, почему сумма двух нечётных чисел чётная.','Write a Python function that returns the longest consecutive run in a list of integers.']
ids=[tok.apply_chat_template([{'role':'user','content':s}],tokenize=True,add_generation_prompt=True,enable_thinking=True) for s in prompts]
ids=[x if isinstance(x,list) else x['input_ids'] for x in ids]
results={};reference_logits=[]
with torch.inference_mode():
 for variant in ('reference_split','fused_split','fused_single'):
  if variant=='fused_split':enable()
  if variant=='fused_single':
   for module in model.modules():
    module._forward_pre_hooks.clear();module._forward_pre_hooks_with_kwargs.clear()
   model.to('cuda:0');torch.cuda.empty_cache()
  # Untimed warm-up includes JIT compilation and first forward.
  generate_ids(model,ids[0],2,set())
  rows=[]
  for j,prompt in enumerate(ids):
   logits=model.lm_head(model.model(input_ids=torch.tensor([prompt],device='cuda:0'),use_cache=False).last_hidden_state[:,-1]).float().cpu()
   if variant=='reference_split':reference_logits.append(logits)
   assert torch.equal(logits,reference_logits[j]),(variant,j,float((logits-reference_logits[j]).abs().max()))
   r=generate_ids(model,prompt,64,set())
   if variant!='reference_split':assert r['token_ids']==results['reference_split'][j]['token_ids'],(variant,j)
   r['tokens_per_second']=len(r['token_ids'])/r['seconds'];rows.append(r)
  results[variant]=rows
  (out/'results.json').write_text(json.dumps(results,indent=2))
  print(variant,[(round(r['tokens_per_second'],2),round(r['seconds'],2)) for r in rows],flush=True)
print('PASS: logits and generated token IDs exactly match in all variants',flush=True)
(out/'PASSED').write_text('Exact prompt logits and 128 generated tokens match across all three variants. Limited correctness check, not full benchmark.\n')
