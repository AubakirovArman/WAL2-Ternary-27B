"""E013 QAT mixed-loss forward/backward timing. NO optimizer updates or saves."""
import argparse,inspect,json,statistics,time
from pathlib import Path
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fast
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
import ternary_core as core
from train_v2 import restore,write_json
from fast_ternary import enable_fused_quantizer,gradients_finite_by_device
from train_reasoning_qat import reasoning_loss
from train_kd import load_cache
from chunked_kd import loss_from_hidden

parser=argparse.ArgumentParser();parser.add_argument('--diagnostic',action='store_true');a=parser.parse_args()
core.guard();out=core.ROOT/('reports/fla-full-model-diagnostic' if a.diagnostic else 'reports/fla-full-model-probe');out.mkdir(exist_ok=False)
reference=inspect.unwrap(qwen.torch_chunk_gated_delta_rule)
model=core.load_source();core.ternarize(model,rotate=True);enable_fused_quantizer()
opt=restore(model,core.ROOT/'runs/e013-fresh-5000/candidates/step-2048');core.set_alpha(model,1.)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
rows=[json.loads(l) for l in (core.ROOT/'data/fresh-5000-v1/train.jsonl').read_text().splitlines()]
chosen=[min(rows,key=lambda r:abs(len(r['input_ids'])-n)) for n in [128,512,2048]]
meta=json.loads((core.ROOT/'data/v3-kd/manifest.json').read_text());old=core.load_examples(Path(meta['corpus']),512);byid={r['id']:r for r in old['train']};replay=max((byid[i] for i in meta['order_ids']),key=lambda r:len(r['input_ids']));cache=load_cache(replay,meta)
report={'state':'running','checkpoint':'E013 step2048','optimizer_updates':0,'cases':[],'scope':'full27B mixed CE+KD forward/backward, excludes optimizer, dataset IO and validation; no weight updates; sampled parameter gradients, not all-element identity'}
def sync():
 for i in range(2):torch.cuda.synchronize(i)
def iteration(row):
 opt.zero_grad(set_to_none=True)
 ce,_=reasoning_loss(model,row);(.5*ce).backward()
 h=model.model(input_ids=torch.tensor([replay['input_ids']],device='cuda:0'),use_cache=False).last_hidden_state[0,:-1]
 loss,_,_=loss_from_hidden(model.lm_head,h,torch.tensor(replay['labels'][1:],device=h.device),cache);(.5*loss).backward()
 return float(ce.detach()),float(loss.detach())
def samples():
 pieces=[]
 for p in model.parameters():
  if p.requires_grad:
   assert p.grad is not None
   g=p.grad.detach().flatten();pieces.append(g[::max(1,g.numel()//2048)][:2048].float().cpu())
 return torch.cat(pieces)
for row in chosen:
 result={'reasoning_id':row['id'],'reasoning_tokens':len(row['input_ids']),'replay_tokens':len(replay['input_ids'])}
 refs=None
 for label,fn in [('reference',reference),('fla',fast)]:
  qwen.torch_chunk_gated_delta_rule=fn
  print('Warmup',label,len(row['input_ids']),flush=True);iteration(row);sync()
  elapsed=[]
  for _ in range(3):
   sync();start=time.perf_counter();ce,kd=iteration(row);sync();elapsed.append(time.perf_counter()-start)
  assert gradients_finite_by_device(model.parameters())
  grads=samples();result[label]={'reasoning_ce':ce,'kd_loss':kd,'seconds':elapsed,'median_seconds':statistics.median(elapsed)}
  if refs is None:refs=grads
  else:
   result['gradient_sample_relative_l2']=float((grads-refs).norm()/refs.norm().clamp_min(1e-12))
   result['gradient_sample_cosine']=float(torch.nn.functional.cosine_similarity(grads,refs,dim=0))
  print(label,result[label],flush=True)
 result['speedup']=result['reference']['median_seconds']/result['fla']['median_seconds']
 result['ce_delta']=result['fla']['reasoning_ce']-result['reference']['reasoning_ce']
 result['kd_delta']=result['fla']['kd_loss']-result['reference']['kd_loss']
 result['passed']=abs(result['ce_delta'])<.01 and abs(result['kd_delta'])<.01 and result['gradient_sample_relative_l2']<.05
 report['cases'].append(result);write_json(out/'metrics.json',report)
 if not result['passed'] and not a.diagnostic:
  report['state']='numerical_gate_failed';write_json(out/'metrics.json',report);raise RuntimeError('Full model numerical gate failed')
report['state']='completed' if all(r['passed'] for r in report['cases']) else 'completed_numerical_gate_failed';write_json(out/'metrics.json',report);print('Completed; no optimizer updates.',flush=True)
