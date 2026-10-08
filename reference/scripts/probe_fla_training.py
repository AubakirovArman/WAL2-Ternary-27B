"""Isolated GPU7 GatedDeltaNet forward/backward microbenchmark, no weight updates."""
import inspect,json,os,statistics,time
from pathlib import Path
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fast
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
reference=inspect.unwrap(torch_chunk_gated_delta_rule)
assert reference is not torch_chunk_gated_delta_rule
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU_UUID_REDACTED'
torch.set_num_threads(8);torch.manual_seed(20260921)
assert torch.cuda.mem_get_info()[0]>100*2**30
out=Path('reports/fla-training-probe');out.mkdir(exist_ok=False)
report={'state':'running','gpu':'physical7 H200','torch':torch.__version__,'fla':'0.5.2','shapes':[],'scope':'GatedDeltaNet core only, Q/K/V 48 heads x128, no projections, no QAT/optimizer, not full training speed'}
def save():(out/'metrics.json').write_text(json.dumps(report,indent=2))
def difference(a,b):
 x=a.detach().float();y=b.detach().float();d=x-y
 return {'relative_l2':float(d.norm()/x.norm().clamp_min(1e-12)),'max_abs':float(d.abs().max()),'finite':bool(torch.isfinite(y).all())}
for length in [512,513,2048]:
 print('Testing length',length,flush=True)
 args=[torch.randn(1,length,48,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
 args += [-torch.rand(1,length,48,device='cuda')*.2,torch.rand(1,length,48,device='cuda',dtype=torch.bfloat16)]
 grad=torch.randn_like(args[2]);outputs=[];gradients=[];times={}
 for label,fn in [('reference',reference),('fla',fast)]:
  xs=[x.detach().clone().requires_grad_(True) for x in args]
  def iteration():
   for x in xs:x.grad=None
   y,_=fn(*xs,use_qk_l2norm_in_kernel=True,output_final_state=False)
   y.backward(grad);return y
  print('Warmup',label,flush=True)
  for _ in range(2):y=iteration()
  torch.cuda.synchronize();outputs.append(y.detach().clone());gradients.append([x.grad.detach().clone() for x in xs]);elapsed=[]
  for _ in range(5):
   torch.cuda.synchronize();start=time.perf_counter();iteration();torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
  times[label]={'median_seconds':statistics.median(elapsed),'samples':elapsed};del xs
 row={'length':length,'output':difference(*outputs),'gradients':{k:difference(a,b) for k,a,b in zip(['q','k','v','g','beta'],*gradients)},'timing':times,'speedup':times['reference']['median_seconds']/times['fla']['median_seconds']}
 row['passed']=row['output']['relative_l2']<.02 and all(x['finite'] and x['relative_l2']<.05 for x in row['gradients'].values())
 report['shapes'].append(row);save();print(json.dumps(row),flush=True)
 if not row['passed']:raise RuntimeError('Numerical tolerance failed')
report['state']='completed';save()
