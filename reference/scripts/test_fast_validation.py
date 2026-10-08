"""Checks exact eval results, cache lifetime, exceptions, and subsequent gradients."""
import argparse,json,os
from pathlib import Path
import torch
from torch import nn
import ternary_core as core
from fast_validation import validation_runtime
p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');a=p.parse_args()
if a.cuda:
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==core.GPU_IDS[1], 'Only physical GPU7 for small isolated check'
 assert torch.cuda.mem_get_info()[0]>100*2**30
 device='cuda:0'
else:device='cpu'
torch.set_num_threads(2);torch.manual_seed(471)
m=core.TernaryMatrix(nn.Linear(128,128,bias=False,device=device,dtype=torch.bfloat16),rotate=True)
x=torch.randn(7,128,device=device,dtype=torch.bfloat16)
weights=m.weight.detach().clone();params=list(m.parameters());reference_h=core.Hadamard128
with torch.no_grad():
 expected=m(x)
 with validation_runtime(m,True,cache_gib=.01,reserve_gib=1,fused=a.cuda) as stats:
  assert stats['cached_matrices']==1
  assert torch.equal(m(x),expected)
 assert 'forward' not in m.__dict__ and core.Hadamard128 is reference_h
 with validation_runtime(m,True,cache_gib=0,fused=a.cuda) as stats0:
  assert stats0['cached_matrices']==0 and torch.equal(m(x),expected)
 try:
  with validation_runtime(m,True,cache_gib=.01,reserve_gib=1,fused=a.cuda):raise ValueError('test cleanup')
 except ValueError:pass
 assert 'forward' not in m.__dict__ and core.Hadamard128 is reference_h
assert all(x is y for x,y in zip(params,m.parameters())) and torch.equal(weights,m.weight)
m(x).float().sum().backward();assert m.weight.grad is not None and torch.isfinite(m.weight.grad).all()
try:
 with validation_runtime(m,True):pass
 raise AssertionError('grad-enabled validation accepted')
except RuntimeError:pass
result={'passed':True,'device':device,'fused':a.cuda,'checks':['exact outputs','zero-budget fallback','exception cleanup','parameter identity','weights unchanged','training gradients after cleanup','reject grad-enabled cache']}
Path('reports/fast-validation-'+('gpu7' if a.cuda else 'cpu')+'.json').write_text(json.dumps(result,indent=2));print(result)
