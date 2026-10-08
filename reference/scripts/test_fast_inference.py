import json,torch
from ternary_core import guard,Hadamard128
from fast_inference import hadamard
from pathlib import Path
import triton

guard();results=[]
with torch.inference_mode():
 for device in ('cuda:0','cuda:1'):
  for shape in ((1,5120),(1,17408),(17,5120),(128,5120)):
   x=torch.randn(shape,device=device,dtype=torch.bfloat16)
   ref=Hadamard128.apply(x);got=hadamard(x)
   assert torch.equal(ref,got),(device,shape,float((ref-got).abs().max()))
   a=triton.testing.do_bench(lambda:Hadamard128.apply(x),warmup=50,rep=100)
   b=triton.testing.do_bench(lambda:hadamard(x),warmup=50,rep=100)
   results.append(dict(device=device,shape=shape,exact=True,reference_ms=a,fused_ms=b,speedup=a/b))
Path('reports/fast-hadamard-check.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
