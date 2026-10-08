import sys,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/prism-source/gguf-py'))
import gguf
base,variant,out=map(Path,sys.argv[1:]);a=gguf.GGUFReader(base);b=gguf.GGUFReader(variant)
x={t.name:t for t in a.tensors};y={t.name:t for t in b.tensors};assert x.keys()==y.keys();changed=[]
for name,t in x.items():
 u=y[name];assert np.array_equal(t.shape,u.shape)
 if re.fullmatch(r'blk\.\d+\.ssm_(alpha|beta)\.weight',name):
  assert t.tensor_type==gguf.GGMLQuantizationType.F32 and u.tensor_type==gguf.GGMLQuantizationType.BF16
  assert np.array_equal(t.data,gguf.quants.dequantize(u.data,u.tensor_type));changed.append(name)
 else:assert t.tensor_type==u.tensor_type and np.array_equal(t.data,u.data),name
assert len(changed)==96
out.write_text(json.dumps(dict(passed=True,changed_dtype_tensors=changed,count=96,alpha_beta_values_exact=True,other755_tensors_byte_identical=True),indent=2));print('Passed:96 dtype changes, same values, other755 tensors identical',flush=True)
