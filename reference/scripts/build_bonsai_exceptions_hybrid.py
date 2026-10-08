"""Hybrid: our PQ2 matrices and H128 contract; Bonsai nonternary values, kept F32."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/prism-source/gguf-py'))
import gguf

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ours,donor,out=map(Path,sys.argv[1:]);assert not out.exists()
 a=gguf.GGUFReader(ours);b=gguf.GGUFReader(donor);x={t.name:t for t in a.tensors};y={t.name:t for t in b.tensors};assert x.keys()==y.keys()
 assert a.fields['prism.hadamard.block_size'].contents()==128 and a.fields['prism.hadamard.sign_mode'].contents()=='identity'
 temp=out.with_suffix('.partial');w=gguf.GGUFWriter(temp,'qwen35',use_temp_file=True)
 for key,v in a.fields.items():
  if key.startswith('GGUF.') or key=='general.name':continue
  w.add_key_value(key,v.contents(),v.types[0],v.types[-1] if v.types[0]==gguf.GGUFValueType.ARRAY else None)
 w.add_name('E018 ternary matrices + Bonsai high precision parameters, experimental hybrid')
 ter=[];exceptions=[]
 for name,t in x.items():
  u=y[name];assert np.array_equal(t.shape,u.shape)
  if t.tensor_type==gguf.GGMLQuantizationType.PQ2_0:
   assert u.tensor_type==t.tensor_type;w.add_tensor(name,np.ascontiguousarray(t.data),raw_dtype=t.tensor_type);ter.append(name)
  else:
   assert t.tensor_type==gguf.GGMLQuantizationType.F32 and u.tensor_type in (gguf.GGMLQuantizationType.BF16,gguf.GGMLQuantizationType.F32)
   v=np.ascontiguousarray(gguf.quants.dequantize(u.data,u.tensor_type),dtype=np.float32);assert np.isfinite(v).all()
   w.add_tensor(name,v);exceptions.append(name)
 assert len(ter)==402 and len(exceptions)==449
 w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file(progress=True);w.close();temp.rename(out)
 c=gguf.GGUFReader(out);z={t.name:t for t in c.tensors};changed=0
 for name,t in z.items():
  assert np.array_equal(t.shape,x[name].shape) and t.tensor_type==x[name].tensor_type
  if name in ter:assert np.array_equal(t.data,x[name].data)
  else:
   assert np.array_equal(t.data,gguf.quants.dequantize(y[name].data,y[name].tensor_type));changed+=int(not np.array_equal(t.data,x[name].data))
 for key,v in a.fields.items():
  if key.startswith('GGUF.') or key=='general.name':continue
  assert v.contents()==c.fields[key].contents(),key
 report=dict(passed=True,our_model=str(ours),donor=str(donor),ours_sha256=sha(ours),donor_sha256=sha(donor),output_sha256=sha(out),ternary402_byte_identical=True,all_metadata_except_name_identical=True,highprecision449_from_bonsai=True,highprecision_dtype='F32 in both models; donor BF16 converted losslessly',numerically_changed_highprecision_tensors=changed,scope='Hybrid changes only nonternary parameter values. Not a pure runtime parity test; quality may change due to coadaptation between weights.')
 out.with_suffix('.hybrid-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
