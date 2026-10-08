"""CPU-only export of a frozen local latent checkpoint using a compatible packed template."""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from ternary_core import pack_trits,unpack_trits

def main():
 p=argparse.ArgumentParser();p.add_argument('latent',type=Path);p.add_argument('template',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 torch.set_num_threads(8);a.output.mkdir(parents=True,exist_ok=False)
 lm=json.loads((a.latent/'manifest.json').read_text());m=copy.deepcopy(json.loads((a.template/'manifest.json').read_text()))
 assert m['group_size']==128
 for i,(name,info) in enumerate(m['matrices'].items()):
  with safe_open(a.latent/lm[name],framework='pt',device='cpu') as f:
   w=f.get_tensor('weight');sc=f.get_tensor('scales');assert list(w.shape)==info['shape']
   assert info['rotation']=='hadamard128'
   flat=w.reshape(-1,128);q=torch.empty(w.numel(),dtype=torch.int8).reshape(-1,128)
   for start in range(0,len(flat),32768):q[start:start+32768]=(flat[start:start+32768]/sc[start:start+32768].float()).round().clamp(-1,1).to(torch.int8)
   q=q.flatten();chunks=[]
   for start in range(0,len(q),5_000_000):
    part=q[start:start+5_000_000];packed=pack_trits(part)
    assert torch.equal(unpack_trits(packed,part.numel()),part);chunks.append(packed)
   tensors={'trits':torch.cat(chunks),'scales':sc}
   save_file(tensors,str(a.output/info['file']))
  del w,sc,flat,q,chunks,tensors
  if i%20==0:print(f'CPU export: {i+1}/{len(m["matrices"])}',flush=True)
 extras={}
 for full,shape in m['exceptions'].items():
  name,key=full.rsplit('.',1)
  with safe_open(a.latent/lm[name],framework='pt',device='cpu') as f:extras[full]=f.get_tensor(key)
  assert list(extras[full].shape)==shape
 save_file(extras,str(a.output/'exceptions.safetensors'))
 for src in a.template.iterdir():
  if src.is_file() and src.suffix!='.safetensors' and src.name!='manifest.json':shutil.copy2(src,a.output/src.name)
 m.update(latent_source=str(a.latent),latent_manifest_sha256=hashlib.sha256((a.latent/'manifest.json').read_bytes()).hexdigest(),export_method='CPU FP32 latent/scales round-clamp, full trit packing roundtrip; no model loading on GPU')
 m['tensor_file_bytes']=sum(p.stat().st_size for p in a.output.glob('*.safetensors'));m['effective_tensor_bpw']=8*m['tensor_file_bytes']/m['total_parameters']
 (a.output/'manifest.json').write_text(json.dumps(m,indent=2));print('CPU export complete',flush=True)
if __name__=='__main__':main()
