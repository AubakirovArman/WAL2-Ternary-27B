"""Rebase frozen H128 latent weights to signed H1024, refit ternary scales, CPU only."""
import argparse,copy,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from ternary_core import fit_scales

def had(x,block):
 shape=x.shape;y=x.float().reshape(-1,block)
 stride=1
 while stride<block:
  z=y.reshape(-1,block//(2*stride),2,stride);a,b=z[:,:,0],z[:,:,1]
  y=torch.stack((a+b,a-b),dim=2).reshape(-1,block);stride*=2
 return (y/(block**.5)).reshape(shape)

def pack(q):
 x=(q.numpy().reshape(-1).astype(np.int16)+1).astype(np.uint8);n=len(x)
 if n%5:x=np.pad(x,(0,5-n%5),constant_values=1)
 z=x.reshape(-1,5);b=z[:,0].copy()
 for i,v in enumerate((3,9,27,81),1):b+=z[:,i]*np.uint8(v)
 for i in range(5):assert np.array_equal((b//(3**i))%3,z[:,i])
 return torch.from_numpy(b)

def signs_for(widths,seed):
 rng=np.random.default_rng(seed)
 return {w:torch.from_numpy(rng.choice(np.array([-1.,1.],dtype=np.float32),size=w)) for w in sorted(widths)}

def self_test():
 torch.set_num_threads(8);g=torch.Generator().manual_seed(19);results=[]
 for width in [5120,6144,17408]:
  signs=signs_for([width],20260927)[width];w=torch.randn(8,width,generator=g);x=torch.randn(3,width,generator=g)
  old=had(w,128);new=had(had(old,128)*signs,1024)
  restored=had(new,1024)*signs
  err=float((restored-w).norm()/w.norm());a=had(x,128)@old.T;b=had(x*signs,1024)@new.T
  linear=float((a-b).norm()/a.norm());assert err<2e-6 and linear<1e-5
  results.append(dict(width=width,inverse_relative_l2=err,linear_relative_l2=linear))
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument('latent',type=Path);p.add_argument('template',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 torch.set_num_threads(8);tests=self_test();a.output.mkdir(parents=True,exist_ok=False)
 lm=json.loads((a.latent/'manifest.json').read_text());m=copy.deepcopy(json.loads((a.template/'manifest.json').read_text()));assert m['group_size']==128
 signs=signs_for({r['shape'][1] for r in m['matrices'].values()},20260927)
 rotation=dict(block_size=1024,sign_mode='explicit',sign_seed=20260927,signs={str(w):v.to(torch.int8).tolist() for w,v in signs.items()})
 m['hadamard']=rotation;audits=[];total=len(m['matrices']);start=time.monotonic()
 for index,(name,info) in enumerate(m['matrices'].items()):
  assert info['rotation']=='hadamard128';shape=info['shape'];width=shape[1];q=torch.empty(shape,dtype=torch.int8);scales=torch.empty((shape[0]*width//128,1),dtype=torch.bfloat16);err=energy=olde=oldenergy=0.
  with safe_open(a.latent/lm[name],framework='pt',device='cpu') as f:
   ws=f.get_slice('weight');ss=f.get_slice('scales')
   for row in range(0,shape[0],128):
    w=ws[row:row+128].float();new=had(had(w,128)*signs[width],1024)
    sc=fit_scales(new);v=(new.reshape(-1,128)/sc.float()).round().clamp(-1,1).to(torch.int8);approx=(v.float()*sc.float()).reshape(new.shape)
    q[row:row+len(w)]=v.reshape(w.shape);lo=row*width//128;scales[lo:lo+len(sc)]=sc
    err+=float((new-approx).square().sum());energy+=float(new.square().sum())
    # Compare the requantized model with the CURRENT effective H128 ternary model too.
    oldsc=ss[lo:lo+len(sc)].float();oldq=((w.reshape(-1,128)/oldsc).round().clamp(-1,1)*oldsc).reshape(w.shape)
    recovered=had(had(approx,1024)*signs[width],128)
    olde+=float((recovered-oldq).square().sum());oldenergy+=float(oldq.square().sum())
  chunks=[pack(q.flatten()[j:j+5_000_000]) for j in range(0,q.numel(),5_000_000)]
  save_file({'trits':torch.cat(chunks),'scales':scales},str(a.output/info['file']));info['rotation']='hadamard1024'
  audits.append(dict(name=name,requantization_relative_l2=(err/max(energy,1e-30))**.5,relative_l2_vs_old_effective=(olde/max(oldenergy,1e-30))**.5,weights=q.numel()))
  (a.output/'conversion-progress.json').write_text(json.dumps(dict(completed=index+1,total=total,last=name,seconds=time.monotonic()-start)))
  if index%10==0:print(f'H1024 {index+1}/{total}: {name}, requant L2 {audits[-1]["requantization_relative_l2"]:.4f}',flush=True)
  del q,scales,chunks
 extras={}
 for full,shape in m['exceptions'].items():
  module,key=full.rsplit('.',1)
  with safe_open(a.latent/lm[module],framework='pt',device='cpu') as f:extras[full]=f.get_tensor(key)
  assert list(extras[full].shape)==shape
 save_file(extras,str(a.output/'exceptions.safetensors'))
 for src in a.template.iterdir():
  if src.is_file() and src.suffix!='.safetensors' and src.name!='manifest.json':shutil.copy2(src,a.output/src.name)
 m.update(latent_source=str(a.latent),export_method='H128 inverse -> own seeded signs -> H1024, CPU FP32, refit BF16 scales/group128 with8 iterations; fresh ternary projection. No training.',tensor_file_bytes=sum(f.stat().st_size for f in a.output.glob('*.safetensors')))
 m['effective_tensor_bpw']=8*m['tensor_file_bytes']/m['total_parameters']
 (a.output/'conversion-audit.json').write_text(json.dumps(dict(self_tests=tests,matrices=audits,warning='Requantization is lossy, quality must be evaluated; signs are ours, not copied from Bonsai.'),indent=2))
 (a.output/'manifest.json').write_text(json.dumps(m,indent=2));print('H1024 conversion completed',flush=True)
if __name__=='__main__':main()
