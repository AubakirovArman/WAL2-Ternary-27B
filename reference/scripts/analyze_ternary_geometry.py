"""CPU-only source-weight diagnostic; no model mutations or quality claims."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]

def hadamard(x, block):
    shape=x.shape; y=x.float().reshape(-1,block)
    stride=1
    while stride<block:
        z=y.reshape(-1,block//(2*stride),2,stride)
        a,b=z[:,:,0],z[:,:,1]
        y=torch.stack((a+b,a-b),dim=2).reshape(-1,block)
        stride*=2
    return (y/block**0.5).reshape(shape)

def iterative_scales(w):
    s=(w.abs().mean(-1,keepdim=True)*1.3).clamp_min(1e-8)
    for _ in range(8):
        q=(w/s).round().clamp(-1,1)
        s=((w*q).sum(-1,keepdim=True)/q.square().sum(-1,keepdim=True).clamp_min(1)).clamp_min(1e-8)
    return s

def exact_scales(w):
    # For k nonzero trits, the best support is the k largest magnitudes.
    # Optimal scale=sum(abs(w_support))/k; maximize saved energy=sum^2/k.
    prefix=w.abs().sort(dim=-1,descending=True).values.cumsum(-1)
    count=torch.arange(1,w.shape[-1]+1,dtype=w.dtype)
    index=(prefix.square()/count).argmax(-1,keepdim=True)
    return (prefix.gather(-1,index)/(index+1)).clamp_min(1e-8)

def error(w,s):
    return ((w-(w/s).round().clamp(-1,1)*s)**2).sum(-1)

def self_check():
    torch.manual_seed(219)
    w=torch.randn(100,128)
    assert torch.all(error(w,exact_scales(w))<=error(w,iterative_scales(w))+1e-4)
    x=torch.randn(3,2048)
    for block in (128,1024):
        assert torch.allclose(hadamard(hadamard(x,block),block),x,atol=1e-6)
    # Exhaustive supports for a small group independently verify the derivation.
    import itertools
    for w in torch.randn(10,5):
        best=min(float(((w-q*(w*q).sum()/q.square().sum().clamp_min(1))**2).sum())
                 for q in (torch.tensor(v,dtype=w.dtype) for v in itertools.product((-1,0,1),repeat=5)))
        assert abs(float(error(w[None],exact_scales(w[None])))-best)<1e-5

def main():
    torch.set_num_threads(4);self_check()
    source=ROOT/'models/qwen3.8-27b-fp8-source'
    index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    matrices=json.loads((ROOT/'runs/e005-data-lr/selected-ternary/manifest.json').read_text())['matrices']
    gen=torch.Generator().manual_seed(20260919)
    signs=torch.randint(0,2,(1024,),generator=gen).float()*2-1
    report={'scope':'Sampled source weight MSE, NOT downstream model quality; no trained weights changed',
            'rows_per_matrix':32,'seed':20260919,'matrices':[]}
    for name,spec in matrices.items():
        key=name+'.weight'
        if key.startswith('model.'):key=key.replace('model.','model.language_model.',1)
        rows,cols=spec['shape']
        selected=torch.linspace(0,rows-1,32).long().unique().tolist()
        with safe_open(source/index[key],framework='pt',device='cpu') as f:
            view=f.get_slice(key)
            w=torch.cat([view[r:r+1].float() for r in selected])
            scale_key=key.replace('.weight','.weight_scale_inv')
            if scale_key in f.keys():
                scales=f.get_tensor(scale_key).float()[torch.tensor(selected)//128]
                w=(w.reshape(len(selected),cols//128,128)*scales[:,:,None]).reshape(len(selected),cols)
                w=w.bfloat16().float()
        results={}
        for mode,block,signed in [('h128',128,False),('h1024',1024,False),('signed_h1024',1024,True)]:
            assert cols%block==0
            v=w if not signed else (w.reshape(-1,1024)*signs).reshape_as(w)
            rotated=hadamard(v,block).reshape(-1,128)
            energy=float(rotated.square().sum())
            for method,fit in [('iterative',iterative_scales),('exact',exact_scales)]:
                scale=fit(rotated).bfloat16().float()
                results[mode+'_'+method]=float(error(rotated,scale).sum())/energy
        row={'name':name,'shape':spec['shape'],'sampled_rows':selected,'relative_mse':results}
        report['matrices'].append(row)
        print(json.dumps(row),flush=True)
    report['mean_matrix_relative_mse']={k:sum(r['relative_mse'][k] for r in report['matrices'])/len(matrices) for k in results}
    (ROOT/'reports/ternary-geometry.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['mean_matrix_relative_mse']),flush=True)

if __name__=='__main__':main()
