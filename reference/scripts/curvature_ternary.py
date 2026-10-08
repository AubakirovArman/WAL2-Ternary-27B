"""Fixed-scale ternary error compensation using input covariance.

Experimental primitive, not connected to training/evaluation jobs.
Algorithmic reference: GPTQ, Frantar et al., https://arxiv.org/abs/2210.17323.
Our variant keeps the existing contiguous group scales and {-1,0,1} alphabet.
"""
import json
from pathlib import Path
import torch


@torch.no_grad()
def quantize(weight,covariance,scales,group=128,block=128,damping=0.01):
    if weight.dtype not in (torch.float32,torch.float64):raise ValueError('FP32/64 working weights required')
    rows,cols=weight.shape
    if cols%group or scales.numel()!=rows*(cols//group):raise ValueError('Invalid group scales')
    if covariance.shape!=(cols,cols) or block<1 or damping<=0:raise ValueError('Invalid covariance/config')
    if not torch.isfinite(weight).all() or not torch.isfinite(covariance).all():raise ValueError('Nonfinite inputs')
    if not (torch.isfinite(scales)&(scales>0)).all():raise ValueError('Invalid scales')
    h=covariance.to(device=weight.device,dtype=weight.dtype)
    h=(h+h.T)/2
    diagonal=h.diagonal();ridge=(damping*diagonal.mean()).clamp_min(torch.finfo(weight.dtype).eps)
    h=h.clone();h.diagonal().add_(ridge)
    chol=torch.linalg.cholesky(h)
    inverse_upper=torch.linalg.cholesky(torch.cholesky_inverse(chol),upper=True)
    work=weight.clone();trits=torch.empty_like(weight,dtype=torch.int8)
    scale=scales.reshape(rows,-1).to(device=weight.device,dtype=weight.dtype)
    for start in range(0,cols,block):
        end=min(start+block,cols)
        tile=work[:,start:end].clone();errors=torch.empty_like(tile)
        for offset in range(end-start):
            index=start+offset
            s=scale[:,index//group]
            q=(tile[:,offset]/s).round().clamp(-1,1)
            trits[:,index]=q.to(torch.int8)
            e=(tile[:,offset]-q*s)/inverse_upper[index,index]
            errors[:,offset]=e
            tile[:,offset:]-=e[:,None]*inverse_upper[index,index:end][None,:]
        if end<cols:
            work[:,end:]-=errors@inverse_upper[start:end,end:]
    return trits


def check():
    from analyze_ternary_geometry import exact_scales
    torch.set_num_threads(4);torch.manual_seed(3821)
    w=torch.randn(12,128,dtype=torch.float64)
    s=exact_scales(w.reshape(-1,128)).bfloat16().double()
    rounded=(w/s).round().clamp(-1,1).to(torch.int8)
    assert torch.equal(quantize(w,torch.eye(128,dtype=w.dtype),s,block=32),rounded)
    a=torch.randn(128,128,dtype=w.dtype);h=a@a.T/128
    scalar=quantize(w,h,s,block=1)
    assert torch.equal(scalar,quantize(w,h,s,block=32))
    # Independently solve the first-coordinate constrained quadratic optimum.
    hd=h+torch.eye(128,dtype=w.dtype)*(.01*h.diagonal().mean())
    upper=torch.linalg.cholesky(torch.linalg.inv(hd),upper=True)
    error=w[:,0]-rounded[:,0]*s[:,0]
    compensated=w[:,1:]-error[:,None]*upper[0,1:]/upper[0,0]
    direct=w[:,1:]+error[:,None]*torch.linalg.solve(hd[1:,1:],hd[1:,0])[None,:]
    assert torch.allclose(compensated,direct,atol=1e-10,rtol=1e-10)
    results=[]
    for seed in range(8):
        gen=torch.Generator().manual_seed(seed+483)
        mix=torch.randn(16,128,generator=gen,dtype=torch.float64)/4
        train=torch.randn(1024,16,generator=gen,dtype=torch.float64)@mix+.1*torch.randn(1024,128,generator=gen,dtype=torch.float64)
        test=torch.randn(2048,16,generator=gen,dtype=torch.float64)@mix+.1*torch.randn(2048,128,generator=gen,dtype=torch.float64)
        weight=torch.randn(32,128,generator=gen,dtype=torch.float64)
        scale=exact_scales(weight).bfloat16().double()
        plain=(weight/scale).round().clamp(-1,1)*scale
        q=quantize(weight,train.T@train/len(train),scale,block=32)
        adjusted=q.double()*scale
        base=float(((test@(weight-plain).T)**2).mean())
        after=float(((test@(weight-adjusted).T)**2).mean())
        results.append({'seed':seed,'plain_holdout_output_mse':base,'compensated_holdout_output_mse':after,'ratio':after/base})
    report={'scope':'Synthetic correlated-input linear layers only; NOT 27B model quality',
            'identity_covariance_matches_rounding':True,'blocked_matches_scalar':True,
            'first_coordinate_matches_constrained_solve':True,'cases':results}
    path=Path(__file__).resolve().parents[1]/'reports/curvature-ternary-cpu-check.json'
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':check()
