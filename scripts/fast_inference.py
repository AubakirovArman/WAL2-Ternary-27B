"""Opt-in inference-only Hadamard fusion; weights and training code unchanged."""
import torch
import triton
import triton.language as tl

@triton.jit
def _hadamard(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B)
    offset=tl.program_id(0)*B+i
    y=tl.load(X+offset,offset<N,other=0).to(tl.float32)
    for k in tl.static_range(7):
        stride=1<<k
        partner=tl.gather(y,i^stride,axis=0)
        y=tl.where((i&stride)==0,y+partner,partner-y)
    y=y*0.08838834764831845
    tl.store(Y+offset,y,offset<N)

def hadamard(x):
    if not x.is_cuda or x.numel()%128:raise ValueError('Expected CUDA groups of 128')
    if torch.is_grad_enabled() and x.requires_grad:raise RuntimeError('Inference only')
    x=x.contiguous();y=torch.empty_like(x)
    with torch.cuda.device(x.device):
        _hadamard[(triton.cdiv(x.numel(),128),)](x,y,x.numel(),128,num_warps=4,enable_fp_fusion=False)
    return y

class FastHadamard:
    apply=staticmethod(hadamard)

def enable():
    import inspect_checkpoint
    inspect_checkpoint.Hadamard128=FastHadamard
