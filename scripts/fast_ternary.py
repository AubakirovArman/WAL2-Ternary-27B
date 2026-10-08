"""Opt-in fused QAT kernels. Reference implementation remains in ternary_core.py."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _quantize(W,S,O,N:tl.constexpr,ALPHA:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    w=tl.load(W+i,i<N,other=0).to(tl.float32)
    s=tl.load(S+i//128,i<N,other=1).to(tl.float32)
    q=libdevice.nearbyint(tl.div_rn(w,s))
    q=tl.minimum(tl.maximum(q,-1.),1.)*s
    # Follow ATen/native/Lerp.h, including its two numerically stable branches.
    if ALPHA<0.5:
        out=tl.fma(ALPHA,q-w,w)
    else:
        out=tl.fma(-(1.-ALPHA),q-w,q)
    tl.store(O+i,out,i<N)

class FusedQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weight,scales,alpha):
        if weight.dtype!=torch.float32 or not weight.is_cuda or not weight.is_contiguous():
            raise ValueError('Fused quantizer requires contiguous FP32 CUDA weights')
        if weight.numel()%128 or scales.numel()!=weight.numel()//128:
            raise ValueError('Expected complete groups of 128')
        if scales.dtype!=torch.bfloat16 or not scales.is_contiguous():
            raise ValueError('Expected contiguous BF16 scales')
        out=torch.empty_like(weight,dtype=torch.bfloat16)
        with torch.cuda.device(weight.device):
            _quantize[(triton.cdiv(weight.numel(),1024),)](weight,scales,out,weight.numel(),float(alpha),1024,
                                                        num_warps=4,enable_fp_fusion=False)
        return out
    @staticmethod
    def backward(ctx,grad):return grad.float(),None,None

def gradients_finite_by_device(parameters):
    flags={}
    for p in parameters:
        if p.grad is not None:
            flags.setdefault(p.grad.device,[]).append(torch.isfinite(p.grad).all())
    # Finish both device reductions before the two host checks.
    reduced=[torch.stack(v).all() for v in flags.values()]
    return all(bool(x) for x in reduced)

def enable_fused_quantizer():
    import ternary_core
    reference=ternary_core.QuantizeSTE
    class FinalStageQuantizeSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx,weight,scales,alpha):
            # Avoid compiling hundreds of alpha-specialized kernels during the short ramp.
            if alpha!=1.0:return reference.forward(ctx,weight,scales,alpha)
            return FusedQuantizeSTE.forward(ctx,weight,scales,alpha)
        @staticmethod
        def backward(ctx,grad):return grad.float(),None,None
    ternary_core.QuantizeSTE=FinalStageQuantizeSTE
