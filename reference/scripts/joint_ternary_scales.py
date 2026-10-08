"""E022 opt-in joint latent/scale QAT; existing trainers remain untouched.

Forward remains round(W/s).clamp(-1,1)*s in groups of128, with BF16
scales. Weight STE remains the parent trainer's identity to isolate scale
learning. Scale STE uses LSQ-style clipped derivative and 1/sqrt(128)
normalization. This is an ablation inspired by ParetoQ, not its SEQ recipe.
"""
import math
import types
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch import nn
from torch.nn import functional as F
from fast_ternary import FusedQuantizeSTE
from ternary_core import TernaryMatrix, Hadamard128


@triton.jit
def _gain_backward(W, S, BASE, GAIN, GRAD, OUT, GROUPS: tl.constexpr):
    groups = tl.program_id(0)*4 + tl.arange(0,4)
    offsets = groups[:,None]*128 + tl.arange(0,128)[None,:]
    valid = groups[:,None] < GROUPS
    w = tl.load(W+offsets,valid,other=0).to(tl.float32)
    grad = tl.load(GRAD+offsets,valid,other=0).to(tl.float32)
    s = tl.load(S+groups,groups<GROUPS,other=1).to(tl.float32)
    v = tl.div_rn(w,s[:,None])
    q = tl.minimum(tl.maximum(libdevice.nearbyint(v),-1.),1.)
    derivative = q - tl.where(tl.abs(v)<1.,v,0.)
    chain = tl.load(BASE+groups,groups<GROUPS,other=0).to(tl.float32)
    chain *= tl.exp(tl.load(GAIN+groups,groups<GROUPS,other=0))
    result = tl.sum(grad*derivative,axis=1)*chain*0.08838834764831845
    tl.store(OUT+groups,result,groups<GROUPS)


class JointQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weight,base_scales,log_gain):
        scales=(base_scales.float()*log_gain.exp()).bfloat16().contiguous()
        ctx.save_for_backward(weight,scales,base_scales,log_gain)
        return FusedQuantizeSTE.forward(ctx,weight,scales,1.)

    @staticmethod
    def backward(ctx,grad):
        weight,scales,base,gain=ctx.saved_tensors
        grad=grad.contiguous()
        gg=torch.empty_like(gain)
        with torch.cuda.device(weight.device):
            _gain_backward[(triton.cdiv(gain.numel(),4),)](
                weight,scales,base,gain,grad,gg,gain.numel(),num_warps=4)
        return grad.float(),None,gg


def joint_forward(self,x):
    if self.alpha!=1.:raise ValueError('E022 requires fully ternary forward')
    weight=JointQuantizeSTE.apply(self.weight,self.scales,self.log_gain)
    if self.embedding:
        out=F.embedding(x,weight)
        return Hadamard128.apply(out) if self.rotate else out
    x=x.to(weight.dtype)
    if self.rotate:x=Hadamard128.apply(x)
    return F.linear(x,weight,self.bias)


def install(model):
    params=[]
    for module in model.modules():
        if isinstance(module,TernaryMatrix):
            module.log_gain=nn.Parameter(torch.zeros_like(module.scales,dtype=torch.float32))
            module.forward=types.MethodType(joint_forward,module)
            params.append(module.log_gain)
    return params


def effective_scales(module):
    if hasattr(module,'log_gain'):
        return (module.scales.float()*module.log_gain.exp()).bfloat16()
    return module.scales


def check(device='cpu'):
    """Independent dense scale STE reference, plus real CUDA fused path."""
    torch.set_num_threads(2);torch.manual_seed(20261007)
    cases=[]
    for groups in (1,7,17):
        w=torch.randn(groups,128,device=device)*.03
        s=(torch.rand(groups,1,device=device)*.03+.01).bfloat16()
        gain=(torch.randn(groups,1,device=device)*.025).requires_grad_()
        g=torch.randn_like(w).bfloat16()
        unrounded=s.float()*gain.exp()
        rounded=unrounded+(unrounded.bfloat16().float()-unrounded).detach()
        v=w/rounded
        q=v.round().clamp(-1,1)
        mask=(v.abs()<1).float().detach()
        q_ste=q.detach()+(v-v.detach())*mask
        dense=q_ste*rounded
        reference=torch.autograd.grad(dense,gain,g.float())[0]/math.sqrt(128)
        expected=(g.float()*(q-v*mask)).sum(-1,keepdim=True)*unrounded/math.sqrt(128)
        assert torch.allclose(reference,expected,atol=2e-7,rtol=3e-5)
        if device!='cpu':
            w=w.detach().requires_grad_();gain=gain.detach().requires_grad_()
            actual=JointQuantizeSTE.apply(w,s,gain)
            target=(q*unrounded.bfloat16().float()).bfloat16()
            assert torch.equal(actual,target)
            gw,gg=torch.autograd.grad(actual,(w,gain),g)
            assert torch.equal(gw,g.float())
            assert torch.allclose(gg,reference,atol=2e-7,rtol=4e-5)
            z=torch.zeros_like(gain)
            assert torch.equal(JointQuantizeSTE.apply(w,s,z),FusedQuantizeSTE.apply(w,s,1.))
        cases.append(groups)
    return dict(passed=True,device=device,groups=cases,
                checks='Independent dense scale STE gradient, group normalization; CUDA: fused BF16 forward, identity weight STE, zero-gain parent equality')


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
    print(json.dumps(check(a.device),ensure_ascii=False))
