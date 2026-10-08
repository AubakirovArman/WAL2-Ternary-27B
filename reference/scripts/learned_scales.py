"""Opt-in fixed-trit, learned group-scale recovery. Does not alter existing trainers."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from ternary_core import Hadamard128


class ScaledTrits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trits, base_scales, log_gain):
        ctx.save_for_backward(trits,base_scales,log_gain)
        flat=trits.reshape(-1,128)
        output=torch.empty_like(flat,dtype=torch.bfloat16)
        for start in range(0,len(flat),32768):
            end=start+32768
            scale=(base_scales[start:end].float()*log_gain[start:end].exp()).bfloat16()
            output[start:end]=flat[start:end].to(torch.bfloat16)*scale
        return output.reshape_as(trits)

    @staticmethod
    def backward(ctx, gradient):
        trits,base,log_gain=ctx.saved_tensors
        q=trits.reshape(-1,128);g=gradient.reshape(-1,128)
        result=torch.empty_like(log_gain)
        # STE only through BF16 scale rounding; trits remain exactly fixed.
        for start in range(0,len(q),32768):
            end=start+32768
            scale=base[start:end].float()*log_gain[start:end].exp()
            result[start:end]=(g[start:end].float()*q[start:end].float()).sum(-1,keepdim=True)*scale
        return None,None,result


class ScaleTunedMatrix(nn.Module):
    def __init__(self,trits,scales,embedding,rotate):
        super().__init__()
        self.register_buffer('trits',trits)
        self.register_buffer('base_scales',scales)
        self.log_gain=nn.Parameter(torch.zeros_like(scales,dtype=torch.float32))
        self.embedding=embedding;self.rotate=rotate

    def forward(self,x):
        w=ScaledTrits.apply(self.trits,self.base_scales,self.log_gain)
        if self.embedding:
            y=F.embedding(x,w)
            return Hadamard128.apply(y) if self.rotate else y
        x=x.to(w.dtype)
        if self.rotate:x=Hadamard128.apply(x)
        return F.linear(x,w)

    def effective_scales(self):
        return (self.base_scales.float()*self.log_gain.exp()).bfloat16()


@torch.no_grad()
def install(model,checkpoint):
    """Replace loaded packed matrices. Zero gain must reproduce every weight exactly."""
    checkpoint=Path(checkpoint)
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    count=0
    for name,info in manifest['matrices'].items():
        if info.get('rotation') not in (None,'hadamard128'):
            raise ValueError('Unsupported rotation in scale adapter')
        original=model.get_submodule(name)
        tensors=load_file(str(checkpoint/info['file']))
        if 'bias' in tensors:raise ValueError('Scale-only adapter currently requires bias-free matrices')
        scales=tensors['scales'].to(original.weight.device)
        q=(original.weight.reshape(-1,128).float()/scales.float()).round().to(torch.int8).reshape_as(original.weight)
        if not bool(((q>=-1)&(q<=1)).all()):raise RuntimeError('Invalid packed ternary values')
        if not torch.equal((q.reshape(-1,128).bfloat16()*scales).reshape_as(original.weight),original.weight):
            raise RuntimeError('Scale adapter changed initial packed weights: '+name)
        replacement=ScaleTunedMatrix(q,scales,info['embedding'],bool(info.get('rotation')))
        parent,key=name.rsplit('.',1) if '.' in name else ('',name)
        setattr(model.get_submodule(parent),key,replacement)
        count+=replacement.log_gain.numel()
    # Replacing the head removes its input transfer hook.
    def move_head(module,args):return (args[0].to(module.trits.device),)
    model.lm_head.register_forward_pre_hook(move_head)
    return count


def check():
    torch.set_num_threads(2);torch.manual_seed(819)
    q=torch.randint(-1,2,(8,128),dtype=torch.int8)
    base=(torch.rand(8,1)*0.1+0.01).bfloat16()
    gain=torch.randn(8,1,requires_grad=True)*0.02;gain=gain.detach().requires_grad_()
    grad=torch.randn(8,128).bfloat16()
    actual=ScaledTrits.apply(q,base,gain)
    assert torch.equal(ScaledTrits.apply(q,base,torch.zeros_like(gain)),q.bfloat16()*base)
    scale=base.float()*gain.exp()
    rounded=scale+(scale.bfloat16().float()-scale).detach()
    reference=q.float()*rounded
    assert torch.equal(actual,reference.bfloat16())
    a=torch.autograd.grad(actual,gain,grad)[0]
    b=torch.autograd.grad(reference,gain,grad.float())[0]
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
    for embedding in (False,True):
        module=ScaleTunedMatrix(q,base,embedding,True)
        x=torch.tensor([0,2,4]) if embedding else torch.randn(3,128).bfloat16()
        output=module(x)
        output.float().square().sum().backward()
        assert torch.isfinite(module.log_gain.grad).all()
    print('PASS: zero-gain packed equality, BF16 forward, independent STE gradient, linear/embedding backward')


if __name__=='__main__':check()
