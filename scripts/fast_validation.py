"""Bounded no-grad QAT validation cache; temporary, never serialized or trained."""
from contextlib import contextmanager
import types
import torch
from torch.nn import functional as F
import ternary_core as core

@contextmanager
def validation_runtime(model, enabled=False, cache_gib=8.0, reserve_gib=12.0, fused=True):
    if not enabled:
        yield {};return
    if torch.is_grad_enabled():raise RuntimeError('Fast validation requires no_grad')
    if cache_gib<0 or reserve_gib<0:raise ValueError('Negative cache budget')
    reference_h=core.Hadamard128
    originals=[];used={};limits={}
    try:
        if fused:
            from fast_inference import FastHadamard
            core.Hadamard128=FastHadamard
        modules=[m for m in model.modules() if isinstance(m,core.TernaryMatrix)]
        for m in sorted(modules,key=lambda m:m.weight.numel()):
            device=m.weight.device
            if device not in limits:
                free=torch.cuda.mem_get_info(device)[0] if device.type=='cuda' else float('inf')
                limits[device]=max(0,min(int(cache_gib*2**30),free-int(reserve_gib*2**30)))
                used[device]=0
            size=m.weight.numel()*2
            if used[device]+size>limits[device]:continue
            # Cached output is precisely the current training quantizer's output.
            weight=core.QuantizeSTE.apply(m.weight,m.scales,m.alpha)
            if weight.dtype!=torch.bfloat16:raise ValueError('Expected BF16 quantizer output')
            old=m.__dict__.get('forward');had='forward' in m.__dict__
            def cached_forward(self,x,w=weight):
                if torch.is_grad_enabled():raise RuntimeError('Validation cache cannot be used for training')
                if self.embedding:
                    y=F.embedding(x,w)
                    return core.Hadamard128.apply(y) if self.rotate else y
                x=x.to(w.dtype)
                if self.rotate:x=core.Hadamard128.apply(x)
                return F.linear(x,w,self.bias)
            originals.append((m,had,old));m.forward=types.MethodType(cached_forward,m)
            used[device]+=size
        yield {'cached_matrices':len(originals),'bytes_by_device':{str(k):v for k,v in used.items()}}
    finally:
        for m,had,old in originals:
            if had:m.forward=old
            else:del m.forward
        core.Hadamard128=reference_h
