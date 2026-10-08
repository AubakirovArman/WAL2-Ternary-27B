import json,time,statistics
import torch
from ternary_core import ROOT,guard,QuantizeSTE,fit_scales
from fast_ternary import FusedQuantizeSTE,gradients_finite_by_device,enable_fused_quantizer

def latency(fn):
    for _ in range(4):fn()
    torch.cuda.synchronize();values=[]
    for _ in range(12):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();out=fn();end.record();end.synchronize();values.append(start.elapsed_time(end))
    return statistics.median(values)

def main():
    guard();results=[]
    for device in ['cuda:0','cuda:1']:
        torch.cuda.set_device(device)
        for shape in [(32,128),(10240,5120),(17408,5120)]:
            w=(torch.randn(shape,device=device)*0.02).requires_grad_(True)
            s=fit_scales(w)
            for alpha in [0.,1/256,.25,.5,.75,1.]:
                reference=QuantizeSTE.apply(w,s,alpha);fast=FusedQuantizeSTE.apply(w,s,alpha)
                mismatches=int((reference!=fast).sum())
                if mismatches:raise AssertionError((device,shape,alpha,mismatches,float((reference-fast).abs().max())))
                if shape==(32,128):
                    upstream=torch.randn_like(reference)
                    a=torch.autograd.grad(reference,w,upstream)[0]
                    b=torch.autograd.grad(fast,w,upstream)[0]
                    assert torch.equal(a,b)
                del reference,fast
            # Ties and their adjacent FP32 numbers, not only random well-separated values.
            tiny_s=torch.full((1,1),0.125,device=device,dtype=torch.bfloat16)
            tiny_w=torch.zeros((1,128),device=device)
            ties=torch.tensor([-.1875,-.0625,.0625,.1875],device=device)
            tiny_w[0,:4]=ties
            tiny_w[0,4:8]=torch.nextafter(ties,torch.full_like(ties,float('inf')))
            tiny_w[0,8:12]=torch.nextafter(ties,torch.full_like(ties,-float('inf')))
            for alpha in [.25,.5,.75,1.]:
                assert torch.equal(QuantizeSTE.apply(tiny_w,tiny_s,alpha),FusedQuantizeSTE.apply(tiny_w,tiny_s,alpha))
            old=latency(lambda:QuantizeSTE.apply(w,s,1.));new=latency(lambda:FusedQuantizeSTE.apply(w,s,1.))
            row={'device':device,'shape':shape,'reference_ms':old,'fused_ms':new,'speedup':old/new,'bit_exact_forward':True}
            results.append(row);print(json.dumps(row),flush=True)
            del w,s
    params=[torch.nn.Parameter(torch.ones(8,device=d)) for d in ['cuda:0','cuda:1']]
    for p in params:p.grad=torch.ones_like(p)
    assert gradients_finite_by_device(params)
    for p in params:
        for value in [float('nan'),float('inf'),-float('inf')]:
            p.grad[0]=value;assert not gradients_finite_by_device(params)
        p.grad[0]=1
    import ternary_core
    enable_fused_quantizer()
    for device in ['cuda:0','cuda:1']:
        w=torch.randn((32,128),device=device,requires_grad=True)
        s=fit_scales(w)
        for alpha in [0.,.5,1.]:
            expected=QuantizeSTE.apply(w,s,alpha)
            actual=ternary_core.QuantizeSTE.apply(w,s,alpha)
            assert torch.equal(expected,actual)
            upstream=torch.randn_like(expected)
            assert torch.equal(torch.autograd.grad(expected,w,upstream)[0],torch.autograd.grad(actual,w,upstream)[0])
    report={'quantizer':results,'finite_checks':'passed on both GPUs, including NaN and ±Inf',
            'opt_in_dispatch':'forward and backward passed for alpha 0, 0.5, 1 on both GPUs',
            'scope':'Kernel microbenchmark, not whole-training speedup.'}
    (ROOT/'reports/performance/fused_quantizer.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
