"""Small mathematical checks for lossy quantization and lossless packing."""
import torch
from torch import nn
from ternary_core import pack_trits,unpack_trits,TernaryMatrix,QuantizeSTE,Hadamard128

torch.manual_seed(11)
for count in [1,4,5,6,127,128,129,1025]:
    q=torch.randint(-1,2,(count,),dtype=torch.int8)
    assert torch.equal(q,unpack_trits(pack_trits(q),count))
linear=nn.Linear(256,32,bias=False)
m=TernaryMatrix(linear)
q=QuantizeSTE.apply(m.weight,m.scales,1.)
codes=(m.weight.detach().reshape(-1,128)/m.scales.float()).round().clamp(-1,1)
expected=(codes*m.scales.float()).reshape_as(q).bfloat16()
assert torch.equal(q,expected)
m.alpha=0.
assert torch.equal(QuantizeSTE.apply(m.weight,m.scales,0.),m.weight.bfloat16())
m.alpha=1.
x=torch.randn(4,256,dtype=torch.bfloat16)
loss=m(x).float().square().mean();loss.backward()
assert m.weight.grad is not None and torch.isfinite(m.weight.grad).all()
assert m.weight.grad.abs().sum()>0
print('PASS: trit packing round-trip, exact ternary forward, alpha=0 identity, STE gradients')
x=torch.randn(7,256,requires_grad=True)
h=Hadamard128.apply(x)
assert torch.allclose(Hadamard128.apply(h),x,atol=1e-6)
h.square().sum().backward()
assert torch.allclose(x.grad,2*x,atol=3e-6)
w=torch.randn(32,256)
assert torch.allclose(Hadamard128.apply(x)@Hadamard128.apply(w).T,x@w.T,atol=2e-5)
print('PASS: orthogonal Hadamard transform, inverse, gradients, linear equivalence')
