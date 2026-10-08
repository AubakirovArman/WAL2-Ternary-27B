"""CPU-only candidate benchmark. Does not modify the running training implementation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from ternary_core import ROOT,pack_trits,unpack_trits

def numpy_pack(q):
    x=(q.numpy().astype(np.int16)+1).astype(np.uint8)
    pad=(-len(x))%5
    if pad:x=np.pad(x,(0,pad),constant_values=1)
    x=x.reshape(-1,5)
    out=x[:,0].copy()
    for i,k in enumerate([3,9,27,81],1):out+=x[:,i]*np.uint8(k)
    return torch.from_numpy(out)

def timed(fn):
    times=[]
    for _ in range(5):
        t=time.perf_counter();out=fn();times.append(time.perf_counter()-t)
    return out,{'median_seconds':statistics.median(times),'runs_seconds':times}

def main():
    torch.set_num_threads(8)
    lut=np.array([[(code//(3**i))%3-1 for i in range(5)] for code in range(243)],dtype=np.int8)
    all_codes=torch.arange(243,dtype=torch.uint8)
    assert np.array_equal(lut,unpack_trits(all_codes,243*5).numpy().reshape(-1,5))
    for n in [1,4,5,6,127,128,129,10001]:
        q=torch.randint(-1,2,(n,),dtype=torch.int8)
        assert torch.equal(numpy_pack(q),pack_trits(q))
    path=ROOT/'runs/e004-rotated-qat/initial-ternary'
    meta=json.loads((path/'manifest.json').read_text())
    name=next(n for n in meta['matrices'] if 'in_proj_qkv' in n)
    info=meta['matrices'][name]
    with safe_open(path/info['file'],framework='pt',device='cpu') as f:packed=f.get_tensor('trits')
    count=int(np.prod(info['shape']))
    q=unpack_trits(packed,count)
    # Match the existing exporter: chunks of five million weights, each round-trip checked.
    def reference():
        chunks=[]
        for start in range(0,len(q),5_000_000):
            part=q[start:start+5_000_000];out=pack_trits(part)
            assert torch.equal(unpack_trits(out,len(part)),part);chunks.append(out)
        return torch.cat(chunks)
    def candidate():
        chunks=[]
        for start in range(0,len(q),5_000_000):
            part=q[start:start+5_000_000];out=numpy_pack(part)
            assert torch.equal(unpack_trits(out,len(part)),part);chunks.append(out)
        return torch.cat(chunks)
    def candidate_lut():
        chunks=[]
        for start in range(0,len(q),5_000_000):
            part=q[start:start+5_000_000];out=numpy_pack(part)
            decoded=lut[out.numpy()].reshape(-1)[:len(part)]
            assert np.array_equal(decoded,part.numpy());chunks.append(out)
        return torch.cat(chunks)
    reference();candidate();candidate_lut()
    a,old=timed(reference);b,new=timed(candidate)
    c,lut_stats=timed(candidate_lut)
    assert torch.equal(a,b) and torch.equal(a,c) and torch.equal(a,packed)
    result={'matrix':name,'shape':info['shape'],'weights':count,'threads':8,
        'reference_pack_and_verify':old,'numpy_pack_and_verify':new,
        'microbenchmark_speedup':old['median_seconds']/new['median_seconds'],
        'numpy_pack_lut_verify':lut_stats,
        'lut_microbenchmark_speedup':old['median_seconds']/lut_stats['median_seconds'],
        'bit_exact':True,'note':'CPU microbenchmark during live training; not end-to-end export timing.'}
    (ROOT/'reports/performance/cpu_packing_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
