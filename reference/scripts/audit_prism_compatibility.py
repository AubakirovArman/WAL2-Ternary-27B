"""CPU audit only: exact scale conversion and sampled lossless PQ2 symbol repacking."""
import json,torch
from pathlib import Path
from safetensors import safe_open
from ternary_core import unpack_trits

torch.set_num_threads(4)
p=Path('runs/e012-fresh-reasoning/selected-ternary');m=json.loads((p/'manifest.json').read_text())
count=changed=overflow=underflow=sampled=0;maximum_error=0.;rows=[]
for name,info in m['matrices'].items():
 with safe_open(p/info['file'],framework='pt',device='cpu') as f:
  scale=f.get_tensor('scales').float();fp16=scale.half().float()
  count+=scale.numel();changed+=int((scale!=fp16).sum());overflow+=int((~torch.isfinite(fp16)).sum());underflow+=int(((scale!=0)&(fp16==0)).sum())
  maximum_error=max(maximum_error,float((scale-fp16).abs().max()))
  n=min(128*20,info['shape'][0]*info['shape'][1]);raw=f.get_slice('trits')[:(n+4)//5]
  q=unpack_trits(raw,n).to(torch.int16)
  codes=(q+1).reshape(-1,4);packed=(codes*torch.tensor([1,4,16,64])).sum(-1).to(torch.uint8)
  decoded=((packed.to(torch.int16)[:,None]>>torch.tensor([0,2,4,6]))&3)-1
  assert torch.equal(decoded.flatten(),q)
  sampled+=n
 if info['shape'][1]%128:raise ValueError('Input width not divisible by 128: '+name)
 rows.append({'name':name,'shape':info['shape'],'rotation':info.get('rotation'),'embedding':info['embedding']})
r={'checkpoint':str(p),'matrices':len(rows),'scales_checked':count,'scales_changed_fp16':changed,'overflow_fp16':overflow,'underflow_fp16':underflow,'max_scale_abs_difference':maximum_error,'sampled_ternary_symbols_pq2_roundtrip':sampled,'sampled_symbols_exact':True,'all_input_widths_divisible_by_128':True,'scope':'CPU format feasibility only. No GGUF model export, native CUDA execution, or full logits comparison performed.'}
Path('reports/prism-compatibility-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
