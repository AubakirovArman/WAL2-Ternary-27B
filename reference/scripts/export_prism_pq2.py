"""Lossless trit repack to Prism PQ2_0; no BF16 matrix reconstruction.
Architecture transforms follow pinned Prism conversion/qwen.py. Scale conversion
BF16 -> FP16 is audited separately. Only metadata is borrowed from the template.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/prism-source/gguf-py'))
import gguf

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def reorder(a,head=128):
 return a.reshape(16,3,head,*a.shape[1:]).swapaxes(0,1).reshape(a.shape).copy()
def transform(a,name):
 if '.linear_attn.' in name:
  if '.in_proj_qkv.' in name or '.conv1d.' in name:
   a=a.squeeze() if '.conv1d.' in name else a
   a=np.concatenate((a[:4096],reorder(a[4096:])))
  elif '.in_proj_z.' in name:a=reorder(a)
  elif any(s in name for s in ('.in_proj_a.','.in_proj_b.','.A_log','.dt_bias')):a=reorder(a,1)
  # out_proj columns stay grouped: runtime permutes activation before H128.
 return a

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);p.add_argument('--name',default='Vol2 ternary H128 PQ2');p.add_argument('--alpha-beta-bf16',action='store_true');a=p.parse_args()
 torch.set_num_threads(8)
 if a.output.exists():raise FileExistsError(a.output)
 template=ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
 reader=gguf.GGUFReader(template);shapes={t.name:tuple(reversed(t.shape.tolist())) for t in reader.tensors}
 names=gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35,64)
 def mapped(n):
  if n.endswith('.dt_bias'):n=n[:-len('.dt_bias')]+'.dt_proj.bias'
  result=names.get_name(n,try_suffixes=('.weight','.bias'))
  if result is None:raise ValueError('Unknown tensor '+n)
  return result
 meta=json.loads((a.checkpoint/'manifest.json').read_text())
 cfg=json.loads((a.checkpoint/'config.json').read_text())
 assert (cfg['hidden_size'],cfg['num_hidden_layers'],cfg['linear_num_value_heads'],cfg['linear_num_key_heads'])==(5120,64,48,16)
 assert meta['group_size']==128 and meta['format']=='vol2-ternary-base3-v1'
 a.output.parent.mkdir(parents=True,exist_ok=True)
 temp=a.output.with_suffix('.gguf.partial')
 writer=gguf.GGUFWriter(temp,'qwen35',use_temp_file=True)
 for k,v in reader.fields.items():
  if k.startswith(('GGUF.','general.','prism.')):continue
  writer.add_key_value(k,v.contents(),v.types[0],v.types[-1] if v.types[0]==gguf.GGUFValueType.ARRAY else None)
 writer.add_name(a.name)
 writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_PQ2_0));writer.add_quantization_version(2)
 rotation=meta.get('hadamard',{'block_size':128,'sign_mode':'identity'})
 block=rotation['block_size'];sign_mode=rotation['sign_mode']
 assert block in (128,1024) and sign_mode in ('identity','explicit')
 writer.add_uint32('prism.hadamard.version',1);writer.add_uint32('prism.hadamard.block_size',block)
 writer.add_string('prism.hadamard.transform','normalized-sylvester-walsh-hadamard')
 writer.add_string('prism.hadamard.axis','input-last-dimension');writer.add_string('prism.hadamard.sign_mode',sign_mode)
 if sign_mode=='explicit':
  widths=sorted(map(int,rotation['signs']));values=[]
  for w in widths:
   vec=rotation['signs'][str(w)];assert w%block==0 and len(vec)==w and all(v in (-1,1) for v in vec);values.extend(vec)
  writer.add_array('prism.hadamard.sign_widths',widths);writer.add_array('prism.hadamard.sign_values',values)
 writer.add_bool('prism.hadamard.gdn_v_grouped',True)
 forward=[];inverse=[];report=[];seen=set()
 for index,(module,info) in enumerate(meta['matrices'].items()):
  name=module+'.weight';dest=mapped(name);shape=tuple(info['shape'])
  assert shapes[dest]==shape and info['rotation']==f'hadamard{block}'
  (inverse if info['embedding'] else forward).append(dest)
  with safe_open(a.checkpoint/info['file'],framework='pt',device='cpu') as f:
   packed=f.get_tensor('trits').numpy();scale=f.get_tensor('scales').float().numpy().reshape(shape[0],shape[1]//128)
  codes=np.empty((packed.size,5),dtype=np.uint8)
  for j,power in enumerate((1,3,9,27,81)):codes[:,j]=(packed//power)%3
  codes=codes.ravel()[:np.prod(shape)].reshape(shape);del packed
  codes=transform(codes,name);scale=transform(scale,name)
  converted=scale.astype('<f2')
  if not np.isfinite(converted).all() or np.any((scale!=0)&(converted==0)):raise ValueError('Scale overflow/underflow')
  delta=np.abs(converted.astype(np.float32)-scale)
  audit=dict(source=name,name=dest,shape=shape,scale_changed=int(np.count_nonzero(delta)),scale_max_abs=float(delta.max()),trits=int(codes.size))
  groups=codes.reshape(-1,128);out=np.empty((len(groups),34),dtype=np.uint8)
  out[:,:2]=converted.reshape(-1,1).view(np.uint8)
  out[:,2:]=groups[:,0::4] | (groups[:,1::4]<<2) | (groups[:,2::4]<<4) | (groups[:,3::4]<<6)
  # Verify every symbol, not a sample. No floating-point weights materialized.
  for j in range(4):
   if not np.array_equal((out[:,2:]>>(2*j))&3,groups[:,j::4]):raise ValueError('PQ2 roundtrip failed')
  writer.add_tensor(dest,out.reshape(shape[0],-1),raw_dtype=gguf.GGMLQuantizationType.PQ2_0)
  seen.add(dest);report.append(audit)
  del codes,groups,out,scale,converted,delta
  if index%20==0:print(f'Packed {index+1}/{len(meta["matrices"])}',flush=True)
 bf16_names=[]
 with safe_open(a.checkpoint/'exceptions.safetensors',framework='pt',device='cpu') as f:
  for name in f.keys():
   # F32 preserves the source BF16 constants, including alpha/beta matrices.
   t=f.get_tensor(name).float()
   if name.endswith('.A_log'):t=-torch.exp(t)
   elif name.endswith('norm.weight') and not name.endswith('linear_attn.norm.weight'):t=t+1
   value=transform(t.numpy(),name);dest=mapped(name)
   assert tuple(value.shape)==shapes[dest],(name,value.shape,shapes[dest])
   value=np.ascontiguousarray(value,dtype=np.float32)
   if a.alpha_beta_bf16 and name.endswith(('.linear_attn.in_proj_a.weight','.linear_attn.in_proj_b.weight')):
    encoded=gguf.quants.quantize(value,gguf.GGMLQuantizationType.BF16)
    if not np.array_equal(gguf.quants.dequantize(encoded,gguf.GGMLQuantizationType.BF16),value):raise ValueError('Unexpected lossy alpha/beta conversion')
    writer.add_tensor(dest,encoded,raw_dtype=gguf.GGMLQuantizationType.BF16);bf16_names.append(dest)
   else:writer.add_tensor(dest,value)
   seen.add(dest)
 if a.alpha_beta_bf16 and len(bf16_names)!=96:raise ValueError('Expected exactly96 BF16 alpha/beta matrices')
 assert seen==set(shapes),(seen-set(shapes),set(shapes)-seen)
 writer.add_array('prism.hadamard.weight_names',forward);writer.add_array('prism.hadamard.inverse_weight_names',inverse)
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file(progress=True);writer.close()
 temp.rename(a.output)
 result=dict(state='exported_not_yet_validated',checkpoint=str(a.checkpoint),checkpoint_manifest_sha256=digest(a.checkpoint/'manifest.json'),output=str(a.output),bytes=a.output.stat().st_size,sha256=digest(a.output),matrices=report,all_trits_roundtrip_exact=True,scales_changed=sum(r['scale_changed'] for r in report),tensor_count=len(seen),hadamard_block=block,hadamard_sign_mode=sign_mode,bf16_alpha_beta_names=bf16_names,runtime_commit='9a9394a895b96003ca842a6041cb28ac49a108f7',note='402 matrices PQ2; exceptions F32 except explicitly listed BF16 alpha/beta; metadata template only, zero Bonsai weights copied')
 a.output.with_suffix('.export.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='matrices'}),flush=True)
if __name__=='__main__':main()
