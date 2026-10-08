"""Experimental text-only Qwen ternary QAT. Packed storage; BF16 compute reference."""
import gc
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'models/qwen3.8-27b-fp8-source'
GPU_IDS = ['GPU_UUID_REDACTED', 'GPU_UUID_REDACTED']

def event(kind, **values):
    record = {'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':kind,**values}
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)

def guard():
    global _GPU_LOCK
    _GPU_LOCK=(ROOT/'runs/gpu67.lock').open('a')
    try:fcntl.flock(_GPU_LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another Vol2 experiment is already using GPU 6/7')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(GPU_IDS):
        raise RuntimeError('Must source configs/environment.sh; only physical GPU 6/7 authorized')
    if torch.cuda.device_count()!=2:raise RuntimeError('Expected exactly two visible GPUs')
    for i in range(2):
        free,total=torch.cuda.mem_get_info(i)
        if free<100*2**30:raise RuntimeError(f'Visible GPU {i} has less than 100 GiB free; will not interfere with existing jobs')
    torch.set_num_threads(8)
    torch.manual_seed(20260919)

def target_device(name):
    if name.startswith('model.layers.'):
        return 'cuda:0' if int(name.split('.')[2])<32 else 'cuda:1'
    return 'cuda:1' if name.startswith(('lm_head','model.norm')) else 'cuda:0'

def move_tree(x, device):
    if isinstance(x,torch.Tensor):return x.to(device)
    if isinstance(x,tuple):return tuple(move_tree(v,device) for v in x)
    if isinstance(x,list):return [move_tree(v,device) for v in x]
    if isinstance(x,dict):return {k:move_tree(v,device) for k,v in x.items()}
    return x

def install_device_hooks(model):
    def hook(device):
        def move(module,args,kwargs):return move_tree(args,device),move_tree(kwargs,device)
        return move
    for i,layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(hook('cuda:0' if i<32 else 'cuda:1'),with_kwargs=True)
    model.model.norm.register_forward_pre_hook(hook('cuda:1'),with_kwargs=True)
    model.lm_head.register_forward_pre_hook(hook('cuda:1'),with_kwargs=True)
    for name,buf in list(model.named_buffers()):
        parent,key=name.rsplit('.',1)
        model.get_submodule(parent)._buffers[key]=buf.to(target_device(name))

def load_source():
    cfg=Qwen3_5TextConfig(**json.loads((SOURCE/'config.json').read_text())['text_config'])
    cfg._attn_implementation='sdpa';cfg.use_cache=False
    with init_empty_weights():model=Qwen3_5ForCausalLM(cfg)
    expected=set(dict(model.named_parameters()))
    loaded=set()
    for path in sorted(SOURCE.glob('*.safetensors')):
        with safe_open(path,framework='pt',device='cpu') as f:
            for key in f.keys():
                name=key.replace('model.language_model.','model.')
                if name not in expected:continue
                device=target_device(name)
                value=f.get_tensor(key).to(device)
                if value.dtype==torch.float8_e4m3fn:
                    scales=f.get_tensor(key.replace('.weight','.weight_scale_inv')).to(device).float()
                    # Source format is block-scaled FP8; scale_inv is the stored dequantization multiplier.
                    rows,cols=value.shape
                    value=(value.float().view(rows//128,128,cols//128,128)*scales[:,None,:,None]).reshape(rows,cols).bfloat16()
                else:value=value.bfloat16()
                set_module_tensor_to_device(model,name,device,value=value,dtype=torch.bfloat16)
                loaded.add(name)
        event('source_shard_loaded',file=path.name,parameters_loaded=len(loaded))
    if loaded!=expected:raise RuntimeError(f'Missing source parameters: {sorted(expected-loaded)}')
    model.requires_grad_(False)
    install_device_hooks(model)
    model.eval()
    event('source_loaded',parameters=sum(p.numel() for p in model.parameters()))
    return model

@torch.no_grad()
def fit_scales(weight, group=128):
    assert weight.shape[-1]%group==0
    flat=weight.reshape(-1,group)
    result=torch.empty((flat.shape[0],1),device=weight.device,dtype=torch.bfloat16)
    for start in range(0,len(flat),32768):
        w=flat[start:start+32768].float()
        s=(w.abs().mean(-1,keepdim=True)*1.3).clamp_min(1e-8)
        for _ in range(8):
            q=(w/s).round().clamp(-1,1)
            s=((w*q).sum(-1,keepdim=True)/q.square().sum(-1,keepdim=True).clamp_min(1)).clamp_min(1e-8)
        result[start:start+len(w)]=s
    return result

class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weight,scales,alpha):
        flat=weight.reshape(-1,128)
        out=torch.empty_like(weight,dtype=torch.bfloat16).reshape(-1,128)
        for start in range(0,len(flat),32768):
            w=flat[start:start+32768]
            s=scales[start:start+len(w)].float()
            q=(w/s).round().clamp(-1,1)*s
            out[start:start+len(w)]=w.lerp(q,alpha)
        return out.reshape_as(weight)
    @staticmethod
    def backward(ctx,grad):return grad.float(),None,None

class Hadamard128(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        shape=x.shape
        y=x.float().reshape(-1,128)
        stride=1
        while stride<128:
            z=y.reshape(-1,128//(2*stride),2,stride)
            a,b=z[:,:,0],z[:,:,1]
            y=torch.stack((a+b,a-b),dim=2).reshape(-1,128)
            stride*=2
        return (y/(128**0.5)).reshape(shape).to(x.dtype)
    @staticmethod
    def backward(ctx,grad):return Hadamard128.apply(grad)

class TernaryMatrix(nn.Module):
    def __init__(self,original,embedding=False,rotate=False):
        super().__init__()
        weight=original.weight.detach().float()
        if rotate:
            # Bound the scratch memory for the vocabulary matrices.
            for start in range(0,len(weight),1024):weight[start:start+1024]=Hadamard128.apply(weight[start:start+1024])
        self.weight=nn.Parameter(weight)
        self.register_buffer('scales',fit_scales(self.weight))
        self.embedding=embedding;self.alpha=1.0;self.rotate=rotate
        self.bias=None
        if getattr(original,'bias',None) is not None:
            self.register_buffer('stored_bias',original.bias.detach().clone());self.bias=self.stored_bias
    def forward(self,x):
        w=QuantizeSTE.apply(self.weight,self.scales,self.alpha)
        if self.embedding:
            out=F.embedding(x,w)
            return Hadamard128.apply(out) if self.rotate else out
        x=x.to(w.dtype)
        if self.rotate:x=Hadamard128.apply(x)
        return F.linear(x,w,self.bias)

def ternarize(model,rotate=False):
    names=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,(nn.Linear,nn.Embedding)) and not name.endswith(('in_proj_a','in_proj_b')):
            parent,key=name.rsplit('.',1) if '.' in name else ('',name)
            replacement=TernaryMatrix(module,isinstance(module,nn.Embedding),rotate=rotate)
            setattr(model.get_submodule(parent),key,replacement)
            names.append(name)
    # New head needs its transfer hook after replacement.
    event('ternarized',matrices=len(names),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    gc.collect();torch.cuda.empty_cache()
    return names

def set_alpha(model,alpha):
    for m in model.modules():
        if isinstance(m,TernaryMatrix):m.alpha=alpha

def loss_for(model,example):
    ids=torch.tensor([example['input_ids']],device='cuda:0')
    labels=torch.tensor(example['labels'][1:],device='cuda:1')
    hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
    mask=labels!=-100
    logits=model.lm_head(hidden[mask]).float()
    return F.cross_entropy(logits,labels[mask]),int(mask.sum())

@torch.no_grad()
def evaluate(model,examples):
    model.eval();total=0.;tokens=0
    for e in examples:
        loss,n=loss_for(model,e);total+=loss.item()*n;tokens+=n
    return {'answer_ce':total/tokens,'tokens':tokens,'examples':len(examples)}

def load_examples(path,max_length=384):
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    result={'train':[],'validation':[],'test':[]};seen=set();dropped=[]
    for line in Path(path).read_text().splitlines():
        row=json.loads(line)
        if row['sha256'] in seen:continue
        seen.add(row['sha256'])
        prompt=tok.apply_chat_template([{'role':'user','content':row['instruction']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        p=tok.encode(prompt,add_special_tokens=False)
        answer=tok.encode(row['response']+'<|im_end|>',add_special_tokens=False)
        if len(p)>max_length-32:dropped.append(row['id']);continue
        answer=answer[:max_length-len(p)]
        result[row['split']].append({'id':row['id'],'input_ids':p+answer,'labels':[-100]*len(p)+answer})
    event('data_tokenized',splits={k:len(v) for k,v in result.items()},dropped_long_prompts=dropped)
    return result

def pack_trits(q):
    flat=(q.flatten().to(torch.int16)+1)
    pad=(-flat.numel())%5
    if pad:flat=F.pad(flat,(0,pad),value=1)
    coeff=torch.tensor([1,3,9,27,81],device=q.device,dtype=torch.int16)
    return (flat.reshape(-1,5)*coeff).sum(-1).to(torch.uint8)

def unpack_trits(packed,count):
    coeff=torch.tensor([1,3,9,27,81],device=packed.device,dtype=torch.int16)
    return ((packed.to(torch.int16)[:,None]//coeff)%3-1).flatten()[:count].to(torch.int8)

@torch.no_grad()
def export_packed(model,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    metadata={'format':'vol2-ternary-base3-v1','group_size':128,'trits_per_byte':5,
              'compute':'reference BF16 unpack; no native low-bit speed claim','matrices':{},'exceptions':{},
              'source':str(SOURCE),'text_only':True}
    excluded=set();total=0
    for index,(name,m) in enumerate((n,m) for n,m in model.named_modules() if isinstance(m,TernaryMatrix)):
        flat=m.weight.reshape(-1,128)
        chunks=[]
        # Pack entire matrix to preserve five-trit alignment; int8 scratch is at most 1.3 GB.
        q=torch.empty_like(m.weight,dtype=torch.int8).reshape(-1,128)
        for start in range(0,len(flat),32768):
            w=flat[start:start+32768]
            q[start:start+len(w)]=(w/m.scales[start:start+len(w)].float()).round().clamp(-1,1).to(torch.int8)
        # CPU chunking avoids giant GPU temporaries for embeddings/lm_head.
        qcpu=q.flatten().cpu(); del q
        for start in range(0,qcpu.numel(),5_000_000):
            part=qcpu[start:start+5_000_000]
            packed=pack_trits(part)
            if not torch.equal(unpack_trits(packed,part.numel()),part):raise RuntimeError('packing mismatch')
            chunks.append(packed)
        filename=f'matrix-{index:03d}.safetensors'
        tensors={'trits':torch.cat(chunks),'scales':m.scales.cpu()}
        if m.bias is not None:tensors['bias']=m.bias.cpu()
        save_file(tensors,str(outdir/filename))
        metadata['matrices'][name]={'shape':list(m.weight.shape),'file':filename,'embedding':m.embedding,'rotation':'hadamard128' if m.rotate else None}
        excluded.add(name+'.weight');total+=m.weight.numel()
        del qcpu,chunks,tensors
    extras={name:p.detach().cpu().contiguous() for name,p in model.named_parameters() if name not in excluded}
    save_file(extras,str(outdir/'exceptions.safetensors'))
    metadata['exceptions']={k:list(v.shape) for k,v in extras.items()}
    metadata['quantized_parameters']=total
    metadata['total_parameters']=sum(p.numel() for p in model.parameters())
    model.config.to_json_file(outdir/'config.json')
    for filename in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt','generation_config.json','LICENSE']:
        shutil.copy2(SOURCE/filename,outdir/filename)
    metadata['tensor_file_bytes']=sum(p.stat().st_size for p in outdir.glob('*.safetensors'))
    metadata['effective_tensor_bpw']=8*metadata['tensor_file_bytes']/metadata['total_parameters']
    metadata['base_model']='Qwen/Qwen3.8-27B-FP8'
    metadata['base_revision']='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
    (outdir/'manifest.json').write_text(json.dumps(metadata,indent=2))
    event('packed_export',path=str(outdir),bytes=metadata['tensor_file_bytes'],bpw=metadata['effective_tensor_bpw'])
    return metadata
