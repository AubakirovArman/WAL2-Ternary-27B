"""Load our packed checkpoint and evaluate/generate with a BF16 reference runtime."""
import argparse
import json
from pathlib import Path
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig,AutoTokenizer
from ternary_core import SOURCE,guard,target_device,install_device_hooks,unpack_trits,load_examples,evaluate,event,Hadamard128

class PackedMatrix(torch.nn.Module):
    def __init__(self,weight,embedding,rotate):
        super().__init__();self.weight=torch.nn.Parameter(weight,requires_grad=False)
        self.embedding=embedding;self.rotate=rotate
    def forward(self,x):
        if self.embedding:
            y=torch.nn.functional.embedding(x,self.weight)
            return Hadamard128.apply(y) if self.rotate else y
        if self.rotate:x=Hadamard128.apply(x)
        return torch.nn.functional.linear(x,self.weight)

def load_packed(path, single_device=None):
    if single_device not in (None, "cuda:0", "cuda:1"):raise ValueError("Invalid inference device")
    placement=target_device if single_device is None else lambda name:single_device
    path=Path(path);meta=json.loads((path/'manifest.json').read_text())
    if meta['format']!='vol2-ternary-base3-v1':raise ValueError('Unknown packed format')
    cfg=Qwen3_5TextConfig.from_json_file(path/'config.json');cfg._attn_implementation='sdpa';cfg.use_cache=False
    with init_empty_weights():model=Qwen3_5ForCausalLM(cfg)
    filled=set()
    for name,info in meta['matrices'].items():
        dev=placement(name);count=info['shape'][0]*info['shape'][1]
        with safe_open(path/info['file'],framework='pt',device='cpu') as f:
            packed=f.get_tensor('trits');scales=f.get_tensor('scales').to(dev)
        q=torch.empty(count,dtype=torch.int8,device=dev)
        for start in range(0,packed.numel(),1_000_000):
            part=packed[start:start+1_000_000].to(dev)
            n=min(count-start*5,len(part)*5)
            q[start*5:start*5+n]=unpack_trits(part,n)
        weight=(q.reshape(-1,128).to(torch.bfloat16)*scales).reshape(info['shape'])
        set_module_tensor_to_device(model,name+'.weight',dev,value=weight,dtype=torch.bfloat16)
        if info.get('rotation'):
            if info['rotation']!='hadamard128':raise ValueError('Unknown rotation')
            parent,key=name.rsplit('.',1) if '.' in name else ('',name)
            setattr(model.get_submodule(parent),key,PackedMatrix(weight,info['embedding'],True))
        filled.add(name+'.weight')
        del q,weight,scales,packed
    with safe_open(path/'exceptions.safetensors',framework='pt',device='cpu') as f:
        for name in f.keys():
            set_module_tensor_to_device(model,name,placement(name),value=f.get_tensor(name),dtype=torch.bfloat16)
            filled.add(name)
    if filled!=set(dict(model.named_parameters())):raise RuntimeError('Incomplete packed checkpoint')
    model.requires_grad_(False)
    if single_device is None:install_device_hooks(model)
    else:model.to(single_device)
    model.eval()
    return model

@torch.no_grad()
def generate_reference(model,prompt,max_new_tokens=48,tokenizer_path=SOURCE):
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if not isinstance(ids,list):ids=ids['input_ids']
    result=[]
    stop=set([tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')])
    for _ in range(max_new_tokens):
        hidden=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False).last_hidden_state[:,-1]
        token=int(model.lm_head(hidden).argmax(-1))
        if token in stop:break
        ids.append(token);result.append(token)
    return tok.decode(result,skip_special_tokens=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('--data');ap.add_argument('--split',default='validation')
    ap.add_argument('--examples',type=int,default=4);ap.add_argument('--max-length',type=int,default=256)
    ap.add_argument('--prompt');ap.add_argument('--max-new-tokens',type=int,default=48);ap.add_argument('--output',required=True)
    a=ap.parse_args();guard();model=load_packed(a.checkpoint);result={'checkpoint':a.checkpoint}
    if a.data:
        data=load_examples(a.data,a.max_length)
        result[a.split]=evaluate(model,data[a.split][:a.examples])
    if a.prompt:
        tp=Path(a.checkpoint) if (Path(a.checkpoint)/'tokenizer.json').exists() else SOURCE
        result['generation']={'prompt':a.prompt,'answer':generate_reference(model,a.prompt,a.max_new_tokens,tp)}
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));event('packed_inspection',**result)

if __name__=='__main__':main()
