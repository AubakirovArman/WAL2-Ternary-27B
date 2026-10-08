"""Offline top-K teacher distributions. Source weights only; GPU 6/7 guard required."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from safetensors.torch import save_file
import ternary_core as core
from data_v2 import build
from train_v2 import schedule,write_json

@torch.no_grad()
def collect(model,example,topk):
    ids=torch.tensor([example['input_ids']],device='cuda:0')
    hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
    tops=[];logs=[];tails=[]
    for start in range(0,len(hidden),64):
        logits=model.lm_head(hidden[start:start+64]).float()
        logp=torch.log_softmax(logits,dim=-1)
        lp,ix=logp.topk(topk,dim=-1)
        tail=(1-lp.exp().sum(-1)).clamp_min(0)
        tops.append(ix.to(torch.int32).cpu());logs.append(lp.half().cpu());tails.append(tail.cpu())
        del logits,logp,lp,ix,tail
    return {'input_ids':torch.tensor(example['input_ids'],dtype=torch.int32),
            'labels':torch.tensor(example['labels'],dtype=torch.int32),
            'top_ids':torch.cat(tops),'top_logp':torch.cat(logs),'tail_prob':torch.cat(tails)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=4096);ap.add_argument('--topk',type=int,default=128)
    args=ap.parse_args();core.guard()
    folder=core.ROOT/'data/v3-kd';folder.mkdir(exist_ok=True);corpus=folder/'corpus.jsonl'
    if not corpus.exists():build(corpus,target=12000)
    digest=hashlib.sha256(corpus.read_bytes()).hexdigest();cache=folder/'cache';cache.mkdir(exist_ok=True)
    manifest_path=folder/'manifest.json'
    meta=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'source':str(core.SOURCE),'corpus':str(corpus),'corpus_sha256':digest,'topk':args.topk,
        'temperature':1.,'positions':'all next-token positions including prompt and answer','count_target':args.count,
        'format':'top log probabilities fp16 plus fp32 residual tail; normalized when consumed','examples':{},'state':'collecting'}
    assert meta['corpus_sha256']==digest and meta['topk']==args.topk and meta['count_target']==args.count
    data=core.load_examples(corpus,512);ordered=schedule(data['train'],corpus,len(data['train'])*3,seed=62061)
    seen=set();selected=[]
    for row in ordered:
        if row['id'] not in seen:seen.add(row['id']);selected.append(row)
        if len(selected)==args.count:break
    assert len(selected)==args.count
    meta['order_ids']=[row['id'] for row in selected];write_json(manifest_path,meta)
    model=core.load_source();model.eval()
    start=time.time();covered=0.;positions=0
    for index,example in enumerate(selected):
        key=str(example['id']);path=cache/(key+'.safetensors')
        if key in meta['examples']:
            if not path.exists():raise RuntimeError('Manifest references missing cache')
            continue
        t=time.time();tensors=collect(model,example,args.topk)
        temp=path.with_suffix('.tmp');save_file(tensors,str(temp));temp.replace(path)
        n=len(example['input_ids'])-1;tail=float(tensors['tail_prob'].mean())
        meta['examples'][key]={'file':str(path),'positions':n,'answer_tokens':sum(x!=-100 for x in example['labels']),
                               'mean_tail_probability':tail,'input_sha256':hashlib.sha256(json.dumps(example['input_ids']).encode()).hexdigest()}
        covered+=(1-tail)*n;positions+=n
        if (index+1)%32==0 or index+1==len(selected):
            meta['cached']=len(meta['examples']);write_json(manifest_path,meta)
            core.event('kd_teacher_cache',cached=meta['cached'],target=args.count,mean_topk_mass=covered/max(positions,1),
                       elapsed_seconds=time.time()-start,last_example_seconds=time.time()-t)
        if (folder/'STOP').exists():write_json(manifest_path,meta);return
    meta['state']='completed';meta['cached']=len(meta['examples']);write_json(manifest_path,meta)
    core.event('kd_teacher_cache_completed',manifest=str(manifest_path),examples=len(meta['examples']))

if __name__=='__main__':
    try:main()
    except Exception as e:core.event('kd_teacher_cache_failed',error=repr(e));raise
