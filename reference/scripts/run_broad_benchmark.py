"""Generate matching tasks with source/student PyTorch or official Bonsai runtime."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,guard,load_source
from inspect_checkpoint import load_packed
from cached_generation import generate_ids
from benchmark_tasks import tasks,final_text

@torch.no_grad()
def check_real_cache(model):
    # Compare teacher-forced logits, rather than masking divergent generation with a format check.
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    ids=tok.encode('A short independent cache check: два плюс три равно пяти.\nprint(1 + 2)',add_special_tokens=False)
    full=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False).last_hidden_state
    expected=model.lm_head(full).float();cache=None;max_abs=0.;agreed=0;max_kl=0.
    for i,token in enumerate(ids):
        output=model.model(input_ids=torch.tensor([[token]],device='cuda:0'),past_key_values=cache,use_cache=True)
        cache=output.past_key_values;actual=model.lm_head(output.last_hidden_state).float()
        target=expected[:,i:i+1]
        max_abs=max(max_abs,float((actual-target).abs().max()))
        agreed+=int(actual.argmax(-1).item()==target.argmax(-1).item())
        logp=target.log_softmax(-1);logq=actual.log_softmax(-1)
        max_kl=max(max_kl,float((logp.exp()*(logp-logq)).sum(-1).max()))
    result={'tokens':len(ids),'top1_agreement':agreed/len(ids),'max_absolute_logit_difference':max_abs,'max_forward_kl':max_kl}
    # Numerical execution differs between recurrent and chunk paths. Reject material drift.
    if max_kl>0.02 or agreed/len(ids)<0.9:raise RuntimeError('Cached/full-prefix disagreement: '+str(result))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['source','student','bonsai2'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--subset',choices=['smoke','final'],default='smoke');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=['core','russian'],default='core')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--fast-hadamard',action='store_true')
    p.add_argument('--single-device',choices=['cuda:0'])
    a=p.parse_args();guard()
    if a.resume:
        if not a.output.is_dir() or a.model!='student':raise ValueError('Resume requires existing student output')
    else:a.output.mkdir(parents=True,exist_ok=False)
    joblist=tasks(a.subset,a.suite);tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    for job in joblist:
        text=tok.apply_chat_template([{'role':'user','content':job['instruction']}],tokenize=False,add_generation_prompt=True,
                                    enable_thinking=job['thinking'],reasoning_effort=job['reasoning_effort'])
        job['prompt']=text;job['input_ids']=tok.encode(text,add_special_tokens=False)
        if len(job['input_ids'])+job['max_new_tokens']>8192:raise RuntimeError('Benchmark prompt exceeds fixed context budget')
    if a.resume:
        if json.loads((a.output/'tasks.json').read_text())!=joblist:raise ValueError('Resume tasks mismatch')
    else:(a.output/'tasks.json').write_text(json.dumps(joblist,ensure_ascii=False,indent=2))
    config={'model':a.model,'checkpoint':str(a.checkpoint) if a.checkpoint else None,'subset':a.subset,'suite':a.suite,
            'decoding':'greedy','max_new_tokens':2048,'context_limit':8192,'reasoning_effort':'medium',
            'tasks_sha256':hashlib.sha256((a.output/'tasks.json').read_bytes()).hexdigest(),'state':'running'}
    done=[]
    if a.resume:
        prior=json.loads((a.output/'config.json').read_text())
        if prior['state']=='completed':raise ValueError('Already completed')
        for key in ('model','checkpoint','subset','suite','decoding','max_new_tokens','context_limit','reasoning_effort','tasks_sha256'):
            if prior[key]!=config[key]:raise ValueError('Resume config mismatch: '+key)
        f=a.output/'responses.jsonl'
        done=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
        if len(done)>len(joblist) or [r['id'] for r in done]!=[r['id'] for r in joblist[:len(done)]]:raise ValueError('Invalid response prefix')
    runtime={'fast_hadamard':a.fast_hadamard,'single_device':a.single_device,'resumed_from':len(done),'time':time.time()}
    with (a.output/'runtime-segments.jsonl').open('a') as f:f.write(json.dumps(runtime)+'\n')
    config['runtime']=runtime
    (a.output/'config.json').write_text(json.dumps(config,indent=2));start=time.time()
    if a.model=='bonsai2':
        tool=ROOT/'tools/bonsai-runtime';backend=tool/'llama-prism-b10709-9a9394a';lines=[]
        for job in joblist:
            ids=job['input_ids'];base=f"{job['id']} {len(ids)} {len(ids)} "+' '.join(map(str,ids))
            lines.extend(['T '+base+' '+job['prompt'].encode().hex(),'G '+base])
        (a.output/'inputs.txt').write_text('\n'.join(lines)+'\n')
        env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(map(str,[backend,tool/'cuda-deps/nvidia/cuda_runtime/lib',tool/'cuda-deps/nvidia/cublas/lib']))
        with (a.output/'native.jsonl').open('w') as out,(a.output/'native.log').open('w') as err:
            subprocess.run([str(tool/'bonsai-eval'),str(ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf'),
                            str(a.output/'inputs.txt'),str(backend),'2048','8192'],env=env,stdout=out,stderr=err,check=True)
        generated={r['id']:r for r in map(json.loads,(a.output/'native.jsonl').read_text().splitlines())}
        for job in joblist:
            r=generated[job['id']];raw=tok.decode(r['tokens'],skip_special_tokens=False);answer,complete=final_text(raw,job['thinking'])
            row={**job,'raw_response':raw,'answer':answer,'thinking_completed':complete,'finish_reason':r['finish_reason'],'generated_tokens':len(r['tokens'])}
            with (a.output/'responses.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    else:
        if a.model=='student' and a.checkpoint is None:raise ValueError('student requires checkpoint')
        if a.fast_hadamard:
            from fast_inference import enable
            enable()
        model=load_source() if a.model=='source' else load_packed(a.checkpoint,single_device=a.single_device)
        model.eval();model.gradient_checkpointing_disable()
        (a.output/'cache-check.json').write_text(json.dumps(check_real_cache(model),indent=2))
        stop={tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')}
        for job in joblist[len(done):]:
            result=generate_ids(model,job['input_ids'],2048,stop)
            result['generated_tokens']=len(result['token_ids'])
            raw=tok.decode(result.pop('token_ids'),skip_special_tokens=False);answer,complete=final_text(raw,job['thinking'])
            row={**job,**result,'runtime':runtime,'raw_response':raw,'answer':answer,'thinking_completed':complete}
            with (a.output/'responses.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({'event':'benchmark_response','family':job['family'],'id':job['id'],'finish_reason':row['finish_reason'],'seconds':row['seconds']}),flush=True)
    config.update(state='completed',seconds=time.time()-start,timing_scope='current segment only; see runtime-segments.jsonl');(a.output/'config.json').write_text(json.dumps(config,indent=2))
if __name__=='__main__':main()
