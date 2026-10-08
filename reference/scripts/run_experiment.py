import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
import torch
from transformers import Adafactor
from ternary_core import ROOT,guard,event,load_source,load_examples,evaluate,ternarize,set_alpha,loss_for,export_packed

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',default='e001-pilot')
    ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--ramp-steps',type=int,default=8)
    ap.add_argument('--eval-every',type=int,default=8)
    ap.add_argument('--max-length',type=int,default=256)
    ap.add_argument('--eval-examples',type=int,default=4)
    ap.add_argument('--lr',type=float,default=3e-5)
    ap.add_argument('--source-only',action='store_true')
    ap.add_argument('--refresh-data',action='store_true')
    ap.add_argument('--save-latent',action='store_true')
    ap.add_argument('--rotate',action='store_true')
    ap.add_argument('--data',default=str(ROOT/'data/teacher_pairs.jsonl'))
    ap.add_argument('--fast-kernels',action='store_true',help='Fused quantization at alpha=1 and grouped finite-gradient checks')
    a=ap.parse_args();guard()
    run=ROOT/'runs'/a.run;run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(json.dumps(vars(a),indent=2))
    (run/'code').mkdir()
    for name in ['ternary_core.py','run_experiment.py','collect_teacher.py','fast_ternary.py']:
        shutil.copy2(ROOT/'scripts'/name,run/'code'/name)
    if a.fast_kernels:
        from fast_ternary import enable_fused_quantizer,gradients_finite_by_device
        enable_fused_quantizer()
        event('fast_kernels_enabled',run=a.run,scope='alpha=1 quantization, grouped finite checks; reference quantizer during ramp')
    corpus=Path(a.data)
    data_bytes=corpus.read_bytes();(run/'teacher_snapshot.jsonl').write_bytes(data_bytes)
    event('run_started',run=a.run,config=vars(a),data_sha256=hashlib.sha256(data_bytes).hexdigest())
    data=load_examples(run/'teacher_snapshot.jsonl',a.max_length)
    if min(len(data[k]) for k in data)<1:raise RuntimeError('Need examples in all splits')
    validation=data['validation'][:a.eval_examples];test=data['test'][:a.eval_examples]
    (run/'split_ids.json').write_text(json.dumps({k:[e['id'] for e in v] for k,v in data.items()},indent=2))
    model=load_source()
    metrics={'source_validation':evaluate(model,validation),'source_test':evaluate(model,test)}
    event('source_validation',run=a.run,**metrics['source_validation'])
    if a.source_only:
        (run/'metrics.json').write_text(json.dumps(metrics,indent=2));return
    ternarize(model,rotate=a.rotate)
    metrics['ternary_initial_validation']=evaluate(model,validation)
    event('ternary_initial_validation',run=a.run,**metrics['ternary_initial_validation'])
    export_packed(model,run/'initial-ternary')
    best=metrics['ternary_initial_validation']['answer_ce'];best_path='initial-ternary'
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2))
    if a.steps==0:
        event('quantization_only_completed',run=a.run,metrics=metrics);return
    opt=Adafactor([p for p in model.parameters() if p.requires_grad],lr=a.lr,
        relative_step=False,scale_parameter=False,warmup_init=False,beta1=None,weight_decay=0.0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    rng=random.Random(20260919);order=list(data['train']);rng.shuffle(order)
    for step in range(1,a.steps+1):
        t=time.time();model.train()
        alpha=min(1.,step/max(1,a.ramp_steps));set_alpha(model,alpha)
        e=order[(step-1)%len(order)]
        opt.zero_grad(set_to_none=True)
        loss,n=loss_for(model,e)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        # Adafactor clips each tensor update internally; reject nonfinite gradients before any update.
        if a.fast_kernels:
            if not gradients_finite_by_device(model.parameters()):raise RuntimeError(f'Nonfinite gradient at {step}')
        else:
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError(f'Nonfinite gradient at {step}')
        opt.step();opt.zero_grad(set_to_none=True)
        record={'step':step,'alpha':alpha,'train_ce':loss.item(),'answer_tokens':n,'example_id':e['id'],
                'seconds':time.time()-t,'allocated_gib':[torch.cuda.memory_allocated(i)/2**30 for i in range(2)],
                'peak_gib':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]}
        with (run/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        event('train_step',run=a.run,**record)
        if step%a.eval_every==0 or step==a.steps:
            set_alpha(model,1.)
            val=evaluate(model,validation);val['step']=step
            metrics.setdefault('validation',[]).append(val);event('validation',run=a.run,**val)
            if val['answer_ce']<best:
                best=val['answer_ce'];best_path=f'ternary-step-{step:04d}'
                export_packed(model,run/best_path)
            metrics['best_validation_ce']=best;metrics['best_checkpoint']=best_path
            metrics['last_step']=step;metrics['state']='training'
            (run/'metrics.json').write_text(json.dumps(metrics,indent=2))
            if a.refresh_data:
                snapshot=run/f'teacher_snapshot_step_{step:04d}.jsonl'
                snapshot.write_bytes(corpus.read_bytes())
                refreshed=load_examples(snapshot,a.max_length)
                order=refreshed['train'];rng.shuffle(order)
                event('training_data_refreshed',run=a.run,step=step,train_examples=len(order),
                      sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        if (run/'STOP').exists():event('stop_requested',run=a.run,step=step);break
    set_alpha(model,1.)
    final_path=f'final-ternary-step-{step:04d}'
    if best_path==f'ternary-step-{step:04d}':
        final_path=best_path
    else:
        export_packed(model,run/final_path)
    metrics['final_test']=evaluate(model,test)
    metrics['best_validation_ce']=best;metrics['best_checkpoint']=best_path
    metrics['final_checkpoint']=final_path
    metrics['state']='completed'
    metrics['test_note']='Test score belongs to final checkpoint, not necessarily best validation checkpoint.'
    if a.save_latent:
        from safetensors.torch import save_file
        latent=run/'resume-latent';latent.mkdir(exist_ok=True)
        manifest={}
        for index,(name,module) in enumerate(model.named_modules()):
            tensors={k:p.detach().cpu().contiguous() for k,p in module.named_parameters(recurse=False)}
            tensors.update({k:b.detach().cpu().contiguous() for k,b in module.named_buffers(recurse=False) if k=='scales'})
            if tensors:
                filename=f'shard-{index:04d}.safetensors';save_file(tensors,str(latent/filename));manifest[name]=filename
        (latent/'manifest.json').write_text(json.dumps(manifest,indent=2))
        torch.save({'optimizer':opt.state_dict(),'step':step,'random_state':rng.getstate(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},latent/'training_state.pt')
        event('latent_checkpoint_saved',run=a.run,path=str(latent))
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2))
    event('run_completed',run=a.run,metrics=metrics)

if __name__=='__main__':
    try:main()
    except Exception as e:
        event('run_failed',error=repr(e));raise
