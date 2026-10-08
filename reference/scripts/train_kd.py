"""E006: matched hard-label control and compressed-logit KD, followed by selected recovery."""
import gc,hashlib,json,math,random,shutil,time
from pathlib import Path
import torch
from safetensors.torch import load_file
import ternary_core as core
from train_v2 import restore,save_latent,write_json
from fast_ternary import enable_fused_quantizer,gradients_finite_by_device
from kd_objective import student_loss

ROOT=core.ROOT;RUN=ROOT/'runs/e006-soft-kd';INITIAL=ROOT/'runs/e005-data-lr/continuation/best-latent'
def load_cache(example,meta):
    info=meta['examples'][str(example['id'])]
    assert info['input_sha256']==hashlib.sha256(json.dumps(example['input_ids']).encode()).hexdigest()
    values=load_file(info['file'])
    assert values['input_ids'].tolist()==example['input_ids'] and values['labels'].tolist()==example['labels']
    assert values['top_ids'].shape==(len(example['input_ids'])-1,meta['topk'])
    return values

def stage(model,parent,folder,order,validation,verified,meta,lr,kd_weight,steps,deadline,decay=False):
    folder.mkdir(exist_ok=False);opt=restore(model,parent);core.set_alpha(model,1.)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    cfg={'parent':str(parent),'lr':lr,'kd_weight':kd_weight,'steps':steps,'alpha':1.,'max_length':512,
         'teacher_manifest':str(ROOT/'data/v3-kd/manifest.json'),'corpus_sha256':meta['corpus_sha256'],
         'teacher_topk':meta['topk'],'teacher_temperature':1.,'kd_positions':'all next tokens',
         'lr_schedule':'cosine_to_0.2' if decay else 'constant','optimizer':'restored Adafactor from parent'}
    write_json(folder/'config.json',cfg);write_json(folder/'order.json',[r['id'] for r in order])
    if folder.name=='hard-lr3e-6':
        # Exercise the real 27B KD backward before a long job; perform no optimizer update.
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
        model.train();opt.zero_grad(set_to_none=True)
        smoke,smoke_kl,smoke_ce=student_loss(model,order[0],load_cache(order[0],meta),0.9)
        if not torch.isfinite(smoke):raise RuntimeError('Nonfinite GPU KD smoke loss')
        smoke.backward()
        if not gradients_finite_by_device(model.parameters()):raise RuntimeError('Nonfinite GPU KD smoke gradients')
        core.event('kd_gpu_smoke_passed',loss=smoke.item(),kl=smoke_kl.item(),ce=smoke_ce.item(),
                   peak_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)])
        opt.zero_grad(set_to_none=True);del smoke,smoke_kl,smoke_ce
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
    baseline=core.evaluate(model,validation);metrics={'state':'running','initial_validation':baseline,'validation':[],
            'best_validation_ce':baseline['answer_ce'],'best_checkpoint':str(parent),'best_step':0}
    write_json(folder/'metrics.json',metrics);core.event('kd_stage_started',stage=folder.name,config=cfg,initial=baseline)
    best=baseline['answer_ce'];stale=0;step=0
    for step in range(1,steps+1):
        if (RUN/'STOP').exists() or time.time()>=deadline:break
        example=order[(step-1)%len(order)];model.train();opt.zero_grad(set_to_none=True);start=time.monotonic()
        rate=lr*(0.2+0.8*(1+math.cos(math.pi*(step-1)/max(steps-1,1)))/2) if decay else lr
        for group in opt.param_groups:group['lr']=rate
        if kd_weight:
            cache=load_cache(example,meta);loss,kl,ce=student_loss(model,example,cache,kd_weight)
        else:
            loss,_=core.loss_for(model,example);kl=torch.tensor(0.);ce=loss.detach()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite KD loss')
        loss.backward()
        if not gradients_finite_by_device(model.parameters()):raise RuntimeError('Nonfinite KD gradients')
        opt.step();opt.zero_grad(set_to_none=True)
        row={'step':step,'loss':loss.item(),'kl':kl.item(),'train_ce':ce.item(),'lr':rate,
             'example_id':example['id'],'positions':len(example['input_ids'])-1,'seconds':time.monotonic()-start}
        with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        core.event('kd_train_step',stage=folder.name,**row)
        interval=256 if decay else 128
        if step%interval==0 or step==steps:
            val=core.evaluate(model,validation);vextra=core.evaluate(model,verified)
            entry={'step':step,**val,'verified_ce':vextra['answer_ce']};metrics['validation'].append(entry)
            meaningful=val['answer_ce']<best-0.002;stale=0 if meaningful else stale+1
            if val['answer_ce']<best:
                best=val['answer_ce']
                save_latent(model,opt,folder/'best-latent',{'step':step,'config':cfg,'validation':entry,
                            'order_ids':[r['id'] for r in order],'next_order_index':step%len(order),'alpha':1.})
                metrics.update(best_validation_ce=best,best_checkpoint=str(folder/'best-latent'),best_step=step)
            metrics['last_step']=step;write_json(folder/'metrics.json',metrics);core.event('kd_validation',stage=folder.name,**entry)
            if decay and stale>=4:core.event('kd_early_stop',stage=folder.name,step=step);break
    metrics['state']='stopped' if (RUN/'STOP').exists() else 'deadline' if time.time()>=deadline else 'completed'
    write_json(folder/'metrics.json',metrics);del opt;gc.collect();torch.cuda.empty_cache();return metrics

def main():
    core.guard();RUN.mkdir(exist_ok=False);(RUN/'code').mkdir()
    for name in ['train_kd.py','kd_objective.py','cache_teacher_logits.py','ternary_core.py','fast_ternary.py','train_v2.py']:
        shutil.copy2(ROOT/'scripts'/name,RUN/'code'/name)
    meta=json.loads((ROOT/'data/v3-kd/manifest.json').read_text());assert meta['state']=='completed'
    assert len(meta['examples'])==meta['count_target']==4096
    assert hashlib.sha256(Path(meta['corpus']).read_bytes()).hexdigest()==meta['corpus_sha256']
    data=core.load_examples(Path(meta['corpus']),512);byid={e['id']:e for e in data['train']}
    order=[byid[i] for i in meta['order_ids']];assert len({r['id'] for r in order})==4096
    verified=core.load_examples(ROOT/'data/v2/verified_holdout.jsonl',512)['validation'];validation=data['validation']
    assert len(validation)==128
    deadline=time.time()+6*3600
    cfg={'initial':str(INITIAL),'candidates':[{'name':'hard-lr3e-6','lr':3e-6,'kd':0.},
         {'name':'kd-lr3e-6','lr':3e-6,'kd':0.9},{'name':'kd-lr1e-5','lr':1e-5,'kd':0.9}],
         'pilot_steps':256,'continuation_steps':4096,'continue_if_gain':0.003,'deadline_epoch':deadline,
         'scope':'Recover own E005 weights toward original Qwen distribution; no Bonsai weights used.'}
    write_json(RUN/'config.json',cfg);model=core.load_source();core.ternarize(model,rotate=True);enable_fused_quantizer()
    results={'state':'pilots','candidates':[]};write_json(RUN/'metrics.json',results)
    for c in cfg['candidates']:
        if (RUN/'STOP').exists():break
        m=stage(model,INITIAL,RUN/c['name'],order,validation,verified,meta,c['lr'],c['kd'],256,deadline)
        results['candidates'].append({**c,**m});write_json(RUN/'metrics.json',results)
    if not results['candidates']:raise RuntimeError('Stopped before pilots')
    winner=min(results['candidates'],key=lambda x:x['best_validation_ce']);results['selected_pilot']=winner
    selected=Path(winner['best_checkpoint'])
    if winner['best_validation_ce']<winner['initial_validation']['answer_ce']-0.003 and not (RUN/'STOP').exists() and time.time()<deadline:
        results['state']='continuation';write_json(RUN/'metrics.json',results)
        # Start after the pilot's consumed prefix. Deterministic fixed corpus for this ablation.
        shifted=order[256:]+order[:256]
        m=stage(model,selected,RUN/'continuation',shifted,validation,verified,meta,winner['lr'],winner['kd'],4096,deadline,True)
        results['continuation']=m;selected=Path(m['best_checkpoint'])
    else:results['continuation_note']='No improvement >=0.003, STOP, or deadline.'
    opt=restore(model,selected);del opt;model.gradient_checkpointing_disable();core.set_alpha(model,1.)
    results['selected_validation']=core.evaluate(model,validation)
    results['selected_test']=core.evaluate(model,data['test'])
    results['selected_latent']=str(selected)
    if selected!=INITIAL:
        core.export_packed(model,RUN/'selected-ternary');results['selected_packed']=str(RUN/'selected-ternary')
    else:results['selected_packed']=str(ROOT/'runs/e005-data-lr/selected-ternary')
    results['state']='completed';write_json(RUN/'metrics.json',results);core.event('kd_completed',metrics=results)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if RUN.exists():write_json(RUN/'failure.json',{'error':repr(e),'time':time.time()})
        core.event('kd_failed',error=repr(e));raise
