"""Bounded LR comparison followed by selected continuation; only physical GPU 6/7."""
import gc, hashlib, json, math, random, shutil, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Adafactor
import ternary_core as core
from fast_ternary import enable_fused_quantizer, gradients_finite_by_device
from data_v2 import build, read_rows

ROOT=core.ROOT;RUN=ROOT/'runs/e005-data-lr'
INITIAL=ROOT/'runs/e004-rotated-qat/resume-latent'

def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def restore(model,path):
    manifest=json.loads((path/'manifest.json').read_text())
    with torch.no_grad():
        for name,filename in manifest.items():
            module=model.get_submodule(name)
            with safe_open(path/filename,framework='pt',device='cpu') as f:
                for key in f.keys():getattr(module,key).copy_(f.get_tensor(key))
    # Trusted local checkpoint produced by our own scripts, never remote pickle.
    state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
    opt=Adafactor([p for p in model.parameters() if p.requires_grad],lr=1e-5,
                 relative_step=False,scale_parameter=False,warmup_init=False,beta1=None,weight_decay=0.)
    opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    if 'random_state' in state:random.setstate(state['random_state'])
    return opt

def save_latent(model,opt,path,metadata):
    """Commit only after all tensors/state are saved; retain previous best until commit."""
    tmp=path.with_name(path.name+'.writing');tmp.mkdir(exist_ok=False)
    manifest={}
    for index,(name,module) in enumerate(model.named_modules()):
        tensors={k:p.detach().cpu().contiguous() for k,p in module.named_parameters(recurse=False)}
        tensors.update({k:b.detach().cpu().contiguous() for k,b in module.named_buffers(recurse=False) if k=='scales'})
        if tensors:
            filename=f'shard-{index:04d}.safetensors';save_file(tensors,str(tmp/filename));manifest[name]=filename
    write_json(tmp/'manifest.json',manifest)
    torch.save({'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'random_state':random.getstate(),'metadata':metadata},tmp/'training_state.pt')
    write_json(tmp/'metadata.json',metadata)
    backup=path.with_name(path.name+'.previous')
    if path.exists():path.rename(backup)
    tmp.rename(path)
    if backup.exists():shutil.rmtree(backup)  # Only this stage's superseded checkpoint.
    core.event('v2_latent_saved',path=str(path),step=metadata['step'])

def schedule(examples,corpus,count,seed=41271):
    languages={r['id']:r.get('language','English') for r in read_rows(corpus)}
    queues={True:[],False:[]}
    for row in examples:queues[languages[row['id']]=='Russian'].append(row)
    rng=random.Random(seed)
    for q in queues.values():rng.shuffle(q)
    pos={True:0,False:0};out=[]
    for i in range(count):
        ru=i%10<3
        if not queues[ru]:ru=not ru
        q=queues[ru]
        if pos[ru]==len(q):rng.shuffle(q);pos[ru]=0
        out.append(q[pos[ru]]);pos[ru]+=1
    return out

def train_stage(model,parent,folder,corpus,lr,steps,validation,verified,decay=False,deadline=None):
    folder.mkdir(exist_ok=False);opt=restore(model,parent)
    core.set_alpha(model,1.);model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    data=core.load_examples(corpus,512)
    order=schedule(data['train'],corpus,steps)
    cfg={'parent':str(parent),'lr':lr,'steps':steps,'max_length':512,'validation_examples':len(validation),
         'data':str(corpus),'data_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),'russian_sampling_fraction':0.3,
         'decay':'cosine_to_0.1' if decay else 'constant','minimum_improvement':0.002,'patience':3 if decay else None}
    write_json(folder/'config.json',cfg);write_json(folder/'order.json',[e['id'] for e in order])
    initial=core.evaluate(model,validation);secondary=core.evaluate(model,verified)
    best=initial['answer_ce'];best_path=parent;best_step=0;stale=0
    metrics={'state':'running','initial_validation':initial,'initial_verified_ce':secondary,'validation':[],
             'best_validation_ce':best,'best_checkpoint':str(best_path)}
    write_json(folder/'metrics.json',metrics);core.event('v2_stage_started',stage=folder.name,config=cfg,initial=initial)
    for step,example in enumerate(order,1):
        if (RUN/'STOP').exists() or (deadline and time.time()>deadline):break
        model.train();t=time.monotonic();current_lr=lr*(0.1+0.9*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))/2) if decay else lr
        for group in opt.param_groups:group['lr']=current_lr
        opt.zero_grad(set_to_none=True)
        loss,n=core.loss_for(model,example)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward()
        if not gradients_finite_by_device(model.parameters()):raise RuntimeError('Nonfinite gradient')
        opt.step();opt.zero_grad(set_to_none=True)
        row={'step':step,'loss':loss.item(),'lr':current_lr,'seconds':time.monotonic()-t,'example_id':example['id'],'answer_tokens':n}
        with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        core.event('v2_train_step',stage=folder.name,**row)
        interval=256 if decay else 128
        if step%interval==0 or step==steps:
            val=core.evaluate(model,validation);extra=core.evaluate(model,verified)
            entry={'step':step,**val,'verified_ce':extra['answer_ce']};metrics['validation'].append(entry)
            core.event('v2_validation',stage=folder.name,**entry)
            meaningful=val['answer_ce']<best-0.002
            if val['answer_ce']<best:
                best=val['answer_ce'];best_step=step
                save_latent(model,opt,folder/'best-latent',{'step':step,'config':cfg,'next_order_index':step,
                            'order_ids':[e['id'] for e in order],'validation':entry,'alpha':1.0})
                best_path=folder/'best-latent'
            stale=0 if meaningful else stale+1
            metrics.update(last_step=step,best_validation_ce=best,best_checkpoint=str(best_path),best_step=best_step)
            write_json(folder/'metrics.json',metrics)
            if decay and stale>=3:
                core.event('v2_early_stop',stage=folder.name,step=step);break
            if decay and step%512==0 and step<steps:
                fresh=folder/f'corpus-step-{step:04d}.jsonl';build(fresh,target=12000)
                data=core.load_examples(fresh,512);corpus=fresh
                remaining=schedule(data['train'],fresh,steps-step,seed=41271+step)
                order[step:]=remaining
                cfg.update(data=str(fresh),data_sha256=hashlib.sha256(fresh.read_bytes()).hexdigest())
                write_json(folder/'order.json',[e['id'] for e in order])
                core.event('v2_data_refreshed',stage=folder.name,step=step,path=str(fresh))
    metrics['state']='stopped' if (RUN/'STOP').exists() else 'deadline' if deadline and time.time()>deadline else 'completed'
    write_json(folder/'metrics.json',metrics)
    del opt;gc.collect();torch.cuda.empty_cache()
    return metrics

def main():
    core.guard();RUN.mkdir(exist_ok=False);(RUN/'code').mkdir()
    for name in ['train_v2.py','ternary_core.py','fast_ternary.py','data_v2.py','collect_teacher_v2.py','inspect_checkpoint.py']:
        shutil.copy2(ROOT/'scripts'/name,RUN/'code'/name)
    deadline=time.time()+10*3600
    config={'candidate_lrs':[1e-5,3e-6,1e-6],'pilot_steps':256,'continuation_steps':4096,
            'deadline_epoch':deadline,'selection':'lowest CE on original 128 validation; test not used',
            'continue_if_improvement':0.005,'initial':str(INITIAL)}
    write_json(RUN/'config.json',config)
    corpus=ROOT/'data/v2/pilot.jsonl';data=core.load_examples(corpus,512)
    validation=data['validation'];assert len(validation)==128
    verified=core.load_examples(ROOT/'data/v2/verified_holdout.jsonl',512)['validation']
    model=core.load_source()
    source_validation=core.evaluate(model,validation)
    result={'state':'pilots','source_validation':source_validation,'candidates':[]}
    write_json(RUN/'metrics.json',result);core.event('v2_source_validation',**source_validation)
    core.ternarize(model,rotate=True);enable_fused_quantizer()
    for lr in config['candidate_lrs']:
        if (RUN/'STOP').exists():break
        metrics=train_stage(model,INITIAL,RUN/f'pilot-lr-{lr:g}',corpus,lr,256,validation,verified,deadline=deadline)
        result['candidates'].append({'lr':lr,**metrics});write_json(RUN/'metrics.json',result)
    if not result['candidates']:raise RuntimeError('Stopped before first candidate')
    winner=min(result['candidates'],key=lambda m:m['best_validation_ce'])
    result['selected_lr']=winner['lr'];result['selected_pilot']=winner
    selected=Path(winner['best_checkpoint'])
    if not (RUN/'STOP').exists() and winner['best_validation_ce']<winner['initial_validation']['answer_ce']-0.005:
        result['state']='continuation';write_json(RUN/'metrics.json',result)
        corpus=RUN/'continuation-data.jsonl';build(corpus,target=12000)
        long=train_stage(model,selected,RUN/'continuation',corpus,winner['lr'],4096,validation,verified,decay=True,deadline=deadline)
        result['continuation']=long;selected=Path(long['best_checkpoint'])
    else:result['continuation_note']='No pilot gain >=0.005, or STOP requested; no long run.'
    # Restore the selected BEST state; never report final-step test as best-state test.
    opt=restore(model,selected);del opt;gc.collect();core.set_alpha(model,1.)
    model.gradient_checkpointing_disable()
    core.export_packed(model,RUN/'selected-ternary')
    result['selected_latent']=str(selected);result['selected_packed']=str(RUN/'selected-ternary')
    result['selected_validation']=core.evaluate(model,validation)
    previous=json.loads((ROOT/'reports/e004-best-validation128.json').read_text())['validation']
    result['e004_best_validation_128']=previous
    result['beats_previous_best']=result['selected_validation']['answer_ce']<previous['answer_ce']
    result['recommended_packed']=str(RUN/'selected-ternary') if result['beats_previous_best'] else str(ROOT/'runs/e004-rotated-qat/ternary-step-0384')
    result['selected_test']=core.evaluate(model,data['test'])
    held=core.load_examples(ROOT/'data/v2/verified_holdout.jsonl',512)
    result['verified_test_ce']=core.evaluate(model,held['test'])
    from inspect_checkpoint import generate_reference
    checks=[]
    for row in [r for r in read_rows(ROOT/'data/v2/verified_holdout.jsonl') if r['split']=='test'][:16]:
        answer=generate_reference(model,row['instruction'],max_new_tokens=64)
        try:correct=json.loads(answer)==json.loads(row['response']) if row['checker']=='json' else answer.strip()==row['response'].strip()
        except (ValueError,TypeError):correct=False
        checks.append({'id':row['id'],'topic':row['topic'],'instruction':row['instruction'],'expected':row['response'],'answer':answer,'correct':correct})
    write_json(RUN/'verified-generation.json',checks)
    result['verified_exact_match']={'correct':sum(r['correct'] for r in checks),'total':len(checks)}
    write_json(RUN/'metrics.json',result)
    del model;gc.collect();torch.cuda.empty_cache()
    source=core.load_source()
    result['source_test']=core.evaluate(source,data['test'])
    result['state']='completed';write_json(RUN/'metrics.json',result);core.event('v2_completed',metrics=result)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if RUN.exists():write_json(RUN/'failure.json',{'error':repr(e),'time':time.time()})
        core.event('v2_failed',error=repr(e));raise
