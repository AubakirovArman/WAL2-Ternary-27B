"""Paired E022 QAT arm, sharded exclusively over authorized physical GPUs2/6."""
import argparse,fcntl,inspect,json,math,os,random,time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
import ternary_core as core
from train_v2 import restore,write_json
from fast_ternary import enable_fused_quantizer,gradients_finite_by_device
from e020_kd import loss_from_hidden
from train_kd import load_cache
from joint_ternary_scales import install,effective_scales,check

RUN=core.ROOT/'runs/e022-joint-scales'
def progress(folder,stage,**kw):
    value=dict(stage=stage,updated=time.time(),**kw)
    write_json(folder/'phase-status.json',value)
    print(time.strftime('%H:%M:%S')+' | '+stage,flush=True)


def locks():
    handles=[]
    for name in ['gpu2.lock','gpu67.lock']:
        f=(core.ROOT/'runs'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
    return handles


@torch.no_grad()
def validation(model,rows):
    model.eval();total=0.;tokens=0;categories={}
    for row in rows:
        ids=torch.tensor([row['input_ids']],device='cuda:0')
        h=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1].to(model.lm_head.weight.device)
        labels=torch.tensor(row['labels'][1:],device=h.device)
        loss,stats=loss_from_hidden(model.lm_head,h,labels,None,1.,0.,chunk_tokens=64)
        n=int((labels!=-100).sum());total+=float(loss)*n;tokens+=n
        c=categories.setdefault(row['category'],dict(nll=0.,tokens=0))
        c['nll']+=float(loss)*n;c['tokens']+=n
        del h,loss
    return dict(ce=total/tokens,tokens=tokens,examples=len(rows),
                categories={k:v['nll']/v['tokens'] for k,v in categories.items()})


@torch.no_grad()
def checkpoint(model,opt,gain_opt,path,metadata):
    # Standard latent format stores effective BF16 scales and can be exported
    # without the adapter. Extra base/gain state exists only for exact E022 resume.
    tmp=path.with_name(path.name+'.writing');tmp.mkdir(exist_ok=False)
    manifest={};scale_state={}
    for index,(name,module) in enumerate(model.named_modules()):
        values={k:p.detach().cpu().contiguous() for k,p in module.named_parameters(recurse=False) if k!='log_gain'}
        if hasattr(module,'scales'):
            values['scales']=effective_scales(module).cpu().contiguous()
        if hasattr(module,'log_gain'):
            scale_state[name+'.base']=module.scales.detach().cpu().contiguous()
            scale_state[name+'.gain']=module.log_gain.detach().cpu().contiguous()
        if values:
            file=f'shard-{index:04d}.safetensors';save_file(values,str(tmp/file));manifest[name]=file
    write_json(tmp/'manifest.json',manifest)
    if scale_state:save_file(scale_state,str(tmp/'scale-state.safetensors'))
    torch.save(dict(optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    random_state=random.getstate(),metadata=metadata),tmp/'training_state.pt')
    if gain_opt is not None:torch.save(gain_opt.state_dict(),tmp/'scale-optimizer.pt')
    write_json(tmp/'metadata.json',metadata)
    tmp.rename(path)


@torch.no_grad()
def scale_stats(model):
    count=0;mean=0.;maximum=0.;changed=0
    for m in model.modules():
        if not hasattr(m,'log_gain'):continue
        g=m.log_gain;n=g.numel();count+=n
        mean+=float(g.abs().sum());maximum=max(maximum,float(g.abs().max()))
        changed+=int((effective_scales(m)!=m.scales).sum())
    return dict(parameters=count,mean_abs_log_gain=mean/max(count,1),max_abs_log_gain=maximum,
                scales_changed_fraction=changed/max(count,1))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['fixed','joint']);a=ap.parse_args()
    cfg=json.loads((RUN/'config.json').read_text());core.GPU_IDS=cfg['gpu_uuids']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(core.GPU_IDS):raise RuntimeError('Only physicalGPU2/6 allowed')
    handles=locks()
    if torch.cuda.device_count()!=2:raise RuntimeError('Need exactly two authorized visible devices')
    caps=[]
    for i in range(2):
        free,total=torch.cuda.mem_get_info(i)
        if free<125*2**30:raise RuntimeError(f'GPU{cfg["gpu_indices"][i]} insufficient free memory; existing services left running')
        cap=free-cfg['safety_reserve_gib']*2**30
        torch.cuda.set_per_process_memory_fraction(cap/total,i);caps.append(cap/2**30)
    torch.set_num_threads(8);torch.manual_seed(20261007)
    folder=RUN/a.arm;folder.mkdir(exist_ok=True)
    for i in range(2):
        result=check(f'cuda:{i}');write_json(folder/f'kernel-check-gpu{cfg["gpu_indices"][i]}.json',result)
    torch.set_num_threads(8)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    qwen.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
    progress(folder,'Загрузка исходной модели и весов E020-B-1024',arm=a.arm,memory_caps_gib=caps)
    model=core.load_source();core.ternarize(model,rotate=True);enable_fused_quantizer()
    model.lm_head.to('cuda:0');torch.cuda.empty_cache()
    saved=[folder/f'checkpoint-{s:03d}' for s in cfg['checkpoints'] if (folder/f'checkpoint-{s:03d}/metadata.json').exists()]
    parent=saved[-1] if saved else Path(cfg['parent_latent'])
    opt=restore(model,parent);core.set_alpha(model,1.)
    start=json.loads((parent/'metadata.json').read_text())['step'] if saved else 0
    gains=[];gain_opt=None
    if a.arm=='joint':
        gains=install(model);gain_opt=torch.optim.Adam(gains,lr=cfg['scale_lr'],foreach=False)
        if saved:
            values=load_file(str(parent/'scale-state.safetensors'))
            with torch.no_grad():
                for name,m in model.named_modules():
                    if hasattr(m,'log_gain'):
                        m.scales.copy_(values[name+'.base']);m.log_gain.copy_(values[name+'.gain'])
            gain_opt.load_state_dict(torch.load(parent/'scale-optimizer.pt',map_location='cpu',weights_only=True))
    for group in opt.param_groups:group['lr']=cfg['weight_lr']
    train=json.loads((RUN/'train.json').read_text());val=json.loads((RUN/'validation.json').read_text())
    cache_meta=json.loads(Path(cfg['cache_manifest']).read_text())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for layer in model.model.layers:layer.gradient_checkpointing=True
    progress(folder,'Короткая контрольная проверка: 32 примера',arm=a.arm,step=start,total=cfg['updates'])
    baseline=validation(model,val)
    write_json(folder/('initial-validation.json' if start==0 else f'resume-validation-{start:03d}.json'),dict(step=start,**baseline))
    started=time.monotonic()
    for step in range(start+1,cfg['updates']+1):
        if (RUN/'STOP').exists():raise RuntimeError('User STOP requested')
        model.train();opt.zero_grad(set_to_none=True)
        if gain_opt is not None:gain_opt.zero_grad(set_to_none=True)
        for i in range(2):torch.cuda.reset_peak_memory_stats(i)
        t=time.monotonic();losses=[];positions=0
        rows=train[(step-1)*cfg['samples_per_update']:step*cfg['samples_per_update']]
        for index,row in enumerate(rows,1):
            if (RUN/'STOP').exists():raise RuntimeError('User STOP requested before optimizer update')
            h=model.model(input_ids=torch.tensor([row['input_ids']],device='cuda:0'),use_cache=False).last_hidden_state[0,:-1]
            h=h.to(model.lm_head.weight.device);labels=torch.tensor(row['labels'][1:],device=h.device)
            loss,stats=loss_from_hidden(model.lm_head,h,labels,load_cache(row,cache_meta),.25,.75,chunk_tokens=64)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; no optimizer update')
            losses.append(dict(id=row['id'],category=row['category'],loss=float(loss.detach()),**stats))
            (loss/len(rows)).backward();positions+=len(h)
            del loss,h,labels
            progress(folder,f'{a.arm}: шаг {step}/{cfg["updates"]}, пример {index}/{len(rows)}',
                     arm=a.arm,step=step,total=cfg['updates'],sample=index,memory_caps_gib=caps)
        if not gradients_finite_by_device(model.parameters()):raise RuntimeError('Nonfinite gradients; no optimizer update')
        if step==1 and gains and not any(bool((p.grad!=0).any()) for p in gains):raise RuntimeError('Scale gradients are all zero')
        torch.cuda.empty_cache();opt.step()
        if gain_opt is not None:
            gain_opt.step()
            with torch.no_grad():
                for p in gains:p.clamp_(-cfg['max_abs_log_gain'],cfg['max_abs_log_gain'])
        opt.zero_grad(set_to_none=True)
        if gain_opt is not None:gain_opt.zero_grad(set_to_none=True)
        elapsed=time.monotonic()-t
        metrics=dict(arm=a.arm,step=step,total=cfg['updates'],samples=len(rows),positions=positions,
                     loss=sum(r['loss'] for r in losses)/len(losses),seconds=elapsed,tokens_per_second=positions/elapsed,
                     peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],
                     peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)],
                     gpu_indices=cfg['gpu_indices'],memory_caps_gib=caps,examples=losses)
        with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(metrics,ensure_ascii=False)+'\n');f.flush()
        progress(folder,f'{a.arm}: шаг {step}/{cfg["updates"]} выполнен; {elapsed:.1f} сек; {positions/elapsed:.0f} ток/с',
                 **{k:v for k,v in metrics.items() if k!='examples'},eta_seconds=(cfg['updates']-step)*(time.monotonic()-started)/(step-start))
        if step in cfg['checkpoints']:
            progress(folder,f'Шаг {step}: проверка 32 контрольных примеров',arm=a.arm,step=step,total=cfg['updates'])
            v=validation(model,val);metrics.update(validation=v,scale_stats=scale_stats(model))
            write_json(folder/f'validation-{step:03d}.json',metrics)
            progress(folder,f'Шаг {step}: ошибка {v["ce"]:.6f}; сохранение весов',arm=a.arm,step=step,total=cfg['updates'],validation=v)
            checkpoint(model,opt,gain_opt,folder/f'checkpoint-{step:03d}',metrics)
            progress(folder,f'Шаг {step}: веса сохранены',arm=a.arm,step=step,total=cfg['updates'],validation=v)
    write_json(folder/'training-completed.json',dict(arm=a.arm,step=cfg['updates'],updated=time.time()))

if __name__=='__main__':main()
