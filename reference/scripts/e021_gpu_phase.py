"""One local teacher-scoring or QAT-update phase; no endpoints or other GPUs."""
import argparse,inspect,json,os,time
from pathlib import Path
import numpy as np
import torch
import ternary_core as core
from train_v2 import restore,save_latent,write_json
from fast_ternary import enable_fused_quantizer,gradients_finite_by_device
from e021_objective import loss_from_hidden

RUN=core.ROOT/'runs/e021-opd'

def progress(folder,stage,**kw):
    write_json(folder/'phase-status.json',dict(stage=stage,updated=time.time(),**kw))
    print(time.strftime('%H:%M:%S')+' | '+stage,flush=True)

def kernel(fla):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    if fla:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        qwen.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
    else:qwen.torch_chunk_gated_delta_rule=inspect.unwrap(qwen.torch_chunk_gated_delta_rule)

@torch.no_grad()
def chosen(model,row):
    p=len(row['prompt_ids']); ids=row['prompt_ids']+row['tokens']
    h=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False).last_hidden_state[0,p-1:-1]
    h=h.to(model.lm_head.weight.device)
    y=torch.tensor(row['tokens'],device=h.device)
    out=[]
    for start in range(0,len(h),64):
        lp=model.lm_head(h[start:start+64]).float().log_softmax(-1)
        out.extend(lp.gather(-1,y[start:start+64,None]).squeeze(-1).cpu().tolist())
    assert len(out)==len(row['tokens'])
    if not np.isfinite(out).all():raise RuntimeError('Nonfinite chosen-token logp')
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['teacher','opd','qad','parity'])
    parser.add_argument('folder',type=Path);parser.add_argument('--parent',type=Path)
    parser.add_argument('--step',type=int,default=0);a=parser.parse_args()
    core.guard();config=json.loads((RUN/'config.json').read_text())
    caps=[]
    for i in range(2):
        free,total=torch.cuda.mem_get_info(i)
        cap=free-config['safety_reserve_gib']*2**30
        torch.cuda.set_per_process_memory_fraction(cap/total,i);caps.append(cap/2**30)
    progress(a.folder,'Загрузка модели',memory_caps_gib=caps)
    kernel(False);model=core.load_source()
    if a.kind in ('teacher',):
        rows=[json.loads(l) for l in (a.folder/'rollouts.jsonl').open()]
        for i,row in enumerate(rows,1):
            row['teacher_logp']=chosen(model,row)
            with (a.folder/'teacher.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            progress(a.folder,f'Учитель: оценено {i}/{len(rows)} ответов',completed=i,total=len(rows))
        write_json(a.folder/'teacher-provenance.json',dict(source=str(core.SOURCE),compute='BF16 dequantized original FP8 source; reference GDN; no FLA',sampled_token_logp='Full vocabulary softmax on exact student token IDs including EOS',gpu_uuids=core.GPU_IDS,rows=len(rows)))
        return
    if a.parent is None:raise ValueError('Explicit parent required')
    core.ternarize(model,rotate=True);enable_fused_quantizer()
    # GPU7 has an unrelated21GiB service. Put the large vocabulary projection
    # AND its gradient/optimizer moments on the roomier physicalGPU6 before
    # restoring the optimizer. The tensors and mathematical objective stay equal.
    model.lm_head.to('cuda:0')
    torch.cuda.empty_cache()
    opt=restore(model,a.parent);core.set_alpha(model,1.)
    if a.kind in ('opd','parity'):
        rows=[json.loads(l) for l in (a.folder/('teacher.jsonl' if a.kind=='opd' else 'rollouts.jsonl')).open()]
        model.eval();deltas=[];seq=[]
        # Compare the actual training forward kernel with deployment behavior.
        kernel(True)
        for i,row in enumerate(rows,1):
            lp=chosen(model,row);d=np.abs(np.asarray(lp)-np.asarray(row['behavior_logp']))
            deltas.extend(d.tolist());seq.append(dict(id=row['id'],mean_abs=float(d.mean()),p95=float(np.quantile(d,.95))))
            progress(a.folder,f'Сверка вероятностей: {i}/{len(rows)}',completed=i,total=len(rows))
        parity=dict(mean_abs=float(np.mean(deltas)),p95=float(np.quantile(deltas,.95)),max_abs=float(np.max(deltas)),sequences=seq)
        gate=config['parity_gate'];parity['passed']=parity['mean_abs']<=gate['mean_abs_logp_max'] and parity['p95']<=gate['p95_abs_logp_max']
        write_json(a.folder/'parity.json',parity)
        if not parity['passed']:raise RuntimeError('Native/train chosen logp drift exceeds gate; no optimizer update: '+str({k:parity[k] for k in ('mean_abs','p95')}))
        if a.kind=='parity':return
    else:
        from train_kd import load_cache
        from e020_kd import loss_from_hidden as qad_loss
        rows=json.loads((a.folder/'qad-rows.json').read_text())
        cache_manifest=json.loads((core.ROOT/'data/e020-kd-v1/manifest.json').read_text())
    kernel(True);model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for layer in model.model.layers:layer.gradient_checkpointing=True
    for group in opt.param_groups:group['lr']=config['learning_rate']
    opt.zero_grad(set_to_none=True)
    for i in range(2):torch.cuda.reset_peak_memory_stats(i)
    started=time.monotonic();losses=[]
    # Do not step until the entire accumulation group and finite-gradient check pass.
    for i,row in enumerate(rows,1):
        if (RUN/'STOP').exists():raise RuntimeError('User STOP requested before optimizer update')
        if a.kind=='opd':
            ids=row['prompt_ids']+row['tokens'];p=len(row['prompt_ids'])
            h=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False).last_hidden_state[0,p-1:-1]
            h=h.to(model.lm_head.weight.device)
            loss=loss_from_hidden(model.lm_head,h,row['tokens'],row['behavior_logp'],row['teacher_logp'],row['task_advantage'])
        else:
            ids=row['input_ids'];h=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False).last_hidden_state[0,:-1]
            h=h.to(model.lm_head.weight.device)
            labels=torch.tensor(row['labels'][1:],device=h.device)
            cache=load_cache(row,cache_manifest)
            loss,_=qad_loss(model.lm_head,h,labels,cache,.25,.75,chunk_tokens=64)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective; no optimizer update')
        value=float(loss.detach());(loss/len(rows)).backward();losses.append(value)
        del loss,h
        progress(a.folder,f'Обучение {a.kind.upper()}: {i}/{len(rows)} ответов',completed=i,total=len(rows),seconds=time.monotonic()-started)
    if not gradients_finite_by_device(model.parameters()):raise RuntimeError('Nonfinite gradients; no optimizer update')
    # Adafactor has its inherited factored second moments; same parent and LR in both arms.
    torch.cuda.empty_cache()
    opt.step();opt.zero_grad(set_to_none=True)
    metrics=dict(step=a.step,arm=a.kind,loss=sum(losses)/len(losses),samples=len(rows),seconds=time.monotonic()-started,
                 peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],
                 peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)],memory_caps_gib=caps,
                 parent=str(a.parent),optimizer_updates=1,alpha=1.,gpu_uuids=core.GPU_IDS)
    metrics['placement']='layers0-31+embedding+lm_head GPU6; layers32-63+norm GPU7; vocabulary-head relocation before optimizer restore'
    progress(a.folder,'Сохранение обновлённых весов',**metrics)
    save_latent(model,opt,RUN/a.kind/'working-latent',metrics)
    write_json(a.folder/'metrics.json',metrics)
    progress(a.folder,'Шаг сохранён',**metrics)

if __name__=='__main__':main()
