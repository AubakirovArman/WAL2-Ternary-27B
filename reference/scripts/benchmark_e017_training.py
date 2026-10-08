"""Matched end-to-end QAT timings on GPUs6/7; updates are discarded, no checkpoints saved."""
import copy,gc,json,random,statistics,time
from pathlib import Path
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
import ternary_core as core
from train_v2 import restore,write_json
from train_reasoning_qat import chunked_ce
from chunked_kd import loss_from_hidden
from train_kd import load_cache
from fast_ternary import enable_fused_quantizer,gradients_finite_by_device

OUT=core.ROOT/'reports/e017-training-speed-20260925'
def sync():
    for i in range(2):torch.cuda.synchronize(i)
def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return copy.deepcopy(x)
def samples(params,gradient=False):
    return torch.cat([(p.grad if gradient else p).detach().flatten()[::max(1,p.numel()//2048)][:2048].float().cpu() for p in params])
def main():
    core.guard();OUT.mkdir(exist_ok=False)
    report=dict(state='loading',checkpoint='runs/e017-fresh10000/candidates/step-10000',scope='24 real mixed-loss QAT optimizer updates per mode from identical initial weights/state/RNG; first4 warmup excluded. Old10000dataset. FLA enabled in every mode. No training checkpoints written; all updates discarded.',modes=[])
    def save(state=None,**kw):
        if state:report['state']=state
        report.update(kw);write_json(OUT/'metrics.json',report)
    save();print('Загрузка E017 для теста скорости, толькоGPU6/7.',flush=True)
    model=core.load_source();core.ternarize(model,rotate=True);enable_fused_quantizer();opt=restore(model,core.ROOT/report['checkpoint']);core.set_alpha(model,1.)
    for g in opt.param_groups:g['lr']=5e-7
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();qwen.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
    params=[p for p in model.parameters() if p.requires_grad]
    initial=[p.detach().cpu() for p in params];initial_opt=cpu_tree(opt.state_dict());cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
    rows=[json.loads(l) for l in (core.ROOT/'data/fresh-10000-v2/train.jsonl').open()];random.Random(20260925).shuffle(rows);chosen=rows[:24]
    meta=json.loads((core.ROOT/'data/v3-kd/manifest.json').read_text());old=core.load_examples(Path(meta['corpus']),512);lookup={r['id']:r for r in old['train']};order=[lookup[i] for i in meta['order_ids']]
    replay=[order[(6784+i)%len(order)] for i in range(24)]
    write_json(OUT/'order.json',[dict(reasoning=r['id'],reasoning_tokens=len(r['input_ids']),replay=e['id'],replay_tokens=len(e['input_ids'])) for r,e in zip(chosen,replay)])
    reference_grad=None;reference_losses=None;initial_samples=samples(params);reference_update=None
    def iteration(row,example,chunk,update):
        opt.zero_grad(set_to_none=True);sync();start=time.perf_counter()
        ids=torch.tensor([row['input_ids']],device='cuda:0');hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
        ce,_=chunked_ce(model.lm_head,hidden,torch.tensor(row['labels'][1:],device=hidden.device),chunk=chunk)
        if not torch.isfinite(ce):raise RuntimeError('Nonfinite CE')
        (.25*ce).backward();ceval=float(ce.detach());del hidden,ce,ids;sync();ceend=time.perf_counter()
        cache=load_cache(example,meta);hidden=model.model(input_ids=torch.tensor([example['input_ids']],device='cuda:0'),use_cache=False).last_hidden_state[0,:-1]
        loss,kl,oldce=loss_from_hidden(model.lm_head,hidden,torch.tensor(example['labels'][1:],device=hidden.device),cache,chunk_tokens=chunk)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite KD')
        (.75*loss).backward();kdval=float(loss.detach());del hidden,loss,kl,oldce,cache;sync();kdend=time.perf_counter()
        if not gradients_finite_by_device(params):raise RuntimeError('Nonfinite gradients')
        sync();checkend=time.perf_counter()
        if update:opt.step();opt.zero_grad(set_to_none=True)
        sync();end=time.perf_counter()
        return dict(seconds=end-start,ce_forward_backward=ceend-start,kd_load_forward_backward=kdend-ceend,gradient_check=checkend-kdend,optimizer=end-checkend,reasoning_ce=ceval,kd_loss=kdval,reasoning_tokens=len(row['input_ids']),kd_tokens=len(example['input_ids']))
    try:
        for chunk in [64,256,512]:
            print(f'Режим: блок{chunk}. Восстановление исходного состояния вRAM.',flush=True)
            save('benchmarking',current_chunk=chunk,current_step=0)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for p,x in zip(params,initial):p.copy_(x)
            opt.load_state_dict(copy.deepcopy(initial_opt));torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
            gc.collect();torch.cuda.empty_cache()
            # Same longest selected example probes numerical differences without updates.
            probe=max(chosen,key=lambda x:len(x['input_ids']));pr=iteration(probe,replay[0],chunk,False);grad=samples(params,True)
            if reference_grad is None:reference_grad=grad;reference_losses=pr;parity=dict(passed=True,baseline=True)
            else:
                relative=float((grad-reference_grad).norm()/reference_grad.norm().clamp_min(1e-12));cos=float(torch.nn.functional.cosine_similarity(grad,reference_grad,dim=0));delta=max(abs(pr[k]-reference_losses[k]) for k in ['reasoning_ce','kd_loss'])
                parity=dict(sampled_gradient_relative_l2=relative,sampled_gradient_cosine=cos,max_loss_delta=delta,passed=relative<.01 and delta<1e-4,scope='2048 sampled gradient elements per trainable tensor; not full tensor identity')
            opt.zero_grad(set_to_none=True)
            if not parity['passed']:
                report['modes'].append(dict(chunk_tokens=chunk,parity=parity,state='numerical_gate_failed'));save();print('Численная проверка не пройдена',parity,flush=True);continue
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
            for i in range(2):torch.cuda.reset_peak_memory_stats(i)
            measured=[]
            for step,(row,example) in enumerate(zip(chosen,replay),1):
                r=iteration(row,example,chunk,True);r['step']=step;r['chunk']=chunk
                measured.append(r)
                with (OUT/'steps.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
                save(current_step=step);print(f'Блок{chunk}: шаг{step}/24, {r["seconds"]:.3f}с, CE {r["reasoning_ce"]:.5f}',flush=True)
            steady=measured[4:];update=samples(params)-initial_samples
            mode=dict(chunk_tokens=chunk,parity=parity,state='completed',warmup_steps=4,measured_steps=20,mean_seconds=statistics.mean(r['seconds'] for r in steady),median_seconds=statistics.median(r['seconds'] for r in steady),tokens_per_second=sum(r['reasoning_tokens']+r['kd_tokens'] for r in steady)/sum(r['seconds'] for r in steady),phase_means={k:statistics.mean(r[k] for r in steady) for k in ['ce_forward_backward','kd_load_forward_backward','gradient_check','optimizer']},peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)])
            if reference_update is None:reference_update=update
            else:mode['sampled_update_relative_l2']=float((update-reference_update).norm()/reference_update.norm().clamp_min(1e-12))
            report['modes'].append(mode);save();print(json.dumps(mode),flush=True)
        save('completed',current_step=24)
    except Exception as e:save('failed',error=str(e));raise
if __name__=='__main__':main()
