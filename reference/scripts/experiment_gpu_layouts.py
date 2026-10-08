"""Controlled GPU6/7 layer/head placement and loss-chunk benchmarks. LR0, no saved weights."""
import argparse,copy,gc,json,statistics,time
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
from benchmark_e017_training import cpu_tree,samples,sync
OUT=core.ROOT/'reports/gpu-layout-experiments-20260925'

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint-probe',action='store_true');args=ap.parse_args()
    if args.checkpoint_probe:OUT=core.ROOT/'reports/selective-checkpoint-experiments-20260925'
    core.guard();OUT.mkdir(exist_ok=False)
    report=dict(state='loading',scope='Identical fixed E017 weights; real forward/backward, finite-gradient checks and Adafactor computation with LR0. Does not train or save new weights. One warmup and two measured iterations per length/mode. FLA in all modes.',results=[])
    def save(**kw):report.update(kw);write_json(OUT/'metrics.json',report)
    save();model=core.load_source();core.ternarize(model,rotate=True);enable_fused_quantizer();opt=restore(model,core.ROOT/'runs/e017-fresh10000/candidates/step-10000');core.set_alpha(model,1.)
    initial_opt=cpu_tree(opt.state_dict());model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();qwen.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
    old=[json.loads(l) for l in (core.ROOT/'data/fresh-10000-v2/train.jsonl').open()]
    fresh=[]
    with (core.ROOT/'data/e018-25000/committed.jsonl').open() as f:
        for l in f:
            if l.endswith('\n'):
                r=json.loads(l)
                if r['row']['split']=='train':fresh.append(r['encoded'])
    cases=[min(old,key=lambda r:abs(len(r['input_ids'])-512)),max(old,key=lambda r:len(r['input_ids'])),min(fresh,key=lambda r:abs(len(r['input_ids'])-4096))]
    write_json(OUT/'cases.json',cases)
    if args.checkpoint_probe:cases=json.loads((core.ROOT/'reports/gpu-layout-experiments-20260925/cases.json').read_text());write_json(OUT/'cases.json',cases)
    meta=json.loads((core.ROOT/'data/v3-kd/manifest.json').read_text());original=core.load_examples(Path(meta['corpus']),512)
    lookup={r['id']:r for r in original['train']};replay=lookup[meta['order_ids'][6784%len(meta['order_ids'])]];cache=load_cache(replay,meta)
    params=[p for p in model.parameters() if p.requires_grad];reference={}
    configs=[('baseline',32,1,512),('chunk1024',32,1,1024),('chunk2048',32,1,2048),('chunk4096',32,1,4096),('layers36_28',36,1,512),('layers28_36',28,1,512),('head_on_gpu6',32,0,512)]
    if args.checkpoint_probe:configs=[('baseline',32,1,512),('checkpoint_half',32,1,512),('checkpoint_quarter',32,1,512)]
    def configure(split,head):
        opt.zero_grad(set_to_none=True);opt.state.clear();gc.collect();torch.cuda.empty_cache()
        for module in list(model.model.layers)+[model.model.norm,model.lm_head]:
            module._forward_pre_hooks.clear();module._forward_pre_hooks_with_kwargs.clear()
        for i,layer in enumerate(model.model.layers):layer.to('cuda:0' if i<split else 'cuda:1')
        model.model.norm.to('cuda:1');model.lm_head.to('cuda:'+str(head))
        def hook(device):
            def move(module,args,kwargs):return core.move_tree(args,device),core.move_tree(kwargs,device)
            return move
        for i,layer in enumerate(model.model.layers):layer.register_forward_pre_hook(hook('cuda:0' if i<split else 'cuda:1'),with_kwargs=True)
        model.model.norm.register_forward_pre_hook(hook('cuda:1'),with_kwargs=True)
        opt.load_state_dict(copy.deepcopy(initial_opt))
        for group in opt.param_groups:group['lr']=0.
        gc.collect();torch.cuda.empty_cache();sync()
    def iteration(row,chunk):
        opt.zero_grad(set_to_none=True);sync();start=time.perf_counter();device=model.lm_head.weight.device
        hidden=model.model(input_ids=torch.tensor([row['input_ids']],device='cuda:0'),use_cache=False).last_hidden_state[0,:-1].to(device)
        ce,_=chunked_ce(model.lm_head,hidden,torch.tensor(row['labels'][1:],device=device),chunk=chunk);(.25*ce).backward();cev=float(ce.detach());del hidden,ce;sync();mid=time.perf_counter()
        h=model.model(input_ids=torch.tensor([replay['input_ids']],device='cuda:0'),use_cache=False).last_hidden_state[0,:-1].to(device)
        kd,kl,ac=loss_from_hidden(model.lm_head,h,torch.tensor(replay['labels'][1:],device=device),cache,chunk_tokens=chunk);(.75*kd).backward();kdv=float(kd.detach());del h,kd,kl,ac
        if not gradients_finite_by_device(params):raise RuntimeError('Nonfinite gradients')
        sync();beforeopt=time.perf_counter();opt.step();sync();end=time.perf_counter()
        return dict(seconds=end-start,ce_seconds=mid-start,kd_and_checks_seconds=beforeopt-mid,optimizer_seconds=end-beforeopt,ce=cev,kd=kdv)
    try:
        for name,split,head,chunk in configs:
            save(state='running',mode=name,length=None);print('Режим',name,flush=True)
            try:configure(split,head)
            except torch.OutOfMemoryError as e:
                report['results'].append(dict(mode=name,state='oom_during_placement',error=str(e)[:200]));save();continue
            for i,layer in enumerate(model.model.layers):
                layer.gradient_checkpointing=(i%2==1) if name=='checkpoint_half' else (i%4==3) if name=='checkpoint_quarter' else True
            initial_sample=samples(params);result=dict(mode=name,split=split,head_gpu=6+head,chunk=chunk,checkpointed_layers=sum(layer.gradient_checkpointing for layer in model.model.layers),cases=[])
            for row in cases:
                n=len(row['input_ids']);save(length=n);print('Длина',n,flush=True)
                try:
                    warm=iteration(row,chunk);grad=samples(params,True)
                    if name=='baseline':reference[n]=(warm,grad);parity=dict(passed=True,baseline=True)
                    else:
                        ref,rg=reference[n];relative=float((grad-rg).double().norm()/rg.double().norm().clamp_min(1e-12));delta=max(abs(warm[k]-ref[k]) for k in ['ce','kd']);parity=dict(relative_gradient_l2=relative,max_loss_delta=delta,passed=relative<.01 and delta<1e-4,scope='2048 sampled elements per trainable tensor')
                    if not parity['passed']:result['cases'].append(dict(tokens=n,state='parity_failed',parity=parity));continue
                    for i in range(2):torch.cuda.reset_peak_memory_stats(i)
                    measurements=[iteration(row,chunk) for _ in range(2)]
                    case=dict(tokens=n,state='completed',parity=parity,mean_seconds=statistics.mean(x['seconds'] for x in measurements),measurements=measurements,peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)])
                    result['cases'].append(case);print(json.dumps(case),flush=True)
                except torch.OutOfMemoryError as e:
                    result['cases'].append(dict(tokens=n,state='oom',error=str(e)[:200]));opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                save(current_result=result)
            assert torch.equal(samples(params),initial_sample),'LR0 unexpectedly changed sampled weights'
            report['results'].append(result);save(current_result=None)
        save(state='completed')
    except Exception as e:save(state='failed',error=str(e));raise
if __name__=='__main__':main()
