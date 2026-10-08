"""Controlled full-model A/B from the same E004 latent weights and optimizer state."""
import argparse,copy,json,statistics,time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import Adafactor
import ternary_core as core
from fast_ternary import FusedQuantizeSTE,gradients_finite_by_device

def sync():
    for i in range(2):torch.cuda.synchronize(i)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeat-reference',action='store_true',help='Measure run-to-run variation of the original implementation')
    args=parser.parse_args()
    core.guard()
    root=core.ROOT;out=root/('reports/performance/reference_repeat.json' if args.repeat_reference else 'reports/performance/training_ab.json')
    latent=root/'runs/e004-rotated-qat/resume-latent'
    model=core.load_source();core.ternarize(model,rotate=True);core.set_alpha(model,1.)
    manifest=json.loads((latent/'manifest.json').read_text())
    with torch.no_grad():
        for name,filename in manifest.items():
            module=model.get_submodule(name)
            with safe_open(latent/filename,framework='pt',device='cpu') as f:
                for key in f.keys():getattr(module,key).copy_(f.get_tensor(key))
        if args.repeat_reference:
            checked=0
            for module in model.modules():
                if isinstance(module,core.TernaryMatrix):
                    expected=core.QuantizeSTE.apply(module.weight,module.scales,1.)
                    actual=FusedQuantizeSTE.apply(module.weight,module.scales,1.)
                    if not torch.equal(expected,actual):raise AssertionError('Quantization mismatch on saved E004 weights')
                    checked+=1
                    del expected,actual
            core.event('speed_ab_actual_weights_exact',matrices=checked)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[p for p in model.parameters() if p.requires_grad]
    opt=Adafactor(params,lr=1e-5,relative_step=False,scale_parameter=False,warmup_init=False,beta1=None,weight_decay=0.)
    # This file was produced by our own E004 run; it contains optimizer and Python RNG objects.
    state=torch.load(latent/'training_state.pt',map_location='cpu',weights_only=False)
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    initial=[p.detach().cpu() for p in params]
    data=core.load_examples(root/'runs/e004-rotated-qat/teacher_snapshot_step_2048.jsonl',512)
    examples=data['train'][:12]
    reference_class=core.QuantizeSTE
    results={};reference_final=None
    modes=['reference','reference_repeat'] if args.repeat_reference else ['reference','fused_quantizer','fused_and_grouped_checks']
    for mode in modes:
        with torch.no_grad():
            for p,value in zip(params,initial):p.copy_(value)
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        torch.manual_seed(20260919)
        core.QuantizeSTE=reference_class if mode.startswith('reference') else FusedQuantizeSTE
        model.train();rows=[];sync()
        torch.cuda.reset_peak_memory_stats(0);torch.cuda.reset_peak_memory_stats(1)
        for step in range(16):
            example=examples[step%len(examples)]
            opt.zero_grad(set_to_none=True);sync();t=time.perf_counter()
            loss,n=core.loss_for(model,example)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward()
            if mode=='fused_and_grouped_checks':
                if not gradients_finite_by_device(params):raise RuntimeError('nonfinite gradients')
            else:
                for p in params:
                    if not torch.isfinite(p.grad).all():raise RuntimeError('nonfinite gradients')
            opt.step();opt.zero_grad(set_to_none=True);sync()
            row={'step':step,'seconds':time.perf_counter()-t,'loss':loss.item(),'answer_tokens':n,'example_id':example['id']}
            rows.append(row);core.event('speed_ab_step',mode=mode,**row)
        measured=rows[4:]
        results[mode]={'rows':rows,'warmup_steps':4,'median_seconds':statistics.median(r['seconds'] for r in measured),
                       'mean_seconds':statistics.mean(r['seconds'] for r in measured),
                       'peak_gib':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]}
        if mode=='reference':reference_final=[p.detach().cpu() for p in params]
        else:
            different=0;max_abs=0.
            # Full comparison of all trainable FP32 weights after 16 matching optimizer updates.
            for p,r in zip(params,reference_final):
                v=p.detach().cpu()
                if not torch.equal(v,r):
                    different+=1;max_abs=max(max_abs,float((v-r).abs().max()))
            results[mode]['updated_weight_comparison']={'different_tensors':different,'max_absolute_error':max_abs}
            results[mode]['max_loss_difference']=max(abs(a['loss']-b['loss']) for a,b in zip(rows,results['reference']['rows']))
            results[mode]['speedup']=results['reference']['median_seconds']/results[mode]['median_seconds']
        out.write_text(json.dumps(results,indent=2))
        core.event('speed_ab_phase_done',mode=mode,result={k:v for k,v in results[mode].items() if k!='rows'})
    core.event('speed_ab_completed',report=str(out))

if __name__=='__main__':main()
