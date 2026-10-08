"""Bounded scale-only KD pilot from a selected packed model; opt-in, GPU6/7 guarded."""
import argparse,gc,hashlib,json,shutil,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
import ternary_core as core
from inspect_checkpoint import load_packed
from learned_scales import install,ScaleTunedMatrix
from kd_objective import student_loss
from train_kd import load_cache
from train_v2 import write_json
from fast_ternary import gradients_finite_by_device

def gains(model):
    return {name:m.log_gain for name,m in model.named_modules() if isinstance(m,ScaleTunedMatrix)}

@torch.no_grad()
def restore_gains(model,path):
    saved=load_file(str(path));current=gains(model)
    if saved.keys()!=current.keys():raise RuntimeError('Scale checkpoint module mismatch')
    for name,p in current.items():p.copy_(saved[name])

@torch.no_grad()
def export(model,parent,output):
    """Preserve exact trit streams/exceptions; replace only BF16 scales in a new bundle."""
    output.mkdir(exist_ok=False)
    meta=json.loads((parent/'manifest.json').read_text())
    matrixfiles={r['file'] for r in meta['matrices'].values()}
    for file in parent.iterdir():
        if file.is_file() and file.name not in matrixfiles and file.name!='manifest.json':shutil.copy2(file,output/file.name)
    for name,info in meta['matrices'].items():
        tensors=load_file(str(parent/info['file']))
        scale=model.get_submodule(name).effective_scales().cpu()
        if scale.shape!=tensors['scales'].shape or not bool((torch.isfinite(scale)&(scale>0)).all()):
            raise RuntimeError('Invalid learned scales')
        tensors['scales']=scale
        save_file(tensors,str(output/info['file']))
    meta['scale_recovery_parent']=str(parent)
    meta['tensor_file_bytes']=sum(p.stat().st_size for p in output.glob('*.safetensors'))
    meta['effective_tensor_bpw']=8*meta['tensor_file_bytes']/meta['total_parameters']
    write_json(output/'manifest.json',meta)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    core.guard();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'code').mkdir()
    # Include transitive local imports (train_kd, train_v2, fast_ternary, etc.)
    # so the snapshot does not silently depend on later workspace edits.
    code_hashes={}
    for source in sorted((core.ROOT/'scripts').glob('*.py')):
        target=args.output/'code'/source.name
        shutil.copy2(source,target)
        code_hashes[source.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(args.output/'code-manifest.json',code_hashes)
    shutil.copy2(args.checkpoint/'manifest.json',args.output/'parent-manifest.json')
    shutil.copy2(core.ROOT/'data/v3-kd/manifest.json',args.output/'teacher-cache-manifest.json')
    meta=json.loads((core.ROOT/'data/v3-kd/manifest.json').read_text())
    assert meta['state']=='completed'
    assert hashlib.sha256(Path(meta['corpus']).read_bytes()).hexdigest()==meta['corpus_sha256']
    data=core.load_examples(Path(meta['corpus']),512);byid={e['id']:e for e in data['train']}
    verified=core.load_examples(core.ROOT/'data/v2/verified_holdout.jsonl',512)['validation']
    order=[byid[i] for i in meta['order_ids']]
    model=load_packed(args.checkpoint);baseline=core.evaluate(model,data['validation'])
    baseline_verified=core.evaluate(model,verified)
    count=install(model,args.checkpoint)
    initial=core.evaluate(model,data['validation'])
    if abs(initial['answer_ce']-baseline['answer_ce'])>1e-5:
        raise RuntimeError('Scale adapter changed initial validation CE')
    params=list(gains(model).values())
    cfg={'parent':str(args.checkpoint),'trainable_scales':count,'lrs':[0.001,0.003],
         'steps_per_pilot':256,'objective':'0.9 top128+tail KL + 0.1 answer CE',
         'fixed_trits':True,'max_abs_log_gain':0.25,'corpus_sha256':meta['corpus_sha256'],
         'selection':'Lower primary validation CE, subject to verified validation CE <= parent + 1e-5',
         'verified_validation':str(core.ROOT/'data/v2/verified_holdout.jsonl')}
    write_json(args.output/'config.json',cfg)
    write_json(args.output/'order.json',[e['id'] for e in order[:256]])
    results={'state':'pilots','baseline':baseline,'baseline_verified':baseline_verified,'initial_adapter':initial,'pilots':[]}
    best=baseline['answer_ce'];selected=None;deadline=time.monotonic()+3*3600
    for lr in cfg['lrs']:
        with torch.no_grad():
            for p in params:p.zero_()
        torch.manual_seed(20260919)
        opt=torch.optim.Adam(params,lr=lr,foreach=False)
        stage=args.output/f'lr-{lr}';stage.mkdir()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        for step,example in enumerate(order[:256],1):
            if time.monotonic()>deadline:raise TimeoutError('Scale pilot deadline exceeded')
            model.train();opt.zero_grad(set_to_none=True)
            loss,kl,ce=student_loss(model,example,load_cache(example,meta),0.9)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite scale loss')
            loss.backward()
            if not gradients_finite_by_device(params):raise RuntimeError('Nonfinite scale gradients')
            opt.step();opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for p in params:p.clamp_(-0.25,0.25)
            row={'step':step,'lr':lr,'loss':loss.item(),'kl':kl.item(),'ce':ce.item()}
            with (stage/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            core.event('scale_train_step',**row)
            if step%128==0:
                val=core.evaluate(model,data['validation'])
                extra=core.evaluate(model,verified)
                eligible=extra['answer_ce']<=baseline_verified['answer_ce']+1e-5
                results['pilots'].append({'lr':lr,'step':step,**val,'verified_ce':extra['answer_ce'],
                                          'eligible_secondary':eligible})
                if val['answer_ce']<best and eligible:
                    best=val['answer_ce'];selected=stage/f'gains-{step}.safetensors'
                    save_file({n:p.detach().cpu().contiguous() for n,p in gains(model).items()},str(selected))
                    torch.save({'optimizer':opt.state_dict(),'step':step,'lr':lr,'torch_rng':torch.get_rng_state(),
                                'cuda_rng':torch.cuda.get_rng_state_all()},stage/f'optimizer-{step}.pt')
                write_json(args.output/'metrics.json',results);core.event('scale_validation',lr=lr,step=step,**val,
                                                                        verified_ce=extra['answer_ce'],eligible_secondary=eligible)
        del opt;gc.collect();torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()
    if selected is not None:restore_gains(model,selected)
    else:
        with torch.no_grad():
            for p in params:p.zero_()
    results.update(selected_gains=str(selected) if selected else None,
                   selected_validation=core.evaluate(model,data['validation']),selected_test=core.evaluate(model,data['test']),
                   selected_verified=core.evaluate(model,verified))
    if selected is not None:
        packed=args.output/'selected-ternary';export(model,args.checkpoint,packed)
        del model;gc.collect();torch.cuda.empty_cache()
        reloaded=load_packed(packed)
        verification=core.evaluate(reloaded,data['validation'])
        if abs(verification['answer_ce']-results['selected_validation']['answer_ce'])>1e-5:
            raise RuntimeError('Packed reload changed scale-tuned validation')
        results.update(selected_packed=str(packed),packed_validation=verification)
    else:results['selected_packed']=str(args.checkpoint)
    results['state']='completed';write_json(args.output/'metrics.json',results)
    core.event('scale_recovery_completed',metrics=results)

if __name__=='__main__':main()
