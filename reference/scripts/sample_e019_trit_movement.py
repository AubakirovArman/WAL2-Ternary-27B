"""Seeded CPU-only latent/threshold/trit diagnostics; estimates, not exact global counts."""
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/e019-trit-movement'


def main(paths=None,matrices=None):
    torch.set_num_threads(8);OUT.mkdir(exist_ok=False)
    if paths is None:
        paths={'e018':ROOT/'runs/e018-fresh25000/candidates/step-25000'}
        paths.update({str(step):ROOT/f'runs/e019-reasoning-recovery/candidates/step-{step}' for step in (2560,3584,4096)})
    labels=[k for k in paths if k!='e018']
    manifests={k:json.loads((p/'manifest.json').read_text()) for k,p in paths.items()}
    if matrices is None:matrices=json.loads((ROOT/'runs/e019-reasoning-recovery/selected-ternary/manifest.json').read_text())['matrices']
    samples=[];total_parameters=0
    for num,name in enumerate(matrices):
        source={}
        for label,path in paths.items():
            with safe_open(path/manifests[label][name],framework='pt',device='cpu') as f:
                weight=f.get_tensor('weight');scales=f.get_tensor('scales').reshape(-1).float()
                if label=='e018':
                    n=weight.numel();size=min(16384,n)
                    seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')%(2**63-1)
                    indices=torch.randint(n,(size,),generator=torch.Generator().manual_seed(seed));total_parameters+=n
                assert weight.numel()==n
                scale=scales[indices//128];assert bool((scale>0).all())
                source[label]=(weight.reshape(-1)[indices].float()/scale,scale)
        base,base_scale=source['e018'];qbase=base.round().clamp(-1,1)
        record=dict(matrix=name,parameters=n,sampled_positions=size,
                    baseline_near_threshold_001=float((torch.abs(base.abs()-.5)<.001).float().mean()),
                    baseline_near_threshold_01=float((torch.abs(base.abs()-.5)<.01).float().mean()),candidates={})
        for label in labels:
            value,scale=source[label];delta=value-base;q=value.round().clamp(-1,1)
            record['candidates'][label]=dict(trit_flips_sampled=int((q!=qbase).sum()),flip_fraction=float((q!=qbase).float().mean()),
                                           mean_abs_latent_movement_over_scale=float(delta.abs().mean()),mean_squared_latent_movement_over_scale=float(delta.square().mean()),
                                           near_threshold_001=float((torch.abs(value.abs()-.5)<.001).float().mean()),scale_changes_sampled=int((scale!=base_scale).sum()))
        samples.append(record)
        if num%40==0:print(f'Проверено матриц {num+1}/{len(matrices)}',flush=True)
    summary={}
    for label in labels:
        weighted=lambda field:sum(r['parameters']*r['candidates'][label][field] for r in samples)/total_parameters
        summary[label]=dict(parameter_weighted_flip_fraction_estimate=weighted('flip_fraction'),
                            mean_abs_latent_movement_over_scale_estimate=weighted('mean_abs_latent_movement_over_scale'),
                            rms_latent_movement_over_scale_estimate=math.sqrt(weighted('mean_squared_latent_movement_over_scale')),
                            sampled_trit_flips=sum(r['candidates'][label]['trit_flips_sampled'] for r in samples),
                            sampled_scale_changes=sum(r['candidates'][label]['scale_changes_sampled'] for r in samples))
        groups={}
        for record in samples:
            kind='.'.join(record['matrix'].split('.')[3:])
            g=groups.setdefault(kind,{'parameters':0,'weighted_flip_sum':0.,'sampled_flips':0})
            g['parameters']+=record['parameters'];g['weighted_flip_sum']+=record['parameters']*record['candidates'][label]['flip_fraction'];g['sampled_flips']+=record['candidates'][label]['trit_flips_sampled']
        summary[label]['by_matrix_type']={k:{'parameters':v['parameters'],'parameter_weighted_flip_fraction_estimate':v['weighted_flip_sum']/v['parameters'],'sampled_flips':v['sampled_flips']} for k,v in groups.items()}
    report=dict(state='completed',matrix_count=len(samples),parameters=total_parameters,total_samples=sum(r['sampled_positions'] for r in samples),
                method='16384 seeded uniform draws with replacement per ternary matrix; parameter-weighted aggregate. q=round(w/s).clamp(-1,1), decision thresholds +/-0.5. This is a sampled estimate, not an exact model-wide flip count or proof of update usefulness.',
                source_paths={k:str(v) for k,v in paths.items()},torch_version=torch.__version__,python_version=platform.python_version(),summary=summary,matrices=samples)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
