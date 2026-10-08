"""Read-only comparison of historic and regenerated old-cache examples."""
import json,time
from pathlib import Path
import torch
from safetensors.torch import load_file
import ternary_core as core
from cache_teacher_logits import collect

ROOT=core.ROOT
def main():
    core.guard()
    manifest=json.loads((ROOT/'data/v3-kd/manifest.json').read_text())
    examples={str(r['id']):r for r in core.load_examples(Path(manifest['corpus']),512)['train']}
    model=core.load_source();model.eval();result=[]
    for identity in manifest['order_ids'][:4]:
        key=str(identity);old=load_file(manifest['examples'][key]['file']);new=collect(model,examples[key],128)
        assert torch.equal(old['input_ids'],new['input_ids']) and torch.equal(old['labels'],new['labels'])
        same=old['top_ids']==new['top_ids'];top1=old['top_ids'][:,0]==new['top_ids'][:,0]
        both_logp=(old['top_logp'].float()-new['top_logp'].float()).abs();tail=(old['tail_prob']-new['tail_prob']).abs()
        record=dict(id=key,positions=len(top1),top1_agreement=float(top1.float().mean()),
                    exact_top128_row_fraction=float(same.all(-1).float().mean()),top128_id_position_agreement=float(same.float().mean()),
                    median_logp_difference_same_rank=float(both_logp.median()),mean_logp_difference_same_rank=float(both_logp.mean()),
                    max_logp_difference_same_rank=float(both_logp.max()),mean_top1_logp_difference=float(both_logp[:,0].mean()),
                    mean_tail_difference=float(tail.mean()),max_tail_difference=float(tail.max()),
                    old_mean_tail=float(old['tail_prob'].mean()),new_mean_tail=float(new['tail_prob'].mean()))
        result.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
    path=ROOT/'reports/e020-teacher-sentinel-diagnostic.json';path.write_text(json.dumps(dict(created=time.time(),scope='Read-only 4 old examples, exact saved input IDs and labels, no optimization',cases=result),indent=2))
if __name__=='__main__':main()
