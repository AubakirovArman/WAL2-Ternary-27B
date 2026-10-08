"""Loss-only content partitions. Inputs stay intact; prose/formula tags are heuristic."""
import re
import torch
from torch.nn import functional as F

GROUPS=('final','service','formula','prose_other')


def token_partition(text,offsets,labels,final_labels):
    services=[m.span() for m in re.finditer(r'This step\b|Checked fact:|</think>',text)]
    formulas=[m.span() for m in re.finditer(r'\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\)|\\(?:frac|binom|sqrt|gcd)\b|[0-9=+*/^<>≤≥±π∞]',text,re.S)]
    overlaps=lambda a,b,spans:any(a<end and b>start for start,end in spans)
    masks={k:[] for k in GROUPS}
    for (start,end),label,final in zip(offsets,labels,final_labels):
        group=None
        if label!=-100:
            group='final' if final!=-100 else 'service' if overlaps(start,end,services) else 'formula' if overlaps(start,end,formulas) else 'prose_other'
        for k in GROUPS:masks[k].append(int(group==k))
    assert all(sum(masks[k][i] for k in GROUPS)==int(label!=-100) for i,label in enumerate(labels))
    return masks


@torch.no_grad()
def diagnostic_ce(head,hidden,labels,masks,chunk=64):
    if len(hidden)!=len(labels) or chunk<1:raise ValueError('Invalid CE inputs')
    if any(len(mask)!=len(labels) or any(x not in (0,1) for x in mask) for mask in masks.values()):raise ValueError('Invalid diagnostic mask')
    if any(sum(mask[i] for mask in masks.values())!=int(label!=-100) for i,label in enumerate(labels.tolist())):raise ValueError('Masks must partition supervised positions exactly')
    counts={k:sum(v) for k,v in masks.items()};n=sum(counts.values())
    if not n:raise ValueError('No supervised tokens')
    tensors={k:torch.tensor(v,device=hidden.device,dtype=torch.bool) for k,v in masks.items()}
    sums={k:hidden.new_zeros((),dtype=torch.float32) for k in masks};total=hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,len(hidden),chunk):
        end=min(start+chunk,len(hidden))
        losses=F.cross_entropy(head(hidden[start:end]).float(),labels[start:end].to(hidden.device),ignore_index=-100,reduction='none')
        total+=losses.sum()
        for k,mask in tensors.items():sums[k]+=losses.masked_fill(~mask[start:end],0).sum()
    result={k:dict(nll=float(sums[k]),tokens=counts[k],ce=float(sums[k])/counts[k] if counts[k] else None) for k in masks}
    return total/n,n,result
