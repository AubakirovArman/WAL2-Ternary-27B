"""Opt-in checkpointed token chunks for the vocabulary head; existing trainers unchanged."""
import json
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from kd_objective import compressed_kl


def loss_from_hidden(head,hidden,labels,cache,kd_weight=.9,chunk_tokens=64):
    """Same global token weighting as student_loss, with recomputed head activations."""
    n=len(hidden)
    if n<1 or chunk_tokens<1 or len(labels)!=n:raise ValueError('Invalid sequence/chunk length')
    if not 0<=kd_weight<=1:raise ValueError('Invalid KD weight')
    answer_count=int((labels!=-100).sum())
    if not answer_count:raise ValueError('No supervised answer tokens')
    for key in ('top_ids','top_logp','tail_prob'):
        if len(cache[key])!=n:raise ValueError('Cache length mismatch')
    def part(h,y,indices,logp,tail):
        kl,logq=compressed_kl(head(h).float(),indices,logp,tail)
        ce=-logq.gather(-1,y.clamp_min(0)[:,None]).squeeze(-1)
        ce=ce.masked_fill(y==-100,0).sum()/answer_count
        return kl*(len(h)/n),ce
    total_kl=hidden.new_zeros((),dtype=torch.float32)
    total_ce=hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,n,chunk_tokens):
        end=min(n,start+chunk_tokens)
        arguments=(hidden[start:end],labels[start:end].to(hidden.device),
                   *(cache[k][start:end].to(hidden.device) for k in ('top_ids','top_logp','tail_prob')))
        if torch.is_grad_enabled():
            kl,ce=checkpoint(part,*arguments,use_reentrant=False)
        else:kl,ce=part(*arguments)
        total_kl=total_kl+kl;total_ce=total_ce+ce
    return kd_weight*total_kl+(1-kd_weight)*total_ce,total_kl.detach(),total_ce.detach()


def check():
    torch.set_num_threads(2);torch.manual_seed(1936)
    results=[]
    for chunk in (1,4,64):
        for frozen_hidden in (False,True):
            head=torch.nn.Linear(7,23)
            hidden=torch.randn(19,7,requires_grad=not frozen_hidden)
            labels=torch.randint(0,23,(19,));labels[:8]=-100;labels[11]=-100
            probs=torch.randn(19,23).softmax(-1);p,ids=probs.topk(5,dim=-1)
            cache={'top_ids':ids,'top_logp':p.log().half(),'tail_prob':1-p.sum(-1)}
            kl,logq=compressed_kl(head(hidden),ids,cache['top_logp'],cache['tail_prob'])
            ce=-logq.gather(-1,labels.clamp_min(0)[:,None]).squeeze(-1)[labels!=-100].mean()
            expected=.9*kl+.1*ce
            parameters=list(head.parameters())+([] if frozen_hidden else [hidden])
            expected_grad=torch.autograd.grad(expected,parameters)
            actual,ak,ac=loss_from_hidden(head,hidden,labels,cache,chunk_tokens=chunk)
            actual_grad=torch.autograd.grad(actual,parameters)
            max_diff=max(float((a-b).abs().max()) for a,b in zip(expected_grad,actual_grad))
            assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6)
            assert torch.allclose(ak,kl,atol=2e-6) and torch.allclose(ac,ce,atol=2e-6)
            for a,b in zip(expected_grad,actual_grad):assert torch.allclose(a,b,atol=2e-6,rtol=2e-5)
            with torch.no_grad():
                inference,_,_=loss_from_hidden(head,hidden,labels,cache,chunk_tokens=chunk)
                assert torch.allclose(inference,expected,atol=2e-6)
            results.append({'chunk_tokens':chunk,'frozen_hidden':frozen_hidden,'max_gradient_difference':max_diff})
    report={'scope':'CPU mathematical equivalence only; real GPU memory/speed not measured','cases':results}
    (Path(__file__).resolve().parents[1]/'reports/chunked-kd-cpu-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':check()
