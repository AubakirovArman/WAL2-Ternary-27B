"""Coarsened forward KL: teacher top-K categories plus one residual-mass category."""
import torch

def compressed_kl(logits,top_ids,top_logp,tail_prob):
    logq=torch.log_softmax(logits.float(),dim=-1)
    selected=logq.gather(-1,top_ids.long())
    p=top_logp.float().exp();tail=tail_prob.float().clamp_min(0)
    total=p.sum(-1)+tail
    p=p/total[:,None];tail=tail/total
    # log(1 - selected mass), with stable expm1 and a floor only at FP32 saturation.
    logmass=torch.logsumexp(selected,dim=-1).clamp(max=-1e-7)
    logtail=torch.log(-torch.expm1(logmass))
    divergence=(p*(p.clamp_min(1e-30).log()-selected)).sum(-1)
    divergence=divergence+tail*(tail.clamp_min(1e-30).log()-logtail)
    return divergence.mean(),logq

def student_loss(model,example,cache,kd_weight=0.9):
    ids=torch.tensor([example['input_ids']],device='cuda:0')
    hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
    logits=model.lm_head(hidden).float()
    kl,logq=compressed_kl(logits,cache['top_ids'].to(logits.device),cache['top_logp'].to(logits.device),cache['tail_prob'].to(logits.device))
    labels=torch.tensor(example['labels'][1:],device=logits.device);mask=labels!=-100
    ce=-logq.gather(-1,labels.clamp_min(0)[:,None]).squeeze(-1)[mask].mean()
    return kd_weight*kl+(1-kd_weight)*ce,kl.detach(),ce.detach()

def check():
    torch.manual_seed(674)
    for k in [3,11]:
        teacher=torch.randn(7,11).softmax(-1);p,ix=teacher.topk(k,-1);tail=(1-p.sum(-1)).clamp_min(0)
        x=torch.randn(7,11,requires_grad=True)
        actual,_=compressed_kl(x,ix,p.log(),tail)
        q=x.softmax(-1);selected=q.gather(-1,ix);qt=(1-selected.sum(-1)).clamp_min(1e-7)
        pt=torch.cat([p,tail[:,None]],-1);pt=pt/pt.sum(-1,keepdim=True)
        qb=torch.cat([selected,qt[:,None]],-1)
        expected=torch.nn.functional.kl_div(qb.log(),pt,reduction='batchmean')
        ga=torch.autograd.grad(actual,x,retain_graph=True)[0];ge=torch.autograd.grad(expected,x)[0]
        assert torch.allclose(actual,expected,atol=2e-6)
        assert torch.allclose(ga,ge,atol=2e-6)
        identical,_=compressed_kl(teacher.log(),ix,p.log(),tail)
        assert abs(float(identical))<2e-6
    print('PASS: top-K+tail KL and full-vocabulary gradients match independent bucket-distribution reference; self-KL is zero')
if __name__=='__main__':check()
