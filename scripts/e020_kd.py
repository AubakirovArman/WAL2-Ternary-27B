"""One vocabulary-head pass for CE + coarse KL; full context, exact token IDs."""
import torch
from torch.utils.checkpoint import checkpoint


def position_kl(logits, indices, teacher_logp, teacher_tail):
    logq = logits.float().log_softmax(-1)
    indices = indices.long()
    selected = logq.gather(-1, indices)
    p = teacher_logp.float().exp()
    tail = teacher_tail.float().clamp_min(0)
    norm = p.sum(-1) + tail
    p, tail = p / norm[:, None], tail / norm
    if indices.shape[-1] == logits.shape[-1]:
        if torch.any(tail > 1e-6):
            raise ValueError('Nonzero teacher tail with full vocabulary')
        logtail = torch.zeros_like(tail)
    else:
        # The complement uses the full student softmax and remains in the graph.
        complement = logq.scatter(-1, indices, -float('inf'))
        logtail = complement.logsumexp(-1)
    divergence = (p * (p.clamp_min(1e-30).log() - selected)).sum(-1)
    divergence = divergence + tail * (tail.clamp_min(1e-30).log() - logtail)
    return divergence, logq


def loss_from_hidden(head, hidden, labels, cache, ce_weight, kl_weight, chunk_tokens=512):
    """Mean KL per real next-token position; CE per supervised target, no padding."""
    n = len(hidden)
    count = int((labels != -100).sum())
    if n != len(labels) or count < 1 or chunk_tokens < 1:
        raise ValueError('Invalid sequence')
    if cache is None and kl_weight:
        raise ValueError('KL requires an aligned cache')
    if cache is not None:
        for key in ('top_ids', 'top_logp', 'tail_prob'):
            if len(cache[key]) != n:
                raise ValueError('Cache next-token shift/length mismatch')
    def part(h, y, ix=None, lp=None, tail=None):
        if ix is None:
            logq = head(h).float().log_softmax(-1)
            kp = h.new_zeros(len(h), dtype=torch.float32)
        else:
            kp, logq = position_kl(head(h), ix, lp, tail)
        answer = y != -100
        ce = -logq.gather(-1, y.clamp_min(0)[:, None]).squeeze(-1)
        ce = ce.masked_fill(~answer, 0).sum() / count
        kl = kp.sum() / n
        prompt_sum = kp.masked_fill(answer, 0).sum()
        answer_sum = kp.masked_fill(~answer, 0).sum()
        return ce_weight * ce + kl_weight * kl, ce.detach(), kl.detach(), prompt_sum.detach(), answer_sum.detach()
    values = [hidden.new_zeros((), dtype=torch.float32) for _ in range(5)]
    for start in range(0, n, chunk_tokens):
        end = min(n, start + chunk_tokens)
        args = (hidden[start:end], labels[start:end].to(hidden.device))
        if cache is not None:
            args += tuple(cache[k][start:end].to(hidden.device) for k in ('top_ids', 'top_logp', 'tail_prob'))
        result = checkpoint(part, *args, use_reentrant=False) if torch.is_grad_enabled() else part(*args)
        values = [a + b for a, b in zip(values, result)]
    loss, ce, kl, prompt, answer = values
    return loss, dict(ce=float(ce), kl=float(kl),
                     kl_prompt=float(prompt) / max(n-count, 1),
                     kl_answer=float(answer) / count,
                     kl_prompt_contribution=float(prompt) / n,
                     kl_answer_contribution=float(answer) / n,
                     prompt_positions=n-count, answer_positions=count, positions=n)


def check():
    import json
    from pathlib import Path
    from kd_objective import compressed_kl
    torch.set_num_threads(2)
    torch.manual_seed(20260927)
    cases=[]
    for k in (1, 5, 23):
        for scale in (.01, 1., 8.):
            teacher=(torch.randn(13,23)*scale).softmax(-1)
            p, ix=teacher.topk(k, -1)
            tail=(1-p.sum(-1)).clamp_min(0) if k<23 else torch.zeros(13)
            # FP16 storage is part of the tested format.
            lp=p.log().half(); pp=lp.float().exp()
            norm=pp.sum(-1)+tail; pp=pp/norm[:,None]; tt=tail/norm
            x=torch.randn(13,23,requires_grad=True)
            actual,_=position_kl(x,ix,lp,tail)
            q=x.softmax(-1); chosen=q.gather(-1,ix)
            included=torch.zeros_like(q,dtype=torch.bool).scatter(-1,ix,True)
            qt=q.masked_fill(included,0).sum(-1)
            expected=(pp*(pp.clamp_min(1e-30).log()-chosen.log())).sum(-1)
            if k<23:expected=expected+tt*(tt.clamp_min(1e-30).log()-qt.log())
            ga=torch.autograd.grad(actual.mean(),x,retain_graph=True)[0]
            ge=torch.autograd.grad(expected.mean(),x)[0]
            assert torch.allclose(actual,expected,atol=3e-6,rtol=2e-5)
            assert torch.allclose(ga,ge,atol=3e-6,rtol=2e-5)
            legacy,_=compressed_kl(x,ix,lp,tail)
            assert torch.allclose(actual.mean(),legacy,atol=3e-6,rtol=2e-5)
            cases.append(dict(topk=k,scale=scale,loss=float(actual.mean().detach()),gradient_max_error=float((ga-ge).abs().max())))
    for chunk in (1, 4, 512):
        head=torch.nn.Linear(7,23); h=torch.randn(13,7,requires_grad=True)
        labels=torch.randint(23,(13,));labels[:4]=-100;labels[7]=-100
        p,ix=torch.randn(13,23).softmax(-1).topk(5,-1)
        cache=dict(top_ids=ix,top_logp=p.log().half(),tail_prob=1-p.sum(-1))
        kp,lq=position_kl(head(h),ix,cache['top_logp'],cache['tail_prob'])
        ce=-lq.gather(-1,labels.clamp_min(0)[:,None]).squeeze(-1)[labels!=-100].mean()
        ref=.25*ce+.3375*kp.mean()
        params=list(head.parameters())+[h]
        gr=torch.autograd.grad(ref,params)
        got,stats=loss_from_hidden(head,h,labels,cache,.25,.3375,chunk)
        gg=torch.autograd.grad(got,params)
        assert torch.allclose(got,ref,atol=3e-6)
        assert all(torch.allclose(a,b,atol=3e-6,rtol=2e-5) for a,b in zip(gr,gg))
        assert abs(stats['kl_prompt_contribution']+stats['kl_answer_contribution']-stats['kl'])<3e-6
    teacher=torch.randn(13,23).softmax(-1); p,ix=teacher.topk(5,-1)
    same,_=position_kl(teacher.log(),ix,p.log(),1-p.sum(-1))
    assert same.abs().max()<3e-6
    report=dict(passed=True,cases=cases,chunk_sizes=[1,4,512],scope='Coarse bucket value and full-vocabulary/backprop gradient vs independently summed dense complement; no division by bucket count; self KL and legacy consistency')
    (Path(__file__).resolve().parents[1]/'reports/e020-kd-cpu-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':check()
