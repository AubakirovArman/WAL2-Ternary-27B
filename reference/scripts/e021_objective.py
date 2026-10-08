"""Score-function OPD + grouped exact-answer reward; prompt tokens never scored."""
import torch
from torch.utils.checkpoint import checkpoint


def surrogate(logq, behavior, teacher, task_advantage, clip=.2, beta=1.):
    # The detached advantage is essential: differentiating a sampled log(q/p)
    # directly on fixed trajectories is not the reverse-KL policy gradient.
    advantage=(beta*(teacher-behavior)+task_advantage).detach()
    ratio=(logq-behavior).exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-clip,1+clip)*advantage)


def grouped_advantages(rewards):
    r=torch.tensor(rewards,dtype=torch.float32)
    return ((r-r.mean())/(r.std(unbiased=False)+1e-6)).tolist()


def loss_from_hidden(head, hidden, targets, behavior, teacher, task_advantage, chunk=64):
    if not (len(hidden)==len(targets)==len(behavior)==len(teacher)>0):
        raise ValueError('Chosen-token/prefix alignment mismatch')
    device=hidden.device
    y=torch.as_tensor(targets,device=device,dtype=torch.long)
    b=torch.as_tensor(behavior,device=device,dtype=torch.float32)
    t=torch.as_tensor(teacher,device=device,dtype=torch.float32)
    def part(h,yy,bb,tt):
        logq=head(h).float().log_softmax(-1).gather(-1,yy[:,None]).squeeze(-1)
        return surrogate(logq,bb,tt,task_advantage).sum()/len(hidden)
    result=hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,len(hidden),chunk):
        args=(hidden[start:start+chunk],y[start:start+chunk],b[start:start+chunk],t[start:start+chunk])
        result=result+checkpoint(part,*args,use_reentrant=False)
    return result


def check():
    torch.manual_seed(21)
    # Exact full-support enumeration: expected score-function gradient equals
    # the independently differentiated dense reverse KL at the behavior policy.
    x=torch.randn(19,dtype=torch.float64,requires_grad=True)
    teacher=torch.randn(19,dtype=torch.float64).log_softmax(-1)
    logq=x.log_softmax(-1); old=logq.detach(); prob=old.exp()
    dense=(logq.exp()*(logq-teacher)).sum()
    expected=torch.autograd.grad(dense,x,retain_graph=True)[0]
    actual=torch.autograd.grad((prob*surrogate(logq,old,teacher,0.)).sum(),x)[0]
    assert torch.allclose(actual,expected,atol=1e-12)
    assert grouped_advantages([0,0,0,0])==[0,0,0,0]
    assert grouped_advantages([1,0,0,0])[0]>0
    # Increasing probability of a uniquely rewarded action decreases the loss.
    x=torch.zeros(2,requires_grad=True); lp=x.log_softmax(-1); b=lp.detach()
    loss=(b.exp()*surrogate(lp,b,b,torch.tensor([1.,-1.]))).sum()
    g=torch.autograd.grad(loss,x)[0]; assert g[0]<0 and g[1]>0
    # Chunking must preserve gradients and exact chosen-token offsets.
    head=torch.nn.Linear(3,19); h=torch.randn(11,3,requires_grad=True)
    y=torch.randint(19,(11,)); b=torch.randn(11).log_softmax(0); t=b+.2
    ref=surrogate(head(h).log_softmax(-1).gather(-1,y[:,None]).squeeze(-1),b,t,.4).mean()
    params=list(head.parameters())+[h]; a=torch.autograd.grad(ref,params)
    loss=loss_from_hidden(head,h,y,b,t,.4,chunk=3); c=torch.autograd.grad(loss,params)
    assert all(torch.allclose(aa,cc,atol=1e-6) for aa,cc in zip(a,c))
    print('OPD: reverse-KL gradient, reward sign, zero-variance groups and chunk gradients PASS')

if __name__=='__main__':check()
