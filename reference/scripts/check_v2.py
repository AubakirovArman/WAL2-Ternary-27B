"""Integration checks: held-out isolation, complete answers, exact optimizer/RNG resume."""
import json, tempfile
from pathlib import Path
import torch
from transformers import Adafactor
import ternary_core as core
from data_v2 import read_rows,digest
from train_v2 import save_latent,restore,schedule

def main():
    core.guard()
    corpus=core.ROOT/'data/v2/pilot.jsonl';rows=read_rows(corpus)
    old=read_rows(core.ROOT/'data/teacher_pairs.jsonl')
    assert [r for r in rows if r['split']!='train']==[r for r in old if r['split']!='train']
    assert len({r['id'] for r in rows})==len(rows)
    train=[r for r in rows if r['split']=='train'];assert len(train)==10000
    reserved={digest(r['instruction']) for r in rows if r['split']!='train'}
    reserved.update(digest(r['instruction']) for r in read_rows(core.ROOT/'data/v2/verified_holdout.jsonl'))
    assert not reserved.intersection(digest(r['instruction']) for r in train)
    assert all(r['prompt_tokens']+r['answer_tokens']<=512 for r in train)
    data=core.load_examples(corpus,512);assert len(data['validation'])==128 and len(data['test'])==128
    assert len(data['train'])==10000
    assert [r['id'] for r in schedule(data['train'],corpus,256)]==[r['id'] for r in schedule(data['train'],corpus,256)]
    for e in data['train']:
        assert e['labels'][-1]!=-100 and len(e['input_ids'])<=512
    print('PASS: 10k complete training examples, unchanged 128/128 held-out, disjoint verified suite, fixed ordering',flush=True)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.a=torch.nn.Linear(8,8,bias=False,device='cuda:0');self.b=torch.nn.Linear(8,2,bias=False,device='cuda:1')
            self.a.register_buffer('scales',torch.ones(1,device='cuda:0',dtype=torch.bfloat16))
        def forward(self,x):return self.b(self.a(x).to('cuda:1'))
    model=Tiny();opt=Adafactor(model.parameters(),lr=1e-5,relative_step=False,scale_parameter=False,warmup_init=False,beta1=None,weight_decay=0.)
    def step(opt):
        opt.zero_grad(set_to_none=True)
        x=torch.randn(4,8,device='cuda:0');target=torch.randn(4,2,device='cuda:1')
        loss=(model(x)-target).square().mean();loss.backward();opt.step();return loss.detach()
    step(opt)
    with tempfile.TemporaryDirectory(prefix='v2-resume-',dir=core.ROOT/'reports') as directory:
        path=Path(directory)/'latent';save_latent(model,opt,path,{'step':1,'order_ids':[1,2,3]})
        expected_loss=step(opt);expected=[p.detach().clone() for p in model.parameters()]
        opt=restore(model,path);actual_loss=step(opt)
        assert torch.equal(expected_loss,actual_loss)
        assert all(torch.equal(p,e) for p,e in zip(model.parameters(),expected))
        assert json.loads((path/'metadata.json').read_text())['order_ids']==[1,2,3]
    print('PASS: saved/restored weights, optimizer, both CUDA RNG states reproduce next update exactly',flush=True)

if __name__=='__main__':main()
