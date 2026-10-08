"""Full small hybrid-model integration of the E008 mixed objective, CPU only."""
import json
from pathlib import Path
import torch
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
from ternary_core import ternarize
from train_reasoning_qat import chunked_ce
from chunked_kd import loss_from_hidden
from kd_objective import compressed_kl

torch.set_num_threads(2);torch.manual_seed(4128)
cfg=Qwen3_5TextConfig(vocab_size=256,hidden_size=128,intermediate_size=256,num_hidden_layers=4,
 num_attention_heads=4,num_key_value_heads=2,head_dim=32,linear_num_key_heads=2,
 linear_num_value_heads=4,linear_key_head_dim=32,linear_value_head_dim=32,
 max_position_embeddings=256,full_attention_interval=4,
 rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,'mrope_section':[2,2,4]})
cfg._attn_implementation='sdpa';cfg.use_cache=False
model=Qwen3_5ForCausalLM(cfg).bfloat16();model.requires_grad_(False);ternarize(model,rotate=True);model.train()
ids=torch.randint(0,256,(1,23));labels=ids[0,1:].clone();labels[:7]=-100
replay=torch.randint(0,256,(1,17));rl=replay[0,1:].clone();rl[:5]=-100
p,ix=torch.randn(16,256).softmax(-1).topk(128,-1);cache={'top_ids':ix,'top_logp':p.log().half(),'tail_prob':1-p.sum(-1)}
h=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
ce=torch.nn.functional.cross_entropy(model.lm_head(h).float(),labels)
rh=model.model(input_ids=replay,use_cache=False).last_hidden_state[0,:-1]
kl,logq=compressed_kl(model.lm_head(rh).float(),ix,cache['top_logp'],cache['tail_prob'])
rce=-logq.gather(-1,rl.clamp_min(0)[:,None]).squeeze(-1)[rl!=-100].mean()
reference=.5*ce+.5*(.9*kl+.1*rce);reference.backward()
params={n:p for n,p in model.named_parameters() if p.requires_grad};expected={n:p.grad.clone() for n,p in params.items()}
model.zero_grad(set_to_none=True);model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
h=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1];actual_ce,_=chunked_ce(model.lm_head,h,labels,chunk=4)
(.5*actual_ce).backward()
rh=model.model(input_ids=replay,use_cache=False).last_hidden_state[0,:-1];actual_kd,_,_=loss_from_hidden(model.lm_head,rh,rl,cache,chunk_tokens=4)
(.5*actual_kd).backward();actual=.5*actual_ce.detach()+.5*actual_kd.detach()
assert torch.allclose(actual,reference.detach(),atol=2e-6)
errors={}
for n,p in params.items():
 assert p.grad is not None and torch.isfinite(p.grad).all()
 errors[n]=float((p.grad-expected[n]).norm()/expected[n].norm().clamp_min(1e-12))
assert max(errors.values())<.02
report={'scope':'CPU small hybrid Qwen3.5, full ternary QAT, BF16, nested checkpoint, sequential mixed backward',
 'trainable_matrices':len(params),'reference_loss':float(reference.detach()),'chunked_loss':float(actual),
 'max_relative_gradient_l2_difference':max(errors.values()),'tolerance':.02,'per_matrix_relative_difference':errors}
(Path(__file__).resolve().parents[1]/'reports/reasoning-qat-hybrid-cpu-check.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='per_matrix_relative_difference'},indent=2))
