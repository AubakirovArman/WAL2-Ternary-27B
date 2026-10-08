"""Greedy text decoding with per-layer KV/recurrent state caching, no training mutation."""
import time
import torch

@torch.no_grad()
def generate_ids(model,prompt_ids,max_new_tokens,stop_ids,input_device='cuda:0',use_cache=True):
    model.eval();generated=[];cache=None;current=list(prompt_ids);start=time.perf_counter();first=None
    reason='length'
    for _ in range(max_new_tokens):
        outputs=model.model(input_ids=torch.tensor([current],device=input_device),
                            past_key_values=cache if use_cache else None,use_cache=use_cache)
        logits=model.lm_head(outputs.last_hidden_state[:,-1])
        token=int(logits.argmax(-1).item())
        if first is None:first=time.perf_counter()-start
        if token in stop_ids:reason='eos';break
        generated.append(token)
        if use_cache:
            cache=outputs.past_key_values
            if cache is None:raise RuntimeError('Model did not return a requested cache')
            current=[token]
        else:current=list(prompt_ids)+generated
    return {'token_ids':generated,'finish_reason':reason,'seconds':time.perf_counter()-start,'first_token_seconds':first}

def check_cpu():
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    torch.set_num_threads(2);torch.manual_seed(712)
    cfg=Qwen3_5TextConfig(vocab_size=64,hidden_size=64,intermediate_size=128,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,linear_num_key_heads=2,
        linear_num_value_heads=4,linear_key_head_dim=16,linear_value_head_dim=16,
        max_position_embeddings=256,full_attention_interval=4,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':0.5,'mrope_section':[1,1,2]})
    cfg._attn_implementation='sdpa';model=Qwen3_5ForCausalLM(cfg).eval()
    for prefix in [[4,12,9],[6,13,11,19,23,8,4]]:
        a=generate_ids(model,prefix,20,set(),input_device='cpu',use_cache=False)
        b=generate_ids(model,prefix,20,set(),input_device='cpu',use_cache=True)
        assert a['token_ids']==b['token_ids'],(a,b)
        c=generate_ids(model,prefix,20,{a['token_ids'][0]},input_device='cpu')
        assert c['token_ids']==[] and c['finish_reason']=='eos'
    print('PASS: cached vs full-prefix greedy agrees on small hybrid Qwen3.5 model; EOS/length handling')
if __name__=='__main__':check_cpu()
