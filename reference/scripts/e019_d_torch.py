"""E018 reference/FLA numerical diagnosis. No optimizer, gradients or changes."""
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from inspect_checkpoint import load_packed
from ternary_core import ROOT, SOURCE
from cached_generation import generate_ids

OUT = ROOT / 'reports/e019-d'


def save(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    torch.manual_seed(20260927)
    reference = inspect.unwrap(qwen.torch_chunk_gated_delta_rule)
    for name in ('torch_recurrent_gated_delta_rule', 'causal_conv1d_fn', 'causal_conv1d_update'):
        setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
    qwen.torch_chunk_gated_delta_rule = reference
    tok = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}
    cfg = json.loads((OUT / 'config.json').read_text())
    print('Загрузка диагностической BF16-копии E018 на GPU6', flush=True)
    model = load_packed(Path(cfg['models']['e018_packed']), single_device='cuda:0')
    cases = json.loads((OUT / 'parity-cases.json').read_text())
    gens = json.loads((OUT / 'parity-generation.json').read_text())
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fast
    results = {}
    with torch.inference_mode():
        for label, fn in [('reference', reference), ('fla', fast)]:
            qwen.torch_chunk_gated_delta_rule = fn
            rows, logits = [], []
            for case in cases:
                t = time.monotonic()
                ids = case['input_ids']
                h = model.model(input_ids=torch.tensor([ids], device='cuda:0'), use_cache=False).last_hidden_state[0]
                last = model.lm_head(h[-1:]).float()[0].cpu().numpy()
                if not np.isfinite(last).all():
                    raise ValueError('Nonfinite endpoint logits')
                logits.append(last)
                nll, count = 0., 0
                if case['kind'] == 'gold':
                    for start in range(case['prompt_tokens']-1, len(ids)-1, 64):
                        end = min(start+64, len(ids)-1)
                        x = model.lm_head(h[start:end]).float()
                        target = torch.tensor(ids[start+1:end+1], device=x.device)
                        nll += float(torch.nn.functional.cross_entropy(x, target, reduction='sum'))
                        count += end-start
                torch.cuda.synchronize()
                row = dict(id=case['id'], kind=case['kind'], source_id=case['source_id'], sequence_tokens=len(ids), nll=nll if count else None, target_tokens=count, seconds=time.monotonic()-t)
                rows.append(row)
                save(label+'-ce.json', rows)
                save('torch-status.json', dict(state='running', variant=label, completed=len(rows), total=len(cases), current_length=len(ids)))
                print(f'{label}: {len(rows)}/{len(cases)}, {len(ids)} токенов', flush=True)
                del h
            np.save(OUT / (label+'-logits.npy'), np.stack(logits))
            results[label] = rows
            generated = []
            for case in gens:
                g = generate_ids(model, case['input_ids'], 128, stop)
                generated.append(dict(id=case['id'], **g, text=tok.decode(g['token_ids'], skip_special_tokens=False)))
                save(label+'-generations.json', generated)
            torch.cuda.empty_cache()
    save('torch-status.json', dict(state='completed', variants=['reference', 'fla'], optimizer_updates=0, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__ == '__main__':
    main()
