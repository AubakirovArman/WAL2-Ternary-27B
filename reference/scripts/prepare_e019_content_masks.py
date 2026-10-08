"""Add diagnostic masks to a separate validation copy; never rewrite the original."""
import collections
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from reasoning_data import SOURCE
from reasoning_diagnostics import token_partition

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/e019-reasoning-pilot-v1'
OUT=ROOT/'reports/e019-content-masks'


def main():
    OUT.mkdir(exist_ok=False);tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    accepted={r['id']:r for r in map(json.loads,(DATA/'accepted.jsonl').open())}
    counts=collections.Counter();n=0
    with (OUT/'validation.jsonl').open('w') as out:
        for item in map(json.loads,(DATA/'validation.jsonl').open()):
            if item['category']=='math':
                row=accepted[item['id']]
                prompt=tok.apply_chat_template([dict(role='user',content=row['instruction'])],tokenize=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='medium')
                text=prompt+row['reasoning'].rstrip()+'\n</think>\n'+row['response']+'<|im_end|>'
                encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
                assert encoded['input_ids']==item['input_ids']
                item['diagnostic_masks']=token_partition(text,encoded['offset_mapping'],item['labels'],item['final_labels'])
                counts.update({k:sum(v) for k,v in item['diagnostic_masks'].items()});n+=1
            out.write(json.dumps(item)+'\n')
    report=dict(math_examples=n,target_tokens_by_group=dict(counts),original_sha256=hashlib.sha256((DATA/'validation.jsonl').read_bytes()).hexdigest(),
                masked_copy_sha256=hashlib.sha256((OUT/'validation.jsonl').read_bytes()).hexdigest(),input_ids_and_loss_labels_changed=False,
                scope='Masks only affect diagnostic aggregation; every original target remains supervised. Formula/prose classes are lexical heuristics; logical transitions are not automatically proved. Math-only partitions.')
    assert n==128 and sum(counts.values())==27342
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
