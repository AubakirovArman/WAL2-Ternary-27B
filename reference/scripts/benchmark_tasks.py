"""Frozen smoke/final task construction; test answers never enter generation prompts."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def tasks(subset='smoke',suite='core'):
    if suite=='russian':
        folder=ROOT/'data/evaluation-ru-v1'
        plan=json.loads((folder/'PLAN.json').read_text());raw=(folder/plan['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==plan['sha256']
        rows=[json.loads(s) for s in raw.splitlines()]
        return [{'id':i,'family':'rummlu','source_index':index,'source_key':rows[index]['meta']['id'],
                 'instruction':rows[index]['instruction'].format(**rows[index]['inputs']),
                 'thinking':True,'reasoning_effort':'medium','max_new_tokens':2048}
                for i,index in enumerate(plan['smoke_indices' if subset=='smoke' else 'final_indices'])]
    if suite!='core':raise ValueError('Unknown benchmark suite')
    plan=json.loads((ROOT/'data/evaluation-v1/PLAN.json').read_text());result=[]
    for family,spec in plan['sets'].items():
        path=ROOT/'data/evaluation-v1'/spec['file'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==spec['sha256']
        rows=[json.loads(s) for s in (gzip.decompress(raw) if path.suffix=='.gz' else raw).splitlines()]
        for index in spec['smoke_indices' if subset=='smoke' else 'final_indices']:
            row=rows[index]
            if family=='gsm8k':
                prompt=row['question']+'\n\nSolve the problem. End your final answer with #### followed by the numeric result.'
                thinking=True
            elif family=='humaneval':
                prompt='Implement the following Python function. Return the complete function and any required imports in a Python code block, with no explanation outside the code.\n\n'+row['prompt']
                thinking=True
            else:prompt=row['prompt'];thinking=False
            result.append({'id':len(result),'family':family,'source_index':index,'source_key':row.get('task_id',row.get('key',index)),
                           'instruction':prompt,'thinking':thinking,'reasoning_effort':'medium','max_new_tokens':2048})
    return result

def final_text(raw,thinking):
    # Keep thinking delimiters but remove terminal chat control tokens from scored text.
    raw=raw.strip()
    while raw.endswith(('<|im_end|>','<|endoftext|>')):
        raw=raw.rsplit('<|',1)[0].rstrip()
    if thinking:
        if '</think>' not in raw:return '',False
        return raw.rsplit('</think>',1)[1].strip(),True
    return raw.strip(),True
