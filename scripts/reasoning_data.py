"""Separate, untruncated thinking-mode data preparation; does not alter E006/E007."""
import argparse,hashlib,json,re
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'models/qwen3.8-27b-fp8-source'

def encode(row,tokenizer,max_length=2048,*,allow_validation=False,allow_teacher_only=False,allow_verified_code=False):
    splits=('train','validation') if allow_validation else ('train',)
    if row.get('split') not in splits or row.get('enable_thinking') is not True:
        raise ValueError('Unexpected split or non-thinking record')
    reasoning=row['reasoning'];answer=row['response']
    if not reasoning.strip() or not answer.strip():raise ValueError('Empty reasoning or final answer')
    for text in (reasoning,answer):
        if any(marker in text for marker in ('<think>','</think>','<|im_start|>','<|im_end|>','<|endoftext|>')):
            raise ValueError('Unexpected control delimiter inside content')
    expected=hashlib.sha256(row['instruction'].encode()).hexdigest()
    if row['sha256']!=expected:raise ValueError('Instruction hash mismatch')
    if 'expected' in row:
        match=re.search(r'ANSWER:\s*(-?\d+)\s*$',answer.strip())
        if not match or int(match.group(1))!=row['expected']:raise ValueError('Final answer verification failed')
    elif allow_verified_code and row.get('verification')=='final_code_passed_fixed_sandbox_tests_reasoning_unverified':
        from verified_code_tasks import tasks
        from collect_reasoning_code import verify
        match=re.fullmatch(r'verified-code-(\d{4})',row['id'])
        if not match:raise ValueError('Unexpected verified-code task ID')
        index=int(match[1]);reference=tasks(max(8,index+1))[index]
        # Regenerate cases from the fixed seed; do not trust tests supplied in an answer row.
        if any(row.get(key)!=value for key,value in reference.items()):
            raise ValueError('Verified-code task or fixtures changed')
        verdict=verify(reference,{'finish_reason':'stop','message':{'reasoning_content':reasoning,'content':answer}})
        if not verdict['passed']:raise ValueError('Final code re-verification failed: '+str(verdict))
    elif not (allow_teacher_only and row.get('verification')=='teacher_answer_and_reasoning_unverified'):
        raise ValueError('Unverified teacher answers require explicit opt-in')
    prompt=tokenizer.apply_chat_template([{'role':'user','content':row['instruction']}],tokenize=False,
        add_generation_prompt=True,enable_thinking=True,reasoning_effort='medium')
    if not prompt.endswith('<think>\n'):raise ValueError('Unexpected thinking prompt format')
    reasoning_part=reasoning.rstrip()+'\n</think>\n'
    text=prompt+reasoning_part+answer+'<|im_end|>'
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids'];offsets=encoded['offset_mapping']
    prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
    if ids[:len(prompt_ids)]!=prompt_ids:raise ValueError('Prompt token boundary changed')
    if len(ids)>max_length:return None
    labels=[-100]*len(prompt_ids)+ids[len(prompt_ids):]
    final_start=len(prompt)+len(reasoning_part)
    final_labels=[token if start>=final_start else -100 for token,(start,end) in zip(ids,offsets)]
    if ids[-1]!=tokenizer.convert_tokens_to_ids('<|im_end|>'):raise ValueError('Missing answer terminator')
    if not any(x!=-100 for x in final_labels):raise ValueError('Empty final-answer mask')
    return {'id':row['id'],'input_ids':ids,'labels':labels,'final_labels':final_labels,
            'prompt_tokens':len(prompt_ids),'reasoning_and_final_tokens':len(ids)-len(prompt_ids),
            'final_answer_tokens':sum(x!=-100 for x in final_labels),'sha256':row['sha256']}

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verified-code',action='store_true',help='Re-execute fixed synthetic code tests before tokenizing')
    p.add_argument('--max-length',type=int,default=2048);a=p.parse_args()
    meta=json.loads((a.folder/'manifest.json').read_text())
    if meta.get('state')!='completed':raise ValueError('Collection must finish before freezing training data')
    rows=[json.loads(s) for s in (a.folder/'accepted.jsonl').read_text().splitlines()]
    if len(rows)!=meta['accepted']:raise ValueError('Accepted count mismatch')
    if a.verified_code:
        for filename,key in [('accepted.jsonl','accepted_sha256'),('tasks.json','tasks_sha256')]:
            if hashlib.sha256((a.folder/filename).read_bytes()).hexdigest()!=meta[key]:
                raise ValueError('Verified-code collection hash mismatch: '+filename)
        if not meta.get('training_only'):raise ValueError('Verified code must be training-only')
        if not all(r.get('verification')=='final_code_passed_fixed_sandbox_tests_reasoning_unverified' for r in rows):
            raise ValueError('Mixed verification types in code collection')
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);prepared=[];dropped=[];seen=set()
    for row in rows:
        if row['sha256'] in seen:raise ValueError('Duplicate instruction')
        seen.add(row['sha256']);item=encode(row,tok,a.max_length,allow_verified_code=a.verified_code)
        if item is None:dropped.append(row['id'])
        else:prepared.append(item)
    a.output.mkdir(parents=True,exist_ok=False)
    data=''.join(json.dumps(r)+'\n' for r in prepared)
    (a.output/'train.jsonl').write_text(data)
    report={'source':str(a.folder),'state':'completed','train_count':len(prepared),'dropped_long':dropped,
            'max_length':a.max_length,'truncated':False,'sha256':hashlib.sha256(data.encode()).hexdigest(),
            'code_tests_reexecuted':a.verified_code,
            'loss_mask':'labels covers reasoning and final answer; final_labels covers final answer only',
            'source_sha256':hashlib.sha256((a.folder/'accepted.jsonl').read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
