"""Evaluate original Bonsai PTQ1_0 with identical HF token IDs and task protocol."""
import hashlib,json,os,subprocess
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,guard,load_examples

def main():
    guard();tool=ROOT/'tools/bonsai-runtime';backend=tool/'llama-prism-b10709-9a9394a'
    model=ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    reportdir=ROOT/'reports/bonsai2';reportdir.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    data=load_examples(ROOT/'data/teacher_pairs.jsonl',512)
    tasks=[json.loads(s) for s in (ROOT/'data/v2/verified_holdout.jsonl').read_text().splitlines() if json.loads(s)['split']=='test'][:16]
    raw={r['id']:r for r in map(json.loads,(ROOT/'data/teacher_pairs.jsonl').read_text().splitlines())}
    lines=[]
    def line(mode,id,p,ids,text=None):
        lines.append(f'{mode} {id} {p} {len(ids)} '+' '.join(map(str,ids))+(' '+text.encode().hex() if text is not None else ''))
    def prompt(text):return tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    for row in tasks:
        text=prompt(row['instruction']);ids=tok.encode(text,add_special_tokens=False)
        line('T',row['id'],len(ids),ids,text);line('G',row['id'],len(ids),ids)
    ids_to_split={}
    for split in ['validation','test']:
        for ex in data[split]:
            ids=ex['input_ids'];p=next(i for i,label in enumerate(ex['labels']) if label!=-100)
            text=prompt(raw[ex['id']]['instruction']);line('T',ex['id'],p,ids[:p],text)
            # Verify answer tokenizer too before scoring identical truncated target IDs.
            answer=raw[ex['id']]['response']+'<|im_end|>';a=tok.encode(answer,add_special_tokens=False)
            if len(a)<=512:line('T',ex['id'],1,a,answer)
            line('E',ex['id'],p,ids);ids_to_split[ex['id']]=split
    inputs=reportdir/'inputs.txt';inputs.write_text('\n'.join(lines)+'\n')
    env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(map(str,[backend,tool/'cuda-deps/nvidia/cuda_runtime/lib',tool/'cuda-deps/nvidia/cublas/lib']))
    with (reportdir/'raw-results.jsonl').open('w') as out,(ROOT/'logs/bonsai-eval-native.log').open('w') as err:
        subprocess.run([str(tool/'bonsai-eval'),str(model),str(inputs),str(backend)],stdout=out,stderr=err,env=env,check=True)
    rows=[json.loads(s) for s in (reportdir/'raw-results.jsonl').read_text().splitlines()]
    result={'repo':'prism-ml/Ternary-Bonsai-2-27B-gguf','revision':'6ed5e12bf84b7a63069882c91dd9e9218647d17b',
            'runtime':'PrismML llama.cpp prism-b10709-9a9394a CUDA12.8','file_bytes':model.stat().st_size,
            'protocol':'identical HF prompt/answer token IDs; native tokenizer checked on inputs; answer-only CE; no thinking; greedy max64',
            'limitation':'Bonsai native CUDA vs source/student PyTorch BF16; compute precision/runtime are not identical.',
            'input_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest()}
    for split in ['validation','test']:
        selected=[r for r in rows if r['kind']=='ce' and ids_to_split[r['id']]==split]
        assert len(selected)==len(data[split])
        result[split]={'answer_ce':sum(r['nll'] for r in selected)/sum(r['tokens'] for r in selected),'tokens':sum(r['tokens'] for r in selected),'examples':len(selected)}
    byid={r['id']:r for r in tasks};generation=[]
    for r in rows:
        if r['kind']!='generation':continue
        task=byid[r['id']];answer=tok.decode(r['tokens'],skip_special_tokens=True)
        try:correct=json.loads(answer)==json.loads(task['response']) if task['checker']=='json' else answer.strip()==task['response'].strip()
        except (ValueError,TypeError):correct=False
        generation.append({'id':r['id'],'topic':task['topic'],'instruction':task['instruction'],'expected':task['response'],'answer':answer,'correct':correct})
    assert len(generation)==16
    result['strict_exact_match']={'correct':sum(r['correct'] for r in generation),'total':len(generation)}
    (reportdir/'generation.json').write_text(json.dumps(generation,ensure_ascii=False,indent=2))
    (reportdir/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
