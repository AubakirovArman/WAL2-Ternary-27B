"""Measure Bonsai on the frozen reasoning validation; no generated benchmark answers."""
import hashlib,json,os,subprocess
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,guard


def main():
    guard()
    out=ROOT/'reports/bonsai-reasoning-ce';out.mkdir(exist_ok=False)
    tool=ROOT/'tools/bonsai-runtime';backend=tool/'llama-prism-b10709-9a9394a'
    binary=tool/'bonsai-chunked-eval'
    model=ROOT/'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=':'.join(map(str,[backend,tool/'cuda-deps/nvidia/cuda_runtime/lib',tool/'cuda-deps/nvidia/cublas/lib']))
    def run(name,lines,chunk=512,context=2048):
        inputs=out/(name+'.txt');inputs.write_text('\n'.join(lines)+'\n')
        with (out/(name+'.jsonl')).open('w') as stdout,(out/(name+'.log')).open('w') as stderr:
            subprocess.run([str(binary),str(model),str(inputs),str(backend),'1',str(context),str(chunk)],
                           env=env,stdout=stdout,stderr=stderr,check=True,timeout=3600)
        return [json.loads(l) for l in (out/(name+'.jsonl')).read_text().splitlines()]
    def compare(a,b,tolerance):
        assert len(a)==len(b)
        differences=[]
        for x,y in zip(a,b):
            assert x['id']==y['id'] and x['tokens']==y['tokens']
            differences.append(abs(x['nll']-y['nll'])/x['tokens'])
        assert max(differences)<=tolerance,(differences,tolerance)
        return max(differences)
    short=[l for l in (ROOT/'reports/bonsai2/inputs.txt').read_text().splitlines() if l.startswith('E ')][:8]
    ids=[int(l.split()[1]) for l in short]
    prior={r['id']:r for r in map(json.loads,(ROOT/'reports/bonsai2/raw-results.jsonl').read_text().splitlines()) if r['kind']=='ce'}
    short512=run('short512',short,context=1024)
    reference_delta=compare(short512,[prior[i] for i in ids],1e-5)
    short128=run('short128',short,chunk=128,context=1024)
    chunk_delta=compare(short512,short128,.005)
    folder=ROOT/'data/reasoning-mixed-v3-tokenized'
    raw=(folder/'validation.jsonl').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    manifest=json.loads((folder/'manifest.json').read_text())
    assert digest==manifest['files']['validation']['sha256']
    rows=[json.loads(l) for l in raw.splitlines()]
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    lines=[];mapping=[]
    for i,row in enumerate(rows):
        tokens=row['input_ids'];labels=row['labels'];p=row['prompt_tokens']
        assert labels==[-100]*p+tokens[p:] and 0<p<len(tokens)<=2048
        text=tok.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        assert tok.encode(text,add_special_tokens=False)==tokens
        prefix=f'{i} {p} {len(tokens)} '+' '.join(map(str,tokens))
        lines.extend(['T '+prefix+' '+text.encode().hex(),'E '+prefix])
        mapping.append({'id':i,'source_id':row['id'],'tokens':len(tokens)-p})
    assert len(rows)==120 and sum(r['tokens'] for r in mapping)==69641
    (out/'mapping.json').write_text(json.dumps(mapping,indent=2))
    scored=run('reasoning512',lines)
    assert len(scored)==len(rows)
    assert all(r['id']==i and r['tokens']==mapping[i]['tokens'] for i,r in enumerate(scored))
    # Exercise both a long and short prefix with different decode batch boundaries.
    indices=sorted({max(range(len(rows)),key=lambda i:len(rows[i]['input_ids'])),0})
    subset=[line for i in indices for line in lines[2*i:2*i+2]]
    checked=run('reasoning128-check',subset,chunk=128)
    long_delta=compare([scored[i] for i in indices],checked,.005)
    result={'state':'completed','model':str(model),'validation_sha256':digest,'examples':len(rows),
            'tokens':sum(r['tokens'] for r in scored),
            'reasoning_and_answer_ce':sum(r['nll'] for r in scored)/sum(r['tokens'] for r in scored),
            'short_reference_max_ce_delta':reference_delta,'short_chunk_max_ce_delta':chunk_delta,
            'long_chunk_max_ce_delta':long_delta,'native_tokenization_checked':True,
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'scope':'Same input IDs and contiguous reasoning-plus-answer mask as E009; native Bonsai vs PyTorch BF16 arithmetic differs. Teacher-forced CE, not independent reasoning accuracy.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
