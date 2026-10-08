"""Same frozen 96-task protocol, streaming native PQ2 generation on GPU6."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from transformers import AutoTokenizer
from ternary_core import ROOT,SOURCE,guard
from benchmark_tasks import final_text

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=ROOT/'models/vol2-e013-prism/vol2-e013-pq2.gguf');p.add_argument('--output',type=Path,default=ROOT/'reports/e013-native-smoke');a=p.parse_args()
 guard();out=a.output;out.mkdir(exist_ok=False)
 prior=ROOT/'reports/e012-smoke';raw=(prior/'tasks.json').read_bytes();jobs=json.loads(raw)
 assert len(jobs)==96 and len({j['id'] for j in jobs})==96
 tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
 lines=[]
 for job in jobs:
  text=tok.apply_chat_template([{'role':'user','content':job['instruction']}],tokenize=False,add_generation_prompt=True,enable_thinking=job['thinking'],reasoning_effort=job['reasoning_effort'])
  ids=tok.encode(text,add_special_tokens=False)
  assert text==job['prompt'] and ids==job['input_ids'] and job['max_new_tokens']==2048
  assert len(ids)+2048<=8192
  base=f"{job['id']} {len(ids)} {len(ids)} "+' '.join(map(str,ids))
  lines.extend(['T '+base+' '+text.encode().hex(),'G '+base])
 (out/'tasks.json').write_bytes(raw);(out/'inputs.txt').write_text('\n'.join(lines)+'\n')
 model=a.model.resolve()
 config={k:json.loads((prior/'config.json').read_text())[k] for k in ('subset','suite','decoding','max_new_tokens','context_limit','reasoning_effort')}
 config.update(model='student-native-pq2',checkpoint=str(model),tasks_sha256=hashlib.sha256(raw).hexdigest(),state='running',runtime='prism 9a9394a PQ2, GPU6, batch512',export_manifest=str(model.with_suffix('.export.json')))
 def save():
  p=out/'config.tmp';p.write_text(json.dumps(config,indent=2));p.replace(out/'config.json')
 save();start=time.monotonic()
 tool=ROOT/'tools/bonsai-runtime';backend=tool/'llama-prism-b10709-9a9394a'
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=env['CUDA_VISIBLE_DEVICES'].split(',')[0]
 env['LD_LIBRARY_PATH']=':'.join(map(str,[backend,tool/'cuda-deps/nvidia/cuda_runtime/lib',tool/'cuda-deps/nvidia/cublas/lib']))
 count=0
 with (out/'native.log').open('w') as err,(out/'native.jsonl').open('w') as native,(out/'responses.jsonl').open('w') as responses:
  with subprocess.Popen([str(tool/'prism-native-eval'),str(model),str(out/'inputs.txt'),str(backend),'2048','8192','512'],env=env,stdout=subprocess.PIPE,stderr=err,text=True) as process:
   for line in process.stdout:
    native.write(line);native.flush();r=json.loads(line);job=jobs[count]
    assert r['kind']=='generation' and r['id']==job['id']
    text=tok.decode(r['tokens'],skip_special_tokens=False);answer,complete=final_text(text,job['thinking'])
    row={**job,'raw_response':text,'answer':answer,'thinking_completed':complete,'finish_reason':r['finish_reason'],'generated_tokens':len(r['tokens']),'seconds':r['seconds'],'prefill_seconds':r['prefill_seconds']}
    responses.write(json.dumps(row,ensure_ascii=False)+'\n');responses.flush();count+=1
    print(f"Проверено {count}/96: {job['family']}, {len(r['tokens'])} токенов, {r['seconds']:.1f} с",flush=True)
   if process.wait()!=0:raise RuntimeError('Native generation failed; see native.log')
 assert count==96
 config.update(state='completed',seconds=time.monotonic()-start);save()
if __name__=='__main__':main()
