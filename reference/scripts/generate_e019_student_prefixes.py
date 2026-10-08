"""New train-only E018 prefixes for repair-SFT, on GPUs6/7. No updates."""
import asyncio
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

import run_e019_d as runtime
from ternary_core import ROOT, SOURCE, GPU_IDS

OUT=ROOT/'data/e019-reasoning-pilot-v1'
STUDENT=OUT/'student'
MODEL=ROOT/'reports/e018-final-aime10/e018-final.gguf'


def state(state,**kw):runtime.write(STUDENT/'status.json',dict(state=state,updated=time.time(),**kw))


async def main():
    STUDENT.mkdir(exist_ok=True);runtime.OUT=STUDENT
    lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for gpu in GPU_IDS:
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>130*1024,'GPU6/7 occupied'
    cfg=json.loads((OUT/'config.json').read_text());raw=(OUT/'tasks.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==cfg['tasks_sha256']
    pool=[json.loads(l) for l in raw.splitlines()];families={f:[t for t in pool if t['repair'] and t['family_index']==f] for f in range(16)}
    # First32 per family in the frozen random ordering =512 new prefixes.
    jobs=[families[f][i] for i in range(32) for f in range(16)];assert len(jobs)==512
    path=STUDENT/'prefixes.jsonl';done={r['id'] for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    assert done <= {t['id'] for t in jobs}
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);start=time.monotonic();tokens=0;initial=len(done)
    state('starting',completed=len(done),total=512)
    servers=[runtime.start_server(MODEL,gpu,port,12) for gpu,port in zip(GPU_IDS,(18896,18897))]
    runtime.write(STUDENT/'protocol.json',dict(student='E018step25000 unchanged PQ2',checkpoint_sha256=hashlib.file_digest(MODEL.open('rb'),'sha256').hexdigest(),scope='768-token diagnostic prefixes, deliberately capped, NOT full solutions or quality benchmark;512new train-only math tasks, no validation/benchmark rollouts',physical_gpus=[6,7],concurrency=24,slots_per_gpu=12,context_per_slot=32768,greedy=True,optimizer_updates=0))
    async with httpx.AsyncClient(timeout=1200,limits=httpx.Limits(max_connections=30),trust_env=False) as client:
        await asyncio.gather(*(runtime.healthy(client,port,proc) for port,proc in zip((18896,18897),servers)))
        sem=[asyncio.Semaphore(12),asyncio.Semaphore(12)]
        async def job(task,gpu):
            nonlocal tokens
            async with sem[gpu]:
                ids=tok.apply_chat_template([dict(role='user',content=task['base_instruction'])],tokenize=True,add_generation_prompt=True,enable_thinking=True,reasoning_effort='medium')
                if not isinstance(ids,list):ids=ids['input_ids']
                try:
                    result=await runtime.completion(client,18896+gpu,ids,768)
                except httpx.HTTPStatusError as error:
                    if error.response.status_code!=500 or 'expected Content-only format' not in error.response.text:raise
                    # Only bypass failed HTTP text parsing; keep original quantized forward.
                    inp=STUDENT/(task['id']+'.txt');inp.write_text(f"G 0 {len(ids)} {len(ids)} "+' '.join(map(str,ids))+'\n')
                    def fallback():
                        with (STUDENT/(task['id']+'.log')).open('w') as log:
                            p=subprocess.run([str(runtime.BIN),str(MODEL),str(inp),str(runtime.BACK),'768','32768','512'],env=runtime.env_for(GPU_IDS[gpu]),stdout=subprocess.PIPE,stderr=log,text=True,check=True,timeout=600)
                        v=json.loads(p.stdout);return dict(tokens=v['tokens'],stop_type='limit' if v['finish_reason']=='length' else 'eos',runtime_fallback='native_raw_tokens_after_HTTP_parser_error')
                    result=await asyncio.to_thread(fallback)
                text=tok.decode(result['tokens'],skip_special_tokens=False);row=dict(id=task['id'],family_index=task['family_index'],prefix_tokens=result['tokens'],draft=text,stop_type=result['stop_type'],physical_gpu=gpu+6,runtime_fallback=result.get('runtime_fallback'),prefix_only=True)
                with path.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                done.add(task['id']);tokens+=len(result['tokens']);elapsed=time.monotonic()-start
                state('running',completed=len(done),total=512,output_tokens=tokens,tokens_per_second=tokens/max(elapsed,1),eta_seconds=(512-len(done))*elapsed/max(1,len(done)-initial))
                if len(done)%16==0:print(f'Новые черновики E018: {len(done)}/512',flush=True)
        await asyncio.gather(*(job(t,i%2) for i,t in enumerate(jobs) if t['id'] not in done))
    state('completed',completed=512,total=512);print('512новыхпрефиксов сохранены; GPU6/7 освобождаются.',flush=True)


if __name__=='__main__':
    try:asyncio.run(main())
    except BaseException as e:
        if STUDENT.exists():state('failed',error=str(e))
        raise
    finally:runtime.stop_servers()
