"""Two E018 replicas: speed calibration, then the frozen release8 benchmark."""
import argparse, asyncio, collections, fcntl, hashlib, json, os, statistics, subprocess, time
from pathlib import Path
import httpx
from transformers import AutoTokenizer
from run_release_benchmarks import ROOT, BACK, BIN, env_for, write, SOURCE
from benchmark_tasks import final_text
OUT=ROOT/'reports/e018-release-parallel'
MODEL=ROOT/'reports/e018-final-aime10/e018-final.gguf'
GPUS=['GPU_UUID_REDACTED','GPU_UUID_REDACTED']
PORTS=[18686,18687]
PROCS=[]
LOGS=[]
def status(state,**kw):
 write(OUT/'status.json',dict(state=state,updated=time.time(),**kw))
 print(json.dumps(dict(state=state,**kw),ensure_ascii=False),flush=True)
def hardware():
 r=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.free,power.draw','--format=csv,noheader,nounits'],text=True)
 return [x for x in r.splitlines() if x.split(',')[0].strip() in ('6','7')]
async def generate(client,job,gpu,budget):
 t=time.monotonic()
 payload=dict(prompt=job['input_ids'],n_predict=budget,temperature=0.0,samplers=['temperature'],repeat_penalty=1.0,presence_penalty=0.0,frequency_penalty=0.0,seed=20260927,cache_prompt=False,return_tokens=True,stream=False)
 r=await client.post(f'http://127.0.0.1:{PORTS[gpu]}/completion',json=payload)
 if r.status_code==500 and 'expected Content-only format' in r.text:
  # Preserve generation protocol; raw-token native runner bypasses only HTTP text parsing.
  d=OUT/'http-parser-fallback';d.mkdir(exist_ok=True)
  (d/f"{job['id']}-error.json").write_text(r.text)
  ids=job['input_ids'];inp=d/f"{job['id']}.txt"
  inp.write_text(f"G {job['id']} {len(ids)} {len(ids)} "+' '.join(map(str,ids))+'\n')
  def fallback():
   with (d/f"{job['id']}.log").open('w') as log:
    result=subprocess.run([str(BIN),str(MODEL),str(inp),str(BACK),str(budget),'32768','512'],env=env_for(GPUS[gpu]),stdout=subprocess.PIPE,stderr=log,text=True,check=True,timeout=1800)
   (d/f"{job['id']}.jsonl").write_text(result.stdout)
   a=json.loads(result.stdout);dt=max(a['seconds']-a['prefill_seconds'],1e-9)
   return dict(tokens=a['tokens'],stop_type='limit' if a['finish_reason']=='length' else 'eos',timings=dict(predicted_per_second=len(a['tokens'])/dt,prompt_ms=a['prefill_seconds']*1000),runtime_fallback='native_raw_tokens_after_HTTP_parser_error')
  v=await asyncio.to_thread(fallback)
 else:
  r.raise_for_status();v=r.json()
 if v.get('error'):raise RuntimeError(str(v['error']))
 assert not v.get('truncated'),('context overflow',job['id'])
 assert v.get('stop_type') in ('eos','limit'),v.get('stop_type')
 assert isinstance(v['tokens'],list) and len(v['tokens'])<=budget+1
 return dict(id=job['id'],gpu=gpu+6,wall_seconds=time.monotonic()-t,**v)
async def batch(client,jobs,concurrency,budget,handle=None):
 queues=[asyncio.Queue(),asyncio.Queue()]
 for i,j in enumerate(jobs):queues[i%2].put_nowait(j)
 rows=[];start=time.monotonic();peaks=[]
 async def monitor():
  while True:
   if any(p.poll() is not None for p in PROCS):raise RuntimeError('Inference server exited; see server GPU logs')
   peaks.append(hardware());await asyncio.sleep(5)
 async def worker(gpu):
  while not queues[gpu].empty():
   job=queues[gpu].get_nowait();r=await generate(client,job,gpu,budget);rows.append(r)
   if handle:handle(job,r,time.monotonic()-start)
 async def workers():
  async with asyncio.TaskGroup() as group:
   for i in range(concurrency):group.create_task(worker(i%2))
 m=asyncio.create_task(monitor());w=asyncio.create_task(workers())
 try:
  done,_=await asyncio.wait([m,w],return_when=asyncio.FIRST_COMPLETED)
  if m in done:await m
  await w
 finally:
  m.cancel();w.cancel();await asyncio.gather(m,w,return_exceptions=True)
 seconds=time.monotonic()-start;tokens=sum(len(r['tokens']) for r in rows)
 per=[r['timings']['predicted_per_second'] for r in rows]
 return rows,dict(concurrency=concurrency,requests=len(rows),output_tokens=tokens,wall_seconds=seconds,aggregate_tokens_per_second=tokens/seconds,mean_request_decode_tokens_per_second=statistics.mean(per),mean_request_seconds=statistics.mean(r['wall_seconds'] for r in rows),p95_request_seconds=sorted(r['wall_seconds'] for r in rows)[min(len(rows)-1,int(len(rows)*.95))],gpu_samples=peaks,errors=0)
async def main():
 OUT.mkdir(exist_ok=True)
 parser=argparse.ArgumentParser();parser.add_argument('--benchmark-only',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 if args.resume:args.benchmark_only=True
 if args.benchmark_only:
  assert args.resume or not (OUT/'responses.jsonl').exists(),'Benchmark already has responses; do not overwrite'
  assert (OUT/'speed-trials.json').exists()
 else:assert not (OUT/'status.json').exists(),'Use a new output directory; do not overwrite a campaign'
 lock=(ROOT/'runs/gpu67.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for gpu in GPUS:
  free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>130*1024,'GPU6/7 occupied'
 raw=(ROOT/'reports/release-benchmarks-v1/tasks.json').read_bytes();jobs=json.loads(raw)
 refs=json.loads((ROOT/'data/release-benchmarks-v1/tasks-with-references.json').read_text())
 assert len(jobs)==len(refs)==9049 and all(j['id']==i and j['instruction']==refs[i]['instruction'] for i,j in enumerate(jobs))
 assert all(len(j['input_ids'])+8192<=32768 for j in jobs)
 (OUT/'tasks.json').write_bytes(raw)
 tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
 assert all(tok.encode(j['prompt'],add_special_tokens=False)==j['input_ids'] for j in jobs)
 cfg=dict(model='e018-step25000',checkpoint=str(MODEL),checkpoint_sha256=hashlib.file_digest(MODEL.open('rb'),'sha256').hexdigest(),total=len(jobs),state='running',decoding='greedy',max_new_tokens=8192,context_limit=32768,reasoning_effort='medium',tasks_sha256=hashlib.sha256(raw).hexdigest(),attempts_per_task=1)
 if args.resume:
  old=json.loads((OUT/'config.json').read_text())
  for key in ('checkpoint_sha256','tasks_sha256','total','max_new_tokens','decoding'):assert old[key]==cfg[key],key
 write(OUT/'config.json',cfg)
 write(OUT/'protocol.json',dict(**cfg,physical_gpus=[6,7],replicas=2,slots_per_replica=25,kv_cache='f16',batch_size=2048,ubatch_size=512,prompt_mode='Exact archived input token IDs; original HuggingFace prompts verified. No chat retemplating.',parallelism_note='Continuous batching can change floating-point rounding and greedy trajectories relative to sequential native runs.',pilot='Same100 stratified prompts, max512 output tokens, no pilot outputs scored; choose throughput only. Total in-flight concurrency across both replicas.',final_reload_ce_skipped=True))
 status('starting_servers',total=len(jobs))
 for gpu,port in zip(GPUS,PORTS):
  log=(OUT/f'server-gpu{6+len(PROCS)}.log').open('a');LOGS.append(log)
  cmd=[str(BACK/'llama-server'),'-m',str(MODEL),'--host','127.0.0.1','--port',str(port),'-ngl','99','-np','25','-c',str(25*32768),'-b','2048','-ub','512','-fa','on','-ctk','f16','-ctv','f16','-t','8','-tb','8','--fit','off','--cache-ram','0','--no-context-shift','--metrics','--threads-http','32','--special']
  PROCS.append(subprocess.Popen(cmd,env=env_for(gpu),stdout=log,stderr=log))
 async with httpx.AsyncClient(timeout=httpx.Timeout(1800,connect=10),limits=httpx.Limits(max_connections=110,max_keepalive_connections=60),trust_env=False) as client:
  for port,p in zip(PORTS,PROCS):
   start=time.monotonic()
   while True:
    if p.poll() is not None:raise RuntimeError(f'Server {port} failed; see log')
    try:
     r=await client.get(f'http://127.0.0.1:{port}/health')
     if r.status_code==200:break
    except httpx.ConnectError:pass
    if time.monotonic()-start>600:raise TimeoutError('Server startup timeout')
    await asyncio.sleep(2)
  await batch(client,jobs[:2],2,32)
  # Compare server output token IDs against the existing native first AIME answer.
  previous=json.loads((ROOT/'reports/e018-final-aime10/aime10-responses.json').read_text())[0]
  expected=tok.encode(previous['raw_response'],add_special_tokens=False)[:128]
  parity=await generate(client,jobs[0],0,128)
  same=parity['tokens']==expected
  write(OUT/'server-parity.json',dict(first128_equal=same,expected=expected,actual=parity['tokens'],scope='Greedy single-request prefix against saved native result. Batched numerical identity is not guaranteed.'))
  if not same:raise RuntimeError('Server/native greedy128 parity differs; inspect before full campaign')
  families=collections.defaultdict(list)
  for j in jobs:families[j['family']].append(j)
  pilot=[]
  for i in range(13):
   for family in sorted(families):
    if len(pilot)<100:pilot.append(families[family][i*len(families[family])//13])
  write(OUT/'pilot-task-ids.json',[j['id'] for j in pilot]);trials=json.loads((OUT/'speed-trials.json').read_text()) if args.benchmark_only else []
  for c in (() if args.benchmark_only else (2,10,20,30,40,50)):
   status('speed_pilot',concurrency=c,completed_trials=len(trials),total_trials=6)
   rs,summary=await batch(client,pilot,c,512)
   write(OUT/f'pilot-c{c}-responses.json',rs);trials.append(summary);write(OUT/'speed-trials.json',trials)
   print(f"Параллельно {c}: {summary['aggregate_tokens_per_second']:.1f} ток/с суммарно; {summary['mean_request_decode_tokens_per_second']:.1f} ток/с на запрос",flush=True)
  best=max(trials,key=lambda t:t['aggregate_tokens_per_second']);concurrency=best['concurrency']
  # Short full-budget smoke test checks sustained memory behaviour; excluded from scores.
  if args.benchmark_only:
   smoke=dict(state='skipped',reason='User requested immediate full benchmark after completed speed pilots')
  else:
   status('long_context_smoke',concurrency=concurrency)
   rs,smoke=await batch(client,pilot[:concurrency],concurrency,8192)
   write(OUT/'long-smoke.json',smoke)
  write(OUT/'selected-speed.json',dict(concurrency=concurrency,selection='Highest aggregate pilot throughput among error-free trials',pilot=best,long_smoke=smoke))
  cfg.update(concurrency=concurrency);write(OUT/'config.json',cfg)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
  scorelog=(OUT/'scoring.log').open('w');LOGS.append(scorelog)
  scorer=subprocess.Popen([str(ROOT/'.venv-release/bin/python'),'-u',str(ROOT/'scripts/score_release_benchmarks.py'),str(OUT),'--follow'],env=env,stdout=scorelog,stderr=scorelog);PROCS.append(scorer)
  previous={}
  if args.resume:
   for line in (OUT/'responses-arrival.jsonl').read_text().splitlines():
    row=json.loads(line);assert row['id'] not in previous;previous[row['id']]=row
  existing=[json.loads(x) for x in (OUT/'responses.jsonl').read_text().splitlines()] if args.resume else []
  assert [r['id'] for r in existing]==list(range(len(existing)))
  for r in existing:assert r==previous[r['id']]
  persisted=len(existing);buffered={i:r for i,r in previous.items() if i>=persisted}
  finished=len(previous);initial_finished=finished;total_tokens=0
  jobs_pending=[j for j in jobs if j['id'] not in previous]
  status('benchmark',concurrency=concurrency,completed=finished,total=len(jobs))
  with (OUT/'responses.jsonl').open('a' if args.resume else 'w') as responses:
   def handle(job,r,seconds):
    nonlocal finished,persisted,total_tokens
    text=tok.decode(r['tokens'],skip_special_tokens=False);answer,closed=final_text(text,True)
    row=dict(**job,raw_response=text,answer=answer,thinking_completed=closed,finish_reason='length' if r['stop_type']=='limit' else 'eos',generated_tokens=len(r['tokens']),seconds=r['wall_seconds'],prefill_seconds=r['timings']['prompt_ms']/1000,physical_gpu=r['gpu'],runtime_fallback=r.get('runtime_fallback'))
    with (OUT/'responses-arrival.jsonl').open('a') as arrival:
     arrival.write(json.dumps(row,ensure_ascii=False)+'\n');arrival.flush();os.fsync(arrival.fileno())
    buffered[job['id']]=row;finished+=1;total_tokens+=len(r['tokens'])
    # Scorer expects ID order even though requests finish out of order.
    while persisted in buffered:
     responses.write(json.dumps(buffered.pop(persisted),ensure_ascii=False)+'\n');persisted+=1
    responses.flush();os.fsync(responses.fileno())
    status('benchmark',concurrency=concurrency,completed=finished,persisted=persisted,total=len(jobs),output_tokens=total_tokens,aggregate_tokens_per_second=total_tokens/seconds,elapsed_seconds=seconds,eta_seconds=(len(jobs)-finished)*seconds/(finished-initial_finished))
   _,summary=await batch(client,jobs_pending,concurrency,8192,handle)
  cfg['state']='completed';write(OUT/'config.json',cfg);write(OUT/'generation-speed.json',summary)
  for p in PROCS[:2]:p.terminate()
  for p in PROCS[:2]:p.wait(timeout=60)
  status('scoring',completed=len(jobs),total=len(jobs))
  while scorer.poll() is None:await asyncio.sleep(5)
  if scorer.returncode:raise RuntimeError('Scoring failed; generations retained')
  status('completed',completed=len(jobs),total=len(jobs))
if __name__=='__main__':
 try:asyncio.run(main())
 except BaseException as e:
  if OUT.exists():
   status('failed',error=str(e))
   if (OUT/'config.json').exists():
    cfg=json.loads((OUT/'config.json').read_text())
    if cfg['state']!='completed':cfg['state']='failed';write(OUT/'config.json',cfg)
  raise
 finally:
  for p in PROCS:
   if p.poll() is None:p.terminate()
  for p in PROCS:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill()
  for f in LOGS:f.close()
