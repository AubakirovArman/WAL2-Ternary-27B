"""Score completed release responses out of order with unchanged official scorers."""
import argparse, collections, json, multiprocessing, os, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from score_release_benchmarks import ROOT, score, write, get_human_eval_plus, get_mbpp_plus
REFS=None
PROBLEMS=None
def init_worker():
 global REFS,PROBLEMS
 REFS=json.loads((ROOT/'data/release-benchmarks-v1/tasks-with-references.json').read_text())
 PROBLEMS={'humaneval_plus':get_human_eval_plus(),'mbpp_plus':get_mbpp_plus()}
def work(row):
 ref=REFS[row['id']]
 assert row['family']==ref['family'] and row['source_key']==ref['source_key']
 return score(row,ref,PROBLEMS)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
 assert 1<=a.workers<=32
 p=a.folder;result_file=p/'parallel-scores.jsonl';scores={};seen=set();pending={};offset=0
 if result_file.exists():
  with result_file.open('rb+') as f:
   while True:
    start=f.tell();line=f.readline()
    if not line:break
    if not line.endswith(b'\n'):f.truncate(start);break
    r=json.loads(line);assert r['id'] not in scores;scores[r['id']]=r
 def publish(state):
  summary={}
  for family in sorted({r['family'] for r in scores.values()}):
   rows=[r for r in scores.values() if r['family']==family];n=len(rows)
   v=dict(correct=sum(r['correct'] for r in rows),total=n,truncated=sum(r['truncated'] for r in rows))
   v['accuracy']=v['correct']/n
   if family=='ifeval':v['prompt_loose']=sum(r.get('loose_correct',False) for r in rows)/n
   summary[family]=v
  write(p/'parallel-summary.json',summary)
  write(p/'parallel-scoring-status.json',dict(state=state,completed=len(scores),pending=len(pending),workers=a.workers,updated=time.time()))
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as pool,result_file.open('a') as dst:
  publish('scoring')
  while True:
   changed=False
   for future in list(pending):
    if not future.done():continue
    task_id=pending.pop(future);r=future.result();assert r['id']==task_id and task_id not in scores
    dst.write(json.dumps(r)+'\n');dst.flush();os.fsync(dst.fileno());scores[task_id]=r;changed=True
   arrival=p/'responses-arrival.jsonl'
   if arrival.exists():
    with arrival.open('rb') as f:
     f.seek(offset)
     while len(pending)<a.workers*4:
      start=f.tell();line=f.readline()
      if not line or not line.endswith(b'\n'):break
      offset=f.tell();r=json.loads(line);idx=r['id'];assert idx not in seen;seen.add(idx)
      if idx not in scores:pending[pool.submit(work,r)]=idx
   if changed:publish('scoring')
   cfg=json.loads((p/'config.json').read_text())
   if cfg['state']=='completed' and len(scores)==cfg['total']:
    assert set(scores)==set(range(cfg['total']));publish('completed');break
   if cfg['state']=='failed' and not pending:publish('generation_failed');break
   time.sleep(.5)
if __name__=='__main__':main()
