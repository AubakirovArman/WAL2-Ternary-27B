"""Frozen retention-oriented mixture; excludes synthetic format-only block."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
 fresh=ROOT/'data/fresh-5000-v1';old=ROOT/'data/reasoning-fresh-v4-tokenized';out=ROOT/'data/e014-retention-mix'
 sources=[];rows=[]
 for folder in [fresh,old]:
  m=json.loads((folder/'manifest.json').read_text());raw=(folder/'train.jsonl').read_bytes();assert digest(raw)==m['files']['train']['sha256'] and m['state']=='completed'
  items=[json.loads(l) for l in raw.splitlines()]
  if folder==fresh:items=[r for r in items if r['category']!='format']
  rows.extend(dict(r,split='train',mixture_source=folder.name) for r in items);sources.append(dict(folder=str(folder),source_sha256=digest(raw),included=len(items)))
 validation=(old/'validation.jsonl').read_bytes();assert digest(validation)=='592a90f098b128af2badad0db018906f4c25998a1c1e6bcc3acdc159ba9236c2'
 held=[json.loads(l) for l in validation.splitlines()]
 if (old/'auxiliary-validation.jsonl').exists():held.extend(json.loads(l) for l in (old/'auxiliary-validation.jsonl').read_text().splitlines())
 assert len({r['sha256'] for r in rows+held})==len(rows)+len(held)
 assert len({r['id'] for r in rows})==len(rows)==5340
 for r in rows:
  n=r['prompt_tokens'];ids=r['input_ids'];assert 0<n<len(ids)<=2048 and r['labels']==[-100]*n+ids[n:]
 random.Random(20260921014).shuffle(rows);out.mkdir(exist_ok=False)
 train=''.join(json.dumps(r)+'\n' for r in rows).encode();(out/'train.jsonl').write_bytes(train);(out/'validation.jsonl').write_bytes(validation)
 m=dict(state='completed',truncated=False,max_length=2048,sources=sources,seed=20260921014,excluded='1500 synthetic format-only rows; efficacy hypothesis, not established cause of E013 regression',files={})
 for split,data,items in [('train',train,rows),('validation',validation,[json.loads(l) for l in validation.splitlines()])]:m['files'][split]=dict(count=len(items),sha256=digest(data),tokens=sum(len(r['input_ids']) for r in items))
 (out/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m),flush=True)
if __name__=='__main__':main()
