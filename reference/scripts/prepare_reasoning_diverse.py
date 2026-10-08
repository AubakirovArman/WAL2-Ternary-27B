"""Freeze completed diverse collection, preserving the preassigned validation split."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from reasoning_data import encode,SOURCE


def prepare(folder,output,max_length=2048):
    meta=json.loads((folder/'manifest.json').read_text())
    if meta.get('state')!='completed':raise ValueError('Collection not completed')
    tasks=json.loads((folder/'tasks.json').read_text());byid={r['id']:r for r in tasks}
    if len(byid)!=len(tasks):raise ValueError('Duplicate scheduled IDs')
    raw=(folder/'accepted.jsonl').read_bytes();rows=[json.loads(s) for s in raw.splitlines()]
    if dict(Counter(r['split'] for r in rows))!=meta['accepted_splits']:raise ValueError('Accepted split count mismatch')
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    prepared={'train':[],'validation':[]};dropped=[];seen=set();seen_ids=set()
    for row in rows:
        if row['id'] in seen_ids or row['sha256'] in seen:raise ValueError('Duplicate accepted record')
        seen_ids.add(row['id']);seen.add(row['sha256'])
        planned=byid[row['id']]
        for field in ('split','instruction','source_id','sha256'):
            if row[field]!=planned[field]:raise ValueError('Changed scheduled field: '+field)
        item=encode(row,tok,max_length,allow_validation=True,allow_teacher_only=True)
        if item is None:dropped.append({'id':row['id'],'split':row['split']})
        else:
            item.update(split=row['split'],source_id=row['source_id'],verification=row['verification'])
            prepared[row['split']].append(item)
    if not all(prepared.values()):raise ValueError('Empty training or validation split')
    output.mkdir(parents=True,exist_ok=False);files={}
    for split,items in prepared.items():
        data=''.join(json.dumps(r)+'\n' for r in items).encode();(output/f'{split}.jsonl').write_bytes(data)
        files[split]={'count':len(items),'sha256':hashlib.sha256(data).hexdigest(),'tokens':sum(len(r['input_ids']) for r in items)}
    report={'state':'completed','files':files,'max_length':max_length,'truncated':False,'dropped_long':dropped,
        'source_accepted_sha256':hashlib.sha256(raw).hexdigest(),'source_schedule_sha256':hashlib.sha256((folder/'tasks.json').read_bytes()).hexdigest(),
        'verification':meta['verification'],'validation_caveat':meta['validation_caveat']}
    (output/'manifest.json').write_text(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-length',type=int,default=2048);a=p.parse_args()
    print(json.dumps(prepare(a.folder,a.output,a.max_length),indent=2))

if __name__=='__main__':main()
