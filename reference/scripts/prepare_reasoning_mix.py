"""Combine frozen reasoning datasets without changing the base validation split."""
import argparse,hashlib,json,random
from pathlib import Path


def digest(raw):return hashlib.sha256(raw).hexdigest()


def load_split(folder,split,supplement=False):
    manifest_raw=(folder/'manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    if manifest.get('state')!='completed' or manifest.get('truncated') is not False:
        raise ValueError('Dataset must be completed and untruncated: '+str(folder))
    if supplement:
        if split!='train' or 'files' in manifest:raise ValueError('Expected a train-only supplement')
        count=manifest['train_count'];expected=manifest['sha256']
    else:
        count=manifest['files'][split]['count'];expected=manifest['files'][split]['sha256']
    raw=(folder/f'{split}.jsonl').read_bytes()
    if digest(raw)!=expected:raise ValueError('Dataset hash mismatch: '+str(folder))
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=count or not rows:raise ValueError('Empty or inconsistent split')
    for row in rows:
        if row.get('split',split)!=split:raise ValueError('Unexpected split')
        ids=row['input_ids'];labels=row['labels'];final=row['final_labels'];n=row['prompt_tokens']
        if not (0<n<len(ids)<=2048 and len(ids)==len(labels)==len(final)):
            raise ValueError('Invalid token lengths')
        if labels[:n]!=[-100]*n or labels[n:]!=ids[n:] or labels.count(248069)!=1:
            raise ValueError('Invalid reasoning supervision')
        if any(y!=-100 and (i<n or y!=ids[i]) for i,y in enumerate(final)) or all(y==-100 for y in final):
            raise ValueError('Invalid final answer supervision')
    return rows,raw,{'folder':str(folder.resolve()),'split':split,'count':count,'sha256':expected,'manifest_sha256':digest(manifest_raw)}


def prepare(base,supplements,output,seed=2026092001,validation_base=None):
    train,_,train_info=load_split(base,'train')
    validation,validation_raw,val_info=load_split(validation_base or base,'validation')
    sources=[train_info,val_info]
    auxiliary=[];auxiliary_raw=None
    if validation_base is not None and validation_base.resolve()!=base.resolve():
        auxiliary,auxiliary_raw,info=load_split(base,'validation');sources.append(info)
    combined=[{**row,'split':'train','mixture_source':base.name} for row in train]
    for folder in supplements:
        rows,_,info=load_split(folder,'train',supplement=True);sources.append(info)
        combined.extend({**row,'split':'train','mixture_source':folder.name} for row in rows)
    all_rows=combined+validation+auxiliary
    for key in ('id','sha256'):
        if len({row[key] for row in all_rows})!=len(all_rows):
            raise ValueError('Duplicate or train/validation overlap: '+key)
    random.Random(seed).shuffle(combined)
    training_raw=''.join(json.dumps(row)+'\n' for row in combined).encode()
    files={}
    for split,rows,raw in [('train',combined,training_raw),('validation',validation,validation_raw)]:
        files[split]={'count':len(rows),'tokens':sum(len(r['input_ids']) for r in rows),'sha256':digest(raw)}
    report={'state':'completed','files':files,'sources':sources,'shuffle_seed':seed,'max_length':2048,'truncated':False,
            'validation_caveat':'Chosen validation source bytes unchanged; teacher traces on formerly training-designated instructions, not independent final benchmark',
            'verification':'Mixed provenance: diverse teacher traces unverified; supplements checked separately; no reasoning trace correctness claim'}
    if auxiliary_raw is not None:
        report['auxiliary_validation']={'count':len(auxiliary),'sha256':digest(auxiliary_raw),
                                      'file':'auxiliary-validation.jsonl','used_for_training':False}
    output.mkdir(parents=True,exist_ok=False)
    (output/'train.jsonl').write_bytes(training_raw);(output/'validation.jsonl').write_bytes(validation_raw)
    if auxiliary_raw is not None:(output/'auxiliary-validation.jsonl').write_bytes(auxiliary_raw)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--supplement',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-base',type=Path,help='Keep an earlier frozen validation; preserve new validation separately')
    p.add_argument('--seed',type=int,default=2026092001);a=p.parse_args()
    print(json.dumps(prepare(a.base,a.supplement,a.output,a.seed,a.validation_base),indent=2))

if __name__=='__main__':main()
