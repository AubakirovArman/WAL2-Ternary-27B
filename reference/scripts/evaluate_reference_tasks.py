"""Reproduce the 16-task no-thinking, greedy reference generation protocol."""
import argparse,json
from pathlib import Path
from ternary_core import ROOT,guard,load_source
from inspect_checkpoint import load_packed,generate_reference

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();guard();model=load_packed(args.checkpoint) if args.checkpoint else load_source()
    rows=[];tasks=[json.loads(s) for s in (ROOT/'data/v2/verified_holdout.jsonl').read_text().splitlines() if json.loads(s)['split']=='test'][:16]
    for task in tasks:
        answer=generate_reference(model,task['instruction'],64)
        try:correct=json.loads(answer)==json.loads(task['response']) if task['checker']=='json' else answer.strip()==task['response'].strip()
        except (ValueError,TypeError):correct=False
        rows.append({'id':task['id'],'topic':task['topic'],'instruction':task['instruction'],'expected':task['response'],'answer':answer,'correct':correct})
        args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    print('RESULT',sum(r['correct'] for r in rows),len(rows))
if __name__=='__main__':main()
