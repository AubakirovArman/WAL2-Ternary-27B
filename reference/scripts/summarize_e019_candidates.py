"""Pairwise immutable task results for the three retained E019 candidates."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/e019-candidate-review'
SOURCES={2560:ROOT/'reports/e019-best-math68',3584:ROOT/'reports/e019-step3584-math68',4096:ROOT/'reports/e019-step4096-math68'}


def main():
    OUT.mkdir(exist_ok=True)
    tasks=json.loads((ROOT/'reports/e019-d/comparison-tasks.json').read_text())
    baseline={(r['model'],r['id']):r for r in map(json.loads,(ROOT/'reports/e019-d/comparison-scores.jsonl').open())}
    candidates={};answers={}
    for step,folder in SOURCES.items():
        assert json.loads((folder/'status.json').read_text())['state']=='completed'
        scores=[r for r in map(json.loads,(folder/'comparison-scores.jsonl').open()) if r['model']=='e019']
        assert len(scores)==68 and all(r['verification_error'] is None for r in scores)
        candidates[step]={r['id']:r for r in scores}
        answers[step]={r['id']:r['answer'] for r in map(json.loads,(folder/'comparison-responses.jsonl').open()) if r['model']=='e019'}
    paired=[]
    for reference in ('e018',2560,3584):
        for subject in (2560,3584,4096):
            if reference==subject or isinstance(reference,int) and subject<reference:continue
            for family in sorted({t['family'] for t in tasks}):
                ids=[t['id'] for t in tasks if t['family']==family]
                a=baseline if reference=='e018' else candidates[reference]
                old=lambda i:a['e018',i]['correct'] if reference=='e018' else a[i]['correct']
                gained=[i for i in ids if candidates[subject][i]['correct'] and not old(i)]
                lost=[i for i in ids if old(i) and not candidates[subject][i]['correct']]
                retained=[i for i in ids if old(i) and candidates[subject][i]['correct']]
                paired.append(dict(reference=reference,subject=subject,family=family,gained=gained,lost=lost,retained=retained))
    (OUT/'paired-summary.json').write_text(json.dumps(paired,indent=2))
    records=[]
    for task in tasks:
        i=task['id'];row=dict(id=i,family=task['family'],gold=task['gold'],question=task.get('instruction',task.get('prompt','')),
                            e018_correct=baseline['e018',i]['correct'])
        for step in SOURCES:
            row.update({f'e019_{step}_correct':candidates[step][i]['correct'],f'e019_{step}_truncated':candidates[step][i]['truncated'],f'e019_{step}_answer':answers[step][i]})
        records.append(row)
    with (OUT/'task-by-task.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (OUT/'task-by-task.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(json.dumps([r for r in paired if r['reference']=='e018'],indent=2))


if __name__=='__main__':main()
