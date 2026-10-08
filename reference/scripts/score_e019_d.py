"""Incremental CPU-only scoring of the frozen E019-D math diagnostics."""
import argparse
import collections
import fcntl
import json
import signal
import time
from pathlib import Path

from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/e019-d'


def write(name, obj):
    path = OUT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--expected-total', type=int, default=204)
    parser.add_argument('--subject', default='e018')
    args = parser.parse_args()
    global OUT
    if args.folder is not None:
        OUT = args.folder.resolve()
    lock = (OUT / 'scoring.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = {r['id']: r for r in json.loads((OUT / 'comparison-tasks.json').read_text())}
    prior = OUT / 'comparison-scores.jsonl'
    scores = [json.loads(l) for l in prior.read_text().splitlines()] if prior.exists() else []
    seen = {(r['model'], r['id']) for r in scores}
    def timed_out(*_):
        raise TimeoutError('Math parser/checker exceeded15s')
    signal.signal(signal.SIGALRM, timed_out)
    while True:
        path = OUT / 'comparison-responses.jsonl'
        rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        for row in rows:
            key = (row['model'], row['id'])
            if key in seen:
                continue
            error = None
            try:
                signal.alarm(15)
                gold = parse('\\boxed{'+jobs[row['id']]['gold']+'}', extraction_config=[LatexExtractionConfig()])
                pred = parse(row['answer'], extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]) if row['thinking_completed'] else []
                correct = bool(gold) and bool(pred) and bool(verify(gold, pred))
            except Exception as exc:
                correct = False
                error = str(exc)
            finally:
                signal.alarm(0)
            score = dict(model=row['model'], id=row['id'], family=row['family'], correct=correct, truncated=row['finish_reason'] == 'length', closed=row['thinking_completed'], generated_tokens=row['generated_tokens'], seconds=row['seconds'], verification_error=error, gold_verification=jobs[row['id']]['verification'])
            with prior.open('a') as stream:
                stream.write(json.dumps(score, ensure_ascii=False)+'\n')
                stream.flush()
            scores.append(score)
            seen.add(key)
        groups = collections.defaultdict(list)
        for row in scores:
            groups[(row['model'], row['family'])].append(row)
        summary = {}
        for (model, family), group in groups.items():
            correct = sum(r['correct'] for r in group)
            summary.setdefault(model, {})[family] = dict(total=len(group), correct=correct, accuracy=correct/len(group), truncated=sum(r['truncated'] for r in group), unclosed=sum(not r['closed'] for r in group), mean_output_tokens=sum(r['generated_tokens'] for r in group)/len(group), mean_request_seconds=sum(r['seconds'] for r in group)/len(group), tokens_per_correct=sum(r['generated_tokens'] for r in group)/correct if correct else None)
        write('comparison-summary.json', summary)
        paired = []
        bykey = {(r['model'], r['id']): r for r in scores}
        for family in ('aime25_diagnostic', 'fresh_olympiad', 'verified_skill_transfer'):
            for other in (('e018', 'teacher', 'bonsai') if args.subject != 'e018' else ('teacher', 'bonsai')):
                ids = [i for i, j in jobs.items() if j['family'] == family and (args.subject,i) in bykey and (other,i) in bykey]
                a = [bykey[args.subject,i]['correct'] for i in ids]
                b = [bykey[other,i]['correct'] for i in ids]
                paired.append(dict(family=family, subject=args.subject, comparator=other, total=len(ids), **{args.subject+'_only_correct':sum(x and not y for x,y in zip(a,b))}, comparator_only_correct=sum(y and not x for x,y in zip(a,b)), both_correct=sum(x and y for x,y in zip(a,b))))
        write('paired-comparison.json', paired)
        state = json.loads((OUT / 'status.json').read_text())['state']
        write('scoring-status.json', dict(state='running' if args.follow and state == 'running' and len(scores) < args.expected_total else 'completed', completed=len(scores), total=args.expected_total, updated=time.time()))
        if not args.follow or len(scores) == args.expected_total or state in ('completed', 'failed', 'stopped'):
            break
        time.sleep(3)


if __name__ == '__main__':
    main()
