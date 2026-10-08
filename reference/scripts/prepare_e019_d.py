"""Freeze diagnostic inputs before running any model. CPU only; no training."""
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1] / '.venv-release/lib/python3.12/site-packages'))
    import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/e019-d'
SOURCE = ROOT / 'models/qwen3.8-27b-fp8-source'


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'config.json').exists(), 'Diagnostic inputs already frozen'
    tok = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    validation = [json.loads(l) for l in (ROOT / 'data/e018-25000/validation.jsonl').read_text().splitlines()]
    cases = []
    used = set()
    for length in (256, 512, 1024, 2048, 4096, 8192):
        selected = sorted((r for r in validation if r['id'] not in used and r['category'] == 'math'), key=lambda r: abs(len(r['input_ids']) - length))[:2]
        for r in selected:
            used.add(r['id'])
            cases.append(dict(id=len(cases), kind='gold', source_id=r['id'], input_ids=r['input_ids'], prompt_tokens=r['prompt_tokens'], desired_length=length))
    responses = [json.loads(l) for l in (ROOT / 'reports/e018-release-parallel/responses-arrival.jsonl').read_text().splitlines()]
    bad = sorted((r for r in responses if r['family'] == 'aime25' and r['finish_reason'] == 'length'), key=lambda r: r['id'])[:2]
    assert len(bad) == 2
    for r in bad:
        ids = r['input_ids'] + tok.encode(r['raw_response'], add_special_tokens=False)
        for length in (512, 2048, 4096, 8192):
            assert len(ids) >= length
            cases.append(dict(id=len(cases), kind='student_prefix', source_id=r['id'], input_ids=ids[:length], prompt_tokens=len(r['input_ids']), desired_length=length))
    write('parity-cases.json', cases)
    generation = [dict(id=r['id'], input_ids=r['input_ids'], source_id=r['source_id'], kind=r['kind']) for r in [cases[0], cases[4], cases[8], cases[-2]]]
    write('parity-generation.json', generation)

    # Exclude even unaccepted old candidate prompts, not only final training rows.
    seen = set()
    grams = set()
    audit_files = []
    paths = list((ROOT / 'data').glob('**/accepted.jsonl')) + [ROOT / 'data/e018-25000/tasks.jsonl', ROOT / 'data/release-benchmarks-v1/tasks-with-references.json']
    for p in sorted(set(paths)):
        raw = p.read_bytes()
        rows = json.loads(raw) if p.suffix == '.json' else [json.loads(l) for l in raw.splitlines()]
        for r in rows:
            text = r.get('instruction', '')
            if text:
                n = normalized(text)
                seen.add(n)
                words = n.split()
                grams.update(' '.join(words[k:k+8]) for k in range(len(words)-7))
            ref = r.get('reference', {})
            if isinstance(ref, dict):
                for k in ('question', 'problem'):
                    if isinstance(ref.get(k), str):
                        seen.add(normalized(ref[k]))
        audit_files.append(dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest()))
    source = ROOT / 'data/e018-25000/sources/numina.parquet'
    pool = pq.read_table(source, columns=['source', 'problem', 'solution']).to_pylist()
    pool.sort(key=lambda r: hashlib.sha256(r['problem'].encode()).hexdigest())
    tasks = []
    rejection = collections.Counter()
    buckets = collections.Counter()
    for r in pool:
        if r['source'] != 'olympiads' or not 600 <= len(r['solution']) <= 16000:
            continue
        if re.search(r'\b(prove|proof|show that)\b', r['problem'], re.I):
            continue
        answers = re.findall(r'\\boxed\{\s*([+-]?\d+)\s*\}', r['solution'])
        if not answers or len(set(answers)) != 1:
            continue
        n = normalized(r['problem'])
        words = n.split()
        ng = {' '.join(words[k:k+8]) for k in range(len(words)-7)}
        if n in seen or sum(g in grams for g in ng) >= max(3, .5*len(ng)):
            rejection['old_prompt_overlap'] += 1
            continue
        bucket = min(2, len(r['solution']) // 2000)
        if buckets[bucket] >= {0: 13, 1: 13, 2: 10}[bucket]:
            continue
        buckets[bucket] += 1
        seen.add(n)
        tasks.append(dict(id=30+len(tasks), family='fresh_olympiad', instruction=r['problem'] + '\n\nSolve the problem. End your final answer with the integer result inside \\boxed{}.', gold=answers[-1], reference_solution=r['solution'], verification='Source silver label, NOT independently proof-verified', source='AI-MO/NuminaMath-CoT/train shard0/olympiads', source_problem_sha256=hashlib.sha256(r['problem'].encode()).hexdigest(), solution_chars=len(r['solution'])))
        if len(tasks) == 30:
            break
    assert len(tasks) == 30, (len(tasks), buckets, rejection)

    # New compositional skill probes with exact independent arithmetic/DP oracles.
    verified = []
    for seed in (19037, 19079):
        lower, upper = seed, seed + 11273
        mods, rems = (11, 13, 17), (seed % 11, (seed+3) % 13, (seed+5) % 17)
        gold = sum(x for x in range(lower, upper+1) if all(x % m == a for m, a in zip(mods, rems)) and math.gcd(x, 30) == 1)
        problem = f'Find the sum of all integers x in [{lower}, {upper}] satisfying x ≡ {rems[0]} (mod 11), x ≡ {rems[1]} (mod 13), x ≡ {rems[2]} (mod 17), and gcd(x,30)=1.'
        verified.append((problem, gold, 'finite exhaustive integer oracle; CRT plus coprimality'))
    for digits in (8, 9):
        dp = {(0, 0, False): 1}
        for pos in range(digits):
            nxt = collections.Counter()
            for (rem, total, prev), count in dp.items():
                for d in range(1 if pos == 0 else 0, 10):
                    if prev and d == 7:
                        continue
                    if total+d <= 31:
                        nxt[((rem*10+d) % 37, total+d, d == 7)] += count
            dp = nxt
        gold = sum(v for (rem, total, prev), v in dp.items() if rem == 0 and total == 31)
        problem = f'How many {digits}-digit positive decimal integers are divisible by 37, have digit sum 31, and do not contain the adjacent pair 77?'
        verified.append((problem, gold, 'exact digit DP: residue, digit sum, last-digit state'))
    for total in (67, 73):
        gold = sum(1 for x in range(total+1) for y in range(total+1) for z in range(total+1) if 2*x+3*y+5*z == total and x < y and math.gcd(y, z) == 1)
        problem = f'Count triples of nonnegative integers (x,y,z) with 2x+3y+5z={total}, x<y, and gcd(y,z)=1 (using gcd(0,0)=0).'
        verified.append((problem, gold, 'finite exhaustive triple oracle'))
    for n in (31, 37):
        # Count 0/1 subsets by cardinality, sum residue and number of even elements.
        dp = {(0, 0, 0): 1}
        for a in range(1, n+1):
            nxt = collections.Counter(dp)
            for (k, r, e), count in dp.items():
                if k < 6 and e + (a % 2 == 0) <= 3:
                    nxt[(k+1, (r+a) % 11, e+(a % 2 == 0))] += count
            dp = nxt
        gold = dp.get((6, 0, 3), 0)
        problem = f'How many 6-element subsets of {{1,2,...,{n}}} have exactly three even elements and a sum divisible by 11?'
        verified.append((problem, gold, 'exact subset generating-function DP'))
    for problem, gold, oracle in verified:
        tasks.append(dict(id=30+len(tasks), family='verified_skill_transfer', instruction=problem + '\n\nGive a finite checked solution and end with the integer answer inside \\boxed{}.', gold=str(gold), verification=oracle, source='New local E019-D probe, not a standardized olympiad benchmark'))
    archived = json.loads((ROOT / 'reports/release-benchmarks-v1/tasks.json').read_text())
    refs = {r['id']: r for r in json.loads((ROOT / 'data/release-benchmarks-v1/tasks-with-references.json').read_text())}
    aime = []
    for r in archived:
        if r['family'] == 'aime25':
            aime.append(dict(id=len(aime), family='aime25_diagnostic', archive_id=r['id'], instruction=r['instruction'], prompt=r['prompt'], input_ids=r['input_ids'], gold=refs[r['id']]['reference']['answer'], verification='Published dataset label; already used for development'))
    assert len(aime) == 30
    for r in tasks:
        r['prompt'] = tok.apply_chat_template([dict(role='user', content=r['instruction'])], tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort='medium')
        r['input_ids'] = tok.encode(r['prompt'], add_special_tokens=False)
    tasks = aime + tasks
    assert len(tasks) == 68 and all(len(r['input_ids'])+8192 <= 32768 for r in tasks)
    write('comparison-tasks.json', tasks)
    write('fresh-exposure-audit.json', dict(files=audit_files, rejected=dict(rejection), olympiad_count=30, skill_count=8, scope='Normalized exact/8gram screening against old candidate and accepted prompts. Does not establish semantic or base-pretraining independence. Numina labels remain silver; report separately from exact probes.'))
    cfg = dict(state='prepared', experiment='E019-D', optimizer_updates=0, models=dict(e018=str(ROOT / 'reports/e018-final-aime10/e018-final.gguf'), e018_packed=str(ROOT / 'reports/e018-final-aime10/direct-packed'), bonsai=str(ROOT / 'models/bonsai2-27b-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf'), teacher='Qwen/Qwen3.8-27B-FP8', teacher_api='http://127.0.0.1:8000/v1'), parity_cases=len(cases), gold_cases=12, own_prefix_cases=8, comparison_tasks=68, decoding='greedy; temperature0; samplers=[temperature]; repeat_penalty1; thinking medium; one attempt', max_new_tokens=8192, context=32768, native_storage='Original PQ2 GGUF, not BF16 expansion', torch_parity='Explicit BF16-expanded diagnostic reference only, no inference speed claim', sequence_parity_scope='Full CE on12 gold rows; endpoint distributions on20 short/long gold and student prefixes; four128-token continuations; cache/batch/order checks.', source_scope='Existing authorized teacher API; no teacher server changes. Teacher token-template equivalence is checked before requests.', fresh_scope='30 unused source olympiad prompts with silver labels plus8 fresh exact skill probes, separate metrics; not a new official AIME evaluation', task_sha256=hashlib.sha256((OUT / 'comparison-tasks.json').read_bytes()).hexdigest(), parity_sha256=hashlib.sha256((OUT / 'parity-cases.json').read_bytes()).hexdigest())
    write('config.json', cfg)
    write('status.json', dict(state='prepared', stage='Подготовлены неизменяемые задания E019-D', completed=0, total=0))
    print(json.dumps({k: cfg[k] for k in ('parity_cases', 'comparison_tasks', 'optimizer_updates')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
