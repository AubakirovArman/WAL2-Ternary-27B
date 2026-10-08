"""Audit the finished E019 pilot without changing training data or weights."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer
from reasoning_data import SOURCE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/e019-reasoning-pilot-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    config = json.loads((DATA / 'config.json').read_text())
    assert manifest['state'] == 'completed'
    assert not manifest['truncated'] and not manifest['training_started']
    assert digest(DATA / 'config.json') == manifest['config_sha256']
    assert digest(DATA / 'tasks.jsonl') == config['tasks_sha256']
    for name, expected in config['task_code'].items():
        assert digest(DATA / 'code' / name) == expected, name
    code = json.loads((DATA / 'collection-code-manifest.json').read_text())
    for name, expected in code.items():
        assert digest(DATA / 'code' / name) == expected, name
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    close = tokenizer.encode('</think>', add_special_tokens=False)
    assert len(close) == 1
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    accepted = [json.loads(line) for line in (DATA / 'accepted.jsonl').open()]
    by_id = {row['id']: row for row in accepted}
    assert len(by_id) == len(accepted) == 4352
    assert len({row['sha256'] for row in accepted}) == len(accepted)
    ids, stats, families = set(), {}, collections.Counter()
    for split, expected_count in (('train', 4096), ('validation', 256)):
        path = DATA / (split + '.jsonl')
        assert digest(path) == manifest['files'][split]['sha256']
        lengths, targets, finals, categories = [], [], [], collections.Counter()
        for line in path.open():
            item = json.loads(line)
            assert item['id'] not in ids
            ids.add(item['id'])
            row = by_id[item['id']]
            assert item['split'] == row['split'] == split
            assert hashlib.sha256(row['instruction'].encode()).hexdigest() == item['sha256'] == row['sha256']
            assert item['verification'] == row['verification']
            seq, labels, final = item['input_ids'], item['labels'], item['final_labels']
            n, prompt = len(seq), item['prompt_tokens']
            assert 0 < prompt < n <= manifest['max_length']
            assert len(labels) == len(final) == n
            assert labels[:prompt] == [-100] * prompt and labels[prompt:] == seq[prompt:]
            assert all(x == -100 or x == token for x, token in zip(final, seq))
            active_final = [i for i, label in enumerate(final) if label != -100]
            assert active_final == list(range(active_final[0], n))
            assert active_final[0] > prompt and seq[-1] == eos
            assert seq.count(close[0]) == 1
            assert seq.index(close[0]) < active_final[0]
            prompt_text = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['instruction']}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True, reasoning_effort='medium')
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            assert seq[:prompt] == prompt_ids
            assert item['reasoning_and_final_tokens'] == n - prompt
            assert item['final_answer_tokens'] == len(active_final)
            assert item['repair'] == row['repair']
            if row['category'] in ('math', 'math_repair'):
                family = row['family_index']
                assert family in (range(16) if split == 'train' else range(16, 20))
                families[(split, row['category'], family)] += 1
            if row['repair']:
                assert split == 'train' and row['category'] == 'math_repair'
                assert row['instruction'] != row['base_instruction']
            categories[row['category']] += 1
            lengths.append(n); targets.append(n - prompt); finals.append(len(active_final))
        assert len(lengths) == expected_count == manifest['files'][split]['count']
        assert sum(lengths) == manifest['files'][split]['tokens']
        expected_categories = config['target' if split == 'train' else 'holdout']
        assert dict(categories) == {k: v for k, v in expected_categories.items() if v}
        ordered = sorted(lengths)
        stats[split] = dict(count=len(lengths), categories=dict(categories), tokens=sum(lengths),
                            supervised_tokens=sum(targets), final_tokens=sum(finals),
                            length_mean=statistics.mean(lengths), length_median=statistics.median(lengths),
                            length_p95=ordered[int(.95 * (len(ordered) - 1))], length_max=max(lengths))
    assert len(ids) == 4352
    for family in range(16):
        assert families[('train', 'math', family)] == 112
        assert families[('train', 'math_repair', family)] == 16
    for family in range(16, 20):
        assert families[('validation', 'math', family)] == 32
    report = dict(state='passed', statistics=stats, checks=[
        'file hashes and counts', 'unique IDs and prompt hashes across splits',
        'whole math-family holdouts and exact quotas', 'prompt and student drafts masked',
        'reasoning/final targets supervised, single closing delimiter and EOS',
        'sequence lengths within limit; no truncation'],
        limitation='Exact answers/listed math facts and task-specific final validators are checked; arbitrary teacher prose is not formally verified. This audit does not establish semantic independence or model improvement.',
        training_started=False)
    (DATA / 'completion-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
