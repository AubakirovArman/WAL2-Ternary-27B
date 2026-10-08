"""Recalculate published numeric evidence, without generating model answers."""
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    return json.loads(path.read_text())


def summarize(rows):
    if len({str(r['id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate score IDs')
    if not all(type(r['correct']) is bool and type(r['truncated']) is bool for r in rows):
        raise ValueError('Scores must use boolean correctness/truncation')
    return dict(correct=sum(r['correct'] for r in rows), total=len(rows),
                truncated=sum(r['truncated'] for r in rows))


def by_family(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['family']].append(row)
    return {family: summarize(items) for family, items in grouped.items()}


def verify(root=ROOT):
    root = Path(root)
    diagnostics = json.loads((root/'evidence/diagnostic136/results.json').read_text())
    common = None
    for model in diagnostics:
        rows = read_rows(root/f"evidence/diagnostic136/{model['model']}.jsonl")
        assert summarize(rows) == {k: model[k] for k in ('correct','total','truncated')}
        assert model['total'] == 136
        assert by_family(rows) == model['families']
        ids = {str(r['id']) for r in rows}
        if common is not None:
            assert ids == common
        common = ids
    for name, expected in [('e017', (1117,1319,25)), ('bonsai2',(1275,1319,6))]:
        rows = read_rows(root/f'evidence/gsm8k1319/{name}-scores.json')
        s = summarize(rows)
        assert tuple(s[k] for k in ('correct','total','truncated')) == expected
        assert {str(r['id']) for r in rows} == {str(i) for i in range(1319)}
    broad = json.loads((root/'evidence/broad-partial/results.json').read_text())
    assert broad['state'] == 'incomplete' and broad['planned_total'] == 9049
    ids = None
    for name, expected in broad['models'].items():
        rows = read_rows(root/f'evidence/broad-partial/{name}-matched.jsonl')
        assert len(rows) == broad['matched_tasks'] == 1569
        assert by_family(rows) == expected
        own = {str(r['id']) for r in rows}
        if ids is not None:
            assert ids == own
        ids = own
    a = json.loads((root/'model/artifact.json').read_text())
    assert a['text_parameters'] == a['ternary_parameters'] + a['exceptions_parameters']
    assert a['matrices'] == 402 and a['packing'] == 'PQ2_0'
    assert a['physical_bits_per_ternary_weight'] == 34*8/128
    assert math.isclose(a['whole_file_bits_per_language_parameter'],
                        a['bytes']*8/a['text_parameters'], rel_tol=1e-12)
    assert math.isclose(a['base3_tensor_effective_bpw'],
                        a['base3_tensor_bytes']*8/a['text_parameters'], rel_tol=1e-12)
    assert math.isclose(a['base3_directory_effective_bpw'],
                        a['base3_directory_bytes']*8/a['text_parameters'], rel_tol=1e-12)
    return dict(passed=True, diagnostic_models=len(diagnostics), tasks_per_diagnostic=136,
                gsm8k_models=2, tasks_per_gsm8k=1319, matched_partial_tasks=1569,
                scope='Recalculated saved scores and storage arithmetic; no model generation')


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
