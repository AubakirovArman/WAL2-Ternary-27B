"""Prepare the complete v3 collection for E009 continuation; never launch training."""
import hashlib
import json
import time
from pathlib import Path
from wait_reasoning_qat import wait_terminal

ROOT = Path(__file__).resolve().parents[1]
OLD_VALIDATION_SHA = '592a90f098b128af2badad0db018906f4c25998a1c1e6bcc3acdc159ba9236c2'


def validate_collection(folder):
    meta = json.loads((folder / 'manifest.json').read_text())
    tasks = json.loads((folder / 'tasks.json').read_text())
    events = [json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()]
    accepted = [json.loads(s) for s in (folder / 'accepted.jsonl').read_text().splitlines()]
    ids = [r['id'] for r in tasks]
    if meta.get('state') != 'completed' or meta.get('attempted') != len(tasks):
        raise ValueError('Collection is incomplete')
    if len(set(ids)) != len(ids) or [r['id'] for r in events] != ids:
        raise ValueError('Collection events do not cover the exact schedule')
    if [r['id'] for r in accepted] != [r['id'] for r in events if r.get('accepted')]:
        raise ValueError('Accepted records disagree with events')


def main():
    wait_terminal('vol2-reasoning-diverse-v3.service', time.monotonic() + 9 * 3600)
    source = ROOT / 'data/reasoning-diverse-v3'
    validate_collection(source)
    old = ROOT / 'data/reasoning-mixed-v3-tokenized'
    if hashlib.sha256((old / 'validation.jsonl').read_bytes()).hexdigest() != OLD_VALIDATION_SHA:
        raise ValueError('Established validation changed')
    # Imports occur only after collection has completed. Neither preparation uses CUDA.
    from prepare_reasoning_diverse import prepare as tokenize
    from prepare_reasoning_mix import prepare as mix
    tokenized = ROOT / 'data/reasoning-diverse-v3-tokenized'
    output = ROOT / 'data/reasoning-fresh-v4-tokenized'
    if tokenized.exists() or output.exists():
        raise FileExistsError('Inspect existing outputs before rerunning preparation')
    tokenize(source, tokenized)
    result = mix(tokenized, [ROOT / 'data/reasoning-verified-v2-tokenized',
                            ROOT / 'data/reasoning-code-verified-v1-tokenized'],
                 output, seed=2026092003, validation_base=old)
    if result['files']['validation']['sha256'] != OLD_VALIDATION_SHA:
        raise RuntimeError('Prepared validation identity failed')
    print(json.dumps({'event': 'fresh_reasoning_data_ready', 'path': str(output),
                      'files': result['files'], 'training_started': False}), flush=True)


if __name__ == '__main__':
    main()
