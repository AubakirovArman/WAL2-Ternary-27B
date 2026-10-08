"""Resume the bounded E020 pilot from a committed validation checkpoint."""
import hashlib
import json
import shutil
import time
from pathlib import Path


def read_resume(output, step, args):
    if not args.ab_branch or args.preflight_only or not args.defer_export:
        raise ValueError('Resume is restricted to the preserved E020 A/B pilot')
    if not 0 < step < args.steps or step % args.validation_every:
        raise ValueError('Resume must use an earlier validation step')
    output = Path(output)
    config = json.loads((output / 'config.json').read_text())
    result = json.loads((output / 'metrics.json').read_text())
    if result['state'] != 'training':
        raise ValueError('Only interrupted training can be resumed')
    checkpoint = output / 'candidates' / f'step-{step:04d}'
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    checks = result['checks']
    if not checks or checks[-1] != metadata or metadata['step'] != step:
        raise ValueError('Resume checkpoint must be the last committed validation')
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    for filename in set(manifest.values()) | {'training_state.pt'}:
        path = checkpoint / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Incomplete resume checkpoint: ' + str(path))
    rows = [json.loads(line) for line in (output / 'training.jsonl').read_text().splitlines()]
    if [row['step'] for row in rows] != list(range(1, len(rows) + 1)) or len(rows) < step:
        raise ValueError('Interrupted log must contain contiguous steps including the checkpoint')
    expected_order = json.loads((output / 'order.json').read_text())
    return dict(config=config, result=result, checkpoint=checkpoint, metadata=metadata,
                rows=rows, order=expected_order, last_logged_step=rows[-1]['step'])


def verify_resume_config(resume, config, order):
    # Reusing a run is only valid with the original objective and provenance.
    for key, value in config.items():
        if resume['config'].get(key) != value:
            raise ValueError('Resume configuration changed: ' + key)
    if order != resume['order']:
        raise ValueError('Resume training order changed')


def commit_resume(output, resume, step):
    """Archive superseded logs; leave all checkpoint tensors untouched."""
    archive = output / ('resume-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    archive.mkdir(exist_ok=False)
    for name in ('training.jsonl', 'metrics.json', 'config.json', 'code-manifest.json'):
        shutil.copy2(output / name, archive / name)
    for name in ('train_reasoning_qat.py', 'run_e020_ab.py', 'e020_resume.py'):
        shutil.copy2(Path(__file__).parent / name, archive / name)
    lines = (output / 'training.jsonl').read_text().splitlines(keepends=True)
    tmp = output / 'training.jsonl.tmp'
    tmp.write_text(''.join(lines[:step]))
    tmp.replace(output / 'training.jsonl')
    record = dict(checkpoint=str(resume['checkpoint']), restored_step=step,
                  superseded_last_logged_step=resume['last_logged_step'],
                  checkpoint_state_sha256=hashlib.sha256((resume['checkpoint'] / 'training_state.pt').read_bytes()).hexdigest(),
                  resumed_at=time.time(), archive=str(archive),
                  policy='Restore latent weights, Adafactor, CPU/CUDA/Python RNG; continue original fixed order')
    (archive / 'resume.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return record
