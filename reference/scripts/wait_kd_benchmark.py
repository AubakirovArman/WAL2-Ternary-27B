"""Run independent smoke comparisons serially after successful E006 termination."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/broad-smoke-e006'


def record(**values):
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'time': time.time(), **values}) + '\n')
    print(json.dumps(values), flush=True)


def main():
    OUT.mkdir(exist_ok=False)
    deadline = time.monotonic() + 7 * 3600
    record(event='waiting', service='vol2-e006-soft-kd.service')
    while True:
        raw = subprocess.check_output([
            'systemctl', '--user', 'show', 'vol2-e006-soft-kd.service',
            '-p', 'LoadState', '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
        fields = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        if fields.get('LoadState') == 'not-found' and fields.get('ActiveState') == 'inactive':
            # systemd garbage-collects successful transient units. Missing handles are
            # terminal, but only the durable completed artifacts below establish success.
            break
        if fields.get('LoadState') != 'loaded':
            raise RuntimeError('Training service missing: ' + raw)
        if fields.get('ActiveState') in ('inactive', 'failed'):
            if fields.get('Result') != 'success' or fields.get('ExecMainStatus') != '0':
                raise RuntimeError('Training terminated unsuccessfully: ' + raw)
            break
        if fields.get('ActiveState') not in ('active', 'activating', 'deactivating'):
            raise RuntimeError('Unknown training state: ' + raw)
        if time.monotonic() >= deadline:
            raise TimeoutError('Training still live after seven hours; no benchmark launched')
        time.sleep(30)
    metrics = json.loads((ROOT / 'runs/e006-soft-kd/metrics.json').read_text())
    if metrics.get('state') != 'completed':
        raise RuntimeError('Training exited without completed metrics')
    checkpoint = Path(metrics['selected_packed'])
    if not checkpoint.is_dir():
        raise RuntimeError('Missing selected packed checkpoint')
    packed = json.loads((checkpoint / 'manifest.json').read_text())
    if len(packed['matrices']) != 402 or any(not (checkpoint / row['file']).is_file() for row in packed['matrices'].values()):
        raise RuntimeError('Incomplete selected packed model')
    (OUT / 'training-selection.json').write_text(json.dumps(metrics, indent=2))
    # Each child independently takes the GPU6/7 lock and checks free memory.
    # A failed guard or correctness check stops the chain, without retries.
    for model in ('bonsai2', 'source', 'student'):
        folder = OUT / model
        command = [str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'scripts/run_broad_benchmark.py'),
                   '--model', model, '--subset', 'smoke', '--output', str(folder)]
        if model == 'student':
            command += ['--checkpoint', str(checkpoint)]
        record(event='generation_started', model=model)
        subprocess.run(command, check=True, timeout=4 * 3600, cwd=ROOT)
        subprocess.run([str(ROOT / '.venv-eval/bin/python'), str(ROOT / 'scripts/score_broad_benchmark.py'),
                        str(folder)], check=True, timeout=1800, cwd=ROOT)
        record(event='scored', model=model, scores=json.loads((folder / 'summary.json').read_text()))
    record(event='completed')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            record(event='failed', error=repr(error))
        raise
