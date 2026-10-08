"""Matched native PQ2 evaluation of the selected E019 on the 68 E019-D tasks."""
import asyncio
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer
import run_e019_d as native
from benchmark_tasks import final_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/e019-best-math68'
BASE = ROOT / 'reports/e019-d'
RUN = ROOT / 'runs/e019-reasoning-recovery'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(state, stage, **kw):
    native.write(OUT / 'status.json', dict(state=state, stage=stage, updated=time.time(), **kw))
    print(time.strftime('%H:%M:%S') + ' | ' + stage, flush=True)


def scorer(follow=False):
    return [str(ROOT / '.venv-release/bin/python'), str(ROOT / 'scripts/score_e019_d.py'),
            '--folder', str(OUT), '--expected-total', '272', '--subject', 'e019'] + (['--follow'] if follow else [])


async def evaluate(model, jobs, gpus=(6,7)):
    native.OUT = OUT
    tokenizer = AutoTokenizer.from_pretrained(native.SOURCE, local_files_only=True)
    ports = tuple(18792+gpu for gpu in gpus)
    servers = [native.start_server(model, native.GPU_IDS[gpu-6], port, 12) for gpu, port in zip(gpus, ports)]
    start = time.monotonic()
    completed = 0
    async with httpx.AsyncClient(timeout=1800, limits=httpx.Limits(max_connections=40), trust_env=False) as client:
        try:
            await asyncio.gather(*(native.healthy(client, port, proc) for port, proc in zip(ports, servers)))
            semaphores = [asyncio.Semaphore(12) for _ in gpus]
            async def job(task, replica):
                nonlocal completed
                async with semaphores[replica]:
                    begin = time.monotonic()
                    raw = await native.completion(client, ports[replica], task['input_ids'], 8192)
                    text = tokenizer.decode(raw['tokens'], skip_special_tokens=False)
                    answer, closed = final_text(text, True)
                    row = dict(model='e019', id=task['id'], family=task['family'], raw_response=text,
                               answer=answer, thinking_completed=closed,
                               finish_reason='length' if raw['stop_type'] == 'limit' else 'eos',
                               generated_tokens=len(raw['tokens']), seconds=time.monotonic()-begin,
                               runtime='native PQ2 batched Prism', physical_gpu=gpus[replica], raw=raw)
                    with (OUT / 'comparison-responses.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
                    completed += 1
                    elapsed = time.monotonic()-start
                    status('running', f'Проверено ответов E019: {completed}/68', completed=completed, total=68,
                           elapsed_seconds=elapsed, eta_seconds=(68-completed)*elapsed/completed)
            await asyncio.gather(*(job(task, i % len(gpus)) for i, task in enumerate(jobs)))
        finally:
            native.stop_servers()


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--checkpoint-step',type=int,choices=(2560,3584,4096))
    parser.add_argument('--gpu',type=int,choices=(6,7))
    parser.add_argument('--gpu-lock-fd',type=int)
    args=parser.parse_args()
    if args.folder is not None:OUT=args.folder.resolve()
    gpus=(args.gpu,) if args.gpu is not None else (6,7)
    OUT.mkdir(exist_ok=False)
    status('waiting', 'Ожидание завершения проверки экспортированной E019')
    start = time.monotonic()
    while True:
        metrics = json.loads((RUN / 'metrics.json').read_text())
        state = subprocess.check_output(['systemctl', '--user', 'show', 'vol2-e019-training.service', '-p', 'ActiveState', '--value'], text=True).strip()
        if metrics['state'] == 'completed' and state not in ('active', 'activating', 'deactivating'):
            break
        if state not in ('active', 'activating', 'deactivating'):
            raise RuntimeError('Trainer stopped before verified export completion')
        if time.monotonic()-start > 1200:
            raise TimeoutError('Final export verification exceeded 20 minutes; trainer left unchanged')
        time.sleep(5)
    latent = Path(metrics['selected_latent']).resolve()
    packed = Path(metrics['selected_packed']).resolve()
    checkpoint = json.loads((latent / 'metadata.json').read_text())
    if args.checkpoint_step is not None:
        latent=(RUN/'candidates'/f'step-{args.checkpoint_step}').resolve()
        checkpoint=json.loads((latent/'metadata.json').read_text())
        assert checkpoint['step']==args.checkpoint_step and checkpoint['eligible']
        packed=OUT/'packed'
        status('exporting','CPU-сборка сохранённой точки',selected_step=checkpoint['step'])
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/export_latent_cpu.py'),str(latent),
                        str(Path(metrics['selected_packed']).resolve()),str(packed)],
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8'),check=True)
    jobs = json.loads((BASE / 'comparison-tasks.json').read_text())
    config = json.loads((BASE / 'config.json').read_text())
    assert len(jobs) == 68 and digest(BASE / 'comparison-tasks.json') == config['task_sha256']
    responses = [json.loads(line) for line in (BASE / 'comparison-responses.jsonl').open()]
    scores = [json.loads(line) for line in (BASE / 'comparison-scores.jsonl').open()]
    expected = {(name, task['id']) for name in ('e018', 'teacher', 'bonsai') for task in jobs}
    assert len(responses) == len(scores) == 204
    assert {(r['model'], r['id']) for r in responses} == {(r['model'], r['id']) for r in scores} == expected
    for filename in ('comparison-tasks.json', 'comparison-responses.jsonl', 'comparison-scores.jsonl'):
        shutil.copy2(BASE / filename, OUT / filename)
    native.write(OUT / 'training-snapshot.json', metrics)
    protocol = dict(model='E019', selected_step=checkpoint['step'], latent=str(latent), packed=str(packed),
                    packed_manifest_sha256=digest(packed / 'manifest.json'), task_sha256=config['task_sha256'],
                    baseline_scores_sha256=digest(BASE / 'comparison-scores.jsonl'),
                    decoding=config['decoding'], max_new_tokens=8192, context=32768,
                    replicas=len(gpus), slots_per_replica=12, physical_gpus=list(gpus),
                    benchmark_scope='Same 30 exposed AIME25 development tasks, 30 silver olympiad tasks, 8 exact probes; not a new unseen benchmark. Prior E018/Bonsai/Qwen results reused unchanged.',
                    scripts_sha256={name:digest(ROOT / 'scripts' / name) for name in ('run_e019_math_check.py','run_e019_d.py','score_e019_d.py','export_prism_pq2.py','export_latent_cpu.py')})
    native.write(OUT / 'protocol.json', protocol)
    model = OUT / 'e019-best.gguf'
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8')
    status('exporting', 'Сборка лучшей E019 в native PQ2', selected_step=checkpoint['step'])
    subprocess.run([str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'scripts/export_prism_pq2.py'),
                    str(packed), str(model), '--name', f'Vol2 E019 best step{checkpoint["step"]}'], env=cpu_env, check=True)
    lock_path=ROOT/'runs/gpu67.lock'
    if args.gpu_lock_fd is None:
        lock = lock_path.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        inherited=os.fstat(args.gpu_lock_fd);expected=lock_path.stat()
        assert (inherited.st_dev,inherited.st_ino)==(expected.st_dev,expected.st_ino)
        fcntl.flock(args.gpu_lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for index in gpus:
        gpu=native.GPU_IDS[index-6]
        free = int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free < 100 * 1024:
            raise RuntimeError('GPU6/7 occupied by another job; existing process left unchanged')
    status('running', 'Запуск параллельной проверки E019', completed=0, total=68, selected_step=checkpoint['step'])
    with (OUT / 'scoring.log').open('w') as log:
        proc = subprocess.Popen(scorer(True), env=cpu_env, stdout=log, stderr=log)
        try:
            asyncio.run(evaluate(model, jobs, gpus))
            status('completed', 'Все 68 ответов получены; итоговый подсчёт', completed=68, total=68)
            proc.wait(timeout=90)
            if proc.returncode != 0:raise RuntimeError('Incremental math scorer failed')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    subprocess.run(scorer(), env=cpu_env, check=True)
    result = json.loads((OUT / 'comparison-summary.json').read_text())
    assert sum(group['total'] for group in result['e019'].values()) == 68
    assert all(r.get('verification_error') is None for r in map(json.loads, (OUT / 'comparison-scores.jsonl').open()))
    status('completed', 'Проверка и подсчёт завершены', completed=68, total=68, selected_step=checkpoint['step'])


if __name__ == '__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():status('failed', 'Ошибка проверки', error=str(exc))
        raise
