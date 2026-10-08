"""E019-D: fixed-model runtime parity, then matched math diagnostics. No training."""
import argparse
import asyncio
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
from transformers import AutoTokenizer

from benchmark_tasks import final_text
from run_release_benchmarks import BACK, BIN, ROOT, SOURCE, env_for, write
from ternary_core import GPU_IDS

OUT = ROOT / 'reports/e019-d'
PROCS = []
LOGS = []


def status(state, stage, **kw):
    value = dict(state=state, stage=stage, updated=time.time(), optimizer_updates=0, **kw)
    write(OUT / 'status.json', value)
    print(time.strftime('%H:%M:%S')+' | '+stage, flush=True)


def run_torch():
    env = env_for(GPU_IDS[0])
    env['PYTHONPATH'] = str(ROOT / 'tools/fla-probe-packages')
    with (OUT / 'torch.log').open('w') as log:
        subprocess.run([str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'scripts/e019_d_torch.py')], env=env, stdout=log, stderr=log, check=True, timeout=7200)


def run_native():
    cfg = json.loads((OUT / 'config.json').read_text())
    cases = json.loads((OUT / 'parity-cases.json').read_text())
    gens = json.loads((OUT / 'parity-generation.json').read_text())
    lines = []
    for case in cases:
        ids = case['input_ids']
        base = f"{case['id']} {len(ids)} {len(ids)} " + ' '.join(map(str, ids))
        lines.append('L '+base)
        if case['kind'] == 'gold':
            lines.append(f"E {case['id']} {case['prompt_tokens']} {len(ids)} "+' '.join(map(str, ids)))
    for case in gens:
        ids = case['input_ids']
        lines.append(f"G {case['id']} {len(ids)} {len(ids)} "+' '.join(map(str, ids)))
    inp = OUT / 'native-inputs.txt'
    inp.write_text('\n'.join(lines)+'\n')
    logits, ce, generated = {}, [], []
    with (OUT / 'native.log').open('w') as log, (OUT / 'native.jsonl').open('w') as output:
        proc = subprocess.Popen([str(BIN), cfg['models']['e018'], str(inp), str(BACK), '128', '32768', '512'], env=env_for(GPU_IDS[1]), stdout=subprocess.PIPE, stderr=log, text=True)
        try:
            for count, line in enumerate(proc.stdout, 1):
                row = json.loads(line)
                output.write(line)
                output.flush()
                if row['kind'] == 'logits':
                    logits[row['id']] = np.asarray(row['logits'], dtype=np.float32)
                elif row['kind'] == 'ce':
                    ce.append(row)
                elif row['kind'] == 'generation':
                    generated.append(row)
                write(OUT / 'native-status.json', dict(state='running', completed=count, total=len(lines)))
            if proc.wait() != 0:
                raise RuntimeError('Native parity failed; see native.log')
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
    assert set(logits) == {c['id'] for c in cases} and len(ce) == 12 and len(generated) == 4
    np.save(OUT / 'native-logits.npy', np.stack([logits[c['id']] for c in cases]))
    write(OUT / 'native-ce.json', ce)
    write(OUT / 'native-generations.json', generated)
    write(OUT / 'native-status.json', dict(state='completed', completed=len(lines), total=len(lines)))


def logsoftmax(x):
    x = x.astype(np.float64)
    x = x-x.max()
    return x-math.log(np.exp(x).sum())


def compare_logits(x, y):
    lp, lq = logsoftmax(x), logsoftmax(y)
    top = np.argsort(x)[-2:][::-1]
    top_y = np.argsort(y)[-2:][::-1]
    return dict(kl=float((np.exp(lp)*(lp-lq)).sum()), top1_equal=int(top[0]) == int(top_y[0]), reference_top1=int(top[0]), other_top1=int(top_y[0]), reference_margin=float(x[top[0]]-x[top[1]]), other_margin=float(y[top_y[0]]-y[top_y[1]]), max_abs_logit=float(np.abs(x-y).max()), rms_logit=float(np.sqrt(np.square(x-y).mean())))


def generation_difference(a, b):
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), n if len(a) != len(b) else None)
    return dict(equal=a == b, first_divergence_token=first, lengths=[len(a), len(b)])


def summarize_parity():
    cases = json.loads((OUT / 'parity-cases.json').read_text())
    arrays = {name: np.load(OUT / (name+'-logits.npy')) for name in ('reference', 'fla', 'native')}
    rows = []
    for i, case in enumerate(cases):
        rows.append(dict(id=case['id'], kind=case['kind'], sequence_tokens=len(case['input_ids']), reference_vs_fla=compare_logits(arrays['reference'][i], arrays['fla'][i]), reference_vs_native=compare_logits(arrays['reference'][i], arrays['native'][i])))
    ce = {}
    byid = {}
    for name in arrays:
        scored = json.loads((OUT / (name+'-ce.json')).read_text())
        scored = [r for r in scored if r.get('target_tokens', r.get('tokens', 0))]
        byid[name] = {r['id']: r for r in scored}
        n = sum(r.get('target_tokens', r.get('tokens', 0)) for r in scored)
        ce[name] = dict(ce=sum(r['nll'] for r in scored)/n, tokens=n, examples=len(scored))
    assert len({v['tokens'] for v in ce.values()}) == 1
    for r in rows:
        if r['kind'] == 'gold':
            r['ce'] = {name: byid[name][r['id']]['nll']/byid[name][r['id']].get('target_tokens', byid[name][r['id']].get('tokens')) for name in arrays}
    reference = {r['id']: r['token_ids'] for r in json.loads((OUT / 'reference-generations.json').read_text())}
    gens = {}
    for label in ('fla', 'native'):
        other = json.loads((OUT / (label+'-generations.json')).read_text())
        gens[label] = [dict(id=r['id'], **generation_difference(reference[r['id']], r.get('token_ids', r.get('tokens')))) for r in other]
    summary = dict(ce=ce, endpoint_checks=rows, generation_checks=gens, scope='E018 step25000:12 gold rows,20 endpoint distributions incl long student states,4 bounded continuations. Divergence at small top2 margins is not automatically an export bug; systematic drift requires investigation. No gradient/backward comparison here.')
    write(OUT / 'parity-summary.json', summary)
    print('CE12:', {k: v['ce'] for k, v in ce.items()}, flush=True)


def start_server(model, gpu, port, slots):
    log = (OUT / f'server-{port}.log').open('w')
    LOGS.append(log)
    cmd = [str(BACK / 'llama-server'), '-m', str(model), '--host', '127.0.0.1', '--port', str(port), '-ngl', '99', '-np', str(slots), '-c', str(slots*32768), '-b', '2048', '-ub', '512', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16', '-t', '8', '-tb', '8', '--fit', 'off', '--cache-ram', '0', '--no-context-shift', '--threads-http', '16', '--special']
    proc = subprocess.Popen(cmd, env=env_for(gpu), stdout=log, stderr=log)
    PROCS.append(proc)
    return proc


async def healthy(client, port, proc):
    t = time.monotonic()
    while time.monotonic()-t < 600:
        if proc.poll() is not None:
            raise RuntimeError(f'Server{port} exited; see server log')
        try:
            if (await client.get(f'http://127.0.0.1:{port}/health')).status_code == 200:
                return
        except httpx.ConnectError:
            pass
        await asyncio.sleep(2)
    raise TimeoutError('Server startup exceeded10min')


async def completion(client, port, ids, budget, **kw):
    payload = dict(prompt=ids, n_predict=budget, temperature=0.0, samplers=['temperature'], repeat_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, seed=20260927, cache_prompt=False, return_tokens=True, stream=False, **kw)
    response = await client.post(f'http://127.0.0.1:{port}/completion', json=payload)
    response.raise_for_status()
    result = response.json()
    if result.get('error') or result.get('truncated'):
        raise RuntimeError(str(result.get('error', 'Context truncation')))
    return result


def stop_servers():
    for proc in PROCS:
        if proc.poll() is None:
            proc.terminate()
    for proc in PROCS:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    PROCS.clear()
    for log in LOGS:
        log.close()
    LOGS.clear()


async def batched_parity():
    cfg = json.loads((OUT / 'config.json').read_text())
    cases = json.loads((OUT / 'parity-cases.json').read_text())
    gens = json.loads((OUT / 'parity-generation.json').read_text())
    proc = start_server(cfg['models']['e018'], GPU_IDS[0], 18796, 8)
    reference = np.load(OUT / 'reference-logits.npy')
    native = np.load(OUT / 'native-logits.npy')
    output = []
    try:
        async with httpx.AsyncClient(timeout=1800, trust_env=False) as client:
            await healthy(client, 18796, proc)
            # Explicit slot assignment; reverse order changes each prefix's slot.
            for mode, ordered in [('single', cases), ('batch_forward', cases), ('batch_reverse', list(reversed(cases)))]:
                size = 1 if mode == 'single' else 8
                for start in range(0, len(ordered), size):
                    batch = ordered[start:start+size]
                    raw = await asyncio.gather(*(completion(client, 18796, c['input_ids'], 1, n_probs=32, post_sampling_probs=False, id_slot=i) for i, c in enumerate(batch)))
                    for c, result in zip(batch, raw):
                        probs = result['completion_probabilities']
                        assert len(probs) == 1, (c['id'], probs)
                        top = probs[0]['top_logprobs']
                        ids = np.array([p['id'] for p in top], dtype=np.int64)
                        q = np.exp(np.array([p['logprob'] for p in top], dtype=np.float64))
                        values = {}
                        for label, x in [('reference', reference[c['id']]), ('native', native[c['id']])]:
                            p = np.exp(logsoftmax(x))[ids]
                            pb = np.append(p, max(1-p.sum(), 0))
                            qb = np.append(q, max(1-q.sum(), 1e-30))
                            values[label] = dict(top32_tail_kl=float(np.sum(pb*(np.log(np.maximum(pb, 1e-30))-np.log(np.maximum(qb, 1e-30))))), top1_equal=int(x.argmax()) == top[0]['id'])
                        output.append(dict(id=c['id'], mode=mode, checks=values, tokens=result['tokens'], top32=top))
                    write(OUT / 'server-prefix-checks.json', output)
                    status('running', f'Проверка пакетов и слотов: {mode}, {min(start+size,len(ordered))}/{len(ordered)}')
            rows = []
            for mode, ordered in [('single', gens), ('batch_forward', gens), ('batch_reverse', list(reversed(gens)))]:
                if mode == 'single':
                    raw = [await completion(client, 18796, c['input_ids'], 128, id_slot=0) for c in ordered]
                else:
                    raw = await asyncio.gather(*(completion(client, 18796, c['input_ids'], 128, id_slot=i) for i, c in enumerate(ordered)))
                rows.extend(dict(id=c['id'], mode=mode, tokens=r['tokens'], stop_type=r['stop_type']) for c, r in zip(ordered, raw))
                write(OUT / 'server-generations.json', rows)
            # Equality is observational, not required bit identity for long reasoning.
            single = {r['id']: r['tokens'] for r in rows if r['mode'] == 'single'}
            checks = [dict(id=r['id'], mode=r['mode'], **generation_difference(single[r['id']], r['tokens'])) for r in rows if r['mode'] != 'single']
            write(OUT / 'batch-summary.json', dict(generation_checks=checks, endpoint_checks=output, scope='Eight slots, forward/reverse assignment after dirty requests; f16 KV, no prompt caching; first-token top32+tail only, not full-vocabulary server KL.'))
    finally:
        stop_servers()


async def comparison():
    cfg = json.loads((OUT / 'config.json').read_text())
    jobs = json.loads((OUT / 'comparison-tasks.json').read_text())
    tok = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    servers = [start_server(cfg['models'][name], gpu, port, 12) for name, gpu, port in [('e018', GPU_IDS[0], 18796), ('bonsai', GPU_IDS[1], 18797)]]
    rows = []
    begin = time.monotonic()
    async with httpx.AsyncClient(timeout=1800, limits=httpx.Limits(max_connections=50), trust_env=False) as client:
        await asyncio.gather(*(healthy(client, port, proc) for port, proc in zip((18796,18797), servers)))
        # Verify complete rendered-template prompt token count on unchanged teacher API.
        probe = await client.post(cfg['models']['teacher_api']+'/completions', json=dict(model=cfg['models']['teacher'], prompt=jobs[0]['prompt'], temperature=0, max_tokens=1))
        probe.raise_for_status()
        write(OUT / 'teacher-template-probe.json', probe.json())
        assert probe.json()['usage']['prompt_tokens'] == len(jobs[0]['input_ids'])
        async def arm(name, port=None):
            semaphore = asyncio.Semaphore(12)
            async def job(task):
                async with semaphore:
                    start = time.monotonic()
                    if name == 'teacher':
                        response = await client.post(cfg['models']['teacher_api']+'/completions', json=dict(model=cfg['models']['teacher'], prompt=task['prompt'], temperature=0, top_p=1, top_k=-1, max_tokens=8192, repetition_penalty=1, presence_penalty=0, frequency_penalty=0))
                        response.raise_for_status()
                        raw = response.json()
                        assert raw['usage']['prompt_tokens'] == len(task['input_ids']), 'Teacher template/token count differs'
                        choice = raw['choices'][0]
                        text = choice['text']
                        finish = 'length' if choice['finish_reason'] == 'length' else 'eos'
                        tokens = raw['usage']['completion_tokens']
                    else:
                        raw = await completion(client, port, task['input_ids'], 8192)
                        text = tok.decode(raw['tokens'], skip_special_tokens=False)
                        finish = 'length' if raw['stop_type'] == 'limit' else 'eos'
                        tokens = len(raw['tokens'])
                    answer, closed = final_text(text, True)
                    row = dict(model=name, id=task['id'], family=task['family'], raw_response=text, answer=answer, thinking_completed=closed, finish_reason=finish, generated_tokens=tokens, seconds=time.monotonic()-start, runtime='unchanged teacher API' if name == 'teacher' else 'native PQ2 batched Prism', raw=raw)
                    with (OUT / 'comparison-responses.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(row)
                    elapsed = time.monotonic()-begin
                    status('running', f'Сравнение моделей: готово {len(rows)}/204', completed=len(rows), total=204, elapsed_seconds=elapsed, eta_seconds=(204-len(rows))*elapsed/len(rows))
                    # Scoring is incremental in a separate CPU service.
            await asyncio.gather(*(job(task) for task in jobs))
        try:
            await asyncio.gather(arm('e018', 18796), arm('bonsai', 18797), arm('teacher'))
        finally:
            stop_servers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parity-only', action='store_true')
    a = parser.parse_args()
    cfg = json.loads((OUT / 'config.json').read_text())
    assert cfg['state'] == 'prepared', 'Do not overwrite an existing run'
    for name, key in [('comparison-tasks.json','task_sha256'), ('parity-cases.json','parity_sha256')]:
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == cfg[key]
    lock = (ROOT / 'runs/gpu67.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for gpu in GPU_IDS:
        free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= 130*1024, f'{gpu} occupied'
    cfg['state'] = 'running'
    cfg['started'] = time.time()
    cfg['hashes'] = {key: hashlib.file_digest(Path(cfg['models'][key]).open('rb'), 'sha256').hexdigest() for key in ('e018','bonsai')}
    cfg['scripts_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'scripts').glob('*e019_d*') if p.is_file()}
    write(OUT / 'config.json', cfg)
    status('running', 'Сверка PyTorch/FLA на GPU6 и native PQ2 на GPU7')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_torch), pool.submit(run_native)]
        for f in futures:
            f.result()
    summarize_parity()
    status('running', 'Проверка состояния, слотов и пакетных запросов')
    asyncio.run(batched_parity())
    if not a.parity_only:
        status('running', 'Сравнение E018, Qwen и Bonsai на68 одинаковых задачах')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1')
        scorer = subprocess.Popen([str(ROOT / '.venv-release/bin/python'), '-u', str(ROOT / 'scripts/score_e019_d.py'), '--follow'], env=env)
        try:
            asyncio.run(comparison())
            if scorer.wait(timeout=600) != 0:
                raise RuntimeError('CPU scoring failed')
        finally:
            if scorer.poll() is None:
                scorer.terminate()
                scorer.wait(timeout=30)
    cfg['state'] = 'completed'
    write(OUT / 'config.json', cfg)
    status('completed', 'Диагностика завершена; GPU6/7 освобождены')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', 'Диагностика остановлена с ошибкой', error=str(error))
        raise
    finally:
        stop_servers()
