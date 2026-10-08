"""Compare retained E0193584/4096 independently, one native replica per GPU."""
import fcntl
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    lock=(ROOT/'runs/gpu67.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8')
    workers=[]
    try:
        for step,gpu in ((3584,6),(4096,7)):
            folder=ROOT/f'reports/e019-step{step}-math68'
            log=(ROOT/f'logs/e019-step{step}-math68.log').open('w')
            proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/run_e019_math_check.py'),
                                   '--folder',str(folder),'--checkpoint-step',str(step),'--gpu',str(gpu),
                                   '--gpu-lock-fd',str(lock.fileno())],cwd=ROOT,env=env,
                                  pass_fds=(lock.fileno(),),stdout=log,stderr=log)
            workers.append((step,proc,log))
        failed=[]
        for step,proc,log in workers:
            code=proc.wait();log.close()
            if code:failed.append(dict(step=step,exit_code=code))
        if failed:raise RuntimeError(json.dumps(failed))
        print('Обе оставшиеся точки проверены: 3584 и 4096.',flush=True)
    finally:
        for step,proc,log in workers:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=40)
            if not log.closed:log.close()


if __name__=='__main__':main()
