"""CPU-only HumanEval test runner with syscall restrictions and bounded output/resources."""
import json,subprocess,tempfile
from pathlib import Path
CHILD=Path(__file__).with_name('code_sandbox_child.py')

def run(code,timeout=8):
    with tempfile.TemporaryDirectory(prefix='vol2-code-') as d:
        with open(Path(d)/'stdout','w+b') as out,open(Path(d)/'stderr','w+b') as err:
            try:
                result=subprocess.run(['/usr/bin/python3','-I',str(CHILD)],input=json.dumps({'code':code}).encode(),
                    stdout=out,stderr=err,cwd=d,env={'PATH':'/usr/bin:/bin','LC_ALL':'C.UTF-8'},
                    timeout=timeout,close_fds=True)
            except subprocess.TimeoutExpired:return {'passed':False,'error':'timeout'}
            out.seek(0);text=out.read(65536).decode(errors='replace');err.seek(0);stderr=err.read(4096).decode(errors='replace')
            if result.returncode:return {'passed':False,'error':'exit '+str(result.returncode),'stderr':stderr}
            marker='VOL2_SANDBOX_RESULT '
            if marker not in text:return {'passed':False,'error':'missing test result'}
            try:return json.loads(text.rsplit(marker,1)[1].strip())
            except ValueError:return {'passed':False,'error':'invalid test result'}

def check():
    assert run('assert sum([1,2,3]) == 6')['passed']
    assert not run('assert False')['passed']
    probes=["open('/etc/passwd').read()",'__import__("socket").socket()',
            '__import__("os").fork()', '__import__("subprocess").run(["/bin/true"])']
    for expression in probes:
        code=f'try:\n    {expression}\nexcept PermissionError:\n    pass\nelse:\n    raise AssertionError("restricted operation succeeded")'
        assert run(code)['passed'],expression
    assert not run('while True: pass')['passed']
    assert not run('print("X"*100000)')['passed']
    print('PASS: success/failure scoring; file/network/spawn denied; CPU and output limits enforced')

if __name__=='__main__':check()
