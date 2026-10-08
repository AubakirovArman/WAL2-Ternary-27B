"""Official EvalPlus test oracles, with seccomp applied inside each test child."""
import ctypes,importlib,os,pickle,resource,tempfile
from contextlib import contextmanager
from pathlib import Path
import evalplus.eval as ev
from evalplus.data import get_human_eval_plus,get_mbpp_plus
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec
ROOT=Path(__file__).resolve().parents[1]
_ORIGINAL=ev.reliability_guard

def guarded(maximum_memory_bytes=None):
    # Preload dependencies before denying file access; applies only to forked test child.
    for name in ['math','cmath','collections','itertools','functools','operator','typing','random','copy','string','re','hashlib','statistics','fractions','decimal','bisect','heapq','array','datetime','json','base64','unicodedata','struct','socket','subprocess','numpy','numpy.linalg','numpy.fft']:
        importlib.import_module(name)
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p]
    ctx=lib.seccomp_init(0x00050000|1)
    if not ctx:raise RuntimeError('seccomp init failed')
    allow=['read','write','close','fstat','lseek','mmap','mprotect','munmap','mremap','brk','madvise','rt_sigaction','rt_sigprocmask','rt_sigreturn','sigaltstack','getpid','gettid','futex','sched_yield','clock_gettime','gettimeofday','getrusage','getrandom','exit','exit_group','setitimer','getitimer']
    for name in allow:
        nr=lib.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and lib.seccomp_rule_add(ctx,0x7fff0000,nr,0)!=0:raise RuntimeError('seccomp rule failed')
    _ORIGINAL(maximum_memory_bytes)
    resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
    if lib.seccomp_load(ctx)!=0:raise RuntimeError('seccomp load failed')

TEST_DIR=None
@contextmanager
def child_tempdir():
    os.chdir(TEST_DIR)
    yield TEST_DIR

def setup():
    ev.reliability_guard=guarded
    ev.create_tempdir=child_tempdir

def safe_check(*args,**kwargs):
    global TEST_DIR
    with tempfile.TemporaryDirectory(prefix='vol2-evalplus-') as tmp:
        TEST_DIR=tmp
        return ev.untrusted_check(*args,**kwargs)

def evaluate(family,problem,code):
    dataset='humaneval' if family=='humaneval_plus' else 'mbpp'
    # Official references only. Persist timings/expected values locally; never unpickle remote data.
    cache=ROOT/'data/release-benchmarks-v1/oracles';cache.mkdir(exist_ok=True)
    p=cache/(problem['task_id'].replace('/','-')+'.pkl')
    if p.exists():oracle=pickle.loads(p.read_bytes())
    else:
        oracle={}
        for kind in ('base','plus'):
            oracle[kind],oracle[kind+'_time']=trusted_exec(problem['prompt']+problem['canonical_solution'],problem[kind+'_input'],problem['entry_point'],record_time=True,output_not_none=dataset=='mbpp' and problem['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS)
        tmp=p.with_suffix('.tmp-'+str(os.getpid()));tmp.write_bytes(pickle.dumps(oracle));tmp.replace(p)
    setup();result={}
    for kind in ('base','plus'):
        status,details=safe_check(dataset,code,problem[kind+'_input'],problem['entry_point'],oracle[kind],problem['atol'],oracle[kind+'_time'],fast_check=True,min_time_limit=0.2,gt_time_limit_factor=4)
        result[kind]=status
        if status!='pass':return {'correct':False,**result}
    return {'correct':True,**result}

if __name__=='__main__':
    setup()
    def check(code):return safe_check('humaneval',code,[[3]],'f',[4],0,[.001],fast_check=True)[0]
    assert check('def f(x): return x+1')=='pass'
    assert check('def f(x): return x')!='pass'
    assert check("def f(x):\n open('/etc/passwd').read()\n return x+1")!='pass'
    assert check("def f(x):\n import socket\n socket.socket()\n return x+1")!='pass'
    p=get_human_eval_plus()['HumanEval/0'];assert evaluate('humaneval_plus',p,p['prompt']+p['canonical_solution'])['correct']
    p=get_mbpp_plus()['Mbpp/2'];assert evaluate('mbpp_plus',p,p['prompt']+p['canonical_solution'])['correct']
    print('PASS: official plus references and correct/incorrect solutions; file/network blocked')
