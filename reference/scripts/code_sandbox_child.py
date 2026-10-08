"""Linux-only isolated code-test child: no file opening, network, spawning or signals to others."""
import sys,json,resource,ctypes,os,importlib

def main():
    payload=json.loads(sys.stdin.buffer.read());sys.stdin.close()
    # Load computational stdlib dependencies before filesystem access is disabled.
    for name in ['math','cmath','collections','itertools','functools','operator','typing','random','copy',
                 'string','re','hashlib','statistics','fractions','decimal','bisect','heapq','array','datetime',
                 'json','base64','unicodedata','struct','socket','subprocess']:
        importlib.import_module(name)
    import hashlib,random
    hashlib.md5(b'warmup').hexdigest();random.seed(0)
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    library=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    library.seccomp_init.argtypes=[ctypes.c_uint32];library.seccomp_init.restype=ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];library.seccomp_syscall_resolve_name.restype=ctypes.c_int
    library.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    library.seccomp_load.argtypes=[ctypes.c_void_p];library.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=library.seccomp_init(0x00050000|1) # Default EPERM.
    if not ctx:raise RuntimeError('seccomp_init failed')
    allow=['read','write','close','fstat','lseek','mmap','mprotect','munmap','mremap','brk','madvise',
           'rt_sigaction','rt_sigprocmask','rt_sigreturn','sigaltstack','getpid','gettid','futex',
           'sched_yield','clock_gettime','gettimeofday','getrusage','getrandom','exit','exit_group']
    for name in allow:
        nr=library.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and library.seccomp_rule_add(ctx,0x7fff0000,nr,0)!=0:raise RuntimeError('seccomp_rule_add failed')
    if library.seccomp_load(ctx)!=0:raise RuntimeError('seccomp_load failed')
    library.seccomp_release(ctx)
    try:
        scope={'__name__':'__code_test__'}
        exec(compile(payload['code'],'<model-code>','exec'),scope)
        result={'passed':True}
    except BaseException as e:result={'passed':False,'error':type(e).__name__+': '+str(e)[:300]}
    print('\nVOL2_SANDBOX_RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
