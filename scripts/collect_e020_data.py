"""Natural teacher solutions, separate reasoning audit; append-only resumable commits."""
import argparse,ast,collections,concurrent.futures as cf,fcntl,hashlib,json,math,os,random,re,subprocess,time,urllib.request
from fractions import Fraction
from pathlib import Path
from e020_math_tasks import make_math
from e018_tasks import verify_task,repeat_reason
from collect_e019_pilot import verified_function
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/e020-natural-v1'
CONTROL=('<think>','</think>','<|im_start|>','<|im_end|>','<|endoftext|>')
def write(name,obj):
    path=OUT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def append(name,obj):
    with (OUT/name).open('a') as stream:stream.write(json.dumps(obj,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
def load(name):
    p=OUT/name
    if not p.exists():return []
    raw=p.read_bytes()
    if raw and not raw.endswith(b'\n'):
        end=raw.rfind(b'\n')+1;(OUT/(name+'.partial-'+str(time.time_ns()))).write_bytes(raw[end:]);p.write_bytes(raw[:end]);raw=raw[:end]
    return [json.loads(line) for line in raw.splitlines()]
def api(prompt,thinking=True):
    body=dict(model='Qwen/Qwen3.8-27B-FP8',messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=8192 if thinking else 4096,
              reasoning_effort='medium',chat_template_kwargs={'enable_thinking':thinking})
    for attempt in range(3):
        try:
            req=urllib.request.Request(os.environ.get('VOL2_TEACHER_URL','http://127.0.0.1:8000/v1/chat/completions'),data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=600) as stream:return json.load(stream)
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)
def exact(expr):
    """Bounded AST calculator, never eval teacher strings or symbolic Python."""
    if len(expr)>500:raise ValueError('Expression too long')
    def run(node,depth=0):
        if depth>20:raise ValueError('Depth')
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)) and not isinstance(node.value,bool):return Fraction(str(node.value))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            v=run(node.operand,depth+1);return v if isinstance(node.op,ast.UAdd) else -v
        if isinstance(node,ast.BinOp):
            a,b=run(node.left,depth+1),run(node.right,depth+1)
            if max(len(str(a)),len(str(b)))>3000:raise ValueError('Magnitude')
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.FloorDiv):return Fraction(a//b)
            if isinstance(node.op,ast.Mod):return a%b
            if isinstance(node.op,ast.Pow) and b.denominator==1 and abs(b)<=100:return a**int(b)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            args=[run(v,depth+1) for v in node.args]
            if node.func.id=='abs' and len(args)==1:return abs(args[0])
            if node.func.id in ('comb','factorial','gcd','lcm') and all(v.denominator==1 and abs(v)<=10000 for v in args):
                if node.func.id=='factorial' and (len(args)!=1 or args[0]>200):raise ValueError('Factorial bound')
                return Fraction(getattr(math,node.func.id)(*(int(v) for v in args)))
        raise ValueError('Unsupported exact arithmetic')
    return run(ast.parse(expr,mode='eval').body)
def boxed_value(answer):
    found=[]
    for match in re.finditer(r'\\boxed\{',answer):
        start=match.end();depth=1;end=start
        while end<len(answer) and depth:
            depth+=(answer[end]=='{')-(answer[end]=='}');end+=1
        if depth:raise ValueError('Unclosed box')
        value=answer[start:end-1].strip().replace(' ','').replace('\\,','')
        value=re.sub(r'\\(?:dfrac|tfrac|frac)\{(-?\d+)\}\{(-?\d+)\}',r'\1/\2',value)
        found.append(Fraction(value))
    if len(found)!=1:raise ValueError('Exactly one rational final box required')
    return found[0]
def solve(task):
    start=time.monotonic();raw=None;audit_raw=None
    try:
        prompt=task['instruction']+'\nUse natural English reasoning. Work through the needed derivation and important computations, verify the result, then finish. Avoid artificial labels such as Checked fact. Do not lengthen a solution with repetition.'
        raw=api(prompt);choice=raw['choices'][0]
        if choice['finish_reason']!='stop':return dict(accepted=False,reason='unfinished',raw=raw)
        msg=choice['message'];reason=msg.get('reasoning_content') or '';answer=msg.get('content') or ''
        if task['category']=='code':
            blocks=list(re.finditer(r'```(?:python|py)\s*\n.*?```',answer,re.S))
            if len(blocks)==1:
                b=blocks[0];prose=(answer[:b.start()]+'\n'+answer[b.end():]).strip()
                if prose:reason=reason.rstrip()+'\n\n'+prose
                answer=b.group(0)
        if not reason.strip() or not answer.strip():return dict(accepted=False,reason='empty_content',raw=raw)
        if any(marker in reason or marker in answer for marker in CONTROL) or 'Checked fact:' in reason or repeat_reason(reason+'\n'+answer):
            return dict(accepted=False,reason='control_template_or_repetition',raw=raw)
        checks=[];audit=None
        if task['category']=='math':
            reference=make_math(task['family_index'],task['serial'],task['split'])
            assert all(reference[k]==task[k] for k in ('facts','gold','base_instruction'))
            if boxed_value(answer)!=Fraction(task['gold']):return dict(accepted=False,reason='wrong_final',raw=raw)
            audit_prompt='Audit the proposed mathematical solution independently. The exact reference facts below have been computed independently; the solver did NOT see them. Check the key reasoning, reductions, counts and arithmetic in the actual solution, not just its final number. Reject incorrect, unsupported or mutually contradictory key steps even if the answer agrees. False starts explicitly abandoned and correctly repaired are allowed. Return only JSON: {"valid":boolean,"errors":[strings],"key_steps":[strings],"numeric_checks":[{"quote":"exact substring from solution containing a calculation","lhs":"arithmetic expression","rhs":"arithmetic expression"}]}. Extract at least one genuine substantive numerical equality from the solution, never invent one. Expressions may use integer constants,+,-,*,/,//,%,**,abs,comb,factorial,gcd,lcm. If an equality is not expressible, omit it; do not reject a correct alternative method merely for differing from the reference.\nPROBLEM:\n'+task['instruction']+'\nINDEPENDENT REFERENCE:\n'+json.dumps(dict(facts=task['facts'],gold=task['gold']),ensure_ascii=False)+'\nACTUAL SOLUTION:\n'+reason+'\nFINAL:\n'+answer
            audit_raw=api(audit_prompt,False)
            if audit_raw['choices'][0]['finish_reason']!='stop':return dict(accepted=False,reason='audit_unfinished',raw=raw,audit_raw=audit_raw)
            audit_text=audit_raw['choices'][0]['message']['content'].strip()
            if audit_text.startswith('```'):audit_text=audit_text.split('\n',1)[1].rsplit('```',1)[0]
            audit=json.loads(audit_text)
            if audit.get('valid') is not True or audit.get('errors') or not audit.get('key_steps'):
                return dict(accepted=False,reason='key_reasoning_audit_rejected',raw=raw,audit_raw=audit_raw)
            for check in audit.get('numeric_checks',[]):
                if not isinstance(check.get('quote'),str) or len(check['quote'])<3 or check['quote'] not in reason+'\n'+answer:
                    raise ValueError('Audit arithmetic quote not in solution')
                lhs,rhs=exact(check['lhs']),exact(check['rhs'])
                if lhs!=rhs:return dict(accepted=False,reason='arithmetic_check_failed',raw=raw,audit_raw=audit_raw)
                if check['lhs']==check['rhs']:continue
                checks.append(dict(**check,exact_result=str(lhs)))
            if not checks:return dict(accepted=False,reason='no_nontrivial_arithmetic_check',raw=raw,audit_raw=audit_raw)
            verification='exact_final_and_reference_regeneration_plus_separate_teacher_key_reasoning_audit_and_extracted_exact_arithmetic; prose_not_formally_proved'
        else:
            passed=verified_function(task,answer) if task['category']=='code' else verify_task(task,answer)
            if not passed:return dict(accepted=False,reason='final_verification_failed',raw=raw)
            verification='sandbox_all_code_fixtures_nonmutation' if task['category']=='code' else 'exact_final_constraints_or_graph'
        row=dict(task,reasoning=reason,response=answer,verification=verification,
                 validation_metadata=dict(reference_facts=task.get('facts'),reasoning_audit=audit,exact_arithmetic_checks=checks,
                                          proof_scope='Teacher audit is an additional heuristic; exact arithmetic checks verify extracted claims, not a formal complete proof'))
        return dict(accepted=True,row=row,raw=raw,audit_raw=audit_raw,seconds=time.monotonic()-start)
    except Exception as e:return dict(accepted=False,reason='exception',detail=str(e)[:400],raw=raw,audit_raw=audit_raw,seconds=time.monotonic()-start)
def key(t):return (t['split'],t['category'],t['family_index'] if t['category']=='math' else None)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=int,default=0);args=ap.parse_args()
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((OUT/'config.json').read_text());raw=(OUT/'tasks.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==cfg['tasks_sha256']
    if (OUT/'manifest.json').exists():print('Набор уже готов.');return
    tasks=[json.loads(l) for l in raw.splitlines()];commits=load('committed.jsonl');events=load('events.jsonl')
    counts=collections.Counter(key(c['row']) for c in commits);attempted={e['id'] for e in events}|{c['row']['id'] for c in commits}
    quota={}
    for split,qs in [('train',cfg['target']),('validation',cfg['holdout'])]:
        for cat,n in qs.items():
            if cat=='math':
                for f in (range(16) if split=='train' else range(16,20)):quota[split,cat,f]=40 if split=='train' else 32
            else:quota[split,cat,None]=n
    queues={k:collections.deque(t for t in tasks if key(t)==k and t['id'] not in attempted) for k in quota}
    rejection=collections.Counter(e['reason'] for e in events if not e['accepted']);pending=collections.Counter();active={}
    start=time.monotonic();initial=len(commits);tokens=0;submitted=0;keys=list(quota);cursor=0
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    encoder=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/e018_encode_server.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
    assert json.loads(encoder.stdout.readline())['ready']
    def update(state='collecting',**extra):
        elapsed=time.monotonic()-start;added=len(commits)-initial
        write('status.json',dict(state=state,accepted=len(commits),target=1280,
                               train=sum(v for (s,c,f),v in counts.items() if s=='train'),validation=sum(v for (s,c,f),v in counts.items() if s=='validation'),
                               counts={str(k):v for k,v in counts.items()},rejected=dict(rejection),attempted=len(attempted),active=len(active),
                               completion_tokens=tokens,tokens_per_second=tokens/max(elapsed,1),eta_seconds=(1280-len(commits))*elapsed/added if added else None,updated=time.time(),**extra))
    try:
        with cf.ThreadPoolExecutor(max_workers=50) as pool:
            while True:
                while len(active)<50 and (not args.pilot or submitted<args.pilot):
                    task=None
                    for _ in keys:
                        k=keys[cursor%len(keys)];cursor+=1
                        if counts[k]+pending[k]<quota[k] and queues[k]:task=queues[k].popleft();break
                    if task is None:break
                    pending[key(task)]+=1;active[pool.submit(solve,task)]=task;submitted+=1
                if not active:break
                ready,_=cf.wait(active,timeout=5,return_when=cf.FIRST_COMPLETED)
                for future in ready:
                    task=active.pop(future);r=future.result();pending[key(task)]-=1;attempted.add(task['id'])
                    tokens+=sum((r.get(k) or {}).get('usage',{}).get('completion_tokens',0) for k in ('raw','audit_raw'))
                    append('requests.jsonl',dict(id=task['id'],result=r))
                    if r['accepted']:
                        row=r['row'];encoder.stdin.write(json.dumps(row)+'\n');encoder.stdin.flush();result=json.loads(encoder.stdout.readline())
                        if not result.get('item'):r.update(accepted=False,reason='encoding_or_overlength',detail=result.get('error'))
                        else:
                            item=result['item'];item.update(split=row['split'],category=row['category'],topic=row.get('topic',row.get('family',row['category'])),verification=row['verification'])
                            c=dict(row=row,encoded=item);append('committed.jsonl',c);commits.append(c);counts[key(row)]+=1
                    event=dict(id=task['id'],accepted=r['accepted'],reason=r.get('reason','accepted'),detail=r.get('detail'),category=task['category'],split=task['split'],time=time.time());append('events.jsonl',event)
                    if not r['accepted']:rejection[event['reason']]+=1
                update()
                print(f"Готово {len(commits)}/1280; {tokens/max(time.monotonic()-start,1):.0f} ток/с",flush=True)
        if args.pilot:update('pilot_completed');return
        missing={str(k):v-counts[k] for k,v in quota.items() if counts[k]<v}
        if missing:update('insufficient_candidates',missing=missing);raise RuntimeError('Not all frozen quotas filled')
        from transformers import AutoTokenizer
        from reasoning_data import SOURCE
        from reasoning_diagnostics import token_partition
        tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
        random.Random(2001024).shuffle(commits);files={}
        for split in ('train','validation'):
            rows=[c['encoded'] for c in commits if c['row']['split']==split]
            for c in commits:
                if c['row']['split']==split and split=='validation' and c['row']['category']=='math':
                    r=c['row'];e=c['encoded'];prompt=tok.apply_chat_template([dict(role='user',content=r['instruction'])],tokenize=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='medium')
                    text=prompt+r['reasoning'].rstrip()+'\n</think>\n'+r['response']+'<|im_end|>'
                    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True);assert encoded['input_ids']==e['input_ids']
                    e['diagnostic_masks']=token_partition(text,encoded['offset_mapping'],e['labels'],e['final_labels'])
            blob=''.join(json.dumps(r)+'\n' for r in rows).encode();(OUT/(split+'.jsonl')).write_bytes(blob)
            lengths=sorted(len(r['input_ids']) for r in rows)
            files[split]=dict(count=len(rows),tokens=sum(lengths),target_tokens=sum(sum(v!=-100 for v in r['labels']) for r in rows),sha256=hashlib.sha256(blob).hexdigest(),min_length=min(lengths),median_length=lengths[len(lengths)//2],max_length=max(lengths),length_over_2048=sum(x>2048 for x in lengths))
        (OUT/'accepted.jsonl').write_text(''.join(json.dumps(c['row'],ensure_ascii=False)+'\n' for c in commits))
        write('manifest.json',dict(state='completed',files=files,max_length=12288,truncated=False,verification=cfg['verification'],rejected=dict(rejection),config_sha256=hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),control_sha256=cfg['control_sha256']))
        update('completed');print('Готово: 1024 обучения + 256 проверки. Естественные ответы; сертификаты только в метаданных.',flush=True)
    except Exception as e:update('failed',error=str(e));raise
    finally:encoder.stdin.close();encoder.wait(timeout=30)
if __name__=='__main__':main()
