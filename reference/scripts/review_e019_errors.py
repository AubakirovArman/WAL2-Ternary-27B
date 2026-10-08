"""Resumable diagnostic review of100 math failures. Never a training corpus."""
import argparse
import ast
import asyncio
import collections
import fcntl
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/e019-error-review'
MODEL = 'Qwen/Qwen3.8-27B-FP8'
KINDS = ('place_value', 'arithmetic', 'algebra', 'unjustified_assumption', 'missing_cases', 'wrong_definition', 'invalid_inference', 'repetition', 'unfinished_only', 'unclear')


def write(name, value):
    p = OUT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(p)


def numeric(expr, assignments):
    """Restricted exact arithmetic for a local counterexample, never eval/exec."""
    if len(expr) > 500:
        raise ValueError('Witness expression too long')
    tree = ast.parse(expr, mode='eval')
    if len(list(ast.walk(tree))) > 120:
        raise ValueError('Too many witness nodes')
    def walk(node):
        if isinstance(node, ast.Expression): return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10**12: return Fraction(node.value)
        if isinstance(node, ast.Name) and node.id in assignments: return Fraction(assignments[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)): return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        if isinstance(node, ast.BinOp):
            a,b = walk(node.left),walk(node.right)
            if isinstance(node.op, ast.Add): value=a+b
            elif isinstance(node.op, ast.Sub): value=a-b
            elif isinstance(node.op, ast.Mult): value=a*b
            elif isinstance(node.op, ast.Div): value=a/b
            elif isinstance(node.op, ast.FloorDiv): value=Fraction(a//b)
            elif isinstance(node.op, ast.Mod): value=a%b
            elif isinstance(node.op, ast.Pow) and b.denominator==1 and -8<=b<=8: value=a**int(b)
            else: raise ValueError('Unsupported witness arithmetic')
            if value.numerator.bit_length() > 1024 or value.denominator.bit_length()>1024: raise ValueError('Witness too large')
            return value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('gcd','abs') and not node.keywords:
            values = [walk(a) for a in node.args]
            if node.func.id=='abs' and len(values)==1: return abs(values[0])
            if node.func.id=='gcd' and len(values)==2 and all(a.denominator==1 for a in values): return Fraction(math.gcd(*map(int,values)))
        raise ValueError('Unsupported witness syntax')
    return walk(tree)


def check_review(task, review):
    if not isinstance(review, dict) or review.get('category') not in KINDS:
        raise ValueError('Invalid review category')
    quote = review.get('first_invalid_quote', '')
    if quote and (not isinstance(quote,str) or quote not in task['raw_response']):
        raise ValueError('Reviewer quote does not occur in original answer')
    if not quote and review['category'] not in ('unfinished_only','unclear'):
        raise ValueError('An error claim must cite a literal passage')
    witness = review.get('counterexample')
    result = dict(literal_quote_checked=bool(quote), quote_offset=task['raw_response'].find(quote) if quote else None, counterexample_checked=False, first_error_independently_confirmed=False)
    if witness:
        assign = witness.get('assignment', {})
        if not isinstance(assign,dict) or len(assign)>8 or any(type(v) is not int or abs(v)>10**12 for v in assign.values()): raise ValueError('Invalid assignments')
        left = numeric(witness['claimed_left'],assign)
        right = numeric(witness['claimed_right'],assign)
        if left == right: raise ValueError('Proposed counterexample does not refute the claimed equality')
        result.update(counterexample_checked=True, witness_left=str(left), witness_right=str(right), witness_scope='Exact inequality only; applicability of expressions/domain to quoted claim and earliest-error status still need review')
    return result


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=100);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'review.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads((ROOT/'reports/e019-d/failure-review100.json').read_text())[:args.limit]
    path=OUT/'reviews.jsonl'
    existing=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done={r['id'] for r in existing}; rows=list(existing); start=time.monotonic()
    write('protocol.json',dict(teacher=MODEL,endpoint='existing unchanged API',source='E019-D100 failures',training_use=False,scope='Teacher-proposed earliest errors; literal quotes and supplied arithmetic counterexamples checked. General proofs and earliest status are not automatically certified. No benchmark examples or corrections enter new training data.'))
    def state(state='running',**kw):
        write('status.json',dict(state=state,completed=len(rows),total=len(tasks),updated=time.time(),elapsed_seconds=time.monotonic()-start,**kw))
    state()
    async with httpx.AsyncClient(timeout=480,limits=httpx.Limits(max_connections=22),trust_env=False) as client:
        sem=asyncio.Semaphore(20)
        async def job(task):
            async with sem:
                prompt='You are auditing a defective mathematical solution. Find the earliest substantive mathematical error, not merely a late repetition. Do not invent an error when none is identifiable. Later corrections do not erase an earlier invalid step. The supplied reference label is not a proof certificate. Return ONLY a JSON object with keys category, first_invalid_quote (literal exact substring, or empty for unclear/unfinished_only), explanation, correct_statement, needed_skill, confidence (low/medium/high), counterexample (null or {assignment:{variable:integer},claimed_left:Python-arithmetic-string,claimed_right:Python-arithmetic-string}). Category must be one of '+', '.join(KINDS)+'. A counterexample must make the claimed equality false and satisfy the actual problem domain. Use ** for powers, explicit *, and only integers/rational operations/gcd/abs in witness expressions. Explain domain applicability. Do not equate truncation with an identified mathematical error.\nPROBLEM:\n'+task['instruction']+'\nREFERENCE LABEL / SOURCE INFORMATION:\n'+json.dumps(task['reference'],ensure_ascii=False)+'\nSTUDENT SOLUTION:\n'+task['raw_response']
                errors=[];result=None
                for attempt in range(2):
                    try:
                        response=await client.post('http://127.0.0.1:8000/v1/chat/completions',json=dict(model=MODEL,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=3072,chat_template_kwargs={'enable_thinking':False}))
                        response.raise_for_status();raw=response.json();ch=raw['choices'][0]
                        if ch['finish_reason']!='stop':raise ValueError('Review hit output limit')
                        text=ch['message'].get('content') or ''
                        if text.strip().startswith('```'):text=text.strip().split('\n',1)[1].rsplit('```',1)[0]
                        review=json.loads(text);checks=check_review(task,review)
                        result=dict(id=task['id'],family=task['family'],review=review,checks=checks,raw=raw,status='candidate_review');break
                    except Exception as e:
                        errors.append(str(e))
                        if attempt==0:await asyncio.sleep(2)
                if result is None:result=dict(id=task['id'],family=task['family'],status='review_failed',errors=errors)
                result['retries']=errors
                with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                rows.append(result);state();print(f"Разобрано {len(rows)}/{len(tasks)}",flush=True)
        try:
            await asyncio.gather(*(job(t) for t in tasks if t['id'] not in done))
        except Exception as e:state('failed',error=str(e));raise
    categories=collections.Counter(r.get('review',{}).get('category','review_failed') for r in rows)
    write('summary.json',dict(total=len(rows),categories=dict(categories),arithmetic_counterexamples_checked=sum(r.get('checks',{}).get('counterexample_checked',False) for r in rows),independently_confirmed_first_errors=0,scope='Candidates for subject review, not certified proof labels or training examples'))
    state('completed')


if __name__=='__main__':asyncio.run(main())
