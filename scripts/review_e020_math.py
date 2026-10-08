"""Separate heuristic reasoning audit of new accepted finals; never train on controls."""
import concurrent.futures as cf,fcntl,json,time
from pathlib import Path
from collect_e020_data import api,exact
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/e020-ab'
def review(item):
    t=item['task'];r=item['response'];text=r['raw_response']
    prompt='Independently audit this mathematical solution. A final-answer checker accepted it, but that does not prove the reasoning. Find unsupported reductions, arithmetic errors, invented assumptions or contradictions. Check key steps, not just the final number. Return ONLY JSON {"valid_key_reasoning":boolean,"errors":[strings],"key_steps":[strings],"numeric_checks":[{"quote":"exact substring from solution","lhs":"numeric expression","rhs":"numeric expression"}]}. Numeric expressions can use integers,+,-,*,/,//,%,**,abs,comb,factorial,gcd,lcm. Omit inexpressible checks rather than inventing arithmetic.\nPROBLEM:\n'+t['instruction']+'\nEXPECTED FINAL:\n'+t['gold']+'\nACTUAL FULL RESPONSE:\n'+text
    raw=api(prompt,False);choice=raw['choices'][0]
    if choice['finish_reason']!='stop':return dict(**item,audit_state='incomplete',raw_audit=raw)
    answer=choice['message']['content'].strip()
    if answer.startswith('```'):answer=answer.split('\n',1)[1].rsplit('```',1)[0]
    try:
        audit=json.loads(answer);arithmetic=[]
        for c in audit.get('numeric_checks',[]):
            verdict={**c}
            try:
                if not c.get('quote') or c['quote'] not in text:raise ValueError('Quoted calculation not found')
                left,right=exact(c['lhs']),exact(c['rhs']);verdict.update(exact_equal=left==right,left=str(left),right=str(right))
            except Exception as e:verdict['check_error']=str(e)
            arithmetic.append(verdict)
        return dict(**item,audit_state='heuristic_teacher_audit_completed',reasoning_audit=audit,exact_arithmetic_audit=arithmetic,
                    scope='Same teacher in a separate request; exact extracted arithmetic checked but no formal proof; final-only benchmark unchanged',raw_audit=raw)
    except Exception as e:return dict(**item,audit_state='audit_parse_error',error=str(e),raw_audit=raw)
def main():
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    all_results=[]
    for branch in ('A','B'):
        for step in (512,1024):
            folder=OUT/f'{branch}-{step}';items=json.loads((folder/'new-correct-math-review.json').read_text());target=folder/'reasoning-review.json'
            if target.exists():results=json.loads(target.read_text())
            else:
                with cf.ThreadPoolExecutor(max_workers=12) as pool:results=list(pool.map(review,items))
                target.write_text(json.dumps(results,ensure_ascii=False,indent=2))
            all_results.append(dict(model=f'{branch}-{step}',new_correct_math=len(results),heuristic_invalid=sum(not r.get('reasoning_audit',{}).get('valid_key_reasoning',False) for r in results if r['audit_state']=='heuristic_teacher_audit_completed'),
                                    numeric_false=sum(any(c.get('exact_equal') is False for c in r.get('exact_arithmetic_audit',[])) for r in results),audit_incomplete=sum(r['audit_state']!='heuristic_teacher_audit_completed' for r in results)))
    (OUT/'reasoning-review-summary.json').write_text(json.dumps(all_results,indent=2));print(json.dumps(all_results,indent=2))
if __name__=='__main__':main()
