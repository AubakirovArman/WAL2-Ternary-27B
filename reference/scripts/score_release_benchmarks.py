"""Full official code test sets, math verification, IFEval, exact multiple choice."""
import argparse,collections,json,os,random,re,sys,time
from pathlib import Path
from math_verify import parse,verify,ExprExtractionConfig,LatexExtractionConfig
from score_broad_benchmark import numeric,python_code
from release_evalplus import evaluate
from evalplus.data import get_human_eval_plus,get_mbpp_plus
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('NLTK_DATA',str(Path.home()/'.cache/vol2-27b-eval/nltk_data'))
sys.path.insert(0,str(ROOT/'tools/ifeval'))
from instruction_following_eval.evaluation_lib import InputExample,test_instruction_following_strict,test_instruction_following_loose
from langdetect import DetectorFactory
DetectorFactory.seed=0;random.seed(0)
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def score(row,task,problems):
 family=row['family'];ref=task['reference'];answer=row['answer'];out=dict(id=row['id'],family=family,source_key=row['source_key'],source_index=row['source_index'],truncated=row['finish_reason']=='length',thinking_completed=row['thinking_completed'])
 if not row['thinking_completed'] or not answer.strip():return dict(out,correct=False,error='unfinished_or_empty_final')
 if family=='gsm8k':
  actual=numeric(answer);expected=numeric(ref['answer']);assert expected is not None
  out.update(correct=actual is not None and actual==expected)
 elif family in ('math500','aime25'):
  gold=parse('\\boxed{'+ref['answer']+'}',extraction_config=[LatexExtractionConfig()]);assert gold,('unparseable reference',row['id'])
  pred=parse(answer,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()]);out.update(correct=bool(pred) and bool(verify(gold,pred)))
 elif family in ('musr','mmlu_redux'):
  matches=re.findall(r'ANSWER:\s*([A-Z])\b',answer);actual=matches[-1] if matches else None
  out.update(correct=actual==chr(65+ref['answer_index']),predicted=actual)
 elif family=='ifeval':
  inp=InputExample(**ref);s=test_instruction_following_strict(inp,{inp.prompt:answer});l=test_instruction_following_loose(inp,{inp.prompt:answer})
  out.update(correct=s.follow_all_instructions,loose_correct=l.follow_all_instructions,instructions_strict=s.follow_instruction_list,instructions_loose=l.follow_instruction_list)
 else:
  p=problems[family][ref['task_id']];code=python_code(answer)
  # HumanEval prompt may supply imports referenced by generated complete function.
  if family=='humaneval_plus':code=p['prompt'].split('def '+p['entry_point'],1)[0]+'\n'+code
  out.update(evaluate(family,p,code))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--follow',action='store_true');a=ap.parse_args()
 refs=json.loads((ROOT/'data/release-benchmarks-v1/tasks-with-references.json').read_text());problems={'humaneval_plus':get_human_eval_plus(),'mbpp_plus':get_mbpp_plus()}
 scores=[];p=a.folder/'scores.jsonl'
 if p.exists():
  text=p.read_text();assert not text or text.endswith('\n');scores=[json.loads(l) for l in text.splitlines()]
 assert [r['id'] for r in scores]==list(range(len(scores)))
 while True:
  rp=a.folder/'responses.jsonl';lines=rp.read_text().splitlines(keepends=True) if rp.exists() else [];responses=[json.loads(l) for l in lines if l.endswith('\n')]
  for row in responses[len(scores):]:
   assert row['id']==len(scores);result=score(row,refs[row['id']],problems)
   with p.open('a') as f:f.write(json.dumps(result)+'\n');f.flush();os.fsync(f.fileno())
   scores.append(result)
   summary={}
   for family in sorted({r['family'] for r in scores}):
    rs=[r for r in scores if r['family']==family];v=dict(correct=sum(r['correct'] for r in rs),total=len(rs),truncated=sum(r['truncated'] for r in rs));v['accuracy']=v['correct']/v['total']
    if family=='ifeval':
     v['prompt_loose']=sum(r.get('loose_correct',False) for r in rs)/len(rs)
    summary[family]=v
   write(a.folder/'summary.json',summary);write(a.folder/'scoring-status.json',dict(state='scoring',completed=len(scores),updated=time.time()))
  cfg=json.loads((a.folder/'config.json').read_text()) if (a.folder/'config.json').exists() else {}
  if cfg.get('state')=='completed' and len(scores)==cfg['total']:break
  if not a.follow or cfg.get('state')=='failed':raise RuntimeError('Incomplete generation; scores retained')
  time.sleep(10)
 write(a.folder/'scoring-status.json',dict(state='completed',completed=len(scores),updated=time.time()))
 write(a.folder/'scores.json',scores);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
