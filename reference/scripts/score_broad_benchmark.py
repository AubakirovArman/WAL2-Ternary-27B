"""CPU scorer: official IFEval checks, sandboxed HumanEval tests, exact GSM8K final number."""
import argparse,gzip,json,os,random,re,sys
from decimal import Decimal,InvalidOperation
from pathlib import Path
from code_sandbox import run
ROOT=Path(__file__).resolve().parents[1]

def numeric(answer):
    matches=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',answer)
    if not matches:return None
    try:return Decimal(matches[-1].replace(',',''))
    except InvalidOperation:return None

def python_code(answer):
    blocks=re.findall(r'```(?:python|py)?\s*\n([\s\S]*?)```',answer)
    return blocks[-1] if blocks else answer.strip()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);args=ap.parse_args()
    os.environ.setdefault('NLTK_DATA',str(Path.home()/'.cache/vol2-27b-eval/nltk_data'));sys.path.insert(0,str(ROOT/'tools/ifeval'))
    from instruction_following_eval.evaluation_lib import InputExample,test_instruction_following_strict,test_instruction_following_loose
    from langdetect import DetectorFactory
    DetectorFactory.seed=0;random.seed(0)
    data={'gsm8k':[json.loads(s) for s in (ROOT/'data/evaluation-v1/gsm8k-test.jsonl').read_text().splitlines()],
          'humaneval':[json.loads(s) for s in gzip.open(ROOT/'data/evaluation-v1/humaneval.jsonl.gz','rt')],
          'ifeval':[json.loads(s) for s in (ROOT/'data/evaluation-v1/ifeval-input.jsonl').read_text().splitlines()]}
    scores=[]
    responses=list(map(json.loads,(args.folder/'responses.jsonl').read_text().splitlines()))
    if any(r['family']=='rummlu' for r in responses):
        data['rummlu']=[json.loads(s) for s in (ROOT/'data/evaluation-ru-v1/rummlu-test.jsonl').read_text().splitlines()]
    for row in responses:
        family=row['family'];reference=data[family][row['source_index']];answer=row['answer']
        result={'id':row['id'],'family':family,'source_index':row['source_index'],'truncated':row['finish_reason']=='length','thinking_completed':row['thinking_completed']}
        if family=='gsm8k':
            expected=numeric(reference['answer']);actual=numeric(answer)
            result.update(correct=actual is not None and actual==expected,predicted=str(actual),expected=str(expected))
        elif family=='rummlu':
            actual=answer.strip()
            result.update(correct=actual==reference['outputs'],predicted=actual,expected=reference['outputs'],
                          domain=reference['meta']['domain'])
        elif family=='humaneval':
            code=python_code(answer)
            if not re.search(r'\bdef\s+'+re.escape(reference['entry_point'])+r'\s*\(',code):
                result.update(correct=False,error='missing complete entry-point function')
            else:
                prefix=reference['prompt'].split('def '+reference['entry_point'],1)[0]
                executed=run(prefix+'\n'+code+'\n'+reference['test']+'\ncheck('+reference['entry_point']+')\n')
                result.update(correct=executed['passed'],sandbox=executed)
        else:
            inp=InputExample(**reference)
            strict=test_instruction_following_strict(inp,{inp.prompt:answer});loose=test_instruction_following_loose(inp,{inp.prompt:answer})
            result.update(correct=strict.follow_all_instructions,loose_correct=loose.follow_all_instructions,
                          instructions_strict=strict.follow_instruction_list,instructions_loose=loose.follow_instruction_list)
        scores.append(result)
    summary={}
    for family in data:
        rows=[r for r in scores if r['family']==family]
        summary[family]={'correct':sum(r['correct'] for r in rows),'total':len(rows),'truncated':sum(r['truncated'] for r in rows)}
    (args.folder/'scores.json').write_text(json.dumps(scores,indent=2));(args.folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
