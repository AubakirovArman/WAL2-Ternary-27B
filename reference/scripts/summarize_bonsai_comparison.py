"""Keep original strict scores; add explicitly post-hoc format-normalized diagnostics."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def normalized_correct(row):
    answer=row['answer'].strip()
    fence=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',answer)
    if fence:answer=fence.group(1).strip()
    if row['topic'] in ['JSON','sorting']:
        try:return json.loads(answer)==json.loads(row['expected'])
        except (ValueError,TypeError):return False
    # Accept a final integer from an otherwise arithmetic-only equality chain.
    if '=' in answer and re.fullmatch(r'[\d\s+*/×²=−-]+',answer):answer=answer.rsplit('=',1)[1].strip()
    return answer==row['expected'].strip()

def main():
    paths={'source':ROOT/'reports/source-verified-generation.json',
           'student':ROOT/'runs/e005-data-lr/verified-generation.json',
           'bonsai2':ROOT/'reports/bonsai2/generation.json'}
    result={'note':'Normalized scores are post-hoc diagnostics: strip JSON fences and allow final integer in arithmetic-only equality chains. Original strict scores are unchanged. 16 small tasks, greedy max64, thinking disabled.'}
    for name,path in paths.items():
        rows=json.loads(path.read_text());assert len(rows)==16
        result[name]={'strict':sum(r['correct'] for r in rows),'format_normalized':sum(normalized_correct(r) for r in rows),'total':16}
    (ROOT/'reports/bonsai2/generation-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
