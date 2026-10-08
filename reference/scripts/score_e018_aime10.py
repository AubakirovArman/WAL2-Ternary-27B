import json,sys
from pathlib import Path
from math_verify import parse,verify,ExprExtractionConfig,LatexExtractionConfig
ROOT=Path(__file__).resolve().parents[1];OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'reports/e018-step3584-check'
refs={r['id']:r for r in json.loads((ROOT/'data/release-benchmarks-v1/tasks-with-references.json').read_text())}
rows=json.loads((OUT/'aime10-responses.json').read_text());scores=[]
for r in rows:
 ref=refs[r['id']];assert ref['family']=='aime25'
 gold=parse('\\boxed{'+ref['reference']['answer']+'}',extraction_config=[LatexExtractionConfig()]);assert gold
 pred=parse(r['answer'],extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()]) if r['thinking_completed'] else []
 scores.append(dict(id=r['id'],correct=bool(pred) and bool(verify(gold,pred)),truncated=r['finish_reason']=='length',thinking_completed=r['thinking_completed']))
old=json.loads((OUT/'e017-aime10-scores.json').read_text());assert [r['id'] for r in old]==[r['id'] for r in scores]
result=dict(e018_correct=sum(r['correct'] for r in scores),e017_correct=sum(r['correct'] for r in old),total=10,truncated=sum(r['truncated'] for r in scores),scores=scores,thinking_enabled=rows[0].get('thinking_enabled',True),scope='Small fixed AIME25 subset, same questions and token budget; thinking mode and rendered prompts recorded in protocol.json.')
(OUT/'aime10-metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
