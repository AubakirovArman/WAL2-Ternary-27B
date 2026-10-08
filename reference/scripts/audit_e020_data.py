"""Final CPU dataset gate before cache/optimizer work; no source data mutation."""
import collections,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from reasoning_data import encode,SOURCE
from e020_math_tasks import make_math
from collect_e020_data import exact,boxed_value
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/e020-natural-v1'
def main():
    cfg=json.loads((DATA/'config.json').read_text());dm=json.loads((DATA/'manifest.json').read_text());assert dm['state']=='completed' and dm['truncated'] is False
    controls={json.loads(l)['sha256'] for l in (DATA/'control-tasks.jsonl').open()}
    rows={r['id']:r for r in (json.loads(l) for l in (DATA/'accepted.jsonl').open())};assert len(rows)==1280
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);seen=set();stats={}
    for split,target in [('train',1024),('validation',256)]:
        blob=(DATA/(split+'.jsonl')).read_bytes();assert hashlib.sha256(blob).hexdigest()==dm['files'][split]['sha256']
        items=[json.loads(l) for l in blob.splitlines()];assert len(items)==target
        counts=collections.Counter();families=collections.Counter();lengths=[]
        for item in items:
            row=rows[item['id']];assert row['split']==item['split']==split
            assert item['sha256']==row['sha256']==hashlib.sha256(row['instruction'].encode()).hexdigest()
            assert item['sha256'] not in seen and item['sha256'] not in controls;seen.add(item['sha256'])
            copy=dict(row,verification='teacher_answer_and_reasoning_unverified');encoded=encode(copy,tok,12288,allow_validation=True,allow_teacher_only=True)
            assert encoded and all(encoded[k]==item[k] for k in ('input_ids','labels','final_labels','prompt_tokens'))
            assert item['labels'][:item['prompt_tokens']]==[-100]*item['prompt_tokens'] and item['labels'][item['prompt_tokens']:]==item['input_ids'][item['prompt_tokens']:]
            assert item['input_ids'].count(tok.convert_tokens_to_ids('</think>'))==1
            assert item['input_ids'][-1]==tok.convert_tokens_to_ids('<|im_end|>')
            if row['category']=='math':
                original=make_math(row['family_index'],row['serial'],split)
                assert all(original[k]==row[k] for k in ('instruction','gold','facts'))
                from fractions import Fraction
                assert boxed_value(row['response'])==Fraction(row['gold'])
                metadata=row['validation_metadata'];assert metadata['reasoning_audit']['valid'] and not metadata['reasoning_audit']['errors']
                assert metadata['exact_arithmetic_checks']
                for c in metadata['exact_arithmetic_checks']:assert c['quote'] in row['reasoning']+'\n'+row['response'] and exact(c['lhs'])==exact(c['rhs'])
                families[row['family_index']]+=1
            if item.get('diagnostic_masks'):
                masks=item['diagnostic_masks'];assert all(sum(m[i] for m in masks.values())==int(y!=-100) for i,y in enumerate(item['labels']))
            counts[row['category']]+=1;lengths.append(len(item['input_ids']))
        assert dict(counts)==cfg['target' if split=='train' else 'holdout']
        assert dict(families)=={f:40 if split=='train' else 32 for f in (range(16) if split=='train' else range(16,20))}
        stats[split]=dict(categories=dict(counts),families=dict(families),mean_length=sum(lengths)/len(lengths),max_length=max(lengths),over_2048=sum(x>2048 for x in lengths))
    report=dict(passed=True,stats=stats,manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
                checks=['Exact encode regeneration','Prompt/target shift','Single closing delimiter and EOS','No truncated sequences','Exact train/val/control prompt separation','Fixed category/family quotas','References and accepted arithmetic checked again'],
                limits='Key reasoning audited by the same teacher in a separate request; arithmetic checks exact but not formal proof. Code/constraints were checked at acceptance; independent code regression held out.')
    (ROOT/'reports/e020-data-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
