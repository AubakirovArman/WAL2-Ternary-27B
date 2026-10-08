"""Build immutable, attributed mixed data; no CUDA and no execution of generated code."""
import argparse, collections, hashlib, json, random, re
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
def normalized(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def digest(s):return hashlib.sha256(normalized(s).encode()).hexdigest()
def read_rows(path):
    if not path.exists():return []
    # An online collector may be appending its last line; only consume complete lines.
    raw=path.read_text();return [json.loads(s) for s in raw.splitlines() if s.strip()] if raw.endswith('\n') else [json.loads(s) for s in raw.splitlines()[:-1] if s.strip()]
def record(i,instruction,response,source,split='train',**kw):
    return dict(id=i,instruction=instruction,response=response,source=source,split=split,
                sha256=hashlib.sha256(instruction.encode()).hexdigest(),**kw)

def verified(seed,count,split='train'):
    rng=random.Random(seed);rows=[]
    for i in range(count):
        ru=i%2==0;kind=(i//2)%6;a,b,c=[rng.randint(11,999) for _ in range(3)]
        vals=rng.sample(range(-999,1000),rng.randint(4,9))
        if kind==0:
            prompt=(f'Вычисли ({a} + {b}) * {c}. Ответь только целым числом.' if ru else f'Calculate ({a} + {b}) * {c}. Return only the integer.')
            answer=str((a+b)*c)
        elif kind==1:
            prompt=(f'Отсортируй числа по возрастанию: {vals}. Верни только JSON-массив.' if ru else f'Sort these numbers in ascending order: {vals}. Return only a JSON array.')
            answer=json.dumps(sorted(vals))
        elif kind==2:
            obj={'item':f'part-{a}','price':b,'stock':c}
            prompt=(f'Из объекта {json.dumps(obj)} извлеки item и stock. Верни только JSON-объект.' if ru else f'From {json.dumps(obj)} extract item and stock. Return only a JSON object.')
            answer=json.dumps({'item':obj['item'],'stock':c})
        elif kind==3:
            code=f'xs = {vals}\nprint(sum(x*x for x in xs if x > 0))'
            prompt=('Какое число напечатает этот Python-код? Верни только число.\n' if ru else 'What number does this Python code print? Return only the number.\n')+code
            answer=str(sum(x*x for x in vals if x>0))
        elif kind==4:
            words=rng.choices(['map','tree','blue','desk','river','cloud','stone','green','lamp'],k=rng.randint(6,15));word=rng.choice(words)
            prompt=(f'Сколько раз слово {word!r} встречается в тексте: "{" ".join(words)}"? Верни только число.' if ru else f'How many times does {word!r} occur in "{" ".join(words)}"? Return only the number.')
            answer=str(words.count(word))
        else:
            prompt=(f'В коробке {a} пакетов по {b} деталей. Отгрузили {c} деталей. Сколько осталось? Допускается отрицательный остаток. Только число.' if ru else f'A box has {a} packs of {b} parts. After shipping {c} parts, what is the balance? Negative balances are allowed. Return only the number.')
            answer=str(a*b-c)
        rows.append(record(seed*10000+i,prompt,answer,'verified_generator_v2',split,language='Russian' if ru else 'English',topic=['arithmetic','sorting','JSON','Python','counting','word_problem'][kind],verification='deterministic_reference',checker='json' if kind in [1,2] else 'exact'))
    return rows

def build(output,target=10000):
    if output.exists():raise FileExistsError(output)
    old=read_rows(ROOT/'data/teacher_pairs.jsonl')
    held=[r for r in old if r['split']!='train']
    # Keep original held-out order and tokenization: all 128 validation and 128 test examples.
    seen={digest(r['instruction']) for r in held}
    tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen3.8-27b-fp8-source',local_files_only=True)
    rejected=collections.Counter();accepted=[];tokens=0
    def add(row):
        nonlocal tokens
        key=digest(row['instruction'])
        if key in seen:rejected['duplicate_instruction']+=1;return False
        prompt=tok.apply_chat_template([{'role':'user','content':row['instruction']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        n=len(tok.encode(prompt,add_special_tokens=False));m=len(tok.encode(row['response']+'<|im_end|>',add_special_tokens=False))
        if n+m>512 or n>480 or m<2:rejected['length']+=1;return False
        seen.add(key);row['prompt_tokens']=n;row['answer_tokens']=m;accepted.append(row);tokens+=m;return True
    extra=verified(407,32,'validation')+verified(408,32,'test')
    seen.update(digest(r['instruction']) for r in extra)
    for row in old:
        if row['split']=='train':add(dict(row,source='teacher_e004',verification='unverified_teacher'))
    for row in verified(301,1200):add(row)
    teacher=read_rows(ROOT/'data/v2/teacher_new.jsonl')
    for row in teacher[:5000]:add(row)
    public=read_rows(ROOT/'data/v2/sources/dolly.jsonl');random.Random(20260920).shuffle(public)
    for i,row in enumerate(public):
        if len(accepted)>=target:break
        instruction=row['instruction'].strip()
        if row.get('context'):instruction+='\n\nContext:\n'+row['context'].strip()
        response=row['response'].strip()
        if not instruction or not response:continue
        add(record(1000000+i,instruction,response,'databricks-dolly-15k',language='English',topic=row['category'],license='CC-BY-SA-3.0',verification='human_authored_not_individually_verified'))
    if len(accepted)<target:raise RuntimeError(f'Only {len(accepted)} valid training rows, need {target}')
    random.Random(7719).shuffle(accepted)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in held+accepted))
    suite=ROOT/'data/v2/verified_holdout.jsonl'
    if not suite.exists():suite.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in extra))
    manifest={'train':len(accepted),'held_out':dict(collections.Counter(r['split'] for r in held)),
              'sources':dict(collections.Counter(r['source'] for r in accepted)),
              'languages':dict(collections.Counter(r.get('language','unknown') for r in accepted)),
              'train_answer_tokens':tokens,'rejected':dict(rejected),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'max_length':512,'training_answers_truncated':False,'deduplication':'normalized exact instruction against all train/heldout; semantic paraphrases not guaranteed excluded',
              'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/v2/sources').iterdir() if p.is_file()}}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--target',type=int,default=10000)
    a=p.parse_args();build(a.output,a.target)
