"""CPU-only resumable data pipeline; one teacher request at a time, no training."""
import collections, fcntl, gzip, hashlib, json, math, os, random, re, time, urllib.request
from pathlib import Path
from transformers import AutoTokenizer
from reasoning_data import encode, SOURCE
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/fresh-5000-v1'
TARGET={'format':1500,'verified_reasoning':1000,'diverse':2500}
VAL=ROOT/'data/reasoning-fresh-v4-tokenized/validation.jsonl'
VAL_SHA='592a90f098b128af2badad0db018906f4c25998a1c1e6bcc3acdc159ba9236c2'
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(tmp,path)
def lines(path):return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def append(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def exclusions():
    seen=set();used=set();files={}
    cache=json.loads((ROOT/'data/v3-kd/manifest.json').read_text());used.update(map(str,cache['order_ids']))
    def walk(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in ('instruction','prompt','question') and isinstance(v,str):seen.add(norm(v))
                elif isinstance(v,(dict,list)):walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    paths=list((ROOT/'data').glob('reasoning-*/tasks.json'))
    paths+=list((ROOT/'reports').glob('**/tasks.json'))
    paths+=list((ROOT/'data/evaluation-v1').glob('*.jsonl*'))
    paths+=list((ROOT/'data/evaluation-ru-v1').glob('**/*.jsonl'))
    paths += [ROOT/'data/v2/verified_holdout.jsonl']
    for p in sorted(set(paths)):
        raw=p.read_bytes();files[str(p.relative_to(ROOT))]=sha(raw)
        text=gzip.decompress(raw).decode() if p.suffix=='.gz' else raw.decode()
        rows=json.loads(text) if p.suffix=='.json' else [json.loads(l) for l in text.splitlines()]
        walk(rows)
        if 'reasoning-' in str(p):
            for r in rows:used.add(str(r.get('source_id')))
    corpus=Path(cache['corpus']).read_bytes()
    if sha(corpus)!=cache['corpus_sha256']:raise ValueError('Corpus changed')
    rows=[json.loads(l) for l in corpus.splitlines()]
    for r in rows:
        if r['split']!='train' or str(r['id']) in used:seen.add(norm(r['instruction']))
    return seen,rows,files,cache['corpus_sha256']
def synthetic(i,reasoning=False):
    rng=random.Random(9127000+i+(100000 if reasoning else 0));ru=i%2==0
    a,b,c=[rng.randint(12,900) for _ in range(3)];xs=rng.sample(range(-99,100),8)
    j=(i//2)%(10 if reasoning else 15)
    if reasoning:
        kinds=['arithmetic','inventory','linear_equation','gcd','lcm','python_filter','python_slice','python_loop','python_dictionary','python_nested_loop'];topic=kinds[j]
        if j==0:p=f'({a} + {b}) * {c}';ans=(a+b)*c
        elif j==1:p=f'{a} * {b} - {c}';ans=a*b-c
        elif j==2:p=f'{b} * x + {c} = {b*a+c}. x = ?';ans=a
        elif j==3:p=f'gcd({a}, {b})';ans=math.gcd(a,b)
        elif j==4:p=f'lcm({a}, {b})';ans=math.lcm(a,b)
        elif j==5:p=f'xs = {xs}\nprint(sum(x*x for x in xs if x > 0))';ans=sum(x*x for x in xs if x>0)
        elif j==6:p=f'xs = {xs}\nprint(sum(xs[1::2]))';ans=sum(xs[1::2])
        elif j==7:p=f'total = {a}\nfor x in {xs}:\n    total += x if x % 2 == 0 else -x\nprint(total)';ans=a+sum(x if x%2==0 else -x for x in xs)
        elif j==8:p=f'd = {{"a": {a}, "b": {b}}}\nd["a"] += d.pop("b")\nprint(sum(d.values()))';ans=a+b
        else:p=f'print(sum(x*y for x in {xs[:4]} for y in {xs[4:]} if x < y))';ans=sum(x*y for x in xs[:4] for y in xs[4:] if x<y)
        instruction=(('Вычисли результат. ' if j<5 else 'Какое число напечатает код Python? ')+ 'Заверши ответ отдельной строкой ANSWER: целое_число.\n'+p) if ru else (('Calculate the result. ' if j<5 else 'What integer does this Python code print? ')+'End with a separate line ANSWER: integer.\n'+p)
        return instruction,topic,int(ans),None
    topics=['json_projection','json_filter','json_sort','json_count','csv','numbered_list','xml','json_boolean','exact_lines','reverse_sequence','label_classification','json_aggregation','key_value','json_unique','case_conversion'];topic=topics[j]
    obj={'item':f'part-{a}','price':b,'stock':c}
    data=json.dumps(obj);words=[f'item{x}' for x in xs]
    if j==0:p=f'Extract item and stock from {data}. Return only a JSON object.';q=f'Из {data} извлеки item и stock. Верни только JSON-объект.';ans=json.dumps({'item':obj['item'],'stock':c})
    elif j==1:p=f'Keep positive values in {xs} in their original order. Return only a JSON array.';q=f'Оставь положительные числа из {xs} в исходном порядке. Только JSON-массив.';ans=json.dumps([x for x in xs if x>0])
    elif j==2:p=f'Sort {xs} in descending order. Return only a JSON array.';q=f'Отсортируй {xs} по убыванию. Только JSON-массив.';ans=json.dumps(sorted(xs,reverse=True))
    elif j==3:p=f'Count even values in {xs}. Return only JSON with key count.';q=f'Посчитай чётные числа в {xs}. Только JSON с ключом count.';ans=json.dumps({'count':sum(x%2==0 for x in xs)})
    elif j==4:p=f'Convert {data} to CSV, header item,stock and one data row. No code fences.';q=f'Преобразуй {data} в CSV: заголовок item,stock и одна строка данных. Без Markdown.';ans=f'item,stock\npart-{a},{c}'
    elif j==5:p=f'List {words[:3]} in order, one item per line, numbered 1. through 3. Nothing else.';q=f'Перечисли {words[:3]} по порядку: три строки с нумерацией 1., 2., 3. Ничего больше.';ans='\n'.join(f'{k+1}. {w}' for k,w in enumerate(words[:3]))
    elif j==6:p=f'Return value {a} wrapped in exactly <result> and </result>. Nothing else.';q=f'Верни значение {a} строго между <result> и </result>. Ничего больше.';ans=f'<result>{a}</result>'
    elif j==7:p=f'Is every value in {xs} positive? Return only a JSON boolean.';q=f'Все ли числа в {xs} положительные? Только логическое значение JSON.';ans=json.dumps(all(x>0 for x in xs))
    elif j==8:p=f'Return exactly two lines: min=minimum and max=maximum for {xs}. Use numeric values.';q=f'Для {xs} верни ровно две строки: min=минимум и max=максимум. Подставь числа.';ans=f'min={min(xs)}\nmax={max(xs)}'
    elif j==9:p=f'Reverse {words[:4]}. Separate items with |, no spaces or other text.';q=f'Запиши {words[:4]} в обратном порядке через | без пробелов и пояснений.';ans='|'.join(reversed(words[:4]))
    elif j==10:p=f'Classify {a}: below 300 LOW, 300 to 600 inclusive MID, above 600 HIGH. Return only the label.';q=f'Классифицируй {a}: меньше 300 LOW, от 300 до 600 включительно MID, больше 600 HIGH. Только метка.';ans='LOW' if a<300 else 'MID' if a<=600 else 'HIGH'
    elif j==11:p=f'For {xs}, return only JSON keys sum and count with their values.';q=f'Для {xs} верни только JSON с ключами sum (сумма) и count (количество).';ans=json.dumps({'sum':sum(xs),'count':len(xs)})
    elif j==12:p=f'From {data}, return exactly stock={c};item=part-{a} with no additional text.';q=f'Из {data} верни строго stock={c};item=part-{a} без другого текста.';ans=f'stock={c};item=part-{a}'
    elif j==13:
        vals=xs[:4]+xs[2:6];p=f'Remove duplicates from {vals}, preserving first appearance. JSON array only.';q=f'Удали повторы из {vals}, сохрани порядок первого появления. Только JSON-массив.';ans=json.dumps(list(dict.fromkeys(vals)))
    else:p=f'Convert Part-{a}-Stock-{c} to uppercase. Return only the converted text.';q=f'Преобразуй Part-{a}-Stock-{c} в верхний регистр. Только результат.';ans=f'PART-{a}-STOCK-{c}'
    reason=(f'Результат преобразования: {ans}. В финальном ответе оставляю только требуемый формат.' if ru else f'The transformation gives: {ans}. Keep only the required format in the final answer.')
    return q if ru else p,topic,ans,reason

def schedule():
    seen,corpus,files,corpus_sha=exclusions();tasks=[]
    def add(inst,topic,category,**kw):
        key=norm(inst)
        if key in seen:return
        seen.add(key);tasks.append(dict(id=f'fresh5000-{len(tasks):05d}',instruction=inst,topic=topic,category=category,split='train',sha256=sha(inst.encode()),**kw))
    for i in range(4000):
        p,t,a,r=synthetic(i);add(p,t,'format',gold=a,gold_reasoning=r,language='Russian' if i%2==0 else 'English',source='deterministic_format_v1')
        if sum(x['category']=='format' for x in tasks)==1500:break
    for i in range(2000):
        p,t,a,r=synthetic(i,True);add(p,t,'verified_reasoning',expected=a,language='Russian' if i%2==0 else 'English',source='deterministic_math_code_v1')
    groups=collections.defaultdict(list)
    for row in corpus:
        if row['split']=='train' and norm(row['instruction']) not in seen:groups[(row.get('language','unknown'),row.get('topic','unknown'))].append(row)
    rng=random.Random(202609209);keys=sorted(groups);rng.shuffle(keys)
    for group in groups.values():rng.shuffle(group)
    while any(groups.values()):
        for key in keys:
            if not groups[key]:continue
            r=groups[key].pop();add(r['instruction'],r['topic'],'diverse',language=r.get('language','unknown'),source=r.get('source'),source_id=r['id'],license=r.get('license'))
    OUT.mkdir(exist_ok=False)
    write(OUT/'tasks.json',tasks)
    write(OUT/'config.json',dict(target=TARGET,seed=202609209,max_length=2048,teacher_requests_concurrent=1,excluded_files=files,corpus_sha256=corpus_sha,validation_sha256=VAL_SHA,deduplication='normalized exact prompts; no semantic-overlap guarantee',verification='1500 deterministic format answers; 1000 numeric final answers checked; 2500 teacher answers unverified; teacher reasoning unverified',training_started=False))
    (OUT/'collector.py').write_bytes(Path(__file__).read_bytes())
    return tasks

def run():
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if sha(VAL.read_bytes())!=VAL_SHA:raise ValueError('Validation changed')
    tasks=json.loads((OUT/'tasks.json').read_text()) if OUT.exists() else schedule()
    if (OUT/'manifest.json').exists():print('Набор уже готов.',flush=True);return
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    # Each commit contains both raw and tokenized data, avoiding cross-file resume ambiguity.
    committed=lines(OUT/'committed.jsonl');done={x['row']['id'] for x in committed}
    if len(done)!=len(committed):raise ValueError('Duplicate commits')
    counts=collections.Counter(x['row']['category'] for x in committed)
    events=lines(OUT/'events.jsonl');attempted={x['id'] for x in events};failures=0
    def status(state='collecting',**kw):
        write(OUT/'status.json',dict(state=state,accepted=sum(counts.values()),target=5000,categories=dict(counts),attempted=len(attempted),updated=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
    status()
    for task in tasks:
        category=task['category']
        if task['id'] in done or task['id'] in attempted or counts[category]>=TARGET[category]:continue
        try:
            row=dict(task,enable_thinking=True)
            if category=='format':
                row.update(response=row['gold'],reasoning=row['gold_reasoning'],verification='deterministic_format_answer')
            else:
                body=dict(model='Qwen/Qwen3.8-27B-FP8',messages=[dict(role='user',content=row['instruction'])],temperature=0,max_tokens=2048,reasoning_effort='medium',chat_template_kwargs={'enable_thinking':True})
                req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                for retry in range(3):
                    try:
                        with urllib.request.urlopen(req,timeout=180) as response:raw=json.load(response)
                        break
                    except Exception:
                        if retry==2:raise
                        time.sleep(15*(retry+1))
                append(OUT/'requests.jsonl',{'id':row['id'],'request':body,'response':raw})
                choice=raw['choices'][0];msg=choice['message']
                if choice['finish_reason']!='stop':raise ValueError('unfinished_response')
                row.update(reasoning=msg.get('reasoning_content') or '',response=msg.get('content') or '',verification='numeric_final_checked_reasoning_unverified' if category=='verified_reasoning' else 'teacher_answer_and_reasoning_unverified')
            # Deterministic format rows come directly from the generator, never from teacher output.
            encoding_row=dict(row)
            if category=='format':encoding_row['verification']='teacher_answer_and_reasoning_unverified'
            item=encode(encoding_row,tok,2048,allow_teacher_only=True)
            if item is None:raise ValueError('sequence_over_2048')
            item.update(split='train',category=category,verification=row['verification'],language=row['language'],topic=row['topic'])
            append(OUT/'committed.jsonl',dict(row=row,encoded=item));done.add(row['id']);counts[category]+=1;failures=0
            event=dict(id=row['id'],accepted=True)
        except ValueError as e:
            event=dict(id=task['id'],accepted=False,reason=str(e));failures=0
        except Exception as e:
            failures+=1;status('api_error',error=str(e))
            if failures>=3:raise
            time.sleep(15)
            # API failure is left retryable on resume; stop instead of silently losing coverage.
            raise
        append(OUT/'events.jsonl',event);attempted.add(task['id']);status()
        if sum(counts.values())%25==0 and event['accepted']:print(f'Готово {sum(counts.values())}/5000: {dict(counts)}',flush=True)
        if category!='format':time.sleep(1)
    if dict(counts)!=TARGET:status('insufficient_accepted');raise RuntimeError(f'Candidate pool exhausted: {counts}')
    committed=lines(OUT/'committed.jsonl');rng=random.Random(202609209);rng.shuffle(committed)
    rows=[x['encoded'] for x in committed]
    val=VAL.read_bytes();held=[json.loads(l) for l in val.splitlines()]
    if len({r['sha256'] for r in rows+held})!=len(rows)+len(held):raise ValueError('Train/validation overlap')
    train=''.join(json.dumps(r)+'\n' for r in rows).encode();raw=''.join(json.dumps(x['row'],ensure_ascii=False)+'\n' for x in committed).encode()
    (OUT/'train.jsonl').write_bytes(train);(OUT/'accepted.jsonl').write_bytes(raw);(OUT/'validation.jsonl').write_bytes(val)
    files={s:dict(count=len(rr),sha256=sha(bb),tokens=sum(len(r['input_ids']) for r in rr)) for s,rr,bb in [('train',rows,train),('validation',held,val)]}
    report=dict(state='completed',files=files,truncated=False,max_length=2048,categories=dict(counts),languages=dict(collections.Counter(r['language'] for r in rows)),topics=dict(collections.Counter(r['topic'] for r in rows)),source_accepted_sha256=sha(raw),config_sha256=sha((OUT/'config.json').read_bytes()),verification=json.loads((OUT/'config.json').read_text())['verification'],training_started=False)
    write(OUT/'manifest.json',report);status('completed');print('Готово: 5000 обучающих примеров. Обучение не запускалось.',flush=True)
if __name__=='__main__':
    try:run()
    except Exception as e:
        if OUT.exists():
            state=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
            write(OUT/'status.json',dict(state, state='failed',error=str(e)))
        raise
