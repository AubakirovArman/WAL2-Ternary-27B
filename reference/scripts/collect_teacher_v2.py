"""One concurrent teacher API request; bounded, resumable and separate from E004 data."""
import argparse, fcntl, hashlib, json, random, re, time, urllib.request
from pathlib import Path
from collect_teacher import TOPICS
ROOT=Path(__file__).resolve().parents[1]
def key(s):return hashlib.sha256(' '.join(re.findall(r'\w+',s.casefold())).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=5000);a=ap.parse_args()
    folder=ROOT/'data/v2';out=folder/'teacher_new.jsonl'
    lock=(folder/'teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[]
    for path in [ROOT/'data/teacher_pairs.jsonl',out,folder/'verified_holdout.jsonl']:
        if path.exists():rows.extend(json.loads(s) for s in path.read_text().splitlines())
    seen={key(r['instruction']) for r in rows};done={r['id'] for r in rows if r.get('source')=='teacher_v2'}
    topics=TOPICS+['сжатие длинного делового письма с сохранением фактов','ответ на вопрос по приведённому документу','исправление неоднозначной инструкции','составление русского технического объяснения','написание unit-теста Python','поиск логической ошибки в псевдокоде','соблюдение точного формата ответа','диалог уточнения требований','выбор данных для SQL запроса','преобразование табличных данных в JSON','перевод технического текста','объяснение фрагмента кода']
    schedule=list(range(a.count*2));random.Random(99231).shuffle(schedule)
    deadline=time.monotonic()+18*3600;failures=0
    for i in schedule:
        if len(done)>=a.count or time.monotonic()>deadline or (folder/'STOP_COLLECTOR').exists():break
        uid=2000000+i
        if uid in done:continue
        lang='Russian' if i%3 else 'English';topic=topics[(i//3)%len(topics)]
        prompt=(f'Write one diverse self-contained instruction/answer training pair in {lang}. Topic: {topic}. '
                f'Variation {uid}: invent a specific fresh scenario with its source text, code or data included in the instruction. '
                'Return ONLY JSON with instruction and response string fields. The response must answer the instruction precisely. '
                'Prefer grounded transformations of supplied data, explanations, and short code. Avoid unverifiable current facts, '
                'missing context, medical/legal advice, and generic repeated questions. Check your answer before returning it. '
                'No thinking trace. Keep the whole pair below 220 words, with a complete response and no markdown around JSON.')
        body={'model':'Qwen/Qwen3.8-27B-FP8','messages':[{'role':'user','content':prompt}],
              'temperature':0.75,'max_tokens':900,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_object'}}
        try:
            request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            start=time.time()
            with urllib.request.urlopen(request,timeout=180) as r:raw=json.load(r)
            with (ROOT/'logs/teacher-v2-requests.jsonl').open('a') as f:f.write(json.dumps({'id':uid,'request':body,'response':raw,'seconds':time.time()-start},ensure_ascii=False)+'\n')
            choice=raw['choices'][0]
            if choice['finish_reason']!='stop':raise ValueError('truncated')
            pair=json.loads(choice['message']['content'])
            if not all(isinstance(pair.get(k),str) and len(pair[k])>=8 for k in ['instruction','response']):raise ValueError('invalid fields')
            h=key(pair['instruction'])
            if h in seen:raise ValueError('duplicate instruction')
            row={'id':uid,'split':'train','source':'teacher_v2','language':lang,'topic':topic,
                 'instruction':pair['instruction'],'response':pair['response'],'verification':'teacher_self_checked_not_independently_verified',
                 'sha256':hashlib.sha256(pair['instruction'].encode()).hexdigest()}
            with out.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            seen.add(h);done.add(uid);failures=0
            print(json.dumps({'event':'accepted','id':uid,'accepted_total':len(done),'seconds':time.time()-start}),flush=True)
        except Exception as e:
            failures+=1;print(json.dumps({'event':'rejected_or_failed','id':uid,'error':str(e),'consecutive_failures':failures}),flush=True)
            if failures>=20:raise RuntimeError('20 consecutive collector failures')
            time.sleep(min(30,2*failures))
        time.sleep(1)
    print(json.dumps({'event':'collector_stopped','accepted':len(done),'target':a.count}),flush=True)
if __name__=='__main__':main()
