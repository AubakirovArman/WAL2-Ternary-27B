"""Bounded, resumable, single-request teacher collection using only its public API."""
import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TOPICS = [
    'Python: обработка списков и словарей', 'Python: генераторы и итераторы',
    'SQL: соединения и агрегации', 'SQL: оконные функции',
    'JavaScript: async/await и обработка ошибок', 'TypeScript: типы и интерфейсы',
    'алгоритмы: бинарный поиск', 'алгоритмы: обход графа',
    'отладка: ошибка на единицу', 'отладка: изменяемые значения по умолчанию',
    'тестирование: граничные случаи', 'Linux: безопасный анализ текстовых логов',
    'арифметическая текстовая задача', 'проценты и пропорции',
    'линейное уравнение', 'элементарная вероятность',
    'комбинаторика', 'геометрия: площади и расстояния',
    'логическая задача с ограничениями', 'анализ аргумента и контрпример',
    'физика: скорость и ускорение', 'информатика: двоичное представление',
    'структурированное извлечение в JSON', 'классификация текстов по заданным правилам',
    'краткое резюме предоставленного текста', 'редактирование делового письма',
    'перевод с русского на английский', 'перевод с английского на русский',
    'объяснение технического понятия новичку', 'сравнение двух инженерных решений',
    'планирование расписания при ограничениях', 'выбор функции и её JSON-аргументов',
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=256)
    ap.add_argument('--delay', type=float, default=1.0)
    a = ap.parse_args()
    out = ROOT / 'data/teacher_pairs.jsonl'
    raw = ROOT / 'logs/teacher_requests.jsonl'
    done = {json.loads(s)['id'] for s in out.read_text().splitlines()} if out.exists() else set()
    rng = random.Random(20260919)
    schedule = list(range(a.count)); rng.shuffle(schedule)
    for i in schedule:
        if i in done:
            continue
        topic = TOPICS[i % len(TOPICS)]
        lang = 'Russian' if (i // len(TOPICS)) % 2 == 0 else 'English'
        split = 'test' if i % 8 == 7 else 'validation' if i % 8 == 6 else 'train'
        # Split follows independently varied example IDs; topic coverage is checked downstream.
        split = ['train'] * 6 + ['validation', 'test']
        split = split[(i // len(TOPICS) + (i % len(TOPICS))) % 8]
        prompt = (f'Create one self-contained training example in {lang}. Topic: {topic}. '
                  f'Variation seed: {9137+i*7919}. Vary concrete numbers, entities and formulation. '
                  'Return ONLY a JSON object with two string fields: "instruction" and "response". '
                  'instruction must contain all source text/data needed to solve the task. '
                  'response must solve it accurately with a brief explanation or working code. '
                  'Do not refer to a missing attachment or external source. No medical or legal advice. '
                  'Keep the entire JSON below 350 words. Do not wrap JSON in markdown.')
        body = {'model':'Qwen/Qwen3.8-27B-FP8',
                'messages':[{'role':'user','content':prompt}], 'temperature':0.7,
                'max_tokens':1024, 'chat_template_kwargs':{'enable_thinking':False},
                'response_format':{'type':'json_object'}}
        for attempt in range(3):
            started = time.time()
            try:
                req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
                    data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req, timeout=180) as res:
                    response = json.load(res)
                with raw.open('a') as f:
                    f.write(json.dumps({'id':i,'utc':datetime.now(timezone.utc).isoformat(),
                        'request':body,'response':response,'seconds':time.time()-started},ensure_ascii=False)+'\n')
                choice = response['choices'][0]
                if choice['finish_reason'] != 'stop':
                    raise ValueError('truncated teacher response')
                pair = json.loads(choice['message']['content'])
                if not all(isinstance(pair.get(k),str) and len(pair[k])>8 for k in ['instruction','response']):
                    raise ValueError('invalid pair')
                record = {'id':i,'split':split,'topic':topic,'language':lang,
                    'instruction':pair['instruction'],'response':pair['response'],
                    'teacher':body['model'],'sha256':hashlib.sha256(pair['instruction'].encode()).hexdigest()}
                with out.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                print(json.dumps({'id':i,'split':split,'seconds':round(time.time()-started,2)},ensure_ascii=False),flush=True)
                break
            except Exception as e:
                print(json.dumps({'id':i,'attempt':attempt,'error':str(e)}),flush=True)
                time.sleep(5*(attempt+1))
        time.sleep(a.delay)

if __name__ == '__main__':main()
