"""Bounded training-only reasoning pilot, one teacher request at a time."""
import argparse,fcntl,hashlib,json,math,random,re,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def tasks(count=32):
    rng=random.Random(2026091919)
    result=[]
    for i in range(count):
        a,b,c=rng.randint(17,71),rng.randint(4,16),rng.randint(3,11)
        kind=i%4
        if kind==0:
            prompt=f'A stockroom has {a} boxes with {b} parts each. It ships {c} full boxes, then receives {b+3} loose parts. How many parts remain?'
            expected=(a-c)*b+b+3
        elif kind==1:
            prompt=f'Compute the remainder when ({a} ** {b} + {c} ** {b+2}) is divided by 97. Here ** means integer exponentiation.'
            expected=(pow(a,b,97)+pow(c,b+2,97))%97
        elif kind==2:
            values=[rng.randint(-25,40) for _ in range(c+7)]
            prompt=f'For the list {values}, remove duplicate values, keep only even values, and sum the squares of the remaining values. What is the result?'
            expected=sum(x*x for x in set(values) if x%2==0)
        else:
            prompt=f'A counter starts at {a}. Repeat these operations in order {c} times: multiply the counter by 2; add {b}; replace the counter by its remainder modulo 101. What is its final value?'
            expected=a
            for _ in range(c):expected=(expected*2+b)%101
        prompt+=' Explain your solution. End the final answer with ANSWER: followed by the integer result.'
        result.append({'id':f'reasoning-pilot-{i:03d}','split':'train','topic':['inventory','modular_power','list_transform','recurrence'][kind],
                       'instruction':prompt,'expected':expected})
    assert len({r['instruction'] for r in result})==count
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'data/reasoning-pilot')
    parser.add_argument('--count',type=int,default=32)
    parser.add_argument('--skip',type=int,default=0)
    parser.add_argument('--deadline-hours',type=float,default=0.5)
    args=parser.parse_args()
    if args.count<1 or args.skip<0 or not 0<args.deadline_hours<=8:
        raise ValueError('Invalid bounded collection parameters')
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=args.output;out.mkdir(exist_ok=False)
    schedule=tasks(args.skip+args.count)[args.skip:];(out/'tasks.json').write_text(json.dumps(schedule,indent=2))
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    (out/'config.json').write_text(json.dumps({'count':args.count,'skip':args.skip,'deadline_hours':args.deadline_hours,
        'scope':'Four deterministic arithmetic/data-transformation families; independent final integer check only'},indent=2))
    accepted=0;started=time.monotonic()
    for row in schedule:
        if time.monotonic()-started>args.deadline_hours*3600:raise TimeoutError('Verified collection exceeded deadline')
        body={'model':'Qwen/Qwen3.8-27B-FP8','messages':[{'role':'user','content':row['instruction']}],
              'temperature':0,'max_tokens':2048,'reasoning_effort':'medium','chat_template_kwargs':{'enable_thinking':True}}
        request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response:raw=json.load(response)
        with (out/'requests.jsonl').open('a') as f:f.write(json.dumps({'id':row['id'],'request':body,'response':raw})+'\n')
        choice=raw['choices'][0];message=choice['message'];reasoning=message.get('reasoning_content') or '';answer=message.get('content') or ''
        matches=re.findall(r'ANSWER:\s*(-?\d+)\s*$',answer.strip())
        controls=('<think>','</think>','<|im_start|>','<|im_end|>','<|endoftext|>')
        valid=choice['finish_reason']=='stop' and bool(reasoning.strip()) and bool(matches) and int(matches[-1])==row['expected'] and not any(m in reasoning or m in answer for m in controls)
        event={'id':row['id'],'accepted':valid,'finish_reason':choice['finish_reason'],'usage':raw.get('usage')}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if valid:
            saved={**row,'reasoning':reasoning,'response':answer,'enable_thinking':True,'source':'teacher_reasoning_pilot',
                   'verification':'Final integer independently checked; reasoning trace itself is not verified',
                   'sha256':hashlib.sha256(row['instruction'].encode()).hexdigest()}
            with (out/'accepted.jsonl').open('a') as f:f.write(json.dumps(saved)+'\n')
            accepted+=1
        print(json.dumps(event),flush=True);time.sleep(1)
    (out/'manifest.json').write_text(json.dumps({'state':'completed','attempted':len(schedule),'accepted':accepted,
        'training_only':True,'scope':'Small English arithmetic/data-transformation pilot, not broad reasoning coverage',
        'seconds':time.monotonic()-started,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
