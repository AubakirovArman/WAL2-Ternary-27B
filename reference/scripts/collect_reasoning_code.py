"""Bounded teacher collection with precomputed code tests; train-only, one request at a time."""
import argparse,fcntl,hashlib,json,re,time,urllib.request
from pathlib import Path
from code_sandbox import run
from verified_code_tasks import tasks,test_program
ROOT=Path(__file__).resolve().parents[1]


def verify(row, choice):
    message=choice['message'];reasoning=message.get('reasoning_content') or '';answer=message.get('content') or ''
    if choice['finish_reason']!='stop' or not reasoning.strip():
        return {'passed':False,'error':'unfinished or missing reasoning'}
    if any(t in reasoning or t in answer for t in ('<think>','</think>','<|im_start|>','<|im_end|>','<|endoftext|>')):
        return {'passed':False,'error':'control token in content'}
    match=re.fullmatch(r'\s*```(?:python|py)\s*\n(.*?)```\s*',answer,re.S)
    if not match or '```' in match[1]:return {'passed':False,'error':'expected exactly one Python block'}
    return run(test_program(row,match[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=128);p.add_argument('--deadline-hours',type=float,default=3)
    a=p.parse_args()
    if not 0<a.deadline_hours<=8:raise ValueError('Invalid deadline')
    schedule=tasks(a.count)
    lock=(ROOT/'data/v2/teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'tasks.json').write_text(json.dumps(schedule,indent=2)+'\n')
    hashes={}
    for filename in ('collect_reasoning_code.py','verified_code_tasks.py','code_sandbox.py','code_sandbox_child.py'):
        raw=Path(__file__).with_name(filename).read_bytes();(a.output/filename).write_bytes(raw);hashes[filename]=hashlib.sha256(raw).hexdigest()
    (a.output/'config.json').write_text(json.dumps({'count':a.count,'deadline_hours':a.deadline_hours,'script_sha256':hashes,
        'scope':'Eight synthetic list/integer transformation families, English, training only; finite tests verify final code, not reasoning or universal correctness'},indent=2)+'\n')
    (a.output/'accepted.jsonl').touch();started=time.monotonic();accepted=0
    for row in schedule:
        if time.monotonic()-started>a.deadline_hours*3600:raise TimeoutError('Code collection deadline')
        body={'model':'Qwen/Qwen3.8-27B-FP8','messages':[{'role':'user','content':row['instruction']}],
              'temperature':0,'max_tokens':2048,'reasoning_effort':'medium','chat_template_kwargs':{'enable_thinking':True}}
        request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response:raw=json.load(response)
        with (a.output/'requests.jsonl').open('a') as f:f.write(json.dumps({'id':row['id'],'request':body,'response':raw})+'\n')
        choice=raw['choices'][0];verdict=verify(row,choice)
        event={'id':row['id'],'accepted':bool(verdict['passed']),'finish_reason':choice['finish_reason'],'test_result':verdict,'usage':raw.get('usage')}
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if verdict['passed']:
            message=choice['message'];saved={**row,'reasoning':message['reasoning_content'],'response':message['content'],
                'enable_thinking':True,'source':'teacher_verified_code_v1','verification':'final_code_passed_fixed_sandbox_tests_reasoning_unverified',
                'sha256':hashlib.sha256(row['instruction'].encode()).hexdigest(),'test_result':verdict}
            with (a.output/'accepted.jsonl').open('a') as f:f.write(json.dumps(saved)+'\n')
            accepted+=1
        print(json.dumps(event),flush=True);time.sleep(1)
    manifest={'state':'completed','attempted':len(schedule),'accepted':accepted,'training_only':True,
              'seconds':time.monotonic()-started,'accepted_sha256':hashlib.sha256((a.output/'accepted.jsonl').read_bytes()).hexdigest(),
              'tasks_sha256':hashlib.sha256((a.output/'tasks.json').read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
