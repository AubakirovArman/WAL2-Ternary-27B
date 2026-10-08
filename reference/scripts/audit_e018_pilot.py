"""Recheck all saved pilot responses after correcting APPS singleton output wrappers."""
import collections,json,os,shutil,subprocess,time
from pathlib import Path
from collect_e018 import ROOT,OUT,validate

def main():
    old=json.loads((OUT/'status.json').read_text());assert old['state']=='pilot_completed'
    requests=[json.loads(l) for l in (OUT/'requests.jsonl').open()];assert len(requests)==100
    ids={r['id'] for r in requests};tasks={t['id']:t for l in (OUT/'tasks.jsonl').open() if (t:=json.loads(l))['id'] in ids}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false');encoder=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/e018_encode_server.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env);assert json.loads(encoder.stdout.readline())['ready']
    commits=[];events=[];counts=collections.Counter()
    try:
        for req in requests:
            t=tasks[req['id']];v=validate(t,req['result'])
            if v['accepted']:
                row=dict(t,reasoning=v['reasoning'],response=v['response'],verification='final_'+t['validator']+'_checked_reasoning_unverified');encoder.stdin.write(json.dumps(row)+'\n');encoder.stdin.flush();encoded=json.loads(encoder.stdout.readline())
                if encoded.get('item') is None:v=dict(accepted=False,reason='encoding_or_length')
                else:
                    item=encoded['item'];item.update(split=t['split'],category=t['category'],topic=t['topic'],verification=row['verification']);commits.append(dict(row=row,encoded=item))
            events.append(dict(id=t['id'],accepted=v['accepted'],reason=v.get('reason','accepted'),category=t['category'],split=t['split'],time=time.time(),audit='APPS_wrapper_fix'));counts[t['category'],v['accepted']]+=1
    finally:encoder.stdin.close();encoder.wait(timeout=30)
    backup=OUT/'pilot-initial-audit';backup.mkdir(exist_ok=False)
    for name,rows in [('committed.jsonl',commits),('events.jsonl',events)]:
        shutil.copyfile(OUT/name,backup/name);temp=OUT/(name+'.audited');temp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));temp.replace(OUT/name)
    shutil.copyfile(OUT/'status.json',backup/'status.json')
    summary=dict(accepted=len(commits),attempts=100,by_category={c:{str(ok):counts[c,ok] for ok in [True,False]} for c in {t['category'] for t in tasks.values()}},rejected=collections.Counter(e['reason'] for e in events if not e['accepted']),token_lengths=[len(x['encoded']['input_ids']) for x in commits],scope='Final task checks and encoding; no complete proof of reasoning correctness.')
    (OUT/'pilot-audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
