"""Recover correctly solved graph holdout answers rejected only for a JSON wrapper.

Use immutable saved teacher responses; require one JSON fence and exact oracle equality.
No new teacher requests and no relaxation of mathematical or graph correctness.
"""
import hashlib,json,os,random,re,subprocess
from pathlib import Path
from e018_tasks import verify_task
from collect_e020_data import OUT,append,write
ROOT=Path(__file__).resolve().parents[1]
FENCE=re.compile(r'```(?:json)?\s*\n(.*?)```',re.S|re.I)

def main():
    if (OUT/'manifest.json').exists():raise RuntimeError('Corpus already frozen; recovery must not alter it')
    pool={r['id']:r for r in (json.loads(l) for l in (OUT/'tasks.jsonl').open())}
    original={r['id']:r for r in (json.loads(l) for l in (OUT/'events.jsonl').open())}
    committed={r['row']['id'] for r in (json.loads(l) for l in (OUT/'committed.jsonl').open())}
    assert len(committed)==1253 and len([r for r in committed if pool[r]['split']=='validation' and pool[r]['category']=='logic'])==5
    requests={r['id']:r for r in (json.loads(l) for l in (OUT/'requests.jsonl').open())}
    candidates=[];reasons={}
    for identity,event in original.items():
        if event['split']!='validation' or event['category']!='logic' or event['accepted'] or identity in committed:continue
        task=pool[identity];raw=requests[identity]['result']['raw'];choice=raw['choices'][0]
        if choice['finish_reason']!='stop':continue
        msg=choice['message'];full=msg.get('content') or '';reason=msg.get('reasoning_content') or ''
        blocks=list(FENCE.finditer(full))
        if len(blocks)!=1 or not reason.strip():continue
        b=blocks[0];answer=b.group(1).strip();prose=(full[:b.start()]+'\n'+full[b.end():]).strip()
        try:
            parsed=json.loads(answer)
            if not isinstance(parsed,dict) or set(parsed)!={'reachable','minimum_hops','reachable_count'}:continue
            if not verify_task(task,answer):continue
        except Exception:continue
        final_reason=reason.rstrip()+(('\n\n'+prose) if prose else '')
        if any(marker in final_reason or marker in answer for marker in ('<think>','</think>','<|im_start|>','<|im_end|>','<|endoftext|>')):continue
        candidates.append((identity,task,final_reason,answer,prose,full))
    if len(candidates)<27:raise RuntimeError(f'Only {len(candidates)} valid saved graph solutions; need27')
    random.Random(2001024).shuffle(candidates)
    chosen=candidates[:27]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    encoder=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/e018_encode_server.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
    assert json.loads(encoder.stdout.readline()).get('ready')
    accepted=[]
    try:
        for identity,task,reason,answer,prose,full in chosen:
            row=dict(task,reasoning=reason,response=answer,verification='exact_final_graph_after_unique_json_fence_normalization',
                     validation_metadata=dict(recovery='Single fenced JSON extracted from stored teacher content; surrounding teacher prose moved verbatim to reasoning',
                                              raw_teacher_content_sha256=hashlib.sha256(full.encode()).hexdigest(),prior_event=original[identity],
                                              exact_oracle_checked=True,external_teacher_requests=0))
            encoder.stdin.write(json.dumps(row)+'\n');encoder.stdin.flush();encoded=json.loads(encoder.stdout.readline())
            if not encoded.get('item'):raise RuntimeError('Lossless tokenizer rejected recovered row '+identity+': '+str(encoded.get('error')))
            item=encoded['item'];item.update(split='validation',category='logic',topic=task['topic'],verification=row['verification'])
            append('committed.jsonl',dict(row=row,encoded=item))
            append('recovery-events.jsonl',dict(id=identity,accepted=True,source_request_sha256=hashlib.sha256(json.dumps(requests[identity],ensure_ascii=False).encode()).hexdigest(),
                                                prior_reason=original[identity]['reason'],exact_oracle_checked=True,format_only=True,sequence_tokens=len(item['input_ids'])))
            accepted.append(identity)
    finally:encoder.stdin.close();encoder.wait(timeout=30)
    report=dict(state='completed',recovered=len(accepted),needed=27,eligible_saved_responses=len(candidates),chosen_ids=accepted,
                selection='Seed2001024 random sample from all exact-correct saved rejected graph responses; before generating more tasks or modifying model',
                constraints='Unique fenced JSON, valid graph schema, exact regenerated oracle, finish_reason stop, nonempty reasoning, full sequence accepted without truncation; teacher prose preserved',
                original_data_commits=1253,original_event_log_untouched=True,additional_teacher_requests=0)
    write('recovery-report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
