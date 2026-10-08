"""Freeze a bounded, paired OPD/QAD diagnostic from E020-B-1024."""
import hashlib,json,random
from pathlib import Path
from ternary_core import ROOT,GPU_IDS
from train_v2 import write_json

RUN=ROOT/'runs/e021-opd'
def main():
    RUN.mkdir(exist_ok=False)
    rows=[json.loads(l) for l in (ROOT/'data/e020-natural-v1/train.jsonl').open()]
    tasks={r['id']:r for r in (json.loads(l) for l in (ROOT/'data/e020-natural-v1/tasks.jsonl').open())}
    selected=[]; rng=random.Random(20261002)
    families=json.loads((ROOT/'data/e020-natural-v1/config.json').read_text())['math_families'][:16]
    for family in families:
        pool=[r for r in rows if r['category']=='math' and r['topic']==family and len(r['input_ids'])<=4096]
        rng.shuffle(pool)
        if len(pool)<8:raise ValueError('Insufficient full-length reference rows: '+family)
        selected.extend(pool[:8])
    rng.shuffle(selected)
    evaluation=json.loads((ROOT/'reports/e020-ab/tasks.json').read_text())
    norm=lambda s:' '.join(s.lower().split())
    assert not ({norm(tasks[r['id']]['instruction']) for r in selected}&{norm(t['instruction']) for t in evaluation})
    # Each group uses 8 different questions, four independent completions/question.
    prepared=[]
    for i,row in enumerate(selected):
        task=tasks[row['id']]
        p=row['prompt_tokens'];assert all(y==-100 for y in row['labels'][:p])
        prepared.append(dict(index=i,update=i//8+1,task=task,reference=row,prompt_ids=row['input_ids'][:p]))
    raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared)
    (RUN/'questions.jsonl').write_text(raw)
    config=dict(experiment='E021 OPD vs paired ordinary QAD, bounded diagnostic',gpu_uuids=GPU_IDS,
                parent_latent=str(ROOT/'runs/e020-ab-b/candidates/step-1024'),
                parent_gguf=str(ROOT/'reports/e020-ab/B-1024/model.gguf'),
                packed_template=str(ROOT/'reports/e020-ab/B-1024/packed'),updates=16,prompts_per_update=8,
                completions_per_prompt=4,rollout_slots=8,max_new_tokens=4096,context_per_slot=8192,
                retained_updates=[4,8,16],learning_rate=1e-6,clip=.2,beta=1.,reward_weight=1.,
                sampler=dict(temperature=1.,top_k=0,top_p=1.,repeat_penalty=1.,include_eos=True),
                teacher='Local frozen FP8-source dequantized to BF16 reference compute, exclusively GPU6/7; no live teacher API calls',
                training='FP32 latent masters, BF16 ternary QAT compute, STE, all64layer activation checkpointing, head chunks64',
                opd_objective='Mean clipped importance-ratio policy loss with detached (teacher_logp-behavior_logp)+group-standardized exact final reward. One update per freshly sampled group; no silent removal of truncated/incorrect rollouts.',
                control_objective='.25 full-answer CE + .75 top128+tail teacher forward KL on the same prompts, four repeats each. Same16 optimizer updates, sample count and LR; generated tokens/wall time differ.',
                safety_reserve_gib=4,questions_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                evaluation_tasks_sha256=hashlib.sha256((ROOT/'reports/e020-ab/tasks.json').read_bytes()).hexdigest(),
                promotion='No replacement of E018/E020. Final paired136question generation review vs parent; training rewards are not benchmark scores.',
                parity_gate=dict(mean_abs_logp_max=.15,p95_abs_logp_max=.75),
                checkpoint_retention='Only4,8,16 per arm plus atomic working snapshot; original checkpoints untouched')
    write_json(RUN/'config.json',config)
    write_json(RUN/'status.json',dict(state='prepared',stage='Подготовлены 128 вопросов для парного OPD/QAD',update=0,total_updates=16))
    print('Prepared128 training prompts,16families;16updates,32trajectories/update; benchmark prompts excluded')
if __name__=='__main__':main()
