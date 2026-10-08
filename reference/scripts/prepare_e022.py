"""Freeze paired joint-scale QAT diagnostic, authorized physical GPUs2/6."""
import collections,hashlib,json,random,shutil,time
from pathlib import Path
from train_v2 import write_json
from ternary_core import ROOT

RUN=ROOT/'runs/e022-joint-scales'
GPUS=['GPU_UUID_REDACTED','GPU_UUID_REDACTED']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    if RUN.exists():raise RuntimeError('E022 already prepared; do not overwrite')
    data=ROOT/'data/e020-natural-v1';cache=ROOT/'data/e020-kd-v1/manifest.json'
    meta=json.loads(cache.read_text());assert meta['state']=='completed'
    assert sha(Path(meta['corpus']))==meta['corpus_sha256']
    train=[json.loads(l) for l in (data/'train.jsonl').open()]
    validation=[json.loads(l) for l in (data/'validation.jsonl').open()]
    rng=random.Random(20261007);selected=[];val=[]
    for category,count in [('math',384),('code',64),('instructions',32),('logic',32)]:
        pool=[r for r in train if r['category']==category and str(r['id']) in meta['examples'] and len(r['input_ids'])<=8192]
        rng.shuffle(pool);assert len(pool)>=count;selected.extend(pool[:count])
        pool=[r for r in validation if r['category']==category and len(r['input_ids'])<=8192]
        rng.shuffle(pool);n={'math':16,'code':8,'instructions':4,'logic':4}[category]
        assert len(pool)>=n;val.extend(pool[:n])
    rng.shuffle(selected);rng.shuffle(val)
    assert len(selected)==512 and len(val)==32
    tasks=json.loads((ROOT/'reports/e020-ab/tasks.json').read_text())
    source_tasks={r['id']:r for r in (json.loads(l) for l in (data/'tasks.jsonl').open())}
    norm=lambda x:' '.join(x.lower().split())
    selected_prompts={norm(source_tasks[r['id']]['instruction']) for r in selected}
    assert not selected_prompts & {norm(t['instruction']) for t in tasks}
    assert not {r['id'] for r in selected}&{r['id'] for r in val}
    RUN.mkdir();(RUN/'code').mkdir()
    for script in sorted((ROOT/'scripts').glob('*.py')):shutil.copy2(script,RUN/'code'/script.name)
    for name in ['start_e022.sh','check_e022.sh','stop_e022.sh']:shutil.copy2(ROOT/'scripts'/name,RUN/'code'/name)
    write_json(RUN/'code-manifest.json',{p.name:sha(p) for p in (RUN/'code').iterdir()})
    write_json(RUN/'train.json',selected);write_json(RUN/'validation.json',val)
    cfg=dict(experiment='E022 fixed scales vs joint FP32 latent/learned group scales',
             created=time.time(),gpu_indices=[2,6],gpu_uuids=GPUS,
             parent_latent=str(ROOT/'runs/e020-ab-b/candidates/step-1024'),
             parent_gguf=str(ROOT/'reports/e020-ab/B-1024/model.gguf'),
             packed_template=str(ROOT/'reports/e020-ab/B-1024/packed'),
             train_sha256=sha(RUN/'train.json'),validation_sha256=sha(RUN/'validation.json'),
             cache_manifest=str(cache),cache_manifest_sha256=sha(cache),
             tasks_file=str(ROOT/'reports/e020-ab/tasks.json'),tasks_sha256=sha(ROOT/'reports/e020-ab/tasks.json'),
             train_categories=dict(collections.Counter(r['category'] for r in selected)),
             training_tokens=sum(len(r['input_ids']) for r in selected),max_sequence_tokens=max(len(r['input_ids']) for r in selected),
             validation_examples=32,updates=128,samples_per_update=4,checkpoints=[64,128],
             weight_lr=1e-6,scale_lr=1e-4,max_abs_log_gain=.1,scale_gradient_normalization='1/sqrt(128)',
             objective='.25 answer CE + .75 top128+tail forward KL, all real next-token positions',
             forward='Fully ternary H128/g128 with BF16 group scales; same parent identity weight STE in both arms',
             scale_backward='LSQ-style q - (W/s)*I(abs(W/s)<1), chain through log gain; not exact ParetoQ SEQ',
             teacher='Existing frozen aligned cache from original local FP8 source with BF16 reference computation; no API requests',
             safety_reserve_gib=6,checkpoint_retention='Two checkpoints per arm64/128; no edits/deletion of previous experiments',
             evaluation='Same136 development tasks greedy temp0, thinking medium, output8192; both checkpoints per arm. No promotion by CE alone.',
             old_scale_pilot='E007 trained only scales with fixed trits, LR.001/.003, and regressed; E022 jointly updates weights with LR1e-6 and gains1e-4.')
    write_json(RUN/'config.json',cfg)
    write_json(RUN/'status.json',dict(state='prepared',stage='Подготовлены две ветки, 512 общих примеров',updated=time.time(),step=0,total=128))
    print(json.dumps({k:cfg[k] for k in ['gpu_indices','train_categories','training_tokens','max_sequence_tokens','updates']},ensure_ascii=False))
if __name__=='__main__':main()
