"""Fixed source offline BF16-reference cache, raw IDs, no live-server changes."""
import concurrent.futures as cf,hashlib,importlib.util,json,os,time
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
import ternary_core as core
from cache_teacher_logits import collect
from train_v2 import write_json
ROOT=core.ROOT;DATA=ROOT/'data/e020-natural-v1';OUT=ROOT/'data/e020-kd-v1'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def validate(row,tensors,topk):
    n=len(row['input_ids'])-1
    assert tensors['input_ids'].tolist()==row['input_ids'] and tensors['labels'].tolist()==row['labels']
    assert tensors['top_ids'].shape==tensors['top_logp'].shape==(n,topk) and tensors['tail_prob'].shape==(n,)
    ids=tensors['top_ids'].long();vocab=json.loads((core.SOURCE/'config.json').read_text())['text_config']['vocab_size']
    assert bool(((ids>=0)&(ids<vocab)).all()) and bool((ids.sort(-1).values.diff(dim=-1)!=0).all())
    assert bool(torch.isfinite(tensors['top_logp']).all()) and bool(torch.isfinite(tensors['tail_prob']).all())
    assert bool(((tensors['tail_prob']>=0)&(tensors['tail_prob']<=1)).all())
    mass=tensors['top_logp'].float().exp().sum(-1)+tensors['tail_prob']
    assert bool((mass-1).abs().max()<.01)
    # collect() takes hidden[:-1]: this position predicts ids[t+1]; cache stores exact full input/labels.
    return dict(positions=n,answer_tokens=sum(v!=-100 for v in row['labels'][1:]),input_sha256=hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest(),
                labels_sha256=hashlib.sha256(json.dumps(row['labels']).encode()).hexdigest(),mean_tail_probability=float(tensors['tail_prob'].mean()))
def main():
    if importlib.util.find_spec('fla') is not None or 'fla-probe-packages' in os.environ.get('PYTHONPATH',''):
        raise RuntimeError('Offline teacher cache must use the original Transformers reference forward, without FLA on PYTHONPATH')
    dm=json.loads((DATA/'manifest.json').read_text());assert dm['state']=='completed'
    corpus=DATA/'train.jsonl';assert sha(corpus)==dm['files']['train']['sha256']
    rows=[json.loads(l) for l in corpus.open()];old_path=ROOT/'data/v3-kd/manifest.json';old=json.loads(old_path.read_text())
    assert Path(old['source']).resolve()==core.SOURCE.resolve() and old['temperature']==1 and old['topk']==128
    OUT.mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True)
    source_files=sorted(core.SOURCE.glob('*.safetensors'))+[core.SOURCE/n for n in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json','merges.txt')]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:hashes=dict(zip((p.name for p in source_files),pool.map(sha,source_files)))
    provenance=dict(source=str(core.SOURCE),source_file_sha256=hashes,source_format='block128 FP8 e4m3 + stored weight_scale_inv multiplier',
                    compute='Existing core.load_source: FP32 multiply source block scales then BF16 weights, SDPA, two GPUs split32/32; original BF16 pre-quantization weights unknown',
                    collector_sha256=sha(ROOT/'scripts/cache_teacher_logits.py'),loader_sha256=sha(ROOT/'scripts/ternary_core.py'),
                    temperature=1,topk=128,position_shift='hidden[t] predicts input_ids[t+1], all L-1 including prompt',
                    old_historical_source_hashes='Not recorded in original old manifest; current file hashes plus regenerated old-cache sentinel agreement, not retroactive proof of every old entry',torch_version=torch.__version__)
    meta_path=OUT/'manifest.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text());assert meta['corpus_sha256']==sha(corpus) and meta['provenance']==provenance
        if meta['state']=='completed':print('Кэш уже готов.');return
        if meta.get('examples') and meta.get('reference_backend')!='Transformers reference gated delta, FLA unavailable':
            raise RuntimeError('Cannot resume a mixed-backend teacher cache')
    else:meta=dict(state='collecting',source=str(core.SOURCE),corpus=str(corpus),corpus_sha256=sha(corpus),topk=128,temperature=1.,count_target=len(rows),
                   old_cache_manifest_sha256=sha(old_path),provenance=provenance,examples={})
    meta['reference_backend']='Transformers reference gated delta, FLA unavailable'
    write_json(meta_path,meta);core.guard();model=core.load_source();model.eval();start=time.monotonic()
    # Verify existing old-cache teacher identity before new collection. Do not modify old files.
    old_rows={str(r['id']):r for r in core.load_examples(Path(old['corpus']),512)['train']}
    checks=[]
    for identity in old['order_ids'][:4]:
        row=old_rows[str(identity)];actual=collect(model,row,128);saved=load_file(old['examples'][str(identity)]['file'])
        same=bool(torch.equal(actual['top_ids'],saved['top_ids']));delta=float((actual['top_logp'].float()-saved['top_logp'].float()).abs().max());td=float((actual['tail_prob']-saved['tail_prob']).abs().max())
        checks.append(dict(id=identity,top_ids_exact=same,max_logp_difference=delta,max_tail_difference=td))
        if not same or delta>1e-3 or td>1e-5:raise RuntimeError('Old-cache teacher sentinel mismatch; fresh cache not accepted')
    meta['old_cache_sentinel_checks']=checks;write_json(meta_path,meta)
    for i,row in enumerate(rows):
        identity=str(row['id']);path=OUT/'cache'/(identity+'.safetensors')
        if identity in meta['examples']:
            assert sha(path)==meta['examples'][identity]['file_sha256'];validate(row,load_file(str(path)),128);continue
        tensors=collect(model,row,128);info=validate(row,tensors,128)
        tmp=path.with_suffix('.writing');save_file(tensors,str(tmp));tmp.replace(path)
        meta['examples'][identity]=dict(**info,file=str(path),file_sha256=sha(path))
        meta['cached']=len(meta['examples']);meta['updated']=time.time();write_json(meta_path,meta)
        if i%16==0 or i+1==len(rows):print(f'Кэш учителя: {i+1}/{len(rows)}; {time.monotonic()-start:.0f} сек',flush=True)
    meta.update(state='completed',cached=len(rows),seconds=time.monotonic()-start);write_json(meta_path,meta)
    print('Свежий top128+tail ID-кэш проверен и готов.',flush=True)
if __name__=='__main__':main()
