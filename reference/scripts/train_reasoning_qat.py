"""Full ternary QAT: thinking CE plus original non-thinking KD.

Separate branch from a completed run's selected latent state.
Validation selects an experimental candidate, not a claim of broad parity.
"""
import argparse,inspect,functools,gc,hashlib,json,shutil,time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import ternary_core as core
from train_v2 import restore,save_latent,write_json
from train_kd import load_cache
from fast_ternary import enable_fused_quantizer,gradients_finite_by_device
from chunked_kd import loss_from_hidden
from inspect_checkpoint import load_packed
from fast_validation import validation_runtime


def resolve_parent(run,candidate="selected"):
    """Resolve both representations together so fallback cannot select stale weights."""
    path=run/'metrics.json';raw=path.read_bytes();metrics=json.loads(raw)
    if metrics.get('state')!='completed':raise ValueError('Parent run must be completed')
    if candidate not in ('selected','reasoning_candidate'):raise ValueError('Invalid parent candidate')
    metrics=dict(metrics)
    for suffix in ('latent','packed','validation','reasoning'):
        key=candidate+'_'+suffix
        if key in metrics:metrics['selected_'+suffix]=metrics[key]
        elif suffix!='reasoning':raise ValueError('Missing parent candidate field: '+key)
    latent=Path(metrics['selected_latent']).resolve()
    packed=Path(metrics['selected_packed']).resolve()
    for required in (latent/'manifest.json',latent/'training_state.pt',packed/'manifest.json'):
        if not required.is_file():raise ValueError('Missing parent artifact: '+str(required))
    expected=metrics['selected_validation']['answer_ce']
    if not isinstance(expected,(int,float)) or not 0<=expected<float('inf'):
        raise ValueError('Invalid parent validation CE')
    return latent,packed,metrics,hashlib.sha256(raw).hexdigest()


@functools.lru_cache(maxsize=1)
def closing_token_id():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(core.SOURCE,local_files_only=True)
    encoded=tok.encode('</think>',add_special_tokens=False)
    if len(encoded)!=1:raise ValueError('Unexpected thinking delimiter tokenization')
    return encoded[0]


def selection_status(metric,old_ce,verified_ce,reasoning_ce,baseline_old,baseline_verified,baseline_reasoning):
    if metric=='old':
        return (reasoning_ce<=baseline_reasoning*1.01 and verified_ce<=baseline_verified*1.01),old_ce
    if metric in ('reasoning','math'):
        return (old_ce<=baseline_old*1.01 and verified_ce<=baseline_verified*1.01),reasoning_ce
    raise ValueError('Unknown selection metric')


def closing_metrics(head,hidden,labels,token_id):
    mask=labels==token_id
    if int(mask.sum())!=1:raise ValueError('Expected one supervised closing delimiter')
    logits=head(hidden[mask]).float();targets=labels[mask]
    return float(F.cross_entropy(logits,targets,reduction='sum')),int((logits.argmax(-1)==targets).sum())


def chunked_ce(head,hidden,labels,chunk=64):
    count=int((labels!=-100).sum())
    if count<1 or len(hidden)!=len(labels) or chunk<1:raise ValueError('Invalid CE inputs')
    def part(h,y):return F.cross_entropy(head(h).float(),y,ignore_index=-100,reduction='sum')/count
    loss=hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,len(hidden),chunk):
        args=(hidden[start:start+chunk],labels[start:start+chunk].to(hidden.device))
        loss=loss+(checkpoint(part,*args,use_reentrant=False) if torch.is_grad_enabled() else part(*args))
    return loss,count


def reasoning_loss(model,row,label_key='labels',chunk_tokens=64):
    ids=torch.tensor([row['input_ids']],device='cuda:0')
    hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
    return chunked_ce(model.lm_head,hidden,torch.tensor(row[label_key][1:],device=hidden.device),chunk=chunk_tokens)


@torch.no_grad()
def reasoning_eval(model,rows):
    model.eval();total=0.;tokens=0;closing_ce=0.;closing_correct=0
    categories={};families={};content={}
    close_id=closing_token_id()
    for row in rows:
        ids=torch.tensor([row['input_ids']],device='cuda:0')
        hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
        labels=torch.tensor(row['labels'][1:],device=hidden.device)
        if row.get('diagnostic_masks'):
            from reasoning_diagnostics import diagnostic_ce
            loss,n,parts=diagnostic_ce(model.lm_head,hidden,labels,{k:v[1:] for k,v in row['diagnostic_masks'].items()})
            for key,part in parts.items():
                group=content.setdefault(key,{'nll':0.,'tokens':0});group['nll']+=part['nll'];group['tokens']+=part['tokens']
        else:loss,n=chunked_ce(model.lm_head,hidden,labels)
        ce_value=float(loss);total+=ce_value*n;tokens+=n
        ce,correct=closing_metrics(model.lm_head,hidden,labels,close_id)
        closing_ce+=ce;closing_correct+=correct
        category=row.get('category','unspecified')
        if category=='math_repair':category='math'
        family=category+'/'+str(row.get('topic',category))
        for groups,key in ((categories,category),(families,family)):
            group=groups.setdefault(key,{'loss_sum':0.,'tokens':0,'example_ce_sum':0.,'examples':0})
            group['loss_sum']+=ce_value*n;group['tokens']+=n
            group['example_ce_sum']+=ce_value;group['examples']+=1
    def finish(groups):
        return {key:{'reasoning_and_answer_ce':g['loss_sum']/g['tokens'],
                     'example_mean_ce':g['example_ce_sum']/g['examples'],
                     'tokens':g['tokens'],'examples':g['examples']} for key,g in groups.items()}
    return {'reasoning_and_answer_ce':total/tokens,'tokens':tokens,'examples':len(rows),
            'by_category':finish(categories),'by_family':finish(families),
            'content_diagnostics':{k:{**g,'ce':g['nll']/g['tokens'] if g['tokens'] else None} for k,g in content.items()},
            'content_diagnostic_scope':'Optional non-overlapping token masks; formula/prose separation is heuristic, not proof verification; full input context retained',
            'closing_teacher_forced_ce':closing_ce/len(rows),
            'closing_teacher_forced_correct':closing_correct,
            'closing_metric_scope':'Gold-prefix next-token prediction, not free-generation closure success'}


def rank_candidates(existing,candidate,limit):
    return sorted(existing+[candidate],key=lambda r:(r['score'],-r['step']))[:limit]


def thinking_score(metric,thinking):
    if metric=='math':
        group=thinking.get('by_category',{}).get('math')
        if not group or not group['examples']:raise ValueError('Math selection needs a nonempty labelled math validation group')
        return group['reasoning_and_answer_ce']
    return thinking['reasoning_and_answer_ce']


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent-run',type=Path,default=core.ROOT/'runs/e006-soft-kd')
    p.add_argument('--parent-candidate',choices=['selected','reasoning_candidate'],default='selected')
    p.add_argument('--fla-training',action='store_true')
    p.add_argument('--reasoning-weight',type=float,default=.5)
    p.add_argument('--ab-branch',choices=['A','B'],help='E020 controlled old/fresh-KL pilot, fixed objective coefficients')
    p.add_argument('--fresh-kd-manifest',type=Path)
    p.add_argument('--preflight-only',action='store_true',help='Check full objective memory/backward and parent identity without optimizer updates')
    p.add_argument('--resume-step',type=int,help='Resume an interrupted E020 pilot from its last committed validation checkpoint')
    p.add_argument('--defer-export',action='store_true',help='Preserved pilot checkpoints exported by subsequent CPU generation review, without redundant final CE loops')
    p.add_argument('--loss-chunk-tokens',type=int,choices=[64,256,512],default=64,help='Training-only vocabulary loss chunk; 256/512 experimentally faster, validation stays64')
    p.add_argument('--checkpoint-policy',choices=['all','half','quarter','adaptive'],default='all',help='Activation recomputation: all64layers(default), odd32layers, or every fourth16layers. Fewer checkpoints require more VRAM; long-context preflight remains mandatory.')
    p.add_argument('--max-length',type=int,default=2048,help='Reject longer data; never truncate')
    p.add_argument('--save-final-candidate',action='store_true')
    p.add_argument('--save-every-validation',action='store_true')
    p.add_argument('--preserve-validation-candidates',action='store_true',help='Bounded pilots: save every validation candidate for later generation review; do not prune by CE')
    p.add_argument('--keep-best',type=int,default=0)
    p.add_argument('--max-hours',type=float,default=6)
    p.add_argument('--steps',type=int,default=256);p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--selection-metric',choices=['reasoning','math','old'],default='reasoning')
    p.add_argument('--validation-every',type=int,default=64)
    p.add_argument('--order-offset',type=int,default=0)
    p.add_argument('--kd-order-offset',type=int,help='Replay offset; defaults to --order-offset')
    p.add_argument('--fast-validation',action='store_true',help='Temporary bounded weight cache and fused Hadamard during validation only')
    p.add_argument('--validation-cache-gib',type=float,default=8.0)
    p.add_argument('--target-old-ce',type=float)
    p.add_argument('--target-reasoning-ce',type=float)
    a=p.parse_args()
    if a.preserve_validation_candidates:
        if a.keep_best:raise ValueError('Pending generation review cannot be combined with CE top-k pruning')
        a.save_every_validation=True
    if not 2<=a.max_length<=12288:raise ValueError('Invalid sequence length limit')
    if a.keep_best<0 or not 0<a.max_hours<=72:raise ValueError('Invalid retention or deadline')
    if a.keep_best and (a.save_every_validation or a.save_final_candidate):raise ValueError('Top-k mode cannot retain extra candidates')
    if not 0<a.reasoning_weight<1:raise ValueError('Invalid mixture weight')
    fresh_meta=None
    if a.ab_branch:
        if a.reasoning_weight!=.25 or not a.fresh_kd_manifest or not a.preserve_validation_candidates:
            raise ValueError('A/B requires fixed .25 CE, frozen fresh cache and preservation')
        fresh_meta=json.loads(a.fresh_kd_manifest.read_text())
        if fresh_meta['state']!='completed' or fresh_meta['topk']!=128 or fresh_meta['temperature']!=1.:
            raise ValueError('Incomplete or incompatible fresh cache')
        from e020_kd import loss_from_hidden as ab_loss
        weights={'ce_new':.25,'ce_old':.075,'kl_old':.675 if a.ab_branch=='A' else .3375,'kl_new':0. if a.ab_branch=='A' else .3375}
    elif a.fresh_kd_manifest or a.preflight_only or a.defer_export:
        raise ValueError('Pilot-only options require an explicit A/B branch')
    def training_kernel(enabled):pass
    if a.fla_training:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_kernels
        reference_chunk=inspect.unwrap(qwen_kernels.torch_chunk_gated_delta_rule)
        def training_kernel(enabled):
            qwen_kernels.torch_chunk_gated_delta_rule=fla_chunk if enabled else reference_chunk
        training_kernel(False)
    if a.steps<1 or not 0<a.lr<float('inf'):raise ValueError('Invalid steps or learning rate')
    if a.validation_every<1:raise ValueError('Invalid validation interval')
    if a.order_offset<0:raise ValueError('Negative order offset')
    kd_offset=a.order_offset if a.kd_order_offset is None else a.kd_order_offset
    if kd_offset<0:raise ValueError('Negative KD order offset')
    if a.target_old_ce is not None and not 0<a.target_old_ce<float('inf'):raise ValueError('Invalid CE target')
    if a.target_reasoning_ce is not None and not 0<a.target_reasoning_ce<float('inf'):raise ValueError('Invalid reasoning CE target')
    if a.validation_cache_gib<0:raise ValueError('Negative validation cache budget')
    dm=json.loads((a.data/'manifest.json').read_text());assert dm['state']=='completed'
    datasets={}
    for split in ('train','validation'):
        raw=(a.data/f'{split}.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==dm['files'][split]['sha256']
        datasets[split]=[json.loads(s) for s in raw.splitlines()]
        assert len(datasets[split])==dm['files'][split]['count']
        assert all(r['split']==split and 2<=len(r['input_ids'])<=a.max_length for r in datasets[split])
    if len(datasets['train'])<64 or len(datasets['validation'])<16:raise ValueError('Insufficient complete reasoning data')
    assert not ({r['sha256'] for r in datasets['train']} & {r['sha256'] for r in datasets['validation']})
    meta=json.loads((core.ROOT/'data/v3-kd/manifest.json').read_text());assert meta['state']=='completed'
    assert hashlib.sha256(Path(meta['corpus']).read_bytes()).hexdigest()==meta['corpus_sha256']
    if fresh_meta:
        assert fresh_meta['corpus_sha256']==dm['files']['train']['sha256']
        assert fresh_meta['old_cache_manifest_sha256']==hashlib.sha256((core.ROOT/'data/v3-kd/manifest.json').read_bytes()).hexdigest()
        assert set(fresh_meta['examples'])=={str(r['id']) for r in datasets['train']}
    parent,parent_packed,parent_metrics,parent_metrics_sha=resolve_parent(a.parent_run,a.parent_candidate)
    resume=None
    if a.resume_step is not None:
        from e020_resume import read_resume,verify_resume_config,commit_resume
        resume=read_resume(a.output,a.resume_step,a)
    core.guard();a.output.mkdir(parents=True,exist_ok=bool(resume))
    if not resume:(a.output/'code').mkdir()
    def status_ru(message):
        with (a.output/'status-ru.log').open('a') as stream:
            stream.write(time.strftime('%H:%M:%S')+' | '+message+'\n')
    status_ru('Загрузка модели E009 и состояния обучения' if 'e009' in str(a.parent_run) else 'Загрузка модели и состояния обучения')
    if not resume:
        hashes={}
        for source in sorted((core.ROOT/'scripts').glob('*.py')):
            target=a.output/'code'/source.name;shutil.copy2(source,target);hashes[source.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        write_json(a.output/'code-manifest.json',hashes);write_json(a.output/'reasoning-manifest.json',dm)
    elif json.loads((a.output/'reasoning-manifest.json').read_text())!=dm:
        raise ValueError('Resume dataset manifest changed')
    old=core.load_examples(Path(meta['corpus']),512);lookup={r['id']:r for r in old['train']};order=[lookup[i] for i in meta['order_ids']]
    verified=core.load_examples(core.ROOT/'data/v2/verified_holdout.jsonl',512)['validation']
    config={'parent':str(parent),'steps':a.steps,'lr':a.lr,'objective':f'{a.reasoning_weight} thinking CE + {1-a.reasoning_weight} original top128+tail KD', 'reasoning_weight':a.reasoning_weight,'loss_chunk_tokens':a.loss_chunk_tokens,'fla_training':a.fla_training,'save_final_candidate':a.save_final_candidate,'save_every_validation':a.save_every_validation,
            'keep_best':a.keep_best,'max_hours':a.max_hours,'checkpoint_policy':a.checkpoint_policy,'adaptive_half_max_tokens':4096,'max_length':a.max_length,'parent_run':str(a.parent_run.resolve()),'parent_candidate':a.parent_candidate,'parent_packed':str(parent_packed),'parent_metrics_sha256':parent_metrics_sha,
            'fast_validation':a.fast_validation,'validation_cache_gib':a.validation_cache_gib,'reasoning_data':str(a.data.resolve()),'selection_metric':a.selection_metric,
            'validation_every':a.validation_every,'order_offset':a.order_offset,'kd_order_offset':kd_offset,'target_old_ce':a.target_old_ce,
            'target_reasoning_ce':a.target_reasoning_ce,
            'selection':'Lower chosen validation CE; other validation metrics must remain within 1% of parent',
            'additional_candidate':'Also retain best thinking-validation candidate without the old-CE gate for matched generation evaluation',
            'generation_review_policy':'Preserve every validation checkpoint until generation review; selected export is provisional by CE' if a.preserve_validation_candidates else 'Configured legacy CE selection and retention',
            'scope':'Experimental full QAT branch; existing best models preserved; generation evaluation still required'}
    if a.keep_best:
        config['additional_candidate']='Disabled: only top-k eligible checkpoints beating parent validation retained'
    if not resume:write_json(a.output/'config.json',config)
    if a.ab_branch:
        config.update(ab_branch=a.ab_branch,weights=weights,
                      objective='Explicit CE_new + CE_old + KL_old + KL_new',
                      fresh_cache_manifest=str(a.fresh_kd_manifest.resolve()),
                      fresh_cache_manifest_sha256=hashlib.sha256(a.fresh_kd_manifest.read_bytes()).hexdigest(),
                      teacher_provenance=fresh_meta['provenance'],
                      parent_training_state_sha256=hashlib.sha256((parent/'training_state.pt').read_bytes()).hexdigest(),
                      kl_mask='All L-1 next-token positions; prompt/answer diagnostics separately, never per-bucket mean')
        if not resume:write_json(a.output/'config.json',config)
    model=core.load_source();core.ternarize(model,rotate=True);enable_fused_quantizer();opt=restore(model,resume['checkpoint'] if resume else parent);core.set_alpha(model,1.)
    for group in opt.param_groups:group['lr']=a.lr
    initial_cpu_rng=torch.get_rng_state();initial_gpu_rng=torch.cuda.get_rng_state_all()
    config['initial_rng_sha256']=resume['config']['initial_rng_sha256'] if resume else hashlib.sha256(initial_cpu_rng.numpy().tobytes()+b''.join(x.cpu().numpy().tobytes() for x in initial_gpu_rng)).hexdigest()
    if not resume:write_json(a.output/'config.json',config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    def configure_checkpointing(tokens):
        policy=('half' if tokens<=4096 else 'all') if a.checkpoint_policy=='adaptive' else a.checkpoint_policy
        for index,layer in enumerate(model.model.layers):
            layer.gradient_checkpointing=(index%2==1) if policy=='half' else (index%4==3) if policy=='quarter' else True
    def ab_forward(row,kind):
        ids=torch.tensor([row['input_ids']],device='cuda:0')
        hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
        labels=torch.tensor(row['labels'][1:],device=hidden.device)
        if kind=='new':
            cache=load_cache(row,fresh_meta) if weights['kl_new'] else None
            return ab_loss(model.lm_head,hidden,labels,cache,weights['ce_new'],weights['kl_new'],a.loss_chunk_tokens)
        return ab_loss(model.lm_head,hidden,labels,load_cache(row,meta),weights['ce_old'],weights['kl_old'],a.loss_chunk_tokens)
    # Small fixed-position gradient panel; labelled as a sample, never a full norm.
    gradient_panel=[]
    if a.ab_branch:
        choices=[(n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim==2]
        for index in (0,len(choices)//2,len(choices)-1):
            name,param=choices[index];generator=torch.Generator(device='cpu');generator.manual_seed(2001024+index)
            positions=torch.randint(param.numel(),(2048,),generator=generator).to(param.device)
            gradient_panel.append((name,param,positions))
        config['gradient_panel']=[{'name':n,'positions':2048,'scope':'Fixed uniform sampled entries; norms of weighted new and replay gradients, not full-matrix or full-model norms'} for n,p,ix in gradient_panel]
        if not resume:write_json(a.output/'config.json',config)
    expected_order=[{'reasoning':datasets['train'][(i+a.order_offset)%len(datasets['train'])]['id'],'kd':order[(i+kd_offset)%len(order)]['id']} for i in range(a.steps)]
    if resume:verify_resume_config(resume,config,expected_order)
    def panel_values():
        return {name:p.grad.detach().view(-1)[ix].float().clone() for name,p,ix in gradient_panel if p.grad is not None}
    smoke_rows=[max(datasets['train'],key=lambda r:len(r['input_ids']))]
    if a.checkpoint_policy=='adaptive':
        short=[r for r in datasets['train'] if len(r['input_ids'])<=4096]
        if short and len(smoke_rows[0]['input_ids'])>4096:smoke_rows.append(max(short,key=lambda r:len(r['input_ids'])))
    preflights=[]
    for longest in smoke_rows:
        status_ru(f"Проверка памяти и градиентов: {len(longest['input_ids'])} токенов…")
        # Exercise the worst available lengths before changing any weights.
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
        training_kernel(a.fla_training);model.train();opt.zero_grad(set_to_none=True)
        for device in range(2):torch.cuda.reset_peak_memory_stats(device)
        smoke_start=time.monotonic()
        configure_checkpointing(len(longest['input_ids']))
        if a.ab_branch:
            smoke_ce,new_stats=ab_forward(longest,'new')
            if not torch.isfinite(smoke_ce):raise RuntimeError('Nonfinite full new objective')
            smoke_ce.backward();smoke_ce_value=new_stats['ce'];del smoke_ce
        else:
            smoke_ce,_=reasoning_loss(model,longest,chunk_tokens=a.loss_chunk_tokens)
            if not torch.isfinite(smoke_ce):raise RuntimeError('Nonfinite long-context preflight CE')
            (a.reasoning_weight*smoke_ce).backward();smoke_ce_value=float(smoke_ce.detach());del smoke_ce
        replay=max(order,key=lambda r:len(r['input_ids']));cache=load_cache(replay,meta)
        if a.ab_branch:
            smoke_kd,old_stats=ab_forward(replay,'old');h=smoke_kl=smoke_oldce=None
            if not torch.isfinite(smoke_kd):raise RuntimeError('Nonfinite old objective')
            smoke_kd.backward()
        else:
            h=model.model(input_ids=torch.tensor([replay['input_ids']],device='cuda:0'),use_cache=False).last_hidden_state[0,:-1]
            smoke_kd,smoke_kl,smoke_oldce=loss_from_hidden(model.lm_head,h,torch.tensor(replay['labels'][1:],device=h.device),cache,chunk_tokens=a.loss_chunk_tokens)
            if not torch.isfinite(smoke_kd):raise RuntimeError('Nonfinite preflight KD')
            ((1-a.reasoning_weight)*smoke_kd).backward()
        if not gradients_finite_by_device(model.parameters()):raise RuntimeError('Nonfinite long-context preflight gradients')
        for device in range(2):torch.cuda.synchronize(device)
        preflight={'reasoning_id':longest['id'],'reasoning_tokens':len(longest['input_ids']),
                   'replay_id':replay['id'],'replay_tokens':len(replay['input_ids']),
                   'reasoning_ce':smoke_ce_value,'replay_loss':float(smoke_kd.detach()),
                   'seconds':time.monotonic()-smoke_start,'optimizer_updates':0,
                   'peak_allocated_gib':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],
                   'peak_reserved_gib':[torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)]}
        if a.ab_branch:preflight.update(new_objective=new_stats,old_objective=old_stats,weights=weights)
        del h,cache,smoke_kd,smoke_kl,smoke_oldce
        opt.zero_grad(set_to_none=True);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
        preflights.append(preflight);write_json(a.output/'gpu-preflight.json',{'cases':preflights,'passed':len(preflights)==len(smoke_rows)});core.event('reasoning_qat_preflight_passed',**preflight)
    training_kernel(False);opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    status_ru('Проверка памяти и градиентов пройдена; веса ещё не изменялись.')
    status_ru('Начальная проверка перед обучением: 1500 новых примеров и старые контрольные наборы' if len(datasets['validation'])==1500 else 'Начальная проверка перед обучением')
    def validate_all():
        training_kernel(False)
        start_validation=time.monotonic()
        with torch.no_grad(),validation_runtime(model,a.fast_validation,a.validation_cache_gib) as cache_info:
            values=(core.evaluate(model,old['validation']),core.evaluate(model,verified),reasoning_eval(model,datasets['validation']))
        core.event('validation_runtime',seconds=time.monotonic()-start_validation,cache=cache_info)
        return values
    baseline,secondary,thinking=validate_all()
    if a.fast_validation and not resume:
        status_ru('Однократная сверка ускоренной проверки с обычной…')
        reference=(core.evaluate(model,old['validation']),core.evaluate(model,verified),reasoning_eval(model,datasets['validation']))
        accelerated=(baseline,secondary,thinking)
        for actual,expected,key in zip(accelerated,reference,('answer_ce','answer_ce','reasoning_and_answer_ce')):
            if abs(actual[key]-expected[key])>1e-5:raise RuntimeError('Fast validation changed CE: '+key)
        if thinking['closing_teacher_forced_correct']!=reference[2]['closing_teacher_forced_correct']:
            raise RuntimeError('Fast validation changed closing-token predictions')
        write_json(a.output/'fast-validation-parity.json',{'passed':True,'reference':reference,'accelerated':accelerated,'ce_tolerance':1e-5})
        status_ru('Ускоренная проверка совпала с обычной; можно обучать.')
    # Evaluate the restored weights on the same old validation corpus before any update.
    if resume:
        actual=(baseline,secondary,thinking)
        saved=resume['metadata']
        for value,expected,key in zip(actual,(saved['old'],saved['verified'],saved['reasoning']),('answer_ce','answer_ce','reasoning_and_answer_ce')):
            if abs(value[key]-expected[key])>1e-5:raise RuntimeError('Resume checkpoint validation identity failed: '+key)
        if thinking['closing_teacher_forced_correct']!=saved['reasoning']['closing_teacher_forced_correct']:
            raise RuntimeError('Resume checkpoint closing-token identity failed')
        baseline=resume['result']['baseline'];secondary=resume['result']['baseline_verified'];thinking=resume['result']['baseline_reasoning']
        status_ru(f'Точка {a.resume_step} восстановлена; проверка совпала. Оптимизатор и RNG восстановлены.')
    elif abs(baseline['answer_ce']-parent_metrics['selected_validation']['answer_ce'])>1e-5:
        raise RuntimeError('Restored parent validation identity check failed')
    if a.preflight_only:
        write_json(a.output/'metrics.json',{'state':'preflight_passed','optimizer_updates':0,'baseline':baseline,'baseline_reasoning':thinking})
        status_ru('Проверки пройдены. Обновлений весов: 0.');return
    torch.set_rng_state(initial_cpu_rng);torch.cuda.set_rng_state_all(initial_gpu_rng)
    result=resume['result'] if resume else {'state':'training','baseline':baseline,'baseline_verified':secondary,'baseline_reasoning':thinking,'checks':[]}
    write_json(a.output/'metrics.json',result)
    status_ru(f"Начальная ошибка рассуждений: {thinking['reasoning_and_answer_ce']:.6f}. Меньше — лучше.")
    best=baseline['answer_ce'] if a.selection_metric=='old' else thinking_score(a.selection_metric,thinking);selected=None
    best_reasoning=thinking['reasoning_and_answer_ce'];reasoning_candidate=None;final_candidate=None
    if not resume:
        (a.output/'candidates').mkdir()
        write_json(a.output/'order.json',expected_order)
    retained=[]
    parent_score=best
    if resume:
        for check in result['checks']:
            candidate=a.output/'candidates'/f"step-{check['step']:04d}"
            value=check['old']['answer_ce'] if a.selection_metric=='old' else thinking_score(a.selection_metric,check['reasoning'])
            if check['eligible'] and value<best:best=value;selected=candidate
            if check['reasoning']['reasoning_and_answer_ce']<best_reasoning:
                best_reasoning=check['reasoning']['reasoning_and_answer_ce'];reasoning_candidate=candidate
        record=commit_resume(a.output,resume,a.resume_step)
        result.setdefault('resume_history',[]).append(record)
        write_json(a.output/'metrics.json',result)
    deadline=time.monotonic()+a.max_hours*3600
    start_step=a.resume_step or 0
    status_ru(f'Обучение началось с шага {start_step+1}. План: {a.steps} шагов. Проверка каждые {a.validation_every} шагов.')
    for step in range(start_step+1,a.steps+1):
        if time.monotonic()>deadline:raise TimeoutError('Reasoning QAT configured training time bound')
        training_kernel(a.fla_training);model.train();opt.zero_grad(set_to_none=True);start=time.monotonic()
        row=datasets['train'][(step-1+a.order_offset)%len(datasets['train'])]
        configure_checkpointing(len(row['input_ids']))
        if a.ab_branch:
            ce,new_stats=ab_forward(row,'new')
            if not torch.isfinite(ce):raise RuntimeError('Nonfinite new objective')
            ce.backward();ce_value=new_stats['ce'];del ce
            new_gradient_sample=panel_values() if step==1 or step%128==0 else None
        else:
            ce,_=reasoning_loss(model,row,chunk_tokens=a.loss_chunk_tokens)
            if not torch.isfinite(ce):raise RuntimeError('Nonfinite reasoning CE')
            (ce*a.reasoning_weight).backward();ce_value=float(ce.detach());del ce
        example=order[(step-1+kd_offset)%len(order)];cache=load_cache(example,meta)
        if a.ab_branch:
            loss,old_stats=ab_forward(example,'old');hidden=kl=oldce=None
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite replay objective')
            loss.backward()
        else:
            hidden=model.model(input_ids=torch.tensor([example['input_ids']],device='cuda:0'),use_cache=False).last_hidden_state[0,:-1]
            loss,kl,oldce=loss_from_hidden(model.lm_head,hidden,torch.tensor(example['labels'][1:],device=hidden.device),cache,chunk_tokens=a.loss_chunk_tokens)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite replay loss')
            (loss*(1-a.reasoning_weight)).backward()
        if not gradients_finite_by_device(model.parameters()):raise RuntimeError('Nonfinite mixed gradients')
        gradient_stats=None
        if a.ab_branch and new_gradient_sample is not None:
            total_gradient=panel_values();gradient_stats={}
            for name,new_vector in new_gradient_sample.items():
                old_vector=total_gradient[name]-new_vector
                gradient_stats[name]={'weighted_new_sample_norm':float(new_vector.norm()),'weighted_old_sample_norm':float(old_vector.norm()),
                                      'weighted_total_sample_norm':float(total_gradient[name].norm()),'sample_entries':len(new_vector)}
            del total_gradient,new_gradient_sample
        opt.step();opt.zero_grad(set_to_none=True)
        event={'step':step,'reasoning_ce':ce_value,'replay_loss':float(loss.detach()),'seconds':time.monotonic()-start}
        if a.ab_branch:
            event.update(ab_branch=a.ab_branch,ce_new=new_stats['ce'],ce_old=old_stats['ce'],
                         kl_old=old_stats['kl'],kl_new=new_stats['kl'] if a.ab_branch=='B' else None,
                         new_distribution=new_stats,old_distribution=old_stats,weights=weights)
            if gradient_stats is not None:event['gradient_sample']=gradient_stats
        del hidden,loss,kl,oldce,cache
        with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        core.event('reasoning_qat_step',**event)
        if step==1 or step%16==0:status_ru(f'Обучение: шаг {step}/{a.steps}')
        if step%a.validation_every==0 or step==a.steps:
            status_ru(f'Шаг {step}: проверка…')
            v,sv,rv=validate_all()
            eligible,selection_value=selection_status(a.selection_metric,v['answer_ce'],sv['answer_ce'],thinking_score(a.selection_metric,rv),
                                                     baseline['answer_ce'],secondary['answer_ce'],thinking_score(a.selection_metric,thinking))
            check={'step':step,'old':v,'verified':sv,'reasoning':rv,'eligible':eligible};result['checks'].append(check)
            improve_selected=eligible and selection_value<best
            improve_reasoning=rv['reasoning_and_answer_ce']<best_reasoning
            status_ru(f"Шаг {step}: ошибка рассуждений {rv['reasoning_and_answer_ce']:.6f}; обычных ответов {v['answer_ce']:.6f}; контроль {sv['answer_ce']:.6f}. "+('Новый лучший результат.' if improve_selected else 'Лучший результат не изменился.'))
            if 'math' in rv['by_category']:
                status_ru(f"Математика отдельно: CE {rv['by_category']['math']['reasoning_and_answer_ce']:.6f}; это ещё не точность самостоятельных решений.")
            if a.keep_best:
                candidate={'step':step,'score':selection_value,'path':str(a.output/'candidates'/f'step-{step:04d}')}
                ranked=rank_candidates(retained,candidate,a.keep_best) if eligible and selection_value<parent_score else retained
                if candidate in ranked:
                    saved=Path(candidate['path'])
                    status_ru('Сохранение точки из трёх лучших…')
                    save_latent(model,opt,saved,check)
                    removed=[r for r in retained if r not in ranked]
                    retained=ranked;selected=Path(retained[0]['path']);best=retained[0]['score']
                    result.update(selected_latent=str(selected),retained_candidates=retained)
                    write_json(a.output/'metrics.json',result)
                    write_json(a.output/'checkpoint-ranking.json',{'metric':a.selection_metric,'parent_score':parent_score,'retained':retained})
                    for obsolete in removed:
                        path=Path(obsolete['path'])
                        if path.parent!=a.output/'candidates' or path.is_symlink() or path==parent:raise RuntimeError('Unsafe retention path')
                        shutil.rmtree(path)
                        core.event('checkpoint_pruned',path=str(path),step=obsolete['step'],score=obsolete['score'])
                    status_ru(f'Сохранено лучших точек: {len(retained)}/{a.keep_best}; лучший шаг {retained[0]["step"]}.')
            elif a.save_every_validation or improve_selected or improve_reasoning or (a.save_final_candidate and step==a.steps):
                status_ru('Сохранение контрольной точки…')
                saved=a.output/'candidates'/f'step-{step:04d}'
                save_latent(model,opt,saved,check)
                if a.save_final_candidate and step==a.steps:final_candidate=saved
                if improve_selected:best=selection_value;selected=saved
                if improve_reasoning:best_reasoning=rv['reasoning_and_answer_ce'];reasoning_candidate=saved
                result.update(selected_latent=str(selected or parent),
                              reasoning_candidate_latent=str(reasoning_candidate) if reasoning_candidate else None)
            write_json(a.output/'metrics.json',result);core.event('reasoning_qat_validation',**check)
            if eligible and a.target_old_ce is not None and v['answer_ce']<=a.target_old_ce:
                result['target_reached_at_step']=step
                core.event('reasoning_qat_target_reached',step=step,old_ce=v['answer_ce'])
                break
            if eligible and a.target_reasoning_ce is not None and rv['reasoning_and_answer_ce']<=a.target_reasoning_ce:
                result['target_reached_at_step']=step
                core.event('reasoning_qat_target_reached',step=step,reasoning_ce=rv['reasoning_and_answer_ce'])
                break
    if a.defer_export:
        result.update(state='completed',optimizer_updates=step,export_state='deferred_to_cpu_generation_review',
                      generation_review_pending=True,candidates=[str(p) for p in sorted((a.output/'candidates').glob('step-*'))])
        write_json(a.output/'metrics.json',result)
        status_ru('Все шаги выполнены. Точки сохранены; далее CPU-экспорт и сравнение самостоятельных ответов.');return
    status_ru('Обучение закончено. Экспорт и проверка сохранённой модели…')
    training_kernel(False)
    del opt;gc.collect();torch.cuda.empty_cache()
    selected=selected or parent;model.gradient_checkpointing_disable()
    exports=[('selected',selected)]
    if reasoning_candidate is not None and reasoning_candidate!=selected:
        exports.append(('reasoning_candidate',reasoning_candidate))
    if final_candidate is not None and final_candidate not in [latent for _,latent in exports]:
        exports.append(('final_candidate',final_candidate))
    verify=[]
    for name,latent in exports:
        opt=restore(model,latent);del opt
        with torch.no_grad(),validation_runtime(model,a.fast_validation,a.validation_cache_gib):
            v=core.evaluate(model,old['validation']);rv=reasoning_eval(model,datasets['validation'])
        result.update({name+'_latent':str(latent),name+'_validation':v,name+'_reasoning':rv})
        if latent==parent:packed=parent_packed
        else:
            packed=a.output/(name.replace('_','-')+'-ternary');core.export_packed(model,packed)
            verify.append((name,packed,v,rv))
        result[name+'_packed']=str(packed)
    if final_candidate is not None:
        for name,latent in exports:
            if latent==final_candidate:
                result['final_candidate_latent']=str(final_candidate);result['final_candidate_packed']=result[name+'_packed'];break
    if reasoning_candidate==selected:result['reasoning_candidate_packed']=result['selected_packed']
    del model;gc.collect();torch.cuda.empty_cache()
    for name,packed,expected_v,expected_rv in verify:
        reloaded=load_packed(packed)
        with torch.no_grad(),validation_runtime(reloaded,a.fast_validation,a.validation_cache_gib):
            rv=reasoning_eval(reloaded,datasets['validation']);v=core.evaluate(reloaded,old['validation'])
        if abs(v['answer_ce']-expected_v['answer_ce'])>1e-5 or abs(rv['reasoning_and_answer_ce']-expected_rv['reasoning_and_answer_ce'])>1e-5:
            raise RuntimeError('Packed reload changed candidate metrics: '+name)
        result.update({name+'_packed_reasoning':rv,name+'_packed_validation':v})
        del reloaded;gc.collect();torch.cuda.empty_cache()
    status_ru('Готово: экспорт и проверка модели завершены.')
    result['state']='completed';write_json(a.output/'metrics.json',result);core.event('reasoning_qat_completed',metrics=result)

if __name__=='__main__':main()
