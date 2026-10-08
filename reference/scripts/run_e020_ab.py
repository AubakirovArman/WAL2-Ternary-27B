"""Authorized detached E020 pipeline; fail closed on any required gate, bounded A/B."""
import argparse,collections,fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/e020-pipeline';REPORT=ROOT/'reports/e020-ab'
GPU_IDS=['GPU_UUID_REDACTED','GPU_UUID_REDACTED']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def status(state,stage,**extra):
    value=dict(state=state,stage=stage,updated=time.time(),**extra);tmp=OUT/'status.tmp';tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(OUT/'status.json')
    with (OUT/'status-ru.log').open('a') as stream:stream.write(time.strftime('%H:%M:%S')+' | '+stage+'\n')
    print(time.strftime('%H:%M:%S')+' | '+stage,flush=True)
def run(stage,args,gpu=False):
    for name,digest in json.loads((OUT/'code-manifest.json').read_text()).items():
        if sha(ROOT/'scripts'/name)!=digest:raise RuntimeError('Код изменён после фиксации запуска: '+name)
    status('running',stage)
    # The historical offline teacher cache used Transformers' reference forward.
    # Importing FLA during cache collection changes the teacher distributions;
    # only the experimental QAT backward may see the FLA package.
    qat=Path(args[0]).name=='train_e020.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(GPU_IDS) if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='8',
             PYTHONPATH=str(ROOT/'tools/fla-probe-packages') if qat else '')
    with (OUT/(stage.split(':')[0]+'.log')).open('a') as log:
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-u',*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=log)
    if result.returncode:raise RuntimeError(f'{stage}: процесс завершился с кодом {result.returncode}; подробности в журнале этапа')
def train_args(branch,folder,preflight=False):
    args=[ROOT/'scripts/train_e020.py','--data',ROOT/'data/e020-natural-v1','--output',folder,
          '--parent-run',ROOT/'runs/e018-fresh25000','--parent-candidate','selected',
          '--steps','1024','--lr','5e-7','--validation-every','512','--selection-metric','math',
          '--order-offset','0','--kd-order-offset','25000','--fla-training','--reasoning-weight','.25',
          '--max-hours','12','--fast-validation','--validation-cache-gib','8',
          '--loss-chunk-tokens','512','--checkpoint-policy','adaptive','--max-length','12288',
          '--preserve-validation-candidates','--defer-export','--ab-branch',branch,'--fresh-kd-manifest',ROOT/'data/e020-kd-v1/manifest.json']
    if preflight:args+=['--preflight-only']
    return args
def check_pair():
    a=ROOT/'runs/e020-ab-a';b=ROOT/'runs/e020-ab-b'
    ca=json.loads((a/'config.json').read_text());cb=json.loads((b/'config.json').read_text())
    for key in ['parent','parent_packed','parent_metrics_sha256','parent_training_state_sha256','initial_rng_sha256','lr','steps','order_offset','kd_order_offset','reasoning_data','fresh_cache_manifest_sha256','fla_training','checkpoint_policy','loss_chunk_tokens']:
        assert ca[key]==cb[key],('A/B mismatch',key)
    assert sha(a/'order.json')==sha(b/'order.json')
    for folder in (a,b):
        m=json.loads((folder/'metrics.json').read_text());assert m['state']=='completed' and m['optimizer_updates']==1024
        assert [json.loads(l)['step'] for l in (folder/'training.jsonl').open()]==list(range(1,1025))
        assert len(m['candidates'])==2
    (REPORT/'matched-training-audit.json').write_text(json.dumps(dict(passed=True,parent=ca['parent'],initial_rng_sha256=ca['initial_rng_sha256'],order_sha256=sha(a/'order.json'),weights_A=ca['weights'],weights_B=cb['weights']),indent=2))
def report():
    summary={}
    for name in ['baseline','A-512','A-1024','B-512','B-1024']:
        folder=REPORT/name
        summary.update(json.loads((folder/'summary.json').read_text()))
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2))
    families=['aime25_diagnostic','fresh_olympiad','verified_skill_transfer','e020_math_control','e020_code_control','e020_instructions_control','e020_logic_control']
    lines=['# E020: короткий контролируемый A/B из E018','',
           'Обе ветки выполнили 1024 шага. Веса, состояние Adafactor, RNG и порядок данных совпадали в начале. Проверки каждые 512 шагов; все четыре точки сохранены. A: 0.25 CEnew + 0.075 CEold + 0.675 KLold. B: 0.25 CEnew + 0.075 CEold + 0.3375 KLold + 0.3375 KLnew. Оба KL по всем next-token позициям; T=1; top128 с точными ID и хвостом. Новый KL и CE используют один проход новой последовательности.',
           '', 'Результаты по финальным ответам; подтверждение обоснований отдельно. Прежний математический development-контроль использован повторно, новый контроль исключён из обучающего пула. Это не официальный скрытый тест.','',
           '| Набор | E018 | A512 | A1024 | B512 | B1024 | Учитель |','|---|---:|---:|---:|---:|---:|---:|']
    for family in families:
        cells=[]
        for name in ['e018','A-512','A-1024','B-512','B-1024','teacher']:
            r=summary[name][family];cells.append(f"{r['correct']}/{r['total']}")
        lines.append('| '+family+' | '+' | '.join(cells)+' |')
    lines+=['','Точки не удалены по CE и не объявлены заменой E018. Новые правильные математические финалы сохранены для отдельного аудита доказательств. Если B выигрывает, нужен независимый генерационный контроль перед расширением бюджета; если снижает KL/CE без улучшения генерации, следующий опыт — собственные префиксы student. 1024 шага проверяют только данную смесь и корпус.',
            '', 'Ограничения: учитель-аудитор тот же Qwen в отдельном запросе; точные проверки извлеченной арифметики не доказывают весь текст. Offline учитель — BF16 reference после деквантования фиксированного FP8-источника, численно не идентичен живому API. FLA backward остаётся экспериментальным и одинаков в обеих ветках.']
    (REPORT/'RESULT_RU.md').write_text('\n'.join(lines)+'\n')
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-b-step',type=int,help='Continue interrupted B from its committed validation; preserve completed A')
    options=parser.parse_args()
    OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if options.resume_b_step is not None:
        if options.resume_b_step!=512:raise ValueError('This pilot resumes B at its committed step 512')
        for name,digest in json.loads((OUT/'code-manifest.json').read_text()).items():
            if name not in ('run_e020_ab.py','train_reasoning_qat.py') and sha(ROOT/'scripts'/name)!=digest:
                raise RuntimeError('Unrelated original experiment code changed: '+name)
        required=[(ROOT/'data/e020-natural-v1/manifest.json','state','completed'),
                  (ROOT/'data/e020-kd-v1/manifest.json','state','completed'),
                  (REPORT/'baseline/status.json','state','completed'),
                  (ROOT/'runs/e020-preflight-b/metrics.json','state','preflight_passed'),
                  (ROOT/'runs/e020-ab-a/metrics.json','optimizer_updates',1024)]
        for path,key,expected in required:
            if json.loads(path.read_text()).get(key)!=expected:raise RuntimeError('Resume prerequisite missing: '+str(path))
        if json.loads((ROOT/'runs/e020-ab-a/metrics.json').read_text())['state']!='completed':raise RuntimeError('A not complete')
        archive=OUT/('code-before-resume-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
        shutil.copytree(OUT/'code',archive)
        shutil.copy2(OUT/'code-manifest.json',archive/'code-manifest.json')
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in ['run_e020_ab.py','e020_resume.py','prepare_e020_data.py','e020_math_tasks.py','collect_e020_data.py','recover_e020_logic_format.py','audit_e020_data.py','cache_e020_teacher.py','diagnose_e020_teacher_sentinels.py','e020_kd.py','train_e020.py','train_e019.py','train_reasoning_qat.py','run_e020_generation.py','score_e020_response.py','sample_e020_trits.py','sample_e019_trit_movement.py','review_e020_math.py']:
        shutil.copy2(ROOT/'scripts'/name,code/name)
    (OUT/'code-manifest.json').write_text(json.dumps({p.name:sha(p) for p in code.glob('*.py')},indent=2))
    try:
        status('waiting_data','Подготовка естественного корпуса: 1024 обучения + 256 проверки')
        while not (ROOT/'data/e020-natural-v1/manifest.json').exists():
            p=ROOT/'data/e020-natural-v1/status.json'
            if p.exists() and json.loads(p.read_text())['state'] in ('failed','insufficient_candidates'):raise RuntimeError('Сбор данных остановился: '+p.read_text())
            active=subprocess.check_output(['systemctl','--user','show','vol2-e020-data-full.service','-p','ActiveState','--value'],text=True).strip()
            if active not in ('active','activating'):raise RuntimeError('Сбор данных завершился без готового manifest')
            time.sleep(5)
        run('audit: Проверка полного корпуса', [ROOT/'scripts/audit_e020_data.py'])
        if options.resume_b_step is None:
            run('baseline: Самостоятельные ответы E018 и учителя на зафиксированном контроле',[ROOT/'scripts/run_e020_generation.py','--baseline'])
            run('cache: Свежие top128+tail вероятности учителя с точными ID',[ROOT/'scripts/cache_e020_teacher.py'],True)
            run('preflight: Полная смесь B, память и градиенты без обновления весов',train_args('B',ROOT/'runs/e020-preflight-b',True),True)
            run('train-A: 1024 шага, старый KL',train_args('A',ROOT/'runs/e020-ab-a'),True)
        args=train_args('B',ROOT/'runs/e020-ab-b')
        if options.resume_b_step is not None:args+=['--resume-step',str(options.resume_b_step)]
        run('train-B: Возобновление с 512 до 1024, половина KL на новых рассуждениях' if options.resume_b_step else 'train-B: 1024 шага, половина KL на новых рассуждениях',args,True)
        check_pair()
        run('trits: Выборочное изменение тритов по типам матриц',[ROOT/'scripts/sample_e020_trits.py'])
        for branch in ('A','B'):
            for step in (512,1024):run(f'generate-{branch}-{step}: native PQ2, 136 самостоятельных ответов',[ROOT/'scripts/run_e020_generation.py','--branch',branch,'--step',str(step)])
        run('proofs: Отдельный аудит новых правильных математических финалов',[ROOT/'scripts/review_e020_math.py'])
        report();status('completed','A/B завершён: четыре точки проверены; отчёт готов, E018 остаётся основной моделью')
    except Exception as e:status('failed',str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
