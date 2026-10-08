"""Compare genuinely new E007/E008 candidates only after the GPU training queue ends."""
import argparse,json,subprocess,time
from pathlib import Path
from wait_reasoning_qat import wait_terminal,ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-service',default='vol2-e008-reasoning-qat.service')
    parser.add_argument('--experiments',nargs='+',default=['e007-scale-recovery','e008-reasoning-qat'])
    parser.add_argument('--output',type=Path,default=ROOT/'reports/recovery-smoke')
    parser.add_argument('--student-reference',type=Path,default=ROOT/'reports/broad-smoke-e006/student')
    args=parser.parse_args()
    wait_terminal(args.wait_service,time.monotonic()+20*3600)
    candidates=[]
    for experiment in args.experiments:
        path=ROOT/'runs'/experiment/'metrics.json';metrics=json.loads(path.read_text())
        if metrics.get('state')!='completed':raise RuntimeError('Incomplete candidate: '+experiment)
        choices=[(experiment,metrics['selected_packed'])]
        if metrics.get('reasoning_candidate_packed') and metrics['reasoning_candidate_packed']!=metrics['selected_packed']:
            choices.append((experiment+'-reasoning-candidate',metrics['reasoning_candidate_packed']))
        for label,path in choices:
            checkpoint=Path(path).resolve()
            manifest=json.loads((checkpoint/'manifest.json').read_text())
            if len(manifest['matrices'])!=402 or any(not (checkpoint/r['file']).is_file() for r in manifest['matrices'].values()):
                raise RuntimeError('Incomplete packed export: '+str(checkpoint))
            candidates.append((label,checkpoint,metrics))
    baseline=ROOT/'reports/broad-smoke-e006'
    references={'student':args.student_reference.resolve(),'bonsai2':baseline/'bonsai2'}
    for reference in references.values():
        config=json.loads((reference/'config.json').read_text())
        if config.get('state')!='completed':raise RuntimeError('Incomplete reference generation')
        summary=json.loads((reference/'summary.json').read_text())
        if any(summary[f]['total']!=32 for f in ('gsm8k','humaneval','ifeval')):raise RuntimeError('Incomplete reference scores')
    out=args.output.resolve();out.mkdir(exist_ok=False)
    seen={Path(json.loads((references['student']/'config.json').read_text())['checkpoint']).resolve():references['student']}
    report={'state':'running','candidates':[]};python=str(ROOT/'.venv/bin/python');scorer=str(ROOT/'.venv-eval/bin/python')
    def save():
        temporary=out/'summary.json.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(out/'summary.json')
    save()
    for experiment,checkpoint,metrics in candidates:
        entry={'experiment':experiment,'checkpoint':str(checkpoint)}
        (out/f'{experiment}-training-metrics.json').write_text(json.dumps(metrics,indent=2))
        if checkpoint in seen:
            entry.update(reused_evaluation=str(seen[checkpoint]),reason='Selection retained a checkpoint already evaluated')
        else:
            folder=out/experiment
            subprocess.run([python,'-u',str(ROOT/'scripts/run_broad_benchmark.py'),'--model','student','--checkpoint',str(checkpoint),
                            '--subset','smoke','--output',str(folder)],check=True,cwd=ROOT,timeout=4*3600)
            subprocess.run([scorer,str(ROOT/'scripts/score_broad_benchmark.py'),str(folder)],check=True,cwd=ROOT,timeout=1800)
            seen[checkpoint]=folder;entry['evaluation']=str(folder)
        folder=seen[checkpoint]
        for reference,reference_folder in references.items():
            compared=out/f'{experiment}-vs-{reference}.json'
            subprocess.run([python,str(ROOT/'scripts/compare_benchmarks.py'),str(folder),str(reference_folder),'--output',str(compared)],
                           check=True,cwd=ROOT,timeout=60,stdout=subprocess.DEVNULL)
        entry['scores']=json.loads((folder/'summary.json').read_text());report['candidates'].append(entry);save()
        print(json.dumps({'event':'recovery_candidate_scored',**entry}),flush=True)
    report['state']='completed';save()
    print(json.dumps({'event':'recovery_comparison_completed','scope':'96-task smoke comparison, not broad parity proof'}),flush=True)

if __name__=='__main__':main()
