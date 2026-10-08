"""Paired descriptive comparison; deliberately not a parity/completion gate."""
import argparse,json,math
from collections import defaultdict
from pathlib import Path

def wilson(correct,total,z=1.959963984540054):
    if not total:raise ValueError('Empty evaluation')
    p=correct/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-half),min(1.,center+half)]

def mcnemar(student_only,reference_only):
    n=student_only+reference_only
    if not n:return 1.
    # Exact two-sided binomial test among discordant paired outcomes.
    k=min(student_only,reference_only)
    terms=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)]
    peak=max(terms)
    return min(1.,2*math.exp(peak)*sum(math.exp(t-peak) for t in terms))

def read(folder):
    config=json.loads((folder/'config.json').read_text())
    if config.get('state')!='completed':raise ValueError('Generation incomplete: '+str(folder))
    tasks=json.loads((folder/'tasks.json').read_text())
    scores=json.loads((folder/'scores.json').read_text())
    key=lambda row:(row['family'],row['source_index'])
    td={key(t):t for t in tasks};sd={key(s):s for s in scores}
    if len(td)!=len(tasks) or len(sd)!=len(scores) or td.keys()!=sd.keys():
        raise ValueError('Missing or duplicated task scores: '+str(folder))
    return config,td,sd

def compare(student,reference):
    sc,st,ss=read(student);rc,rt,rs=read(reference)
    for field in ('subset','suite','decoding','max_new_tokens','context_limit','reasoning_effort'):
        if sc.get(field)!=rc.get(field):raise ValueError('Protocol differs: '+field)
    if st.keys()!=rt.keys():raise ValueError('Different task sets')
    grouped=defaultdict(list)
    for key,task in st.items():
        for field in ('input_ids','instruction','thinking','max_new_tokens','reasoning_effort'):
            if task.get(field)!=rt[key].get(field):raise ValueError('Different prompt/protocol: '+str((key,field)))
        grouped[key[0]].append((ss[key],rs[key]))
    report={'student':str(student),'reference':str(reference),'families':{},
            'interpretation':'Descriptive paired comparison, not proof of equivalence. Wilson intervals are approximate 95% marginal intervals. A nonsignificant McNemar test does not establish parity.'}
    for family,rows in grouped.items():
        n=len(rows);a=sum(bool(s['correct']) for s,r in rows);b=sum(bool(r['correct']) for s,r in rows)
        student_only=sum(bool(s['correct']) and not r['correct'] for s,r in rows)
        reference_only=sum(bool(r['correct']) and not s['correct'] for s,r in rows)
        report['families'][family]={'n':n,'student_correct':a,'reference_correct':b,
            'student_accuracy':a/n,'reference_accuracy':b/n,'difference_percentage_points':100*(a-b)/n,
            'student_wilson95':wilson(a,n),'reference_wilson95':wilson(b,n),
            'student_only_correct':student_only,'reference_only_correct':reference_only,
            'both_correct':a-student_only,'both_wrong':n-b-student_only,
            'mcnemar_exact_p':mcnemar(student_only,reference_only),
            'student_truncated':sum(bool(s['truncated']) for s,r in rows),
            'reference_truncated':sum(bool(r['truncated']) for s,r in rows)}
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('student',type=Path);p.add_argument('reference',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=compare(a.student,a.reference)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
