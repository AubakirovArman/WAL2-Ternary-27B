"""Independent reference tasks for E018; no benchmark solutions used for training."""
import collections,hashlib,itertools,json,math,random,re

TARGET={'math':7500,'code_apps':1000,'code_synthetic':5000,'instructions':5000,'grounded':4000,'logic':2500}
HOLDOUT={'math':450,'code_apps':60,'code_synthetic':300,'instructions':300,'grounded':240,'logic':150}
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def split_for(group):return 'validation' if int(digest(group)[:8],16)%20==0 else 'train'
def row(category,identity,instruction,validator,group=None,**kw):
    return dict(id='e018-'+category+'-'+str(identity),category=category,split=split_for(group or str(identity)),instruction=instruction,sha256=digest(instruction),validator=validator,enable_thinking=True,topic=group or category,**kw)

def algorithm(kind,x,k):
    if kind==0:return max((sum(x[i:j]) for i in range(len(x)) for j in range(i+1,len(x)+1)),default=0)
    if kind==1:
        d=[]
        for i,v in enumerate(x):d.append(1+max((d[j] for j in range(i) if x[j]<v),default=0))
        return max(d,default=0)
    if kind==2:return sum(x[i]>x[j] for i in range(len(x)) for j in range(i+1,len(x)))
    if kind==3:return sum(sum(x[i:j])==k for i in range(len(x)) for j in range(i+1,len(x)+1))
    if kind==4:return [max(x[i:i+k]) for i in range(max(0,len(x)-k+1))]
    if kind==5:return [min(x[i:i+k]) for i in range(max(0,len(x)-k+1))]
    if kind==6:return [next((x[j] for j in range(i+1,len(x)) if x[j]>v),None) for i,v in enumerate(x)]
    if kind==7:return next((v for v in x if x.count(v)==1),None)
    if kind==8:
        d=[1]+[0]*k
        for v in sorted({abs(v) for v in x if v}):
            for j in range(v,k+1):d[j]+=d[j-v]
        return d[k]
    if kind==9:
        ss={0}
        for v in x:ss|={s+v for s in list(ss)}
        return k in ss
    if kind==10:return [math.prod(x[:i]+x[i+1:]) for i in range(len(x))]
    if kind==11:return sorted(set(x),key=lambda v:(-x.count(v),v))[:k]
    if kind==12:
        p=1
        while p in x:p+=1
        return p
    if kind==13:return sum(x[i]==x[j] for i in range(len(x)) for j in range(i+1,len(x)))
    if kind==14:
        a=b=0
        for v in x:a,b=b,max(b,a+v)
        return b
    if kind==15:return max((j-i for i in range(len(x)+1) for j in range(i,len(x)+1) if len(set(x[i:j]))==j-i),default=0)
    if kind==16:return [sum(abs(v-w) for w in x) for v in x]
    if kind==17:return sum(sum(x[i:j])%k==0 for i in range(len(x)) for j in range(i+1,len(x)+1))
    if kind==18:return sorted([[a,b] for a in set(x) for b in set(x) if a<=b and a+b==k and (a!=b or x.count(a)>=2)])
    if kind==19:return sorted([[a,b,c] for a,b,c in set(tuple(sorted(v)) for v in itertools.combinations(x,3)) if a+b+c==k])
    if kind==20:return sum((v-w)**2 for v,w in zip(x,x[1:]))
    if kind==21:return [sum(w<=v for w in x[:i]) for i,v in enumerate(x)]
    if kind==22:return sum(x[i:j]==x[i:j][::-1] for i in range(len(x)) for j in range(i+1,len(x)+1))
    if kind==23:
        a=sorted(x);return a[k-1] if len(a)>=k else None
    raise ValueError(kind)

def synthetic_code(i):
    rng=random.Random(18018000+i);kind=i%24;k=rng.randint(2,12);a=rng.randint(1,9);b=rng.randint(-50,50);m=rng.randint(2,19);r=rng.randrange(m)
    descriptions=['maximum sum of a nonempty contiguous subarray; return0 for empty input','length of the longest strictly increasing subsequence','number of inversions (pairs i<j with x[i]>x[j])',f'number of nonempty contiguous subarrays with sum exactly{k}',f'maximum of each full sliding window of length{k}',f'minimum of each full sliding window of length{k}','list of the next strictly greater value to the right for each position; use None if absent','first value that occurs exactly once; None if absent',f'number of unordered ways to make{k} using unlimited coins from distinct nonzero absolute input values',f'whether some subset of positions sums to{k}; the empty subset is allowed','product of all other elements for each position',f'first{k} distinct values ordered by descending frequency, ties ascending numeric value','smallest missing positive integer','number of equal-value pairs of positions i<j','maximum sum of nonadjacent positions; choosing no positions is allowed','length of the longest contiguous subarray with all distinct values','for each position, sum of absolute differences to every element',f'number of nonempty contiguous subarrays whose sum is divisible by{k}',f'all unique ascending pairs of values summing to{k}; need two occurrences for equal values; sort pairs lexicographically',f'all unique nondecreasing triples of values summing to{k}, using distinct positions; sort triples lexicographically','sum of squared differences between consecutive values','for each position i, number of earlier values <= the value at i','number of nonempty palindromic contiguous subarrays',f'{k}-th smallest value counting duplicates, or None when too short']
    samples=[[],[0],[1,-1,0,1],[3,3,-2,-2,-2,3],list(range(-5,6)),[0]*10]+[[rng.randint(-12,12) for _ in range(rng.randrange(15))] for _ in range(24)]
    def oracle(v):return algorithm(kind,[a*n+b for n in v if n%m!=r],k)
    fixtures=[dict(input=v,expected=oracle(v)) for v in samples]
    inst=f'Implement Python function solve(values). Do not mutate the input list of integers. First discard each original value v with v % {m} == {r} (Python modulo). Transform each remaining value to {a}*v+({b}), preserving order; call the resulting list x. Return the {descriptions[kind]}. Empty input is valid. Use only the standard library. Example: solve({samples[3]!r}) == {fixtures[3]["expected"]!r}. Your final answer must contain exactly one fenced Python code block with the full function.'
    # Entire algorithm families held out, rather than parameter-only splitting.
    t=row('code_synthetic',i,inst,'function',group='algorithm-'+str(kind),fixtures=fixtures,entry_point='solve')
    t['split']='validation' if kind in (5,13,18,22) else 'train';return t

TOPICS=['caching','sorting','version control','unit testing','binary search','recursion','databases','data backups','password managers','network latency','compression','checksums','unicode','time zones','file permissions','queues','scheduling','memory allocation','search engines','data visualization','sampling','probability','measurement error','recycling','photosynthesis','erosion','volcanoes','ocean currents','the water cycle','solar panels','batteries','insulation','public transport','urban gardens','language learning','note taking','team meetings','project planning','accessibility','clear writing','library catalogues','museum exhibits','music practice','photography','cooking','bread fermentation','stargazing','maps','budget tracking','online learning']
def instruction_task(i):
    rng=random.Random(18028000+i);topic=rng.choice(TOPICS);aud=rng.choice(['a beginner','a teacher','a small business owner','a software developer','a student','a librarian','a volunteer','a parent','a researcher','a designer']);setting=rng.choice(['a workshop','a newsletter','a classroom discussion','an onboarding guide','a community project','a short presentation','a training handout','a FAQ','a help page','a planning session']);kind=i%8;lo=rng.randint(45,80);hi=lo+55;key=rng.choice(['Practical guide','Useful steps','A clear start','Quick explanation']);word=rng.choice(['example','practice','benefit','choice']);ban=rng.choice(['revolutionary','obviously','game-changing','unprecedented'])
    if kind==0:spec=f'Write exactly3 nonempty paragraphs separated by blank lines; no headings or bullet lists.'
    elif kind==1:spec='Write exactly4 lines, each beginning with "- ", with no other lines.'
    elif kind==2:spec='Return only a JSON object with exactly keys "explanation", "example", "limitation", each a nonempty string.'
    elif kind==3:spec=f'First line must be exactly "{key}". Follow it with a paragraph. End with the exact line "End of guide."'
    elif kind==4:spec='Write exactly3 numbered lines beginning "1. ", "2. ", "3. ", respectively.'
    elif kind==5:spec='Return exactly two sections, headed "Benefits:" and "Limitations:", in that order.'
    elif kind==6:spec='Return a JSON array containing exactly3 nonempty strings, with no extra text.'
    else:spec='Write one paragraph without any newline characters and without Markdown formatting.'
    inst=f'Explain {topic} to {aud} for {setting}. Use English. Requirements for the final answer: {spec} Use {lo} to {hi} whitespace-separated words in the complete final output. Include the lowercase standalone word "{word}" exactly once. Never use the word "{ban}" in any case. Be factually careful and concrete. Do not discuss these formatting rules.'
    return row('instructions',i,inst,'constraints',group='instruction-topic-'+topic,kind=kind,lo=lo,hi=hi,key=key,word=word,ban=ban)

def logic_task(i):
    rng=random.Random(18038000+i);n=rng.randint(5,9);names=rng.sample(['Ada','Ben','Cleo','Dara','Evan','Faye','Gus','Hana','Iris','Jude','Kira','Leo'],n);edges=[(names[a],names[b]) for a in range(n) for b in range(a+1,n) if rng.random()<.3];src,dst=rng.sample(names,2);reachable={src}
    for _ in range(n):reachable|={b for a,b in edges if a in reachable}
    shortest={src:0}
    for _ in range(n):
        for a,b in edges:
            if a in shortest:shortest[b]=min(shortest.get(b,999),shortest[a]+1)
    gold={'reachable':dst in reachable,'minimum_hops':shortest.get(dst),'reachable_count':len(reachable)-1}
    inst=f'A directed communication network has nodes {names!r}. The only directed links are {edges!r}. Starting at {src}, determine whether {dst} can be reached, the minimum number of links needed (null if impossible), and the number of other reachable nodes. Do not reverse links. Final answer: only JSON with keys reachable (boolean), minimum_hops (integer or null), reachable_count (integer).'
    return row('logic',i,inst,'json',group='logic-network-size-'+str(n),gold=gold)

def repeat_reason(text):
    lines=[x.strip() for x in text.splitlines() if len(x.strip())>45]
    if lines and max(collections.Counter(lines).values())>=4:return True
    words=text.split();chunks=[' '.join(words[i:i+32]) for i in range(0,len(words)-31,16)]
    return bool(chunks and max(collections.Counter(chunks).values())>=4)

def verify_task(t,answer):
    v=t['validator']
    if v=='math':
        from math_verify import parse,verify,LatexExtractionConfig,ExprExtractionConfig
        g=parse(t['gold'],extraction_config=[LatexExtractionConfig()],extraction_mode='first_match');p=parse(answer,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()],extraction_mode='first_match')
        return bool(g and p and verify(g,p))
    if v=='json':return json.dumps(json.loads(answer),sort_keys=True)==json.dumps(t['gold'],sort_keys=True)
    if v=='grounded':
        x=json.loads(answer);return set(x)=={'answer','evidence'} and isinstance(x['answer'],str) and isinstance(x['evidence'],str) and norm(x['answer']) in {norm(a) for a in t['answers']} and len(x['evidence'])>=len(x['answer']) and x['evidence'] in t['context'] and x['answer'].casefold() in x['evidence'].casefold()
    if v=='constraints':
        if not t['lo']<=len(answer.split())<=t['hi']:return False
        if len(re.findall(r'\b'+re.escape(t['word'])+r'\b',answer))!=1 or re.search(r'\b'+re.escape(t['ban'])+r'\b',answer,re.I):return False
        k=t['kind'];lines=answer.strip().splitlines()
        if k==0:return len(re.split(r'\n\s*\n',answer.strip()))==3 and not any(x.startswith(('#','- ')) for x in lines)
        if k==1:return len(lines)==4 and all(x.startswith('- ') for x in lines)
        if k==2:
            x=json.loads(answer);return isinstance(x,dict) and set(x)=={'explanation','example','limitation'} and all(isinstance(v,str) and v.strip() for v in x.values())
        if k==3:return lines[0]==t['key'] and lines[-1]=='End of guide.' and len(lines)>=3
        if k==4:return len(lines)==3 and all(x.startswith(str(j+1)+'. ') for j,x in enumerate(lines))
        if k==5:return lines[0]=='Benefits:' and lines.count('Benefits:')==1 and lines.count('Limitations:')==1
        if k==6:
            x=json.loads(answer);return isinstance(x,list) and len(x)==3 and all(isinstance(v,str) and v.strip() for v in x)
        return '\n' not in answer.strip() and not any(x in answer for x in ['**','`','#'])
    from code_sandbox import run
    m=re.fullmatch(r'\s*```(?:python|py)\s*\n(.*?)```\s*',answer,re.S)
    if not m:return False
    code=m[1]
    if v=='function':
        program='import copy\nns={}\nexec('+repr(code)+',ns)\nf=ns['+repr(t['entry_point'])+']\n'
        for case in t['fixtures']:program+=f"x={case['input']!r}\ny=copy.deepcopy(x)\nassert f(x)=={case['expected']!r}\nassert x==y\n"
        return run(program)['passed']
    if v=='apps':
        # Source tests all run; unsupported cases are filtered at pool preparation.
        program='import io,sys,contextlib,json,copy\ncode='+repr(code)+'\ncases='+repr(t['tests'])+'\n'
        if t.get('fn_name'):
            program+='ns={}\nexec(code,ns)\nobj=ns["Solution"]() if "Solution" in ns else None\nfn=getattr(obj,'+repr(t['fn_name'])+') if obj is not None else ns['+repr(t['fn_name'])+']\n'
            program+='for args,expected in cases:\n result=json.loads(json.dumps(fn(*copy.deepcopy(args))))\n assert result==expected or (isinstance(expected,list) and len(expected)==1 and result==expected[0])\n'
        else:
            program+='for inp,expected in cases:\n sys.stdin=io.TextIOWrapper(io.BytesIO(inp.encode()))\n out=io.StringIO()\n try:\n  with contextlib.redirect_stdout(out):exec(code,{"__name__":"__main__"})\n except SystemExit as e:\n  assert e.code in (None,0)\n assert out.getvalue().split()==expected.split()\n'
        return run(program,timeout=10)['passed']
    raise ValueError(v)
