"""E020 exact references, extending held-out derangements beyond exhausted small n."""
import hashlib,math,random
from e019_math_tasks import make_math as original,FAMILIES
def make_math(family,serial,split='train',repair=False):
    if family!=19 or serial<600000:return original(family,serial,split,repair)
    assert split=='validation' and not repair
    rng=random.Random(200000003+serial*101);n=rng.randint(16,32);k=rng.randint(1,n-3);r=n-k
    dp=[1,0]
    for j in range(2,r+1):dp.append((j-1)*(dp[-1]+dp[-2]))
    inclusion=sum((-1)**j*math.comb(r,j)*math.factorial(r-j) for j in range(r+1));assert inclusion==dp[r]
    gold=math.comb(n,k)*dp[r]
    instruction=f'Count permutations of {{1,...,{n}}} with exactly {k} fixed points.\nGive a finite checked solution and end with the result inside \\boxed{{}}.'
    facts=[dict(id='s1',claim=f'Choose fixed positions in C({n},{k})={math.comb(n,k)} ways.',value=str(math.comb(n,k))),
           dict(id='s2',claim=f'The other {r} elements must be deranged; D0=1,D1=0,Dj=(j-1)(D(j-1)+D(j-2)); D{r}={dp[r]}.',value=str(dp[r])),
           dict(id='s3',claim=f'Multiply to get {gold}; inclusion-exclusion independently agrees.',value=str(gold))]
    return dict(id=f'e020-validation-math-19-{serial}',instruction=instruction,base_instruction=instruction,sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                gold=str(gold),facts=facts,family=FAMILIES[19],family_index=19,serial=serial,split=split,category='math',enable_thinking=True,repair=False,validator='exact_certificate')
