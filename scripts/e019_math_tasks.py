"""Fresh compositional math tasks and reproducible exact calculation certificates.

Certificates cover the listed facts/algorithms, not arbitrary teacher prose.
Families0..15 train;16..19 validation. No benchmark solutions are imported.
"""
import collections
import hashlib
import itertools
import math
import random
from fractions import Fraction

FAMILIES = ['place_values_gcd_remainder','crt_coprime_interval_sum','divisors_with_coprimality','nested_modular_powers','quadratic_roots_with_domain','bounded_diophantine_count','subset_residue_parity','digit_sum_divisibility','finite_inclusion_exclusion','conditional_probability','bounded_polynomial_coefficient','reflected_triangle_area','recurrence_prefix_modulus','grid_paths_obstacles','rational_rate_composition','binomial_prime_valuations','digit_pattern_and_parity','lattice_polygon_pick','telescoping_fraction_sum','permutation_fixed_points']


def count_digits(n, target, modulus, forbidden='77', parity=None):
    # States: residue, sum, previous digit, optional parity of count of ones.
    dp={(0,0,-1,0):1}
    for pos in range(n):
        nxt=collections.Counter()
        for (r,s,previous,p),v in dp.items():
            for d in range(1 if pos==0 else 0,10):
                if str(previous)+str(d)==forbidden or s+d>target:continue
                nxt[((10*r+d)%modulus,s+d,d,p^(d==1))]+=v
        dp=nxt
    return sum(v for (r,s,previous,p),v in dp.items() if r==0 and s==target and (parity is None or p==parity))


def make_math(family, serial, split='train', repair=False):
    assert (split=='validation') == (family>=16)
    rng=random.Random(190000003+serial*101+family*7919+(10**8 if repair else 0))
    facts=[]
    def add(claim,value):
        facts.append(dict(id=f's{len(facts)+1}',claim=claim,value=str(value)))
    if family==0:
        base=rng.randint(11,29);digits=rng.sample(range(1,base),5);a,b,c,d,e=digits;prime=rng.choice([7,11,13,17])
        x=a*base**2+b*base+c;y=d*base+e;g=math.gcd(x,y);res=x%prime
        problem=f'In base {base}, the digit VALUES of a three-digit numeral X are {digits[:3]} from left to right. The digit VALUES of a two-digit numeral Y are {digits[3:]} from left to right (values above nine are individual digits). Find gcd(X,Y) plus the least nonnegative remainder of X modulo {prime}, interpreting X,Y as integers.'
        add(f'Positional values: X={a}*{base}^2+{b}*{base}+{c}={x}; Y={d}*{base}+{e}={y}.',[x,y]);add(f'gcd({x},{y})={g}.',g);add(f'{x} modulo {prime} is {res}.',res);gold=g+res
    elif family==1:
        m,n=rng.sample([7,11,13,17,19],2);a=rng.randrange(m);b=rng.randrange(n);lo=rng.randint(1000,12000);hi=lo+rng.randint(250,1800)
        residue=(a+m*((b-a)*pow(m,-1,n)%n))%(m*n)
        first=residue+((lo-residue+m*n-1)//(m*n))*(m*n)
        candidates=list(range(first,hi+1,m*n));valid=[x for x in candidates if math.gcd(x,30)==1]
        independent=[x for x in range(lo,hi+1) if x%m==a and x%n==b and math.gcd(x,30)==1];assert valid==independent
        problem=f'Find the sum of all integers x in [{lo},{hi}] with x ≡ {a} (mod {m}), x ≡ {b} (mod {n}), and gcd(x,30)=1.'
        add(f'The unique common residue modulo {m*n} is {residue}; coprime moduli give all x={residue}+{m*n}k.',residue);add(f'All interval candidates are {candidates}.',candidates);add(f'After excluding multiples of two, three or five, the full list is {valid}.',valid);gold=sum(valid)
    elif family==2:
        p,q=rng.sample([2,3,5,7],2);a,b=rng.randint(1,4),rng.randint(1,3);value=p**a*q**b;avoid=rng.choice([2,3,5,7,11]);divs=sorted({p**i*q**j for i in range(a+1) for j in range(b+1)});valid=[d for d in divs if math.gcd(d,avoid)==1]
        brute=[d for d in range(1,value+1) if value%d==0 and math.gcd(d,avoid)==1];assert valid==brute
        problem=f'Find the sum of positive divisors d of {value} satisfying gcd(d,{avoid})=1.'
        add(f'Prime factorization: {value}={p}^{a}*{q}^{b}; each divisor is {p}^i*{q}^j with 0≤i≤{a},0≤j≤{b}.',value);add(f'The complete divisor list is {divs}.',divs);add(f'The coprime divisors are {valid}.',valid);gold=sum(valid)
    elif family==3:
        a,b=rng.randint(2,19),rng.randint(2,19);u,v=rng.randint(11,90),rng.randint(11,90);mod=rng.choice([97,101,127,143,169]);c=rng.randint(2,17);power=rng.randint(2,7);x=pow(a,u,mod);y=pow(b,v,mod);inner=(x+y+c)%mod;gold=pow(inner,power,mod)
        assert gold==pow(a**u+b**v+c,power,mod)
        problem=f'Find the least nonnegative residue of ({a}^{u}+{b}^{v}+{c})^{power} modulo {mod}.'
        add(f'Repeated modular squaring gives {a}^{u} mod {mod}={x}.',x);add(f'Repeated modular squaring gives {b}^{v} mod {mod}={y}.',y);add(f'The inner sum is congruent to {inner} modulo {mod}.',inner);add(f'{inner}^{power} mod {mod}={gold}.',gold)
    elif family==4:
        r1=rng.randint(-15,10);r2=r1+rng.randint(2,19);a=rng.randint(2,11);coef=-a*(r1+r2);constant=a*r1*r2;bound=rng.randint(r1-3,r2+2);valid=[x for x in [r1,r2] if x>bound];delta=coef**2-4*a*constant
        problem=f'Find the sum of distinct real solutions x>{bound} of {a}x²+({coef})x+({constant})=0. If there are none, return zero.'
        add(f'Discriminant D=({coef})²-4*{a}*({constant})={delta}, with square root {math.isqrt(delta)}.',delta);add(f'The two roots are {r1} and {r2}; substitution makes the polynomial zero at both.',[r1,r2]);add(f'The roots satisfying x>{bound} are {valid}.',valid);gold=sum(valid)
    elif family==5:
        a,b,c=rng.sample([2,3,4,5,7],3);total=rng.randint(35,95)
        valid=[(x,y,(total-a*x-b*y)//c) for x in range(total//a+1) for y in range(x+1,total//b+1) if total-a*x-b*y>=0 and (total-a*x-b*y)%c==0 and math.gcd(y,(total-a*x-b*y)//c)==1]
        brute=[(x,y,z) for x in range(total//a+1) for y in range(total//b+1) for z in range(total//c+1) if a*x+b*y+c*z==total and x<y and math.gcd(y,z)==1];assert valid==brute
        problem=f'Count nonnegative integer triples (x,y,z) with {a}x+{b}y+{c}z={total}, x<y, and gcd(y,z)=1. Use gcd(0,0)=0.'
        add(f'For each 0≤x≤{total//a} and x<y≤{total//b}, z must be ({total}-{a}x-{b}y)/{c}; keep only nonnegative integers and gcd(y,z)=1.','complete constrained enumeration');add(f'The complete solution list is {valid}.',valid);gold=len(valid)
    elif family==6:
        n=rng.randint(11,23);k=rng.randint(3,5);mod=rng.choice([3,5,7,11]);ev=rng.randint(1,k-1);target_r=rng.randrange(mod);dp={(0,0,0):1}
        for x in range(1,n+1):
            nxt=collections.Counter(dp)
            for (j,r,e),v in dp.items():
                if j<k and e+(x%2==0)<=ev:nxt[(j+1,(r+x)%mod,e+(x%2==0))]+=v
            dp=nxt
        gold=dp.get((k,target_r,ev),0);brute=sum(sum(xs)%mod==target_r and sum(x%2==0 for x in xs)==ev for xs in itertools.combinations(range(1,n+1),k));assert gold==brute
        problem=f'How many {k}-element subsets of {{1,...,{n}}} have exactly {ev} even elements and sum congruent to {target_r} modulo {mod}?'
        add('Use DP[j,r,e], counting subsets of processed elements with size j, sum residue r and e evens; initialize DP[0,0,0]=1 and all other states zero.','subset DP initial state');add(f'Processing a adds a skip branch and, if j<{k}, an include branch to (j+1,(r+a) mod {mod},e+[a even]). Each subset occurs exactly once.','subset DP transition');add(f'The required final entry DP[{k},{target_r},{ev}]={gold}; independent subset enumeration agrees.',gold)
    elif family==7 or family==16:
        n=rng.randint(4,7);target=rng.randint(10,42);mod=rng.choice([5,7,11,13,17,19]);pattern=rng.choice(['77','22','55','88']) if family==7 else '13';parity=None if family==7 else 1;gold=count_digits(n,target,mod,pattern,parity)
        extra='' if family==7 else ' and have an odd number of occurrences of the digit one'
        problem=f'Count {n}-digit positive decimal integers divisible by {mod}, with digit sum {target}, not containing the adjacent pattern {pattern}{extra}.'
        add(f'Build exactly {n} digits; the first digit is 1..9, later digits 0..9. Track residue, digit sum, previous digit'+(', and parity of the number of ones.' if parity is not None else '.'),'digit DP boundary');add(f'Appending d updates residue to (10r+d) mod {mod}, sum to s+d, and rejects a previous-digit/d pair equal to {pattern}.'+(' Toggle parity if d=1.' if parity is not None else ''),'digit DP transition');add(f'Sum the final states with residue zero, digit sum {target}'+(', odd parity' if parity is not None else '')+f': {gold}.',gold)
    elif family==8:
        n=rng.randint(100,1100);a,b,c=rng.sample([3,4,5,6,7,8,9,10,11],3);singles=sum(n//m for m in [a,b,c]);pairs=sum(n//math.lcm(*xs) for xs in itertools.combinations([a,b,c],2));triple=n//math.lcm(a,b,c);union=singles-pairs+triple;gold=n-union;assert gold==sum(all(x%m for m in [a,b,c]) for x in range(1,n+1))
        problem=f'How many integers in {{1,...,{n}}} are divisible by none of {a},{b},{c}?'
        add(f'Sum of individual multiple counts is {singles}.',singles);add(f'Sum of pairwise-intersection counts using least common multiples is {pairs}.',pairs);add(f'Triple-intersection count is {triple}.',triple);add(f'Inclusion-exclusion gives {union} integers divisible by at least one of the three numbers.',union)
    elif family==9:
        sides=rng.randint(4,9);value=rng.randint(1,sides);mod=rng.choice([3,4,5,7]);target_r=rng.randrange(mod);events=[xs for xs in itertools.product(range(1,sides+1),repeat=3) if xs.count(value)==1];wins=[xs for xs in events if sum(xs)%mod==target_r];gold=Fraction(len(wins),len(events))
        problem=f'Three independent fair dice each have faces 1..{sides}. Given that exactly one die shows {value}, find the probability that the sum is congruent to {target_r} modulo {mod}. Return a reduced fraction.'
        add(f'The conditioning event has 3*({sides}-1)^2={len(events)} equally likely ordered outcomes.',len(events));add(f'Enumerating the ordered outcomes in that event gives {len(wins)} with sum congruent to {target_r} modulo {mod}.',len(wins));add(f'Conditional probability is {len(wins)}/{len(events)}={gold}.',gold)
    elif family==10:
        bounds=[rng.randint(2,8) for _ in range(4)];target=rng.randint(6,sum(bounds)-2);dp=[1]+[0]*target
        for limit in bounds:dp=[sum(dp[i-j] for j in range(min(limit,i)+1)) for i in range(target+1)]
        gold=dp[target];brute=sum(sum(xs)==target for xs in itertools.product(*(range(u+1) for u in bounds)));assert gold==brute
        factors=' * '.join('(1+x+...+x^'+str(u)+')' for u in bounds)
        problem=f'Find the coefficient of x^{target} in {factors}.'
        add(f'The coefficient counts tuples (a,b,c,d) with respective bounds {bounds}, all nonnegative, and sum {target}.',bounds);add('Multiply factors by bounded convolution: new[i]=sum(old[i-j]) over allowed exponents j; start with old[0]=1.','bounded convolution');add(f'After all four factors the coefficient is {gold}; direct tuple enumeration agrees.',gold)
    elif family==11:
        b,c=rng.randint(12,55),rng.randint(12,55);d=Fraction(b,rng.randint(2,5));f=Fraction(c,rng.randint(2,5));m=(-d,2*f);nn=(2*d,-f);cross=m[0]*nn[1]-m[1]*nn[0];gold=abs(cross)/2
        problem=f'A=(0,0), B=({b},0), C=(0,{c}). D=({d},0) and F=(0,{f}). M is the reflection of D through F, N is the reflection of F through D. Find the area of triangle AMN exactly as an integer or reduced fraction.'
        add(f'M=2F-D=({m[0]},{m[1]}), N=2D-F=({nn[0]},{nn[1]}).',[str(x) for x in m+nn]);add(f'The determinant Mx*Ny-My*Nx={cross}.',cross);add(f'Area is half the absolute determinant, {gold}.',gold)
    elif family==12:
        x,y=rng.randint(1,8),rng.randint(1,8);a,b=rng.randint(1,4),rng.randint(1,3);n=rng.randint(9,18);mod=rng.choice([97,101,127]);seq=[x,y]
        for i in range(2,n+1):seq.append((a*seq[-1]+b*seq[-2])%mod)
        exact=[x,y]
        for i in range(2,n+1):exact.append(a*exact[-1]+b*exact[-2])
        gold=sum(seq)%mod;assert gold==sum(exact)%mod
        problem=f'u0={x}, u1={y}, and un={a}u(n-1)+{b}u(n-2) for n≥2. Find the least nonnegative residue of u0+...+u{n} modulo {mod}.'
        add(f'Keeping each recurrence value modulo {mod} preserves the final sum residue.','modular recurrence');add(f'The residues u0..u{n} are {seq}.',seq);add(f'Their sum modulo {mod} is {gold}.',gold)
    elif family==13:
        w,h=rng.randint(4,10),rng.randint(4,10);blocked=set(rng.sample([(x,y) for x in range(1,w) for y in range(1,h)],rng.randint(1,4)));dp={(0,0):1}
        for x in range(w+1):
            for y in range(h+1):
                if (x,y)==(0,0):continue
                dp[x,y]=0 if (x,y) in blocked else dp.get((x-1,y),0)+dp.get((x,y-1),0)
        gold=dp[w,h]
        problem=f'Count lattice paths from (0,0) to ({w},{h}), using only steps (1,0) and (0,1), that do not visit any point in {sorted(blocked)}.'
        add('Set ways(0,0)=1, ways=0 at forbidden points or outside the grid.','grid DP initial state');add('Every allowed interior point has ways(x,y)=ways(x-1,y)+ways(x,y-1), since the two possible last steps are disjoint.','grid DP transition');add(f'The final row at x={w}, indexed by y=0..{h}, is {[dp[w,y] for y in range(h+1)]}.',[dp[w,y] for y in range(h+1)]);add(f'ways({w},{h})={gold}.',gold)
    elif family==14:
        volume=rng.randint(100,450);r1,r2=rng.randint(3,16),rng.randint(3,16);drain=rng.randint(1,min(r1,r2)-1);fraction=Fraction(rng.randint(1,3),4);first=volume*fraction;remaining=volume-first;t1=first/r1;t2=remaining/(r1+r2-drain);gold=t1+t2
        problem=f'A tank needs {volume} liters to fill. First, only an inlet flowing {r1} liters/minute operates until the tank is {fraction} full. Then a second inlet of {r2} liters/minute and a drain of {drain} liters/minute both start; the first inlet keeps running. Find the total filling time as a reduced fraction of a minute.'
        add(f'First-stage volume={first}, time={first}/{r1}={t1}.',t1);add(f'Remaining volume={remaining}; second-stage net rate={r1+r2-drain}, time={t2}.',t2);add(f'Total time={t1}+{t2}={gold}.',gold)
    elif family==15:
        n=rng.randint(30,85);k=rng.randint(4,n-4);p,q=rng.sample([2,3,5,7],2);a,b=rng.randint(1,3),rng.randint(1,2);base=p**a*q**b
        def vf(t,prime):
            value=0
            while t:t//=prime;value+=t
            return value
        vp=vf(n,p)-vf(k,p)-vf(n-k,p);vq=vf(n,q)-vf(k,q)-vf(n-k,q);gold=min(vp//a,vq//b);x=math.comb(n,k);z=0
        while x%base==0:z+=1;x//=base
        assert gold==z
        problem=f'How many trailing zero digits does the integer binomial coefficient C({n},{k}) have when written in base {base}?'
        add(f'Base {base} has prime factorization {p}^{a}*{q}^{b}.',base);add(f'Legendre factorial valuations give v_{p}(C({n},{k}))={vp}, v_{q}(C({n},{k}))={vq}.',[vp,vq]);add(f'The maximum whole power of the base dividing the coefficient is min(floor({vp}/{a}),floor({vq}/{b}))={gold}.',gold)
    elif family==17:
        w,h=rng.randint(7,40),rng.randint(7,40);s=rng.randint(2,w-2);t=rng.randint(2,h-2);vertices=[(0,0),(w,0),(w-s,h),(0,h-t)];twice=abs(sum(x*vertices[(i+1)%4][1]-y*vertices[(i+1)%4][0] for i,(x,y) in enumerate(vertices)));boundary=sum(math.gcd(abs(vertices[(i+1)%4][0]-x),abs(vertices[(i+1)%4][1]-y)) for i,(x,y) in enumerate(vertices));gold=Fraction(twice-boundary,2)+1;assert gold.denominator==1;gold=int(gold)
        problem=f'A simple lattice polygon has vertices {vertices} in that order. How many integer lattice points lie strictly inside it?'
        add(f'Shoelace doubled area={twice}.',twice);add(f'Boundary lattice-point count is the sum of gcd(|dx|,|dy|) over edges, {boundary}.',boundary);add(f'Pick theorem I=area-B/2+1 gives {gold}.',gold)
    elif family==18:
        lo=rng.randint(2,17);hi=lo+rng.randint(5,26);a=rng.randint(2,9);terms=[Fraction(a,k*(k+1)) for k in range(lo,hi+1)];gold=sum(terms,Fraction(0));assert gold==a*(Fraction(1,lo)-Fraction(1,hi+1))
        problem=f'Find the reduced fraction sum of {a}/(k(k+1)) for integer k from {lo} through {hi} inclusive.'
        add(f'{a}/(k(k+1))={a}(1/k-1/(k+1)) for every positive k.','exact rational partial fractions');add(f'All interior reciprocals cancel, leaving {a}(1/{lo}-1/{hi+1}).','telescoping boundary');add(f'The reduced fraction is {gold}.',gold)
    elif family==19:
        n=rng.randint(7,15);k=rng.randint(1,n-3);r=n-k;ds=[1,0]
        for i in range(2,r+1):ds.append((i-1)*(ds[-1]+ds[-2]))
        inclusion=sum((-1)**j*math.comb(r,j)*math.factorial(r-j) for j in range(r+1));assert ds[r]==inclusion;gold=math.comb(n,k)*ds[r]
        problem=f'Count permutations of {{1,...,{n}}} with exactly {k} fixed points.'
        add(f'Choose the {k} fixed positions in C({n},{k})={math.comb(n,k)} ways.',math.comb(n,k));add(f'The remaining {r} positions must form a derangement; D0=1,D1=0,Dj=(j-1)(D(j-1)+D(j-2)) gives D{r}={ds[r]}.',ds[r]);add(f'Multiply the two independent choices to get {gold}.',gold)
    else:raise ValueError('Unknown family')
    add(f'The required final answer is {gold}.',gold)
    suffix='\nGive a finite checked solution and end with the result inside \\boxed{}.'
    instruction=problem+suffix
    return dict(id=f'e019-{split}-math-{family:02d}-{serial:06d}'+('-repair' if repair else ''),instruction=instruction,base_instruction=instruction,sha256=hashlib.sha256(instruction.encode()).hexdigest(),gold=str(gold),facts=facts,family=FAMILIES[family],family_index=family,serial=serial,split=split,category='math_repair' if repair else 'math',enable_thinking=True,repair=repair,validator='exact_certificate',verification_scope='Exact computed reference and listed calculation/algorithm facts; any natural-language explanation outside those facts is not formally verified')
