"""Deterministic synthetic compositions; references computed before teacher requests."""
import hashlib,itertools,json,math,random,re
from fractions import Fraction
from verified_code_tasks import solve

TARGET={'math':4000,'code':2500,'instructions':2000,'grounded':1500}
HOLDOUT={'math':200,'code':125,'instructions':100,'grounded':75}
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def make_task(category,i,split):
    rng=random.Random(20260922000+i+1000000*list(TARGET).index(category)+(10000000 if split=='validation' else 0))
    ru=i%2==0; held=split=='validation'; language='Russian' if ru else 'English'
    a,b,c,d=[rng.randint(3,70) for _ in range(4)]
    row={'id':f'fresh10k-{split}-{category}-{i:06d}','split':split,'category':category,'language':language,'enable_thinking':True}
    if category=='math':
        # Held-out families, not merely new parameters.
        family=(i%4+12) if held else i%12
        n=rng.randint(4,12); k=rng.randint(2,n-1)
        if family==0:
            c=min(c,a*b)
            en=f'A shop buys {a} boxes of {b} items and sells {c} items. It then buys {d} more boxes of {b} items. How many items remain?';ruq=f'Магазин купил {a} коробок по {b} предметов и продал {c} предметов. Затем купил ещё {d} коробок по {b} предметов. Сколько предметов осталось?';answer=(a+d)*b-c
        elif family==1:
            en=f'A train travels {a} km at {b} km/h, then {c} km at {d} km/h. Find the total travel time in hours as a reduced fraction.';ruq=f'Поезд проехал {a} км со скоростью {b} км/ч, затем {c} км со скоростью {d} км/ч. Найди общее время в часах в виде несократимой дроби.';answer=Fraction(a,b)+Fraction(c,d)
        elif family==2:
            total=a*b+c
            en=f'A rectangle has width {a} and perimeter {2*(a+b)}. Find its area minus {c}.';ruq=f'Ширина прямоугольника {a}, периметр {2*(a+b)}. Найди площадь минус {c}.';answer=a*b-c
        elif family==3:
            en=f'A price of {a*100} is reduced by {b} percent then increased by {c} percent. Find the final price exactly.';ruq=f'Цену {a*100} снизили на {b} процентов, затем повысили на {c} процентов. Найди точную итоговую цену.';answer=Fraction(a*(100-b)*(100+c),100)
        elif family==4:
            en=f'How many unordered groups of {k} people can be chosen from {n} people if one particular person must be included?';ruq=f'Сколько групп из {k} человек можно выбрать из {n}, если один конкретный человек обязательно должен входить в группу?';answer=math.comb(n-1,k-1)
        elif family==5:
            en=f'A bag contains {a} red and {b} blue balls. Two balls are drawn without replacement. What is the probability both are red?';ruq=f'В мешке {a} красных и {b} синих шаров. Два извлекают без возвращения. Какова вероятность, что оба красные?';answer=Fraction(a*(a-1),(a+b)*(a+b-1))
        elif family==6:
            en=f'A sequence starts at {a} and increases by {b} each term. Find the sum of its first {n} terms.';ruq=f'Первый член последовательности {a}, каждый следующий больше на {b}. Найди сумму первых {n} членов.';answer=n*(2*a+(n-1)*b)//2
        elif family==7:
            en=f'Find x+y if {a}*x+{b}*y={a*c+b*d} and {a+1}*x+{b}*y={(a+1)*c+b*d}.';ruq=f'Найди x+y, если {a}*x+{b}*y={a*c+b*d} и {a+1}*x+{b}*y={(a+1)*c+b*d}.';answer=c+d
        elif family==8:
            duration=Fraction(a*b*k,(a+b)*n)
            en=f'Two workers finish a job alone in {a} and {b} hours. They work together for {duration} hours. What fraction of the job remains?';ruq=f'Два работника выполняют работу по отдельности за {a} и {b} часов. Они работают вместе {duration} часов. Какая доля работы осталась?';answer=Fraction(n-k,n)
        elif family==9:
            en=f'A tank initially has {a*b} liters. It gains {c+d} liters/minute and loses {c} liters/minute for {b} minutes. How much remains?';ruq=f'В баке {a*b} литров. В течение {b} минут в него поступает {c+d} л/мин и вытекает {c} л/мин. Сколько литров стало?';answer=b*(a+d)
        elif family==10:
            en=f'Find the smallest integer greater than {a*b} that is divisible by both {b} and {c}.';ruq=f'Найди наименьшее целое число больше {a*b}, делящееся и на {b}, и на {c}.';l=math.lcm(b,c);answer=(a*b//l+1)*l
        elif family==11:
            en=f'{a} values have average {b}. Add {c} values each equal to {d}. What is the new average?';ruq=f'Среднее {a} чисел равно {b}. Добавили {c} чисел, каждое равно {d}. Каково новое среднее?';answer=Fraction(a*b+c*d,a+c)
        elif family==12:
            en=f'How many integer lattice points lie strictly inside a rectangle with vertices (0,0), ({a},0), ({a},{b}), (0,{b})?';ruq=f'Сколько точек с целыми координатами строго внутри прямоугольника (0,0), ({a},0), ({a},{b}), (0,{b})?';answer=(a-1)*(b-1)
        elif family==13:
            en=f'Find the sum of all positive divisors of {a*b}.';ruq=f'Найди сумму всех положительных делителей {a*b}.';answer=sum(x for x in range(1,a*b+1) if a*b%x==0)
        elif family==14:
            en=f'How many length-{n} binary strings contain exactly {k} ones and start with 0?';ruq=f'Сколько двоичных строк длины {n} содержат ровно {k} единиц и начинаются с 0?';answer=math.comb(n-1,k)
        else:
            en=f'An urn has labels 1 through {a}. Select one uniformly. Find the expected square of the selected label.';ruq=f'Равновероятно выбирают одно число от 1 до {a}. Найди математическое ожидание его квадрата.';answer=Fraction((a+1)*(2*a+1),6)
        row.update(instruction=(ruq if ru else en)+('\nОбъясни кратко. Заверши строкой ANSWER: число или несократимая дробь.' if ru else '\nExplain briefly. End with ANSWER: number or reduced fraction.'),topic=f'math_{family}',gold=str(answer),validator='fraction')
    elif category=='code':
        combos=list(itertools.permutations(range(8),3));pool=[x for j,x in enumerate(combos) if (j%5==0)==held];combo=pool[i%len(pool)]
        a=rng.randint(2,7);b=rng.randint(11,43);name='transform'
        specs=[f'count each remainder modulo {a}, yielding a list of length {a}',f'compute sums of all consecutive windows of exactly {a} elements', 'run-length encode consecutive equal values and flatten the [value,count] pairs',f'compute polynomial values of every nonempty prefix modulo {b}, weighting index j by {a}**j',f'reverse each block of at most {a} elements, including the last short block',f'keep the first element, then only elements whose absolute difference from the last kept is at least {a}',f'for index i compute the sum of up to {a} preceding elements, excluding i',f'start t={b}, update t=({a}*t+v)%101 for each value, return all updated t values']
        def reference(v):
            for kind in combo:
                v=solve(kind,v,a,b)
                if kind==2:v=[x for pair in v for x in pair]
            return v
        values=[[],[0],[1,-1,0,1],[3,3,-2,-2,-2,3],list(range(-8,9)),[0]*12]+[[rng.randint(-20,25) for _ in range(rng.randint(0,36))] for _ in range(24)]
        fixtures=[{'input':v,'expected':reference(v)} for v in values]
        inst=('Реализуй Python-функцию transform(values). Вход — список целых, не изменяй его. Последовательно выполни три операции, каждая получает результат предыдущей:\n' if ru else 'Implement Python function transform(values). Input is a list of integers; do not mutate it. Apply these three operations in order, each to the result of the previous:\n')+'\n'.join(f'{j+1}. {specs[k]}' for j,k in enumerate(combo))
        inst+=f'\nExample: transform({values[3]}) == {fixtures[3]["expected"]}. Empty input is valid. Final answer: exactly one fenced Python code block. Use only the standard library.'
        row.update(instruction=inst,topic='code_'+ '_'.join(map(str,combo)),validator='code',entry_point=name,fixtures=fixtures,parameters={'combo':combo,'a':a,'b':b})
    else:
        records=[{'id':f'item-{rng.randrange(10000,99999)}','group':rng.choice(['A','B','C']),'price':rng.randint(10,500),'qty':rng.randint(0,15)} for _ in range(7)]
        family=(i%2+6) if held else i%6
        if family==0:operation=f'Select records with price >= {a*5}. Sort by (price,id) ascending. Return their ids.';gold=[r['id'] for r in sorted(records,key=lambda r:(r['price'],r['id'])) if r['price']>=a*5]
        elif family==1:operation='Return total price*qty for each group A,B,C, including zero totals.';gold={g:sum(r['price']*r['qty'] for r in records if r['group']==g) for g in 'ABC'}
        elif family==2:operation='Return ids of the two highest price*qty records, sorted descending by price*qty, breaking ties by ascending id.';gold=[r['id'] for r in sorted(records,key=lambda r:(-r['price']*r['qty'],r['id']))[:2]]
        elif family==3:operation=f'Return the number of records with qty>{a%10} and price<{b*5}.';gold=sum(r['qty']>a%10 and r['price']<b*5 for r in records)
        elif family==4:operation='Return ids in original order where qty is zero OR group is B.';gold=[r['id'] for r in records if r['qty']==0 or r['group']=='B']
        elif family==5:operation='Return maximum price minus minimum price for each group A,B,C; use null for absent groups.';gold={g:(max(v)-min(v) if (v:=[r['price'] for r in records if r['group']==g]) else None) for g in 'ABC'}
        elif family==6:operation='Return the ids of all records whose price is strictly above the mean price of the records in their own group, in original order.';gold=[r['id'] for r in records if r['price']*sum(t['group']==r['group'] for t in records)>sum(t['price'] for t in records if t['group']==r['group'])]
        else:operation='Return total qty of records having strictly positive qty for each group, but only include groups with at least two such records.';gold={g:sum(r['qty'] for r in records if r['group']==g and r['qty']>0) for g in 'ABC' if sum(r['group']==g and r['qty']>0 for r in records)>=2}
        prefix='Используй только исходные записи. ' if ru else 'Use only the supplied records. '
        if category=='instructions':
            inst=prefix+operation+'\nRecords: '+json.dumps(records)+'\nFinal answer must be exactly a JSON object with the single key result. No Markdown or extra keys.';gold={'result':gold}
        else:
            inst=prefix+operation+'\nRecords: '+json.dumps(records)+'\nExplain the calculation briefly, then end with a separate line ANSWER_JSON: followed by the JSON result. Do not introduce facts beyond these records.'
        row.update(instruction=inst,topic=category+'_'+str(family),gold=gold,validator='json' if category=='instructions' else 'grounded_json')
    row['sha256']=digest(row['instruction']);return row

def verify(row,answer):
    if row['validator']=='fraction':
        m=re.search(r'ANSWER:\s*(-?\d+(?:/\d+|\.\d+)?)\s*$',answer)
        return bool(m) and Fraction(m[1])==Fraction(row['gold'])
    if row['validator']=='code':
        from code_sandbox import run
        from verified_code_tasks import test_program
        m=re.fullmatch(r'\s*```(?:python|py)\s*\n(.*?)```\s*',answer,re.S)
        return bool(m) and '```' not in m[1] and run(test_program(row,m[1]))['passed']
    try:
        text=answer.strip() if row['validator']=='json' else re.search(r'(?:^|\n)ANSWER_JSON:\s*(.+)\s*$',answer)[1]
        actual=json.loads(text)
        # JSON booleans must not pass as integer answers.
        return json.dumps(actual,sort_keys=True)==json.dumps(row['gold'],sort_keys=True)
    except (ValueError,TypeError,AttributeError):return False
