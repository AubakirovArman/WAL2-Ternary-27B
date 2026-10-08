"""Independent boundary checks for the new exact references. CPU only."""
import ast
import collections
import itertools
import json
import math
import re
from fractions import Fraction
from pathlib import Path

from e019_math_tasks import make_math
from collect_e019_pilot import validate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e019-reasoning-pilot-v1'


def digit_mitm(n,target,mod,pattern,parity):
    ln=n//2;rn=n-ln;left=collections.Counter();right=collections.Counter()
    for x in range(10**(ln-1),10**ln):
        s=str(x);total=sum(map(int,s))
        if pattern not in s and total<=target:left[x%mod,total,s[-1],s.count('1')%2]+=1
    for x in range(10**rn):
        s=str(x).zfill(rn);total=sum(map(int,s))
        if pattern not in s and total<=target:right[x%mod,total,s[0],s.count('1')%2]+=1
    total=0;factor=pow(10,rn,mod)
    for (r,s,last,p),v in left.items():
        need=(-r*factor)%mod
        for first in '0123456789':
            if last+first==pattern:continue
            for rp in (0,1):
                if parity is None or p^rp==parity:total+=v*right.get((need,target-s,first,rp),0)
    return total


def polygon_interior(vertices):
    # Direct strict half-plane enumeration for this convex quadrilateral.
    total=0
    for x in range(max(a for a,b in vertices)+1):
        for y in range(max(b for a,b in vertices)+1):
            cross=[]
            for i,(a,b) in enumerate(vertices):
                c,d=vertices[(i+1)%len(vertices)];cross.append((c-a)*(y-b)-(d-b)*(x-a))
            total+=all(v>0 for v in cross) or all(v<0 for v in cross)
    return total


def grid_combinatorial(w,h,blocked):
    def paths(a,b):
        dx,dy=b[0]-a[0],b[1]-a[1]
        return math.comb(dx+dy,dx) if dx>=0 and dy>=0 else 0
    ordered=sorted(blocked,key=lambda p:(sum(p),p));arrivals={}
    for i,p in enumerate(ordered):arrivals[p]=paths((0,0),p)-sum(arrivals[q]*paths(q,p) for q in ordered[:i])
    return paths((0,0),(w,h))-sum(arrivals[p]*paths(p,(w,h)) for p in ordered)


def main():
    tasks=[json.loads(l) for l in (OUT/'tasks.jsonl').read_text().splitlines()];byfamily={f:[t for t in tasks if t.get('family_index')==f and t['category']=='math'] for f in range(20)}
    checks=[]
    # Every certificate must regenerate identically from the frozen task fields.
    for task in tasks:
        if task['category'] not in ('math','math_repair'):continue
        rebuilt=make_math(task['family_index'],task['serial'],task['split'],task['repair'])
        assert task['facts']==rebuilt['facts'] and task['gold']==rebuilt['gold'] and task['base_instruction']==rebuilt['base_instruction']
    for family in (0,7,13,16,17):
        for task in byfamily[family][:12]:
            text=task['base_instruction']
            if family==0:
                base=int(re.search(r'In base (\d+)',text)[1]);digits=re.findall(r'\[[^\]]+\]',text);x=y=0
                for d in ast.literal_eval(digits[0]):x=x*base+d
                for d in ast.literal_eval(digits[1]):y=y*base+d
                modulus=int(re.search(r'modulo (\d+)',text)[1]);actual=math.gcd(x,y)+x%modulus
            elif family in (7,16):
                n=int(re.search(r'Count (\d+)-digit',text)[1]);mod=int(re.search(r'divisible by (\d+)',text)[1]);target=int(re.search(r'digit sum (\d+)',text)[1]);pattern=re.search(r'adjacent pattern (\d+)',text)[1];actual=digit_mitm(n,target,mod,pattern,1 if family==16 else None)
            elif family==13:
                w,h=map(int,re.search(r'to \((\d+),(\d+)\)',text).groups());blocked=ast.literal_eval(re.search(r'point in (\[.*?\])',text)[1]);actual=grid_combinatorial(w,h,blocked)
            elif family==17:
                vertices=ast.literal_eval(re.search(r'vertices (\[.*?\])',text)[1]);actual=polygon_interior(vertices)
            assert str(actual)==task['gold'],(task['id'],actual,task['gold'])
            checks.append(dict(id=task['id'],family=family,gold=task['gold'],independent_value=str(actual),passed=True))
    sample=byfamily[0][0]
    obj=dict(steps=[dict(id=f['id'],explanation='This follows directly from the stated definition and the checked calculation.') for f in sample['facts']],final=sample['gold'])
    def result(x):return dict(raw=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(x)))]))
    assert validate(sample,result(obj))['accepted']
    wrong=dict(obj,final=str(int(sample['gold'])+1));assert not validate(sample,result(wrong))['accepted']
    missing=dict(obj,steps=obj['steps'][:-1]);assert not validate(sample,result(missing))['accepted']
    bad=dict(obj,steps=[dict(s) for s in obj['steps']]);bad['steps'][0]['explanation']='The unchecked equality is 1=2.';assert not validate(sample,result(bad))['accepted']
    repair=next(t for t in tasks if t['category']=='math_repair');rep=dict(steps=[dict(id=f['id'],explanation='This follows directly from the stated definition and the checked calculation.') for f in repair['facts']],final=repair['gold']);assert validate(repair,result(rep))['accepted']
    report=dict(state='passed',certificate_regeneration_count=sum(t['category'] in ('math','math_repair') for t in tasks),independent_checks=checks,negative_checks=['wrong final rejected','missing certificate step rejected','new unchecked equation rejected'],repair_target_checked=True,scope='Listed calculation reference checks; does not formally validate future free teacher prose')
    (OUT/'reference-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(f"PASS: {len(checks)}independent checks; all math certificates regenerated; invalid targets rejected.")


if __name__=='__main__':main()
