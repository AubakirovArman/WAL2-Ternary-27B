"""Reproduce six subject-level checks; benchmark examples remain reports only."""
import itertools
import json
import math
from pathlib import Path

import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/e019-error-review'


def main():
    tasks={r['id']:r for r in json.loads((ROOT/'reports/e019-d/failure-review100.json').read_text())};rows=[]
    def add(i,quote,category,correction,evidence,earliest):
        text=tasks[i]['raw_response'];assert quote in text,(i,quote)
        rows.append(dict(id=i,family=tasks[i]['family'],first_invalid_quote=quote,quote_offset=text.index(quote),category=category,correct_statement=correction,independent_evidence=evidence,error_independently_confirmed=True,earliest_error_status=earliest,training_use=False))
    bases=[b for b in range(10,57) if (9*b+7)%(b+7)==0];assert bases==[21,49]
    add(0,'97_b = 9·b² + 7·b = 9b² + 7b','place_value','97_b=9b+7. Since9b+7=9(b+7)-56, b+7 must divide56. Valid bases21,49; sum70.',dict(bases=bases,sum=70,b10_true=97,b10_false=970,finite_bound='b+7 positive divisor56 impliesb<=49'),'Earliest substantive error confirmed')
    triples=[(c,v,s) for c in range(1,10) for v in range(1,c) for s in range(1,v) if c+v+s==9];N=sum(math.factorial(9)//(math.factorial(c)*math.factorial(v)*math.factorial(s)) for c,v,s in triples);assert N==2016
    add(16,'C(9,6) × C(3,2) × C(1,1) = 3 × 3 × 1 = 9','arithmetic','C(9,6)=84, so this contribution is252. Full count2016 gives remainder16, not72.',dict(triples=triples,total=N,remainder=N%1000,binomial96=math.comb(9,6)),'Arithmetic error confirmed; earliest claim not asserted because preceding abandoned candidates need interpretation')
    x,y=sp.symbols('x y');factor=(3*x+2*y)*(4*x-3*y);assert sp.expand(factor)==12*x*x-x*y-6*y*y
    pairs=[(a,b) for a in range(-100,101) for b in range(-100,101) if 12*a*a-a*b-6*b*b==0];assert len(pairs)==117
    add(24,'(3x - y)(4x - 2y) = 12x^2 - 6xy - 4xy - 2y^2','sign_in_expansion','The last product term is+2y², not-2y². The correct form factors as(3x+2y)(4x-3y). Branches(2t,-3t),(3t,4t) give67+51-1=117pairs.',dict(correct_factor_identity_zero=str(sp.expand(factor-(12*x*x-x*y-6*y*y))),wrong_expansion_counterexample=dict(x=0,y=1,true_product=2,claimed_product=-2),finite_pair_count=len(pairs),teacher_nonfactorization_rejected=True),'Earliest consequential expansion error confirmed')
    chosen=[1,2,4,5,7,8,10,11];assert len(chosen)==8 and all(not(i-1 in chosen and i+1 in chosen) for i in chosen)
    count=sum(not any(a+1 in xs and a+2 in xs for a in xs) for xs in itertools.combinations(range(16),8));blocks=sum(math.comb(8-k,k)*math.comb(9,8-k) for k in range(5));assert count==blocks==2907
    add(192,'the constraint is that no two selected chairs are adjacent','wrong_definition','The constraint excludes three consecutive occupied chairs; adjacent pairs are allowed. Count2907 gives remainder907.',dict(valid_counterexample_chairs=chosen,exhaustive_count=count,independent_block_formula=blocks),'Earliest misinterpretation confirmed')
    add(140,'The smallest possible value of $c$ would be $0$, since $c$ is a positive constant.','domain_constraint','Zero is not positive. For the supplied graph the phases arepi+2k*pi; smallest positive phasepi.',dict(zero_satisfies_positive=bool(0>0),source_graph='2*sin(3*x+pi)+1'),'Earliest incorrect bound confirmed; phase conclusion follows from explicit source function')
    x,p,q,r=sp.symbols('x p q r');f=x**3-3*x**2+4*x-1;g=x**9+p*x**6+q*x**3+r;rem=sp.rem(g,f,x);solution=sp.solve(sp.Poly(rem,x).all_coeffs(),(p,q,r));assert solution=={p:6,q:31,r:-1} and sp.rem(g.subs(solution),f,x)==0
    add(453,'Let h(x) = ax^5 + bx^4 + cx^3 + dx^2 + ex + f, where a ≠ 0.','polynomial_degree','The quotient must have degree6:9minus3. A degree5quotient produces degree8, not9. Exact polynomial remainders give(p,q,r)=(6,31,-1).',dict(dividend_degree=9,divisor_degree=3,required_quotient_degree=6,claimed_product_degree=8,solution={str(k):int(v) for k,v in solution.items()},substituted_remainder='0'),'Earliest degree mismatch confirmed')
    (OUT/'manual-confirmations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    lines=['# Проверенные ошибки E018 и ограничения автоматического разбора','', 'Qwen просмотрел100сохранённыхошибочныхответов.83разбора прошли проверку цитат и формата;17отклонены. Эти83разбора не являются83доказаннымипервымиошибками. Арифметическое неравенство в witness не устанавливает его связь с условием.','', 'Шесть примеров дополнительно проверены кодом или непосредственной проверкой условия. У пяти найден ранний содержательный сбой; у одного подтверждён арифметический дефект без утверждения о первом месте. Все эти бенчмарковые примеры исключены из обучения.','', '| ID | Подтверждённый дефект и правильное действие |','|---|---|']
    for row in rows:lines.append(f"|{row['id']}|{row['category']}: {row['correct_statement']}|")
    lines+=['','Сам Qwen ошибся при разборе: уID0принял неверное прочтение97_b и указал позднее сложение; уID16назвал72правильным вместо16; уID24заявил, что многочлен не раскладывается над целыми. Эти утверждения отклонены и не становятся учебными метками.','', 'Остальные94примера пока не имеют индивидуального подтверждения предметным разбором. Их диагнозы остаются кандидатными. Новый пилот обучающих задач построен независимо: точные ответы и перечисленные вычислительные факты воспроизводятся кодом; свободная проза Qwen не объявляется формально доказанной.']
    (OUT/'REVIEW_RU.md').write_text('\n'.join(lines)+'\n');print('PASS:six errors checked; benchmark training_use=False')


if __name__=='__main__':main()
