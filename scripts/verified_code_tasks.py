"""Synthetic training tasks and CPU fixtures; independent of benchmark files."""
import random


def solve(kind, values, a, b):
    if kind == 0:
        return [sum(v % a == r for v in values) for r in range(a)]
    if kind == 1:
        return [sum(values[i:i+a]) for i in range(max(0, len(values)-a+1))]
    if kind == 2:
        out = []
        for v in values:
            if out and out[-1][0] == v:
                out[-1][1] += 1
            else:
                out.append([v, 1])
        return out
    if kind == 3:
        return [sum(v * a**i for i, v in enumerate(values[:end])) % b for end in range(1, len(values)+1)]
    if kind == 4:
        return [v for start in range(0, len(values), a) for v in values[start:start+a][::-1]]
    if kind == 5:
        out = []
        for v in values:
            if not out or abs(v-out[-1]) >= a:
                out.append(v)
        return out
    if kind == 6:
        return [sum(values[max(0, i-a):i]) for i in range(len(values))]
    if kind == 7:
        total = b
        out = []
        for v in values:
            total = (a*total+v) % 101
            out.append(total)
        return out
    raise ValueError(kind)


def tasks(count=128):
    if not 8 <= count <= 1024:
        raise ValueError('Use 8..1024 tasks')
    rng = random.Random(2026091923)
    rows = []
    for i in range(count):
        kind=i%8; a=rng.randint(2,7); b=rng.randint(11,43); name=f'transform_{i:03d}'
        specs=[
            f'Return a list of {a} counts: entry r counts the input integers whose Python remainder modulo {a} is r, in ascending remainder order.',
            f'Return sums of all consecutive windows of exactly {a} elements, left to right. Return [] if the input is shorter than {a}.',
            'Run-length encode consecutive equal integers into a list of [value, count] pairs. Nonconsecutive equal values must remain separate.',
            f'For each nonempty prefix, return its polynomial value modulo {b}, with the element at original zero-based index j multiplied by {a}**j. Return these values in prefix order.',
            f'Split into consecutive blocks of at most {a} elements, reverse each block including the last short block, and concatenate the blocks.',
            f'Keep the first element if present. Scan subsequent elements left to right and keep an element exactly when its absolute difference from the most recently kept element is at least {a}.',
            f'For each index i return the sum of up to {a} elements immediately before i, excluding the element at i. The first output is 0.',
            f'Start an accumulator at {b}. For each input element v, replace it with ({a} * accumulator + v) modulo 101. Return the accumulator after every update.'
        ]
        values=[[],[0],[1,-1,0,1],[3,3,-2,-2,-2,3],list(range(-8,9)),[0]*12]
        values += [[rng.randint(-15,20) for _ in range(rng.randint(0,32))] for _ in range(18)]
        fixtures=[{'input':v,'expected':solve(kind,v,a,b)} for v in values]
        prompt=(f'Implement Python function {name}(values). The input is a list of integers and must not be mutated. '
                +specs[kind]+' Use only the Python standard library. Empty input is valid. '
                f'Example: {name}({values[3]!r}) returns {fixtures[3]["expected"]!r}. '
                'Provide exactly one fenced Python code block containing the complete function as your final answer.')
        rows.append({'id':f'verified-code-{i:04d}','split':'train','topic':f'synthetic_code_family_{kind}',
                     'language':'English','instruction':prompt,'entry_point':name,'fixtures':fixtures,
                     'generator_parameters':{'kind':kind,'a':a,'b':b}})
    return rows


def test_program(row, code):
    # Expectations are fixed before the teacher sees the task. Inputs are fresh per call.
    lines=[code, '\n']
    for fixture in row['fixtures']:
        lines += [f"_original = {fixture['input']!r}", '_argument = list(_original)',
                  f"_actual = {row['entry_point']}(_argument)",
                  f"assert _actual == {fixture['expected']!r}, 'wrong result'",
                  "assert _argument == _original, 'mutated input'"]
    return '\n'.join(lines)
