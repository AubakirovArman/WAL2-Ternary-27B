"""Read-only display of a training step and original teacher answer."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('step',type=int);a=p.parse_args()
order=json.loads((ROOT/'runs/e018-fresh25000/order.json').read_text())
if not 1<=a.step<=len(order):p.error('Шаг должен быть от1 до25000')
j=order[a.step-1]
with (ROOT/'data/e018-25000/accepted.jsonl').open() as f:
 for line in f:
  r=json.loads(line)
  if r['id']==j['reasoning']:break
 else:raise RuntimeError('Пример не найден')
print(f"Шаг {a.step}; ID {r['id']}; категория {r['category']}; источник {r.get('source','синтетический')}; replay ID {j['kd']}")
print('\nВОПРОС\n'+r['instruction']+'\n\nРАССУЖДЕНИЕ УЧИТЕЛЯ\n'+r['reasoning']+'\n\nОТВЕТ УЧИТЕЛЯ\n'+r['response'])
