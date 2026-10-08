"""Read-only progress observer; does not restart or modify the frozen collector."""
import argparse,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/e018-25000'
def metrics():
    s=json.loads((DATA/'status.json').read_text());now=time.time();end=s['updated'];start=max(end-600,end-s['segment_seconds'])
    train=val=0
    with (DATA/'events.jsonl').open() as f:
        for line in f:
            if not line.endswith('\n'):continue
            e=json.loads(line)
            if e.get('accepted') and start<e.get('time',0)<=end:
                if e['split']=='train':train+=1
                else:val+=1
    elapsed=end-start;rate=(train+val)/elapsed if elapsed>0 else 0
    remaining=max(0,s['target_train']-s['train'])+max(0,s['target_validation']-s['validation'])
    available=s['state']=='collecting' and now-end<120 and elapsed>=60 and train+val>=20
    eta=remaining/rate if available and rate else (0 if s['state']=='completed' else None)
    return dict(updated=end,state=s['state'],train=s['train'],validation=s['validation'],train_per_minute=train/max(elapsed,1)*60,validation_per_minute=val/max(elapsed,1)*60,accepted_per_minute=rate*60,window_seconds=elapsed,tokens_per_second=s['tokens_per_second'],eta_seconds=eta,remaining=remaining,stale=now-end>=120)
def duration(seconds):
    minutes=max(1,round(seconds/60)) if seconds>0 else 0
    return f'{minutes//60} ч {minutes%60:02d} мин'
def lines(m):
    period=f"{m['window_seconds']/60:.0f} мин"
    rate=f"Скорость за последние {period}: {m['accepted_per_minute']:.1f} принятых примеров/мин (обучение {m['train_per_minute']:.1f}, контроль {m['validation_per_minute']:.1f})"
    eta='Осталось примерно: '+duration(m['eta_seconds'])+' до полного набора 25000 + 1500' if m['eta_seconds'] is not None else 'Осталось примерно: оценка недоступна — мало данных, сбор остановлен или статус устарел'
    return [rate,eta,'Оценка по текущему темпу; состав оставшихся задач и отбраковка могут изменить срок.']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args();previous=None
    while True:
        m=metrics()
        if not a.watch:
            print('\n'.join(lines(m)));return
        key=(m['updated'],m['state'],m['stale'])
        if key!=previous:
            text=time.strftime('%H:%M:%S')+f" | Готово {m['train']}/25000; контроль {m['validation']}/1500 | "+lines(m)[0]+' | '+lines(m)[1]+f" | Токены: {m['tokens_per_second']:.0f}/с в среднем с запуска\n"
            fd=os.open(ROOT/'logs/e018-data.log',os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o644)
            try:os.write(fd,text.encode())
            finally:os.close(fd)
            temp=DATA/'progress-estimate.json.tmp';temp.write_text(json.dumps(m,ensure_ascii=False,indent=2));temp.replace(DATA/'progress-estimate.json');previous=key
        if m['state']=='completed':return
        time.sleep(60)
if __name__=='__main__':main()
