"""One bounded CPU response scorer. Official final-only score, no proof promotion."""
import json,signal,sys
from e018_tasks import verify_task
from collect_e019_pilot import verified_function
def score(task,response):
    def timeout(*_):raise TimeoutError('Scoring exceeded20s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(20)
    try:
        if not response['thinking_completed']:return dict(correct=False,verification_error=None)
        answer=response['answer']
        if task.get('category','math')=='math':
            from math_verify import parse,verify,LatexExtractionConfig,ExprExtractionConfig
            gold=parse('\\boxed{'+task['gold']+'}',extraction_config=[LatexExtractionConfig()])
            pred=parse(answer,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()])
            correct=bool(gold and pred and verify(gold,pred))
        elif task['category']=='code':correct=verified_function(task,answer)
        else:correct=verify_task(task,answer)
        return dict(correct=correct,verification_error=None)
    except Exception as e:return dict(correct=False,verification_error=str(e)[:300])
    finally:signal.alarm(0)
if __name__=='__main__':
    value=json.loads(sys.stdin.read());print(json.dumps(score(value['task'],value['response'])))
