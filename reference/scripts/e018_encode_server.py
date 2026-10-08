"""Isolated tokenizer process using the existing training environment."""
import json,sys
from transformers import AutoTokenizer
from reasoning_data import encode,SOURCE
tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
    try:
        row=json.loads(line);row['verification']='teacher_answer_and_reasoning_unverified'
        item=encode(row,tok,12288,allow_validation=True,allow_teacher_only=True)
        print(json.dumps({'item':item}),flush=True)
    except Exception as e:print(json.dumps({'error':str(e)}),flush=True)
