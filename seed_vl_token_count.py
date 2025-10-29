import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer

path = '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/data/oodvqa-8488_seed_1.6_vision.jsonl'
qwen_tokenizer_path = '/home/fangdong.wang/mlm-evaluator/pretrained_vlm/Qwen3-4B'
try:
    tokenizer = AutoTokenizer.from_pretrained(
            qwen_tokenizer_path,
            trust_remote_code=True,
            use_fast=True  
        )
except Exception as e:
        raise RuntimeError(f"加载中文Tokenizer失败！错误：{str(e)}")

data = []
with open(path,'r',encoding='utf-8') as f:
    for l in f:
        data.append(json.loads(l))
data = data[:20]

input_tokens = 0
output_tokens = 0 
for item in tqdm(data):
    question = item.get("question",None)
    response  = item["rstar"][next(reversed(item["rstar"].keys()))]['text']
    print(f'question: {question}')
    print(f'response: {response}')


    input_encodings = tokenizer.encode_plus(
        text=question,
        add_special_tokens=True,
        return_tensors="pt"  
    )
    input_token_count = input_encodings["input_ids"].shape[1]  

    # print(f'question: {question}')
    # print(f'response: {response}')

    
    
    output_encodings = tokenizer.encode_plus(
        text=str(response),
        add_special_tokens=True, 
        return_tensors="pt"
    )
    output_token_count = output_encodings["input_ids"].shape[1]

    input_tokens += input_token_count
    output_tokens += output_token_count 

print(f'average input tokens: {input_tokens/20}')
print(f'average output tokens: {output_tokens/20}')

