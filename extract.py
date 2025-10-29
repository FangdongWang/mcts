import os
import re
import ast
import json
import subprocess
import pdb
import pandas as pd
from tqdm import tqdm
from glob import glob
from typing import List,Union

# ast.literal_eval(str_list). 将字符串转换为实际列表
 
def compute_iou(box1,box2):
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: List or tuple of 4 values [x1, y1, x2, y2] representing first box
        box2: List or tuple of 4 values [x1, y1, x2, y2] representing second box
        
    Returns:
        float: IoU value between the two boxes (0.0 to 1.0)
    """
    # Find coordinates of intersection
    try:
        inter_x1 = max(box1[0],box2[0])
        inter_y1 = max(box1[1],box2[1])
        inter_x2 = min(box1[2],box2[2])
        inter_y2 = min(box1[3],box2[3])

        # Check if boxes overlap
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        # Calculate intersection area
        inter_area = (inter_x2-inter_x1) * (inter_y2-inter_y1)

        # Calculate areas of both boxes
        box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
        box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])

        # Calculate union area
        union_area = box1_area + box2_area -inter_area

        # Return IoU
        return float(inter_area)/union_area
    except:
        return 0.0


def extract_box(text,pattern_type):
    """
    extract bounding box from answer and final_answer
    """ 
    if pattern_type == 'answer':
        match = re.search(r'<box>(.*?)</box>', text, re.DOTALL)
        if match:
            box_str = match.group(1).replace('[','').replace(']','').replace(' ','')
            box_str =  box_str.split(',')
            box = [int(x) for x in box_str]
            return box
    elif pattern_type == "final_answer": 
        # pdb.set_trace()
        if "<box>" in text and "</box>" in text:
            match = re.search(r'<box>(.*?)</box>', text, re.DOTALL)
            if match:
                # box_str = match.group(1).replace('[','').replace(']','').replace(' ','').replace('(','').replace(')','')
                box_str = match.group(1).replace('[','').replace(']','')
                if ',' in box_str:
                    box_str =  box_str.split(',')
                else:
                    box_str = box_str.split()
                box = [int(x) for x in box_str]
                return box
        else:
            match  = re.findall(r'-?\d+', text)
            return list(map(int, match))
    else:
        return None


def read_jsonl(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    if item  not in data:
                        data.append(item)
                    # data.append(item)
                except json.JSONDecodeError:
                    continue
    
    return data

def write_list_to_jsonl(data_list: List, file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            try:
                json_str = json.dumps(item, ensure_ascii=False)
                f.write(json_str + '\n')
            except (json.JSONEncodeError,json.JSONDecodeError) as e:
                print(f"{e}")


def get_save_path(path:str, add_name:str, save_dir:str='')->str:
    dir = os.path.dirname(path)
    base_name = os.path.basename(path)
    save_name = base_name.split('.')[0] + add_name
    if save_dir:
        return(os.path.join(save_dir,save_name))
    return os.path.join(dir,save_name)


def clean_cot(cot:str)->str:
    cot = cot.replace('\x08','\\b').replace('<end_of_step>','').replace('<end_of_think>','</think>').replace('<end_of_answer>','</answer>')
    return cot

# text = "<think>\n# Step 1: Identify the Acceptable bar value for Total from the chart.\nThe Acceptable bar for Total is 66.\n<end_of_step>\n\n<end_of_think>\n\n<answer>66<end_of_answer>"
# text = "<think>\nOkay, let's see. The question is asking how many people in Denmark had never been married as of January 1, 2021. The data is presented in a bar chart. Let me look at the chart.\n\nThe chart has different categories: Never married, Married/separated, Widowed, Divorced. Each year from 2017 to 2021 is shown. The blue bars represent \"Never married.\" For 2021, the blue bar's value is around 2,800,000. Let me check the exact number. The y-axis is labeled \"Number of inhabitants,\" and the 2021 blue bar reaches up to just under 3 million. The exact value for 2021 under \"Never married\" is 2,800,000. Wait, the numbers might be in thousands. Let me check the increments. The vertical axis goes up by 500,000 increments. The 2021 blue bar is at 2,800,000. So the answer should be 2,800,000. But let me confirm the exact value from the chart. The 2021 value for Never married is the top of the blue bar. The chart's source is Statista 2021. So according to the chart, the number is 2,800,000. So the answer is 2,800,000.\n<end_of_step>\n\n<end_of_think>\n\n<answer>\\boxed{2800000}<end_of_answer>"
# print(clean_cot(text))


def is_equal(gt:str,pred:str,Grouning:bool=False)-> bool:
    """
    如果是grounding 任务 提取bounding box 计算iou 按照iou阈值进行计算
    如果通过则 替换cot中bounding box 
    """
    if Grouning:
        # pdb.set_trace()
        try:
            gt_box = extract_box(gt,"answer")
            pred_box = extract_box(pred,"final_answer")
            if gt_box and pred_box and compute_iou(gt_box,pred_box) >= 0.5:
                # print(compute_iou(gt_box,pred_box))
                return True
        except (ValueError, SyntaxError, TypeError) as e:
            print('=='*50)
            print(f'gt: {gt}')
            print(f'pred: {pred}')
            print(f"提取失败 error={e}")
        
        return False

    else:
        gt,pred = gt.lower(),pred.lower()
        gt,pred = gt.replace('\\',''),pred.replace('\\','')
        gt,pred = gt.strip('()$'),pred.strip('()$')
        gt,pred = gt.replace(',','').replace('.',''),pred.replace(',','').replace('.','')
        gt,pred = gt.strip(),pred.strip()
        
        if gt == pred:
            return True
        elif gt in pred:
            return True
        elif pred in gt:
            return True
        return False

def math_clean_and_filter(path:str)->List:
    data = read_jsonl(path)

    clean_data = []
    filter_data = []
    for item in tqdm(data):
       
        reversed_iter = reversed(item["rstar"].keys())
        think_step = item["rstar"][next(reversed_iter)] #获取最终推理轨迹
        cot = clean_cot(think_step["text"])
        pred = think_step["final_answer"]
        gt = item["answer"]
        if pred and gt and is_equal(gt,pred):
            clean_data.append(
                {
                    "image":item["image"],
                    "question":item["question"],
                    "answer":gt,
                    "final_answer":pred,
                    "cot":  cot.split('</think>')[0]+f'</think>\n<answer>{pred}</answer>'
                }
            )
        elif pred == "Fail to sove the problem within limited steps.":
            ## 提取 answer 中的内容 并修正cot内容
            match_1 = re.search(r'<answer>(.*?)</answer>',cot,re.DOTALL)
            if match_1 :
                ans = None
                if "boxed" in match_1.group(1):
                    match_2 = re.search(r'\\boxed\{(.*?)\}', match_1.group(1), re.DOTALL)
                    ans = match_2.group(1) if match_2 else None
                final_ans = ans if ans else  match_1.group(1)

                if gt and final_ans and is_equal(gt,final_ans):
                    
                    clean_data.append(
                    {
                        "image":item["image"],
                        "question":item["question"],
                        "answer":gt,
                        "final_answer":final_ans,
                        "cot": cot.split('</think>')[0]+f'</think>\n<answer>{final_ans}</answer>'
                    }
                )
                else:
                    filter_data.append(
                        {
                            # "image":item["image"],
                            # "question":item["question"],
                            "answer":gt,
                            "final_answer":final_ans if final_ans else pred
                            # "cot": cot
                        }
                    )
            else:
                filter_data.append(
                    {
                        # "image":item["image"],
                        # "question":item["question"],
                        "answer":gt,
                        "final_answer":pred
                        # "cot": cot
                    }
                )

        else:
            filter_data.append(
                {
                    # "image":item["image"],
                    # "question":item["question"],
                    "answer":gt,
                    "final_answer":pred
                    # "cot": cot
                }
            )

    clean_save_path = get_save_path(path,'_msct_cot_clean.jsonl')
    filter_save_path = get_save_path(path,'_msct_cot_filter.jsonl')
    write_list_to_jsonl(clean_data,clean_save_path)
    write_list_to_jsonl(filter_data,filter_save_path)
    print('=='*50)
    print(f'all data: {path}, nums:{len(data)}')
    print(f'clean data has saved to: {clean_save_path},nums:{len(clean_data)}')
    print(f'filter data has saved to: {filter_save_path},nums:{len(filter_data)}')
    print('=='*50)

def extrace_ans_remove_box(text:str)->str:
    # 去除<end_of_step>
    # <end_of_think> -> </think>
    # <end_of_answer> -> </answer>
    # 提取 boxed中的答案 直接放到<answer></answer>中
    # example: <answer>The document is dated \\boxed{January 31, 1969}.<end_of_answer> -> <answer>January 31, 1969<end_of_answer>
    text = text.replace('<end_of_step>','').replace('<end_of_think>','</think>').replace('<end_of_answer>','</answer>')
    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
    match = boxed_pattern.search(text)
    if match:
        extracted = match.group(1) 
        # 将整段 <answer>...</answer> 替换为提取后的简洁形式
        text = re.sub(r'<answer>.*?</answer>',
              f'<answer>{re.escape(extracted)}</answer>',
              text, flags=re.DOTALL)
    return text

def remove_specail_token(path):
    data = read_jsonl(path)
    for item in tqdm(data):
        response = item["conversations"][1]["value"]
        item["conversations"][1]["value"] = extrace_ans_remove_box(response)
    path = '/home/fangdong.wang/mlm-evaluator/0918_RL_cool_start/RL_cool_start/data/cool_start/refcocog_docvqa_textvqa_12k_clean.jsonl'
    write_list_to_jsonl(data,path)
    

def merge_ref_ans(gt:str,pred:str,cot:str)->str:
    ref_m = re.search(r'<ref>(.*?)</ref>', gt)
    ref_text = ref_m.group(1) if ref_m else ""
    ref_text = f'<ref>{ref_text}</ref>: '
    box_nums = re.findall(r'\d+', pred)
    box_str = ",".join(box_nums)

    step_count = cot.count('<end_of_step>')

    think_steps = cot.split('<answer>')[0].replace('<end_of_step>','').replace('<end_of_think>','').replace('\n\n\n','\n').replace('\n\n','\n')
    final_ans = f'{ref_text}<box>{box_str}</box>'
    merge_ans = f"<answer>{ref_text}<box>{box_str}</box></answer>" 
    step_4 = f'#Step {step_count+1}: Now align the answer with the format of the question, and the final answer is '
    final_cot = think_steps + step_4 + final_ans + '</think>\n'+ merge_ans
    return final_cot


def grounding_clean_and_filter(path:str):
    data = read_jsonl(path)

    clean_data = []
    filter_data = []
    for item in tqdm(data):
   
        think_step = item["rstar"][next(reversed(item["rstar"].keys()))] #获取最终推理轨迹
        cot = think_step["text"]
        pred = think_step["final_answer"].replace('bbox','box').replace('\\','')
        no_valids = ('Fail to sove the problem within limited steps.','(x1, y1, x2, y2)', '[x1, y1, x2, y2]','None','Not visible')
        if pred in no_valids:
            continue
        gt = item["answer"]
        # if gt == '<ref>man in blue plaid shirt sitting</ref>: <box>815,51,999,988</box>.':
        #     pdb.set_trace()
        if pred and gt and is_equal(gt,pred,True):
            cot = cot.replace('\x08','\\b').replace('bbox','box')
            cot = merge_ref_ans(gt,pred,cot)
            clean_data.append(
                {
                    "image":item["image"],
                    "question":item["question"],
                    "answer":gt,
                    "final_answer":pred,
                    "cot": cot
                }
            )
        else:
            # pdb.set_trace()
            filter_data.append(
                {
                    # "image":item["image"],
                    # "question":item["question"],
                    "answer":gt,
                    "final_answer":pred
                    # "cot": cot
                }
            )

    clean_save_path = get_save_path(path,'_msct_cot_clean.jsonl')
    filter_save_path = get_save_path(path,'_msct_cot_filter.jsonl')
    write_list_to_jsonl(clean_data,clean_save_path)
    write_list_to_jsonl(filter_data,filter_save_path)
    print('=='*50)
    print(f'all data: {path}, nums:{len(data)}')
    print(f'clean data has saved to: {clean_save_path},nums:{len(clean_data)}')
    print(f'filter data has saved to: {filter_save_path},nums:{len(filter_data)}')
    print('=='*50)



def trans_format(files:Union[str, List[str]],save_path:str) -> None:
    data = []
    if isinstance(files,list):
        for f in tqdm(files):
            data += read_jsonl(f)
    else:
        data = read_jsonl(files)
    final_data = []
    for item in tqdm(data):
        image = item['image']
        question = item['question']
        ans = item['cot']
        final_data.append(
            {
                "image":image,
                "conversations":[
                    {
                        "from":"human",
                        "value":question+"\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                    },
                    {
                        "from":"gpt",
                        "value":ans
                    } 
                ]
            }
        )
    write_list_to_jsonl(final_data,save_path)
    print(f'saved to: {save_path}')




if __name__ == '__main__':
    # 统一转换格式 jsonl
    # dir = '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926'
    # save_dir = '/home/fangdong.wang/0918_RL_cool_start/RL_cool_start/data/0926_mcts_cot_aha_moment'
    # files = glob(f'{dir}/*.jsonl')
    # files = [f for f in files if "msct_cot_clean" in f]
    # for f in tqdm(files):
    #     length = subprocess.run(['wc','-l',f],capture_output=True, text=True).stdout.strip().split()[0]
    #     basename = os.path.basename(f)
    #     save_path = os.path.join(save_dir,f'{basename.split("_msct")[0]}_{length}.jsonl')
    #     trans_format(f,save_path)



    path = '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/RefCOCO+.jsonl.mcts.20250929140606.jsonl'
    grounding_clean_and_filter(path)
    


    





   
    





        
    

