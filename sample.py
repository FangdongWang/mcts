import subprocess
import random
import os
from tqdm import tqdm
from glob import glob
from typing import List,Dict
from extract import read_jsonl,write_list_to_jsonl

FILE_DICT = {
    "docvqa": "/perception-hl/neos.yan/l1_jsonls/add_size/docvqa_en.jsonl",
    "chartqa": "/perception-hl/neos.yan/l1_jsonls/add_size/chartqa_en.jsonl",
    "info_vqa": "/lustre/yexun.zhang/datasets_external/OCR/InfoVQA/train_wo_ocr_processed.jsonl",
    "RefCOCO+": "/lustre/MLM_evaluator/data/RefCOCO/jsonls_normalize_0-1000/refcoco+_train.jsonl",
    "textvqa": "/lustre/yexun.zhang/datasets_external/OCR/textvqa/textvqa_train_v1.0_processed.jsonl",
    "refcocog": "/lustre/MLM_evaluator/data/RefCOCO/jsonls_normalize_0-1000/refcocog_train.jsonl",
    "refcoco": "/lustre/MLM_evaluator/data/RefCOCO/jsonls_normalize_0-1000/refcoco_train.jsonl",
}



def count_lines_with_percentage() -> Dict:
    sample_percentage = {}
    line_counts = {}
    for name, path in FILE_DICT.items():
        try:
            result = subprocess.run(
                ['wc', '-l', path],
                capture_output=True,
                text=True,
                check=True
            )
          
            line_count = int(result.stdout.strip().split()[0])
            line_counts[name] = line_count
        except Exception as e:
            print(f"⚠️  {name} 统计失败: {str(e)}")
            line_counts[name] = 0  

    total_lines = sum(line_counts.values())
    if total_lines == 0:
        print("\n❌ 所有文件统计失败，无法计算百分比")
        return


    print("\n📊 各文件行数及占比：")
    # 排序（按行数降序）
    sorted_items = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_items:
        if count == 0:
            percentage = 0.0
        else:
            percentage = (count / total_lines) 
        sample_percentage[name] = percentage
        print(f"{name:10}  nums: {count}  占比: {percentage*100:.2f}%")

    print(f"\n📝 总行数: {total_lines:,}")
    return sample_percentage



def make_sample_benchmark(total_nums:int, save_dir:str, file_dict:Dict,sample_percentage:Dict):
    for file_name,path in tqdm(file_dict.items()):
        # 1.计算采样数量
        percentage = sample_percentage[file_name]
        data = read_jsonl(path)
        sample_num = min(int(total_nums*percentage),len(data))

        # 2.采样并对image 进行处理
        sample_data = random.sample(data,sample_num)
        sample_data = [process_image_field(file_name,item,path) for item in sample_data]

        # 3.写入文件
        save_path = os.path.join(save_dir,f'{file_name}.jsonl')
        # write_list_to_jsonl(sample_data,save_path)
        print(f'{file_name} sampled {sample_num}, saved to: {save_path}')

def mutliqa_to_single(input_file:str, output_file:str):
    data = read_jsonl(input_file)
    single_turn_datas = []
    for item in tqdm(data):
        conversations = item.get('conversations', [])
        if len(conversations) < 2 or len(conversations) % 2 != 0:
                    continue

         # 拆分多轮对话为单轮（每2个元素为一轮：human→gpt）
        for turn_idx in range(0, len(conversations), 2):
            # 提取当前轮的human和gpt对话
            human_turn = conversations[turn_idx]
            gpt_turn = conversations[turn_idx + 1]
            
            # 验证角色是否正确（human在前，gpt在后）
            if human_turn.get('from') != 'human' or gpt_turn.get('from') != 'gpt':
                print(f"警告：第{line_num}行第{turn_idx//2 + 1}轮角色异常，跳过该轮")
                continue
            
            # 构建单轮对话数据（保留原数据的其他字段，仅替换conversations）
            single_turn_data = {
                **item,  # 复制id、image、width等其他字段
                # 'id': f"{item['id']}_{turn_idx//2}",  # 生成新id（原id_轮次）
                "uuid" : f"{item['uuid']}_{turn_idx//2}",
                'conversations': [human_turn, gpt_turn]  # 当前单轮对话
            }
            single_turn_datas.append(single_turn_data)
    write_list_to_jsonl(single_turn_datas,output_file)
          




def process_image_field(key, data_item, file_path):
    tdata = data_item
    if key in ['chartqa', 'docvqa', 'blip3-ocr-004', 'vqa-nle-llava-short', 
               "chartqapro", "deepform", "tatdqa", "robut_sqa_cauldron", 
               "chrome_writing", "ch_ocr", "MMK12", "MMMath", 
               "mavis-math-metagen", "puzzleVQA", "VisualPuzzle", 
               "Hyperphantasia", "COLUMBUS", "VisualSphinx-V1-Raw", 
               "SciVQA", "pr1"]:
        pass 
    
    elif key in ['info_vqa', "textvqa"]:
        tdata['image'] = os.path.join(os.path.dirname(file_path), tdata['image']['UNKNOWN'][0])
        
    elif key in ['RefCOCO+', 'ai2d_train_12k', "refcoco", "refcocog"]:
        tdata['image'] = tdata['image']['UNKNOWN'][0]

    return tdata

if __name__ == "__main__":
    print("开始统计JSONL文件行数及百分比...")
    sample_percentage = count_lines_with_percentage()
    print("\n统计完成")
    print(sample_percentage)

    total_sum = 40000
    save_dir = '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915'
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    make_sample_benchmark(total_sum,save_dir,FILE_DICT,sample_percentage)
    files = glob(os.path.join(save_dir,'*.jsonl'))
    print(files)

    # input_file = '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/info_vqa.jsonl'
    # output_file = '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/info_vqa_single.jsonl'
    # mutliqa_to_single(input_file,output_file)