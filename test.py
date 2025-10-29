from __future__ import annotations
import os
import pdb
import subprocess
import time
import json
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from agents.config import BaseConfig
from agents.beam_search import BS
from agents.mcts import MCTS
from agents.solver import Solver


torch.set_num_threads(12)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data(filename: str):
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
        if "example" in data:
            data = data["example"]
    elif filename.endswith(".jsonl"):
        data = []
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    else:
        raise ValueError(f"Unrecognized file format: {filename}")
    return data

def batch(iterable, n=-1):
    l = len(iterable)
    if n <= 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/config/test_ori.yaml")
    args.add_argument("--qaf", type=str, default="/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/data/oodvqa-8488.jsonl", help="quesuion and answer file")
    args.add_argument('--save_in_model', type=str, default="")
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    
    print('===='*45)
    print(f'config:\n{config}')
    print('===='*45)
    
    data = load_data(args.qaf)
    if "RefCOCO+" in args.qaf:
        result = subprocess.run(
                ['wc', '-l', '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/RefCOCO+.jsonl.mcts.20250929140606.jsonl'],
                capture_output=True,
                text=True,
                check=True
            )
        length = int(result.stdout.strip().split()[0])
        data = data[length:]
    # elif 'chartqa_single' in args.qaf:
    #     result = subprocess.run(
    #             ['wc', '-l', '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/chartqa_single.jsonl.mcts.20250917175557.jsonl'],
    #             capture_output=True,
    #             text=True,
    #             check=True
    #         )
    #     length = int(result.stdout.strip().split()[0])
    #     data = data[length:]
    # elif 'info_vqa_single' in args.qaf:
    #     result = subprocess.run(
    #             ['wc', '-l', '/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/info_vqa_single.jsonl.mcts.20250919051140.jsonl'],
    #             capture_output=True,
    #             text=True,
    #             check=True
    #         )
    #     length = int(result.stdout.strip().split()[0])
    #     data = data[length:]
    


    #### time test ####
 
    
    
    solver = Solver(config=config)

    # init agent
    agent = MCTS

   
    saved_jsonl_file = f"{args.qaf}.{config.mode}.{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl" 
   
    if args.save_in_model:
        saved_jsonl_file = args.save_in_model + '.jsonl'
        saved_jsonl_file_dir = os.path.dirname(saved_jsonl_file)
        os.makedirs(saved_jsonl_file_dir, exist_ok=True)
        
    """
    {"id": 148, 
     "image": "/perception-hl/ad-cn-hlidc-multimodal/open_datasets/blip3-ocr-200m/images/d20448d2fca7578c4b7a3783f4993ec6_172.jpg", 
     "conversations": [
        { 
            "from": "human",
            "value": "<image>\nPlease describe the text in <box>13,446,995,535</box>"
        }, 
        {
            "from": "gpt", 
            "value": "ww.beliramsilverware.co"
        }
    ], 
     "api-result": "<think>\n首先，用户要求描述指定边界框内的文本。边界框是<box>13,446,995,535</box>，这意味着左上角坐标是(13, 446)，右下角坐标是(995, 535)。归一化到0-999的整数，所以x和y值都在这个范围内。\n\n现在，看图像中的文本。图像中有网址：www.beliramsilverware.com。我需要确认这个文本是否在指定的边界框内。\n\n边界框的x范围是13到995，y范围是446到535。在图像中，网址是水平显示的，覆盖了大部分宽度。y坐标446到535应该对应于文本的垂直位置。\n\n在图像中，文本是“www.beliramsilverware.com”，它位于图像的中间部分，覆盖了从左到右的大部分区域。y坐标446到535可能对应于文本的顶部和底部。\n\n由于边界框是归一化的，我需要确保文本在这个区域内。从视觉上看，网址确实在图像的中下部，所以它应该在这个边界框内。\n\n描述文本：文本是“www.beliramsilverware.com”。这是一个网址，指向一个银器网站。\n\n用户要求输出描述，所以我应该直接给出文本内容。\n\n思考过程要短，所以直接回答。\n\n最终输出：文本是“www.beliramsilverware.com”。\n</think>\n\n<box>13,446,995,535</box> 中的文本是 \"www.beliramsilverware.com\"。"}
    
    """
    total_start_time = time.perf_counter()
    with open(saved_jsonl_file, "a+", encoding='utf-8') as writer:
     
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            aligned_data = []
            for d in cur_data:
                # 传入的image可能是str 、[str]
                # question = d["conversations"][0]["value"]
                # answer = str(d["conversations"][1]["value"])
                # d["question"] = question
                # d["answer"] = answer
                # del d["conversations"]
                # # del d["api-result"]
                # if isinstance( d["image"] ,list) and len( d["image"] )==1:
                #     d["image"] = d["image"][0]

                question = d["instruction"]
                answer = str(d["output"])
                d["question"] = question
                d["answer"] = answer
                # del d["api-result"]
                if isinstance( d["image"] ,list) and len( d["image"] )==1:
                    d["image"] = d["image"][0]


                aligned_data.append(d)
            cur_data = aligned_data  
            
            agents = [
                agent(
                    config=config, 
                    image_path=d["image"], 
                    question=d["question"], 
                    ground_truth=str(d["answer"])
                ) 
                for d in cur_data
            ]
            jsonlines = solver.solve(agents, saved_jsonl_file, cur_data)
            
            for d in cur_data:
                question = d["question"]
                d["rstar"] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()
    

    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    print(f"\n==================== Time Test Summary ====================")
    if total_elapsed >= 60:
        total_elapsed_str = f"{total_elapsed // 60:.0f}分{total_elapsed % 60:.2f}秒"
    else:
        total_elapsed_str = f"{total_elapsed:.2f}秒"
    print(total_elapsed_str)
    print(f"===========================================================")
        