from typing import Optional, Any, Dict, List, Callable, Type, Tuple, Union
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import json
import pdb
import cv2
import random
import requests
import io
import os
import requests
import json
import math
import sys
import base64
import numpy as np
import niofs
from openai import OpenAI
from niofs.conf import Cloud
from niofs.read_cache_op_py import read_cache_op_py
from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModel, AutoTokenizer
from xtuner.utils import RewardModelClient
# from ..check_utils.math_equal import math_equal
# from ..check_utils.checker import check_one_answer
# from ..check_utils.util import equiv,strip_string,choice_answer_clean


sys.path.append('/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/check_utils')
sys.path.append('/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2')
from check_utils.math_equal import math_equal
from check_utils.checker import check_one_answer
from check_utils.util import equiv, strip_string, choice_answer_clean

#-----------------------------------------------调用seed1.5-vl进行推理------------------------------------

def file_to_base64(image_path):
    import base64
    with open(image_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def seed_inference_one_sample(image_path,question):
    
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    api_key = "88b97866-80d6-49bf-976b-51caac340461"
    model_name = "doubao-1-5-thinking-vision-pro-250428"
    # model_name = "doubao-seed-1-6-250615"
    # model_name = "doubao-seed-1-6-vision-250815"
    # model_name = "doubao-seed-1-6-thinking-250715"
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    image_base64 = None
    if "huailai" in image_path:    
        image = loadload_image(image_path) 
        image_base64 = "data:image/jpeg;base64,"+ str(image_to_base64_str(image, (image_path.split(' ')[0]).split('.')[-1]))
    else:
        image_base64 = file_to_base64(image_path)

    response = client.chat.completions.create(
        extra_body={"thinking": {"type": "disabled"}},
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": question},
                ],
            }
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()



#-------------------------------------------------发送请求---------------------------------

def send_critic_request(image_path:str,question,response_list_str):
    image_bytes = None
    if "huailai" in image_path:    
        image = loadload_image(image_path) 
        image_bytes = str(image_to_base64_str(image, (image_path.split(' ')[0]).split('.')[-1]))
        image_path = None
    
    if 'www2025' in image_path:
        image = Image.open(image_path).convert('RGB')
        image_bytes = str(image_to_base64_str(image, "JPEG"))
        image_path = None
    data = {
        "image":image_path,
        "question":question,
        "image_bytes": image_bytes,
        "response_list_str": json.dumps(response_list_str) 
    }

    SERVER_URL = "http://172.24.172.98:8081/infer"
    response = requests.post(SERVER_URL, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"请求失败，状态码：{response.status_code}",
            "details": response.text
        }


def send_inference_request(image_path, question, response_list):

    image_bytes = None
    if image_path.startswith("huailai"):
        image = loadload_image(image_path)
        image_path = None
        image_bytes = str(image_to_base64_str(image, (json_data['image'].split(' ')[0]).split('.')[-1]))

    data = {
        "image":image_path,#直接传递图片路径字符串
        "question":question,#问题
        "image_bytes": image_bytes,
        "response_list_str": json.dumps(response_list) # respponse列表(需要转成JSON)
    }
    response = requests.post(SERVER_URL, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error":f"请求失败,状态码:{response.status_code}",
            "details": response.text
        }


def loadload_image(image_path, root=None, tar_saved=False):
    ON_TXSH = 1
    mount_point = (
        "/lustre" if os.path.exists("/lustre") else "/perception-hl/MLM_evaluator"
    )
    if ":" in image_path:
        read_ceph_kwargs = {}
        if ON_TXSH:
            read_ceph_kwargs["mnt_point"] = mount_point
        if tar_saved:
            tar_path, tar_size, img_offset, img_size = image_path.split(" ")
            tar_path = f"{tar_path} {tar_size}"
            ceph_path = "/" + tar_path.split(":")[1]
            raw_bytes = read_cache_op_py([ceph_path], **read_ceph_kwargs)
            image_bytes = raw_bytes[0][int(img_offset) : int(img_offset) + int(img_size)]
        else:
            if not ON_TXSH:
                if ":" in image_path:
                    ceph_path = "/" + image_path.split(":")[1]
                else:
                    ceph_path = image_path
            else:
                if image_path.startswith("/"):
                    ceph_path = image_path.split(" ")[0]
                else:
                    ceph_path = "/" + image_path.split(" ")[0].split(":")[1]

                print(ceph_path)
                print(read_ceph_kwargs)

            raw_bytes = read_cache_op_py([ceph_path], **read_ceph_kwargs)
            image_bytes = raw_bytes[0]

        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)  # BGR
        if decoded.ndim == 2:
            decoded = np.tile(decoded[..., np.newaxis], (1, 1, 3))
        image = Image.fromarray(decoded[:, :, ::-1], "RGB")
    else:
        if "/" in image_path:
            image = Image.open(image_path).convert('RGB')
        else:
            image_path = os.path.join(root, image_path)
            image = Image.open(image_path).convert('RGB')

    return image


def image_to_base64_str(image: Image.Image, format: str = 'PNG') -> str:
    """
    将PIL的Image对象转换为Base64字符串

    :param image: PIL的Image对象
    :param format: 图像格式，如'PNG'、'JPEG'等
    :return: 转换后的Base64字符串
    """
    # 创建一个字节流缓冲区
    buffer = io.BytesIO()

    # 将图像保存到缓冲区
    image.save(buffer, format=format)

    # 获取缓冲区的字节数据
    image_bytes = buffer.getvalue()

    # 关闭缓冲区
    buffer.close()

    # 将字节数据编码为Base64字符串
    base64_str = base64.b64encode(image_bytes).decode('utf-8')

    return base64_str



#---------------------------- 提取ans---------------------


def remove_text_box(text):
    """
    去除Latex文本标记 去除\text{}内的文本描述 保留核心内容
    """
    if text is None:
        return None
    start = text.find(r"\text{")
    if start == -1:
        return text
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start_text = stack.pop()
            if len(stack) == 0:
                end_text = i
                break
    in_text_string = text[start + start_text + 1 : start + end_text]

    if in_text_string.strip() == "and":
        ex_text = text[:start] + text[start + end_text + 1 :]
    else:
        ex_text = (
            text[:start]
            + text[start + start_text + 1 : start + end_text].strip()
            + text[start + end_text + 1 :]
        )
    return ex_text.strip()
    


def extract_boxed_answer(text, debug=False):
    """
    提取Latexboxed答案 提取包裹在\boxed{}内的最终答案
    """
    if text is None:
        return None
    start = text.rfind(r"\boxed{")
    if start == -1:
        return text
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start = stack.pop()  # \boxed start{
            if len(stack) == 0:
                end = i  # \boxed end}
                break
    if end is None and debug:
        print("brack not closing", answer)
        return None
    return answer[start + 1 : end]

INVALID_ANS = "[invalid]"

def extract_answer(answer):
    """
    端到端提取答案
    """
    try:
        ans = answer
        # extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = ans
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        extract_ans = remove_text_box(extract_boxed_answer(extract_ans))
    except:
        extract_ans = INVALID_ANS
    return extract_ans


#----------------------------prompt 相关 --------------------------------
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class PROMPT_RSTAR:
    def __init__(self, config):
        self.pot_format_instructions = None
        self.pot_suffix = None
        self.few_examples = None
        self.num_few_shot = config.num_few_shot # 0
        self.load_prompt(config)

        assert self.num_few_shot <= len(self.few_examples), f"{self.num_few_shot} should less than few_examples."   


    def load_prompt(self, config):
        self.few_examples = load_json(config.few_shot_path)
        prompt = load_json(config.prompt_path)
        # self.pot_format_instructions = prompt['pot_format_instructions']
        ### """"prompt change here!!!""""
        self.pot_format_instructions = """
        
        You are tasked with solving problems in a structured, step-by-step manner.

        Remember:\n
        1.If the problem involves bounding boxs, please note that it is normalized to 0-999.\n
        2.Solve the problem step by step. The solution should include two parts: <think> and <answer>.\n
        3.The thinking process should include using<end_of_step>to divide between multiple steps,thinking process must follow the following template strictly!\n
        4.The process of thinking must follow the following template and provided example!\n
        Template:
        Question: the input question\n
        <think>
        #Step 1: [Perform any necessary calculations for this step.]
        <end_of_step>\n

        #Step 2: [Perform any necessary calculations for this step.]
        <end_of_step>\n
        ... (Continue adding steps as needed, following the same format: #Step N: ... <end_of_step>) ...
       
        <end_of_think>\n

        <answer>After this step of thinking, if you have already come up with the answer, put your current answer in \boxed{}
        The concise answer without verbose context, put your final answer in \\boxed{}.<end_of_answer>

        Example：
        Question：If $G(m, n, p, q) = m^n + p \times q$, what is the value of $y$ such that $G(3, y, 6, 15) = 171$?
        <think>
        # Step 1: Define the function G(m, n, p, q)
        def G(m, n, p, q):
        return m**n + p * q
        <end_of_step>

        # Step 2: Set up the equation G(3, y, 6, 15) = 171
        equation = G(3, y, 6, 15) - 171
        <end_of_step>

        # Step 3: Solve for y
        We need to find y such that 3^y + 6 * 15 = 171
        Simplify the equation: 3^y + 90 = 171
        Therefore, 3^y = 81
        We know that 81 = 3^4, so y = 4
        <end_of_step>

        <end_of_think>

        <answer>The value of y is \\boxed{4}.<end_of_answer>

        """
        # self.pot_format_instructions = """
        # You are tasked with solving problems in a structured, step-by-step manner.

        # Remember:
        # 1. If the problem involves bounding boxes, please note that they are normalized to 0-999.
        # 2. Solve the problem step by step. The solution must include two parts: <think> and <answer>.
        # 3. In the thinking process, **an additional mandatory step** `#Step A: aha moment` must appear before the final answer step.  
        # - aha moment: one sentence that starts with “Aha! I now realize that …” (or similar), capturing the key insight or turning point.
        # 4.The final answer will be packaged using<answer></answer>!
        # 5. Follow the template strictly:

        # Template:
        # Question: the input question
        # <think>
        # #Step 1: [Perform any necessary calculations for this step.]
        # #Step 2: [Perform any necessary calculations for this step.]
        # … (continue as needed)
        # #Step A: aha moment  
        # Aha! I now realize that [key insight / turning-point idea].
        # #Step N: [Final calculation / verification.]
        # </think>

        # <answer>After this step of thinking, if you have already come up with the answer, put your current answer in \\boxed{}
        # The concise answer without verbose context, put your final answer in \\boxed{}.</answer>

        # Example:
        # Question: If $G(m, n, p, q) = m^n + p \times q$, what is $y$ such that $G(3, y, 6, 15) = 171$?
        # <think>
        # #Step 1: Define G(m, n, p, q) = m^n + p·q
        # #Step 2: Plug in values: 3^y + 6·15 = 171 ⟹ 3^y = 81
        # #Step A: aha moment  
        # Aha! I now realize that 81 is exactly 3^4, so y must be 4.
        # #Step 3: Therefore y = 4
        # </think>

        # <answer>The value of y is \\boxed{4}.</answer>
        # """

        
                
        self.pot_suffix = prompt['pot_suffix']


    def random_examples(self):
        if self.num_few_shot == 1:
            return [self.few_examples[1]]
        selected_examples = random.sample(self.few_examples, min(len(self.few_examples), self.num_few_shot))
        return selected_examples

def rstar_prompt_wrap(
    question: str, 
    partial_solution: str,
    config,
) -> str:
    """
    构建LLM prompt
    """
    step_delim = config.step_delim
    prompt_pot = PROMPT_RSTAR(config) # 装载初始prompt
    inputs = f"{question}{step_delim}"  

    rstar_examples = prompt_pot.random_examples() #[]
    
    if len(rstar_examples) > 1:
        example_prefix = "The following are %d demonstration examples." % len(rstar_examples)
    elif len(rstar_examples) == 1:
        example_prefix = "The following is a demonstration example."

    format_instructions = prompt_pot.pot_format_instructions
    
    if len(rstar_examples) > 0:
        prompt = step_delim.join([format_instructions, example_prefix, *rstar_examples, ""])
    else:
        prompt = step_delim.join([format_instructions, ""])
    if prompt.strip() == "":
        prompt = step_delim.join([prompt_pot.pot_suffix.format(input=inputs)])
    else:
        prompt = step_delim.join([prompt, prompt_pot.pot_suffix.format(input=inputs)])
    if partial_solution:
        prompt = "".join([prompt, partial_solution])
    return prompt + ""



def rstar_obs_wrap(observation: str) -> str:
    return f"{OUTPUT}{observation}{OUTPUT_END}"


def rstar_step_result_unwrap(
    text: str,
) -> Tuple[str, Dict[str, str]]:
    parser_result = {
        "action": "",
        "action_input": "",
        "final_answer": "",
    }
    #if ANSWER_END in text or "boxed" in text:
    # 得到答案 or 没有得到答案
    if "boxed" in text:
        parser_result["final_answer"] = extract_answer(text)
        return text, parser_result
    elif "oxed" in text:
        text = text.replace('oxed','\\boxed')
        parser_result["final_answer"] = extract_answer(text)
        return text.replace('\x08',''), parser_result
    else:
        parser_result["action"] = "do next step think"
        parser_result["action_input"] = text
        return text, parser_result


#---------------------------- math_equiv相关 --------------------------------

# math_evaluation.py
# Adapted from: https://github.com/hendrycks/math
# Licensed under MIT

import re
import math
from typing import Optional


def is_equiv(str1: Optional[str], str2: Optional[str], verbose: bool = False) -> bool:
    """
    Returns whether two strings are equivalent mathematical expressions.
    
    Args:
        str1 (str or None): First string (e.g., model output)
        str2 (str or None): Second string (e.g., ground truth)
        verbose (bool): Whether to print debug info
    
    Returns:
        bool: True if the two strings are mathematically equivalent
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    # Step 1: Strip both strings
    try:
        stripped_str1 = strip_string(str1)
        stripped_str2 = strip_string(str2)
    except Exception as e:
        if verbose:
            print(f"Error in strip_string: {e}")
        return False

    if verbose:
        print(f"Stripped: '{str1}' -> '{stripped_str1}'")
        print(f"Stripped: '{str2}' -> '{stripped_str2}'")

    # Step 2: Direct string comparison
    if stripped_str1 == stripped_str2:
        return True

    # Step 3: Try numerical evaluation (for numbers, fractions, etc.)
    try:
        num1 = to_float(stripped_str1)
        num2 = to_float(stripped_str2)
        if abs(num1 - num2) < 1e-3:  # tolerance for floating point
            return True
    except:
        pass

    # Step 4: Try symbolic evaluation with SymPy (optional, powerful)
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify

        latex1 = stripped_str1
        latex2 = stripped_str2

        # Try to parse LaTeX and simplify difference
        expr1 = parse_latex(latex1)
        expr2 = parse_latex(latex2)
        diff = expr1 - expr2
        if simplify(diff) == 0:
            return True
    except:
        pass

    return False


def strip_string(string: str) -> str:
    """
    Strip and normalize a string of math expression.
    Adapted from the MATH dataset evaluation script.
    """
    # Convert to string
    string = str(string)

    # Remove leading/trailing whitespace
    string = string.strip()

    # Remove $ at start/end (common in LaTeX math)
    if string.startswith("$") and string.endswith("$"):
        string = string[1:-1]

    # Remove \boxed{...}
    if string.startswith("\\boxed"):
        string = string[7:]  # len("\boxed") = 7
        # Remove possible leading { and trailing }
        if string.startswith("{") and string.endswith("}"):
            string = string[1:-1]

    # Remove \text{...}, \mbox{...}, etc.
    string = re.sub(r"\\text{.*?}", "", string)
    string = re.sub(r"\\mbox{.*?}", "", string)

    # Replace \left, \right, \big, etc. (just remove)
    string = re.sub(r"\\left", "", string)
    string = re.sub(r"\\right", "", string)
    string = re.sub(r"\\big", "", string)

    # Remove all spaces
    string = string.replace(" ", "")

    # Normalize common expressions
    string = string.replace("^{*}", "*")  # handle ^{*}
    string = string.replace("^{\\circ}", "deg")  # angle degrees
    string = string.replace("%", "")  # remove percent sign

    # Handle fractions: \frac{a}{b} -> a/b
    string = re.sub(r"\\frac{([^}]+)}{([^}]+)}", r"(\1)/(\2)", string)

    # Handle mixed numbers: 1\\frac{1}{2} -> 1+(1/2)
    string = re.sub(r"(\d+)\\frac", r"\1+(\\frac", string)

    # Remove common LaTeX commands
    string = string.replace("\\,", "").replace("\\!", "").replace("\\;", "")  # thin spaces
    string = string.replace("\\newline", "").replace("\\cr", "")

    # Fix common issues
    string = string.replace("=", "")  # often used in alignment
    string = string.replace(":", "")
    string = string.replace("\\dots", "")
    string = string.replace("...", "")

    # Remove any remaining { }
    string = string.replace("{", "").replace("}", "")

    return string.strip()


def to_float(s: str) -> float:
    """
    Convert a string to float, handling common math formats.
    """
    s = s.strip().lower()

    # Handle fractions
    if "/" in s:
        try:
            nums = s.split("/")
            if len(nums) == 2:
                return float(nums[0]) / float(nums[1])
        except:
            pass

    # Handle percentages
    if s.endswith("%"):
        return float(s[:-1]) / 100.0

    # Handle scientific notation
    s = s.replace("e", "e+").replace("×10^", "e+").replace("*10^", "e+")

    # Remove common non-numeric chars
    s = re.sub(r"[^\d\.\-\+e]", "", s)

    return float(s)


# Optional: Use SymPy for more advanced symbolic comparison
def is_equiv_symbolic(str1: str, str2: str) -> bool:
    """
    Experimental: Use SymPy to check symbolic equivalence.
    Requires: pip install sympy
    """
    try:
        import sympy as sp
        from sympy.parsing.latex import parse_latex

        expr1 = parse_latex(str1)
        expr2 = parse_latex(str2)
        return sp.simplify(expr1 - expr2) == 0
    except:
        return False



def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def rstar_equiv(gt, pred):
    # In this function, I integrated multiple open-source evaluation tools
    # each with its own judgment logic and strengths in handling special cases such as LaTeX, units, etc.
    gt = str(gt)
    pred = str(pred)
    try:
        if gt == pred:
            return True
        
        # For college-math and omni-math, the pred and gt positions need to be changed.
        # Because we found that the quality of ground truth in a small subset of problems within benchmarks like college-math is relatively low.
        if any(
            func(x, y) for func in [math_equal, is_equiv, check_one_answer] for x, y in [(gt, pred), (pred, gt)]
        ):
            return True
        # special for college-math, etc.
        gt_strip, pred_strip = strip_string(gt), strip_string(pred)
        if any(
            func(x, y) for func in [math_equal, is_equiv, check_one_answer] for x, y in [(gt_strip, pred_strip), (pred_strip, gt_strip)]
        ):
            return True

        # for choice question
        if gt in ["A", "B", "C", "D", "E"] and pred not in ["A","B","C","D","E"]:
            pred = choice_answer_clean(pred)
            if math_equal(gt, pred):
                return True
        elif is_multi_choice(gt) and not is_multi_choice(pred):
            pred = "".join(
                    [c for c in pred if c in ["A", "B", "C", "D", "E"]]
                )
            if math_equal(gt, pred):
                return True
    except Exception as e:
        print("maroi_equiv error")
        print(e)
        pass
    return False

def math_equiv(grt: Union[str, list[str]], prd: str, question: str, flag: str = "math"):
    if flag == 'math':
        prd = (prd)
        if isinstance(grt, list):
            for g in grt:
                if rstar_equiv(g, prd):
                    return True
            return False
        else:
            return rstar_equiv(grt, prd)

    elif flag == 'Grounding':
        rewards = []
        answer_reward_client = RewardModelClient(
            "internlm/POLAR-7B",
            server_type="vllm",
            server_address="172.24.146.124:30000"
        )

        data = {
            "prompt": [{"role": "user", "content": question}],
            "reference": [{"role": "assistant", "content": response}],
            "output": [{"role": "assistant", "content": answer}]
        }
        try:
            rewards = answer_reward_client(data)
        except:
            pass 
        if rewards and rewards[0]>-2:
            return True
        return False


