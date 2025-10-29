# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import os
import json
import pdb
import os.path as osp
from tqdm import tqdm
from termcolor import colored
from functools import partial
from pebble import ProcessPool
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
from pydantic import BaseModel, ConfigDict, field_validator

from .utils import send_critic_request,seed_inference_one_sample
from .tree import BaseTree
from .mcts import MCTS
from .config import TIMEOUT_SECONDS, ERROR_COLOR



class CompletionOutput:
    """模拟vllm的CompletionOutput，包含生成的文本内容"""
    def __init__(self, text: str, stop_reason: str = ""):
        self.text = text  # LLM 生成的推理文本（核心内容）
        self.stop_reason = stop_reason  # 生成终止原因（如"stop"“length”“none”）

class APIOutput:
    """模拟vllm的RequestOutput，包含生成结果和价值估计"""
    def __init__(self, content: str):
        self.outputs = [CompletionOutput(text=content)]  # 与原结构对齐：outputs[0].text
        self.value_estimate = None  # 预留评分字段



class Solver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: Any
    stop: List[str] = None
    llm: Optional[Callable[[...], List[str]]] = None
    need_value_func: bool = False
    max_agent_steps: int = 1
    reward_model: Optional[Any] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.config.stop:
            self.stop = OmegaConf.to_object(self.config.stop)

        self.need_value_func = self.config.need_value_func
        if self.need_value_func:
            self.reward_model = send_critic_request
        self.max_agent_steps = self.config.iterations
        self.config.step_beam_width = 1
            

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            return cfg
        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")

        
    @staticmethod
    def processor(agent, output) -> BaseTree:
        """
        处理生成结果 拓展节点
        """
        agent.generate_next_step(output)
        return agent


    @staticmethod
    def selector(agent, output) -> BaseTree:
        """
        处理价值评估结果 选择下一步节点
        """
        agent.select_next_step(output)
        return agent


    def generate_preprocess(self, agents):
        prompts = []
        questions = []
        images = []
        rewards = []
        prompts_span = [0]
        valid_agents = []
        invalid_agents = []
        expanded_agents = []

        for agent in agents:
            if agent.should_generate_next():
                if agent.has_expanded():
                    expanded_agents.append(agent)
                else:
                    images.append(agent.image_path)
                    questions.append(agent.question)

                    agent_prompts = agent.create_prompt()
                    rewards.extend(agent.get_rewards())
                    prompts.extend(agent_prompts)
                    prompts_span.append(prompts_span[-1] + len(agent_prompts))
                    valid_agents.append(agent)
            else:
                invalid_agents.append(agent)
        return prompts, questions, images, prompts_span, valid_agents, invalid_agents, expanded_agents, rewards

    # generate_postproces -> processor -> generate_next_step
    def generate_postprocess(
        self, 
        outputs: List[List[RequestOutput]], 
        valid_agents: List[BaseTree],
    ) -> List[BaseTree]:
        post_agents = []
        #with ProcessPool(max_workers=min(len(valid_agents), os.cpu_count())) as pool:
        with ProcessPool(max_workers=12) as pool:
            future = pool.map(self.__class__.processor, valid_agents, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()
        
        progress_bar = tqdm(total=len(valid_agents), desc="generate_postprocess")  
        while True:
            try:
                result = next(iterator)
                post_agents.append(result)
            except StopIteration:
                break
            except Exception as error:
                print(colored(f"{error}\n", ERROR_COLOR))
                post_agents.append(None)
            progress_bar.update(1) 
        progress_bar.close() 
            
        # update agents
        updated_agents = [
            post_agent if post_agent is not None else valid_agent
            for post_agent, valid_agent in zip(post_agents, valid_agents)
        ]
        return updated_agents
    

    def value_preprocess(self, agents: List[BaseTree]) -> Tuple[List[str], List[int]]:
        prompts = []
        prompts_span = [0]
        for agent in agents:
            agent_prompts = agent.create_prompt(is_value_only=True)
            prompts.extend(agent_prompts)
            prompts_span.append(prompts_span[-1] + len(agent_prompts))
        return prompts, prompts_span
    
    
    def value_postprocess(
        self, 
        outputs, 
        valid_agents,
    ) -> List[BaseTree]:
        for agent, output in zip(valid_agents, outputs):
            if agent is not None:
                self.selector(agent, output)
        return valid_agents
    
    def save_intermediate_metric(self, path: str, agents: List[MCTS], rollout) -> None:
        """
        保存中间指标
        """
        if self.config.is_sampling: 
            return
        states = [s.intermediate_metric for s in agents]
        statics = []
        # 计算每个rollout的通过率
        for i in range(rollout + 1):
            pass1, passn = 0, 0
            for idx, state in enumerate(states):
                max_value = -100
                max_value_result = False
                pass1_ans = False
                for idx, rollout_index in enumerate(state["rollout_indexs"]):
                    if rollout_index <= i:
                        if state["value_estimate"][idx] > max_value:
                            max_value = state["value_estimate"][idx]
                            max_value_result = state["judgements"][idx]
                        if state["judgements"][idx]:
                            pass1_ans = True
                if max_value_result:
                    pass1 += 1
                if pass1_ans:
                    passn += 1
            statics.append({
                "rollout": i,
                "pass1": pass1,
                "passn": passn,
                "len": len(states),
            })
        with open(path, "w", encoding='utf-8') as f:
            json.dump([statics,states], f, ensure_ascii=False, indent=4)

    
    def save_intermediate_rollouts(self, saved_jsonl_file, cur_data, agents, rollout_idx):
        """
        仅在MCTS模式且配置时开启 保存每个rollout的完整推理轨迹
        """
        if self.config.save_intermediate_rollouts and saved_jsonl_file and self.config.mode == "mcts":
            saved_json_dir = osp.dirname(saved_jsonl_file)
            saved_jsonl_file_name = osp.basename(saved_jsonl_file)
            saved_json_path = osp.join(saved_json_dir, f"rollout")
            if not os.path.exists(saved_json_path):
                os.mkdir(saved_json_path)
            self.save_intermediate_metric(osp.join(saved_json_path, f"intermediate_metric_{saved_jsonl_file_name}"), agents, rollout_idx)
            outs = self.output(agents)
            with open(osp.join(saved_json_path, f"rollout{rollout_idx:02}" + saved_jsonl_file_name), "a+", encoding='utf-8') as writer:
                for d in cur_data:
                    question = d["question"]
                    d["rstar"] = outs[question]
                    writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                    writer.flush()
    
    def output(self, agents: List[BaseTree]):
        """
        输出结果整理 key是 question
        """
        jsonlines = {}
        for i, agent in enumerate(agents):         
            jsonlines[agent.question] = agent.return_states()
        
        return jsonlines
    
    def solve(self, agents: List[BaseTree], saved_jsonl_file: str, cur_data: List[Dict[str, Any]]):
        
        for rollout in tqdm(range(self.max_agent_steps), desc="Rollout Processing"): # rollout times
            # Initialize the initial search starting point of agents, and the initial point of each rollout is root
            for agent in agents:
                agent.select_next_step(from_root=True)
                agent.rollout_idx = rollout

            for step in range(self.config.max_depth):
                print("-----------------Current Rollout: ", rollout, "-----------------")
                print("-----------------Current Step: ", step, "-----------------")

                # 1.生成前预处理 infer(image,prompt) critic(image,question,response_list)
                prompts,questions,images,prompts_span, valid_agents, invalid_agents, expanded_agents, valid_rewards = self.generate_preprocess(agents)
                
                if len(valid_agents + expanded_agents) < 1:
                    break

                # 2.调用llm or api 生成推理步骤
                assert len(prompts)==len(images)
                outputs = []
                for prompt,image in zip(prompts,images):
                    # response = self.llm.vlm_func(image_path=image,q=prompt)
                    response = seed_inference_one_sample(image_path=image,question=prompt)
                    api_output = APIOutput(content=response)
                    outputs.append(api_output)
                print("生成的文本内容:", [out.outputs[0].text for out in outputs])  # 模拟原访问方式

                # 3.对生成内容进行评分 evaluate
                assert len(images)==len(questions)==len(outputs)
                for i, (image, question, output) in enumerate(zip(images, questions, outputs)):
                    # pdb.set_trace()
                    reward_response = self.reward_model(image, question, [output.outputs[0].text])
                    score = reward_response["answer"][0][1]
                    output.value_estimate = score 
                print("带评分的输出:", [(out.outputs[0].text, out.value_estimate) for out in outputs])
                reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
            
                # 4.拓展候选叶子节点 & process output
                valid_agents = self.generate_postprocess(reconstructed_outputs, valid_agents)

                # 5.step evaluation 
                prompts, prompts_span = self.value_preprocess(valid_agents)
                reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
                valid_agents = self.value_postprocess(reconstructed_outputs, valid_agents)
                expanded_agents = self.value_postprocess([None] * len(expanded_agents), expanded_agents) # for expanded agents, just do selection step
                
                # 合并所有的agent
                agents = valid_agents + invalid_agents + expanded_agents

            # Save agents internal rollouts
            self.save_intermediate_rollouts(saved_jsonl_file, cur_data, agents, rollout)
            
        return self.output(agents)
    
    
