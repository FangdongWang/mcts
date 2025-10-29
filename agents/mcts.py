from __future__ import annotations
from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from pydantic import field_validator
import pdb

from .mcts_node import BaseNode,MCTSNode
from .tree import BaseTree
from .beam_search import BS
from .config import TOO_MANY_STEPS,NO_VALID_CHILD,TOO_MANY_CODE_ERRORS
from .utils import math_equiv as is_equiv




"""
函数调用:
    select_next_step -> selection -> select_child
"""
class MCTS(BS):
    search_node: Type[BaseNode] = None

    intermediate_metric: Dict = {
        "question": "",
        "gt": "", 
        "answers": [],
        "judgements": [],
        "value_estimate": [],
        "rollout_indexs": [],
    }

    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent, 
            additional_state_keys=self.NODE_KEYS,
            c_puct=self.config.c_puct,
        )
    
    def selection(self, from_root=False) -> Optional[Type[MCTSNode]]:
        """
        从起始节点出发 调用 select_child 方法 根据puct值 选择子节点 直到找到非终端且未完全拓展的节点
        """
        if from_root:
            start_node = self.root
        else:
            start_node = self.search_node
        # select a child node
        node = start_node
        if node is None: 
            return None
        # 如果当前节点有节点或终端，继续选择子节点 
        if node.has_children() or node.is_terminal:
            next_node = self.select_child(node)     # To encourage exploration, select from non-terminal children
            if next_node is None:                   # if None，it mean all children are terminal
                node.is_terminal = True
            node = next_node
        # 返回非终端节点 继续探索
        return None if (node is None or node.is_terminal) else node

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        """
        根据puct算法选择节点 vs uct
        PUCT = Q(s,a)+c_puct*P(s,a)*sqrt(N(s)/(1+N(s,a)))
        Q(s,a)节点平均价值
        P(s,a)节点的先验概率
        """
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue
            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        #return random.choice(best_childs) if best_childs else None
        return best_childs[0] if best_childs else None

    def select_next_step(self, outputs=None, from_root=False) -> None:
        """
        选择下一步与回溯 将符合selection函数的节点 加入到候选节点中
        """
        self.search_node = self.current_nodes[0] if self.current_nodes else None
        self.current_nodes = []

        # 处理模型输出价值估计更新候选节点价值
        if outputs:
            # 终端节点且采样模式
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                if candidate_node.is_terminal and self.config.is_sampling:
                    continue
                # 获取价值估计
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True

                # 达到终端节点 & 当前节点提取出final_ans backup 回溯
                if candidate_node.is_terminal and candidate_node.state["final_answer"]:
                    # for terminal node: update_recursive
                    # 无效答案 -> 更新reward
                    if candidate_node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS]:
                        candidate_node.update(self.config.negative_reward)
                    else:
                        # final_ans 有效 调用 record_intermediate_metric 函数 save intermediate metric
                        self.record_intermediate_metric(answer=candidate_node.state["final_answer"], value_estimate=value_estimate)

                        candidate_node.update_recursive(value_estimate, self.root)
                else:
                    # for intermediate node: just update the value
                    if self.config.terminal_sample:
                        pass
                    else:
                        candidate_node.update(value_estimate)
                # 判断终端节点是否有效
                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)

        selection_node = self.selection(from_root=from_root)
        if selection_node is not None:
            self.current_nodes.append(selection_node)
    


    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
        """
        遍历模型输出的每个候选结果 生成子节点
        """
        for idx, output in enumerate(outputs):
            if not output.stop_reason: 
                output.stop_reason = ""
            """
            parser_result = {
                "action": "",
                "action_input": "",
                "final_answer": "",
            }
            """
            step_result, parser_result = self.step_unwrap(output.text + output.stop_reason)
            self.create_child(step_result, parser_result, node, idx)

    def create_child(
        self, 
        step_result: str, 
        parser_result: Dict[str, str], 
        node: Type[MCTSNode],
        idx: int,
    ) -> None:
        """
        根据BS的create_child 新增连续错误计数和终端节点估计
        """
        new_node = self.create_node(parent=node)
        parent_child_count = len(node.children)
        new_node.tag = f"{node.tag}.{parent_child_count + 1}"
        new_node.depth = node.depth + 1

        #  parser_result 为空 ->结束推理
        #  parser_result 含 final_ans -> 验证final_ans 并结束
        #  否则更新 new_node 的 text 为上一步的答案
        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        # elif parser_result["action"]:
        #     observation = code_execution(node, parser_result)
        #     new_node.state["action"] = parser_result["action"]
        #     new_node.state["action_input"] = parser_result["action_input"]
        #     new_node.state["observation"] = observation
        #     if CODE_END in parser_result["action_input"]:
        #         observation = self.obs_wrap(observation)
        #         new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
        #     else:
        #         new_node.state["text"] = step_result
                
        #     if "error" in observation.lower():
        #         new_node.consecutive_errors = node.consecutive_errors + 1
        #         if new_node.consecutive_errors >= self.config.errors_threshold:
        #             observation = self.obs_wrap(observation)
        #             step_result = step_result + CODE_END if CODE_END not in step_result else step_result
        #             new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
        #             new_node.is_terminal = True
        #             new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
        #             self.eval_final_answer(new_node)
        else:
            new_node.state["text"] = step_result

        # new_node 有效但超过最大深度 -> 结束推理
        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        """
        todo
        为终端节点打分
        1.无效答案 赋值negative_reward
        2.有效答案 调用is_equiv-> True or Flase 赋值
        """
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            # if the final answer is not valid, update the node with negative reward
            node.update(self.config.negative_reward)
            return 
        
        if self.config.is_sampling:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer, self.question, self.flag)
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
        else:
            # just append the node to candidate_nodes, will update the value in select_next_step()
            self.candidate_nodes.append(node)

    def record_intermediate_metric(self, answer, value_estimate):
        """
        记录每次rollout的中间结果
        """
        self.intermediate_metric["question"] = self.question
        self.intermediate_metric["gt"] = self.ground_truth
        # each rollout retains the answer with the highest value_estimate
        # Check if the rollout's answer is already in the list
        if self.intermediate_metric["rollout_indexs"] and self.rollout_idx in self.intermediate_metric["rollout_indexs"]:
            # Find the index of the existing rollout
            index = self.intermediate_metric["rollout_indexs"].index(self.rollout_idx)
            if value_estimate > self.intermediate_metric["value_estimate"][index]:
                self.intermediate_metric["answers"][index] = answer
                self.intermediate_metric["judgements"][index] = is_equiv(self.ground_truth, answer, self.question, self.flag)
                self.intermediate_metric["value_estimate"][index] = value_estimate
        else:
            # If the rollout's answer is not in the list, add it
            self.intermediate_metric["answers"].append(answer)
            self.intermediate_metric["judgements"].append(is_equiv(self.ground_truth, answer,self.question, self.flag))
            self.intermediate_metric["value_estimate"].append(value_estimate)
            self.intermediate_metric["rollout_indexs"].append(self.rollout_idx)


    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            value_estimate = output.value_estimate
            if value_estimate is not None:  
                self.expand_node(output.outputs, current_node)
            else:
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True

            if self.config.update_leaf_value:
                # if need update leaf node value, just append the node to candidate_nodes, will update the value in select_next_step()
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node) 

    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["visit_count"] = node.visit_count()
            if node.has_children():
                candidates.extend(node.children)
        return states




    

