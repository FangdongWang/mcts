import os
from abc import abstractmethod
from termcolor import colored
from typing import Optional, Any, Dict, List, Callable, Type, Tuple, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from omegaconf import DictConfig, OmegaConf
from .mcts_node import BaseNode

class BaseTree(BaseModel):

    config: Any
    question: str
    ground_truth: Optional[Union[str, List[str]]] = None
    image_path: Optional[str] = None
    flag: Optional[str] = None
    root: Optional[Type[BaseNode]] = None
    current_node: Optional[Type[BaseNode]] = None 
    stop: Optional[List[str]] = None
    node_max_retry: int = 5

    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        if self.config.stop:
            # omegaconf.listconfig.ListConfig -> list
            self.stop = OmegaConf.to_object(self.config.stop)
        self.root = self.create_root()
        self.current_node = self.root

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            return cfg

    def create_root(self) -> Type[BaseNode]:
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        """
        subclass must implement
        """
    
    def collect_partial_solution(self, node: Type[BaseNode]) -> str:
        """
        收集部分解 推理轨迹 从当前节点回溯到根节点 收集每个节点的文本状态
        """
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            node = node.parent
        return "".join(reversed(trajectory))

    def return_states(self) -> Dict[str, Dict[str, str]]:
        """
        收集所有节点的状态 BFS
        """
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            if node.has_children():
                candidates.extend(node.children)
        return states
    
        

