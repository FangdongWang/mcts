import random
from typing import List, Optional, Literal
from enum import Enum, EnumMeta
from dataclasses import dataclass, field


TIMEOUT_SECONDS = 60
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."
ERROR_COLOR = "red"
TOO_MANY_CODE_ERRORS = "Too many consecutive steps have code errors."
TOO_MANY_STEPS = "Fail to sove the problem within limited steps."
NO_VALID_CHILD = "Fail to generate parsable text for next step."


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


_SEARCH_CHOICES = [
    "mcts",  # monte carlo tree search
    "bs",    # beam search
]

SEARCH_CHOICES = ChoiceEnum(_SEARCH_CHOICES)


_PROMPT_CHOICES = [
    "rstar", 
]

PROMPT_CHOICES = ChoiceEnum(_PROMPT_CHOICES)
@dataclass
class BaseConfig:

    mode: SEARCH_CHOICES = field(
        default="mcts", metadata={"help": "search mode for inference"}
    )
    few_shot_path: Optional[str] = field(
        default=None, metadata={"help": "few shot data json"}
    )
    prompt_path: Optional[str] = field(
        default=None, metadata={"help": "prompt config json"}
    )
    num_few_shot: int = field(
        default=0, metadata={"help": "the number of few-shot examples"}
    )

    # ------------------------------------- prompt args -------------------------------------
    prompt_wrap: PROMPT_CHOICES = field(
        default="rstar", metadata={"help": "prompt wrap type"}
    )
    result_unwrap: PROMPT_CHOICES = field(
        default="rstar", metadata={"help": "result unwrap"}
    )
    step_delim: str = field(
        default="\n", metadata={"help": "delimiter between two steps"}
    )
    
    # -------------------------------- mcts args -------------------------------------
    n_generate_sample: int = field(
        default=1, metadata={"help": "how many samples generated for each step. B2 in paper."}
    )
    stop: Optional[List[str]] = field(
        default=None, metadata={"help": "possible stop tokens for each step"}
    )
    step_beam_width: int = field(
        default=1, metadata={"help": "beam width for each step. B1 in paper."}
    )
    max_depth: int = field(
        default=4, metadata={"help": "maximum depth of the tree, ie., maximum steps of completion."}
    )
    iterations: int = field(
        default=1, metadata={"help": "number of simulations in mcts"}
    )
    positive_reward: float = field(
        default=1.0, metadata={"help": "reward for positive example"}
    )
    negative_reward: float = field(
       default=-1.0, metadata={"help": "reward for negative example"}
    )
    errors_threshold: int = field(
        default=0, metadata={"help": "maximum code errors allowed, ie., if errors_count > errors_threshold, the tree growth from this node should be terminated."}
    )
    need_value_func: bool = field(
        default=False, metadata={"help": "whether to use value head in decoding"}
    )
    update_leaf_value: bool = field(
        default=False, metadata={"help": "update leaf value in mcts"}
    )
    c_puct: float = field(
        default=2, metadata={"help": "weight of c_puct in mcts"}
    )
    is_sampling: bool = field(
        default=False, metadata={"help": "solution generation in mcts"}
    )

    #  ------------------------------------ offline inferene args  -------------------------------------
    prune: bool = field(
        default=False, metadata={"help": "prune the tree in a complete mcts tree"}
    )
    # ------------------------------------ other args ------------------------------------ 
    batch_size: int = field(
        default=-1, metadata={"help": "batch size for batch inference"}
    )
    max_model_len: int = field(
        default=8192, metadata={"help": "maximum model length"}
    )
    terminal_sample: bool = field(
        default=False
    )

    save_intermediate_rollouts: bool = field(
        default=True, metadata={"help": "save intermediate rollouts in ./rollout"}
    )

