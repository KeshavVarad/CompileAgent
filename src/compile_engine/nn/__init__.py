"""Neural network agent for Compile.

See docs/nn_design.md for the architecture rationale. This package is
opt-in: install with `pip install -e .[nn]` to pull in torch.
"""

from .encoder import (
    MAX_ACTIONS,
    MAX_HAND,
    MAX_STACK,
    NUM_CARDS,
    NUM_PROTOCOLS,
    encode_actions,
    encode_state,
    state_input_dim,
    action_input_dim,
)
from .model import PolicyValueNet
from .agent import NNAgent

__all__ = [
    "MAX_ACTIONS",
    "MAX_HAND",
    "MAX_STACK",
    "NUM_CARDS",
    "NUM_PROTOCOLS",
    "encode_actions",
    "encode_state",
    "state_input_dim",
    "action_input_dim",
    "PolicyValueNet",
    "NNAgent",
]
