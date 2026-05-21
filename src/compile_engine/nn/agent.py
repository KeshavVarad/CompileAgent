"""NNAgent: wraps the PolicyValueNet in the existing Agent protocol.

`NNAgent.choose(game, legal)` runs one forward pass, samples (training mode)
or argmaxes (inference mode), and optionally records the transition into a
provided buffer for PPO training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..actions import Action
from .encoder import encode_actions, encode_state
from .model import PolicyValueNet


@dataclass
class StepRecord:
    """One transition produced by the agent during rollout.

    Stored fields are exactly what PPO needs to compute the update later.
    """
    state: dict[str, np.ndarray]
    action_raw: np.ndarray
    action_card_ids: np.ndarray
    action_proto_ids: np.ndarray
    action_mask: np.ndarray
    action_idx: int
    log_prob: float
    value: float
    reward: float = 0.0   # filled later by the rollout collector
    done: bool = False    # filled later
    # advantage / return computed in the buffer after the episode ends
    advantage: float = 0.0
    ret: float = 0.0


def _state_to_torch(state: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in state.items()}


def _to_torch_actions(
    raw: np.ndarray, card_ids: np.ndarray, proto_ids: np.ndarray, mask: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(raw).unsqueeze(0).to(device),
        torch.from_numpy(card_ids).unsqueeze(0).to(device),
        torch.from_numpy(proto_ids).unsqueeze(0).to(device),
        torch.from_numpy(mask).unsqueeze(0).to(device),
    )


class NNAgent:
    """Stochastic-during-training, greedy-during-inference policy.

    Parameters
    ----------
    model
        A `PolicyValueNet`. Inference mode runs `torch.no_grad()` and argmaxes.
    device
        Torch device. CPU is fine for a small net; MPS works on Apple Silicon.
    stochastic
        If True, samples from the masked policy distribution. If False,
        argmaxes. Set True during PPO rollouts, False for play/eval.
    record
        Optional list to append `StepRecord`s into for every decision the
        agent makes. The collector later fills in `reward` / `done` /
        advantages / returns and consumes the records for PPO updates.
    """

    def __init__(
        self,
        model: PolicyValueNet,
        *,
        device: torch.device | str = "cpu",
        stochastic: bool = False,
        record: list[StepRecord] | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.stochastic = stochastic
        self.record = record

    def set_record(self, record: list[StepRecord] | None) -> None:
        self.record = record

    def choose(self, game, legal: list[Action]) -> Action:
        if not legal:
            raise RuntimeError("NNAgent.choose called with empty legal actions")
        perspective = game.decider()
        state = encode_state(game, perspective)
        raw, card_ids, proto_ids, mask = encode_actions(game, legal, perspective)
        s = _state_to_torch(state, self.device)
        ar, ac, ap, am = _to_torch_actions(raw, card_ids, proto_ids, mask, self.device)
        with torch.no_grad():
            logits, value = self.model(s, ar, ac, ap, am)
        logits = logits[0]
        value_scalar = float(value[0].item())
        # Renormalise over legal actions only.
        if self.stochastic:
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.distributions.Categorical(probs=probs).sample().item())
        else:
            idx = int(torch.argmax(logits).item())

        # log_prob of chosen action under the current policy (for PPO ratio).
        log_probs = torch.log_softmax(logits, dim=-1)
        log_prob = float(log_probs[idx].item())

        if self.record is not None:
            self.record.append(StepRecord(
                state=state,
                action_raw=raw,
                action_card_ids=card_ids,
                action_proto_ids=proto_ids,
                action_mask=mask,
                action_idx=idx,
                log_prob=log_prob,
                value=value_scalar,
            ))
        return legal[idx]
