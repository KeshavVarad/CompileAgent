"""Rollout buffer + GAE.

A single episode's records are appended in `play_episode`, then this module
computes returns / advantages in-place and stitches everything into a tensor
batch for PPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .agent import StepRecord


@dataclass
class Batch:
    state: dict[str, torch.Tensor]
    action_raw: torch.Tensor
    action_card_ids: torch.Tensor
    action_proto_ids: torch.Tensor
    action_mask: torch.Tensor
    action_idx: torch.Tensor
    old_log_prob: torch.Tensor
    advantage: torch.Tensor
    ret: torch.Tensor


def compute_gae(
    records: list[StepRecord],
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> None:
    """Generalised Advantage Estimation (Schulman et al. 2015). In-place.

    `last_value` is V(s_{T+1}) — the bootstrap value for the post-terminal
    state. Use 0 if the episode actually terminated; use net(s_T) if we cut
    off mid-game for any reason.
    """
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(records))):
        r = records[t]
        non_terminal = 0.0 if r.done else 1.0
        delta = r.reward + gamma * next_value * non_terminal - r.value
        gae = delta + gamma * lam * non_terminal * gae
        r.advantage = gae
        r.ret = gae + r.value
        next_value = r.value


def stack_batch(records: list[StepRecord], device: torch.device) -> Batch:
    """Bundle records into a torch Batch ready for the PPO update step."""
    keys = list(records[0].state.keys())
    state = {}
    for k in keys:
        arr = np.stack([r.state[k] for r in records])
        state[k] = torch.from_numpy(arr).to(device)
    action_raw = torch.from_numpy(np.stack([r.action_raw for r in records])).to(device)
    action_card_ids = torch.from_numpy(np.stack([r.action_card_ids for r in records])).to(device)
    action_proto_ids = torch.from_numpy(np.stack([r.action_proto_ids for r in records])).to(device)
    action_mask = torch.from_numpy(np.stack([r.action_mask for r in records])).to(device)
    action_idx = torch.tensor([r.action_idx for r in records], dtype=torch.long, device=device)
    old_log_prob = torch.tensor([r.log_prob for r in records], dtype=torch.float32, device=device)
    advantage = torch.tensor([r.advantage for r in records], dtype=torch.float32, device=device)
    ret = torch.tensor([r.ret for r in records], dtype=torch.float32, device=device)
    return Batch(
        state=state,
        action_raw=action_raw,
        action_card_ids=action_card_ids,
        action_proto_ids=action_proto_ids,
        action_mask=action_mask,
        action_idx=action_idx,
        old_log_prob=old_log_prob,
        advantage=advantage,
        ret=ret,
    )


def minibatches(batch: Batch, batch_size: int, rng: np.random.Generator):
    n = batch.action_idx.shape[0]
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        sel = idx[start : start + batch_size]
        sel_t = torch.from_numpy(sel.copy()).to(batch.action_idx.device)
        yield Batch(
            state={k: v[sel_t] for k, v in batch.state.items()},
            action_raw=batch.action_raw[sel_t],
            action_card_ids=batch.action_card_ids[sel_t],
            action_proto_ids=batch.action_proto_ids[sel_t],
            action_mask=batch.action_mask[sel_t],
            action_idx=batch.action_idx[sel_t],
            old_log_prob=batch.old_log_prob[sel_t],
            advantage=batch.advantage[sel_t],
            ret=batch.ret[sel_t],
        )
