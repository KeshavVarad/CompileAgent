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
    action_extra_card_ids: torch.Tensor
    action_mask: torch.Tensor
    action_idx: torch.Tensor
    old_log_prob: torch.Tensor
    advantage: torch.Tensor
    ret: torch.Tensor
    # Per-step decision class (DRAFT / PLAY / CHOOSE / COMPILE / DISCARD /
    # SHIFT) — used by per-action-type entropy regularisation.
    action_class: torch.Tensor
    # UNREAL-style aux supervision. Padded to the right shape even if the
    # collector didn't fill them in (default zeros = no-signal target),
    # though the train loop only enables the aux loss when collection is
    # wired up. See StepRecord.
    aux_opp_hand: torch.Tensor          # [B, CARD_VOCAB_SIZE]
    aux_compile_margin: torch.Tensor    # [B]


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
    from .encoder import CARD_VOCAB_SIZE
    keys = list(records[0].state.keys())
    state = {}
    for k in keys:
        arr = np.stack([r.state[k] for r in records])
        state[k] = torch.from_numpy(arr).to(device)
    action_raw = torch.from_numpy(np.stack([r.action_raw for r in records])).to(device)
    action_card_ids = torch.from_numpy(np.stack([r.action_card_ids for r in records])).to(device)
    action_proto_ids = torch.from_numpy(np.stack([r.action_proto_ids for r in records])).to(device)
    action_extra_card_ids = torch.from_numpy(
        np.stack([r.action_extra_card_ids for r in records]),
    ).to(device)
    action_mask = torch.from_numpy(np.stack([r.action_mask for r in records])).to(device)
    action_idx = torch.tensor([r.action_idx for r in records], dtype=torch.long, device=device)
    old_log_prob = torch.tensor([r.log_prob for r in records], dtype=torch.float32, device=device)
    advantage = torch.tensor([r.advantage for r in records], dtype=torch.float32, device=device)
    ret = torch.tensor([r.ret for r in records], dtype=torch.float32, device=device)
    action_class = torch.tensor(
        [r.action_class for r in records], dtype=torch.long, device=device,
    )
    # Aux supervision: stack opp-hand multi-hots (collector fills these in;
    # if unset we emit a zero vector so the aux loss is a no-op on that row).
    aux_oh = np.stack([
        r.aux_opp_hand_multi_hot
        if r.aux_opp_hand_multi_hot is not None
        else np.zeros(CARD_VOCAB_SIZE, dtype=np.float32)
        for r in records
    ])
    aux_opp_hand = torch.from_numpy(aux_oh).to(device)
    aux_compile_margin = torch.tensor(
        [r.aux_compile_margin for r in records], dtype=torch.float32, device=device,
    )
    return Batch(
        state=state,
        action_raw=action_raw,
        action_card_ids=action_card_ids,
        action_proto_ids=action_proto_ids,
        action_extra_card_ids=action_extra_card_ids,
        action_mask=action_mask,
        action_idx=action_idx,
        old_log_prob=old_log_prob,
        advantage=advantage,
        ret=ret,
        action_class=action_class,
        aux_opp_hand=aux_opp_hand,
        aux_compile_margin=aux_compile_margin,
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
            action_extra_card_ids=batch.action_extra_card_ids[sel_t],
            action_mask=batch.action_mask[sel_t],
            action_idx=batch.action_idx[sel_t],
            old_log_prob=batch.old_log_prob[sel_t],
            advantage=batch.advantage[sel_t],
            ret=batch.ret[sel_t],
            action_class=batch.action_class[sel_t],
            aux_opp_hand=batch.aux_opp_hand[sel_t],
            aux_compile_margin=batch.aux_compile_margin[sel_t],
        )
