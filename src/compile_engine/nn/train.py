"""Handwritten PPO loop for the Compile NN agent.

Single-process self-play vs an opponent pool. Designed to be tractable on a
laptop (CPU or MPS). The interesting bits are
  - `play_episode`: rolls out one game, recording transitions from the agent
    seat. The shaped per-step reward comes from the engine's existing
    `Δ compiled_protocols`. The terminal +/-1 is added on the last step.
  - `ppo_update`: standard clipped-surrogate PPO update with value-MSE +
    entropy bonus. Gradient clip 0.5.
  - `train`: iterate (collect, GAE, K-epochs of minibatch PPO, eval, snapshot).
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..agents import GreedyAgent, RandomAgent
from ..env import play_game
from ..game import Game
from ..state import GameConfig
from .agent import (
    ACTION_CLASS_NAMES,
    N_ACTION_CLASSES,
    NNAgent,
    StepRecord,
)
from .buffer import Batch, compute_gae, minibatches, stack_batch
from .encoder import CARD_VOCAB_SIZE
from .model import PolicyValueNet


@dataclass
class TrainConfig:
    iters: int = 500
    games_per_iter: int = 32      # was 16; PPO wants ~2k+ transitions / iter
    ppo_epochs: int = 4
    batch_size: int = 256
    lr: float = 1e-4              # was 3e-4; reduced after observed KL spikes
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.01
    grad_clip: float = 0.5
    # KL early-stop: break out of the PPO epoch loop once approx-KL crosses
    # `target_kl`. Standard PPO trick to keep updates inside the trust region.
    # Set to None to disable.
    target_kl: float | None = 0.03
    # Adaptive entropy regularization. We watched our previous 260-iter run
    # collapse entropy from ~0.83 → ~0.20, which froze the policy onto a
    # narrow strategy (face-down ratio stuck at ~10%, three-protocol draft
    # spam). Holding entropy near `entropy_floor` keeps exploration alive
    # without manually tuning `c_entropy` mid-run: if entropy drops below
    # the floor we scale `c_entropy` up; if it climbs comfortably above
    # `entropy_ceiling` we scale it back down. Clamped to the safety range
    # below. Set entropy_floor to None to disable and use a fixed c_entropy.
    entropy_floor: float | None = 0.4
    entropy_ceiling: float = 0.55
    c_entropy_step: float = 1.15   # multiplicative bump per iter when off-target
    c_entropy_min: float = 0.001
    c_entropy_max: float = 0.5
    # Per-action-type entropy multipliers. The standard `c_entropy` term is
    # multiplied by these per-class scalars when summing entropy across the
    # batch. DRAFT decisions need a much larger entropy bonus because the
    # Spark-v4 analysis (docs/STRATEGY_THESIS_sparkv4.md) showed the policy
    # mode-collapsed to Darkness-98% without any reduction in win rate when
    # blocked — i.e. the global entropy bonus is too weak to keep DRAFT
    # exploration alive while still allowing the PLAY/COMPILE logits to
    # sharpen. CHOOSE also gets a moderate boost (the Love-1-End trade was
    # rejected 47/47 times — a credit-assignment-starved local optimum that
    # more entropy on CHOOSE should help break). Order matches
    # ACTION_CLASS_NAMES: (DRAFT, PLAY, CHOOSE, COMPILE, DISCARD, SHIFT).
    per_class_entropy_mult: tuple[float, ...] = (4.0, 1.0, 2.0, 1.0, 1.0, 1.0)
    # UNREAL-style auxiliary losses on the shared trunk. These don't bias
    # the policy directly — they only give the encoder richer training
    # signal (predict opponent's hidden hand contents + final compile
    # margin). Coefficients are small relative to c_value so the
    # supervised aux objective doesn't drown out RL gradients.
    c_aux_opp_hand: float = 0.05      # 0 disables the aux head
    c_aux_margin: float = 0.05
    snapshot_every: int = 10
    eval_games: int = 60
    # Snapshots are always checkpointed to disk. With PFSP sampling weak
    # snapshots get downweighted automatically, so we drop the old
    # "only add to pool above some WR_random threshold" gate — pool
    # diversity is what we want, PFSP handles quality.
    pool_threshold_wr_random: float = 0.0
    # Pool cap. PFSP weighting (see OpponentPool) keeps per-iter cost
    # bounded since we still only sample one opponent per game, so 20 is
    # cheap (each NN opp is ~2.4 MB on MPS). Was 6 with FIFO; bumped to
    # 16 to let the policy face a real distribution of historical styles.
    max_pool_size: int = 16
    # Prioritized Fictitious Self-Play exponent. p=0 → uniform sampling.
    # p=2 → strongly biased toward opponents we currently lose to. Higher
    # p focuses harder on weak spots but reduces diversity. AlphaStar used
    # p=2.0 for the league. min_weight floors each opponent so the
    # weakest pool members still get sampled occasionally (training
    # against baselines is good for stability).
    pfsp_p: float = 2.0
    pfsp_min_weight: float = 0.04
    # Rolling window size for the win-rate estimate used by PFSP.
    pfsp_window: int = 80
    expansion_prob: float = 0.5    # mix AX01 (Apathy/Hate/Love) in training
    main2_prob: float = 0.4        # mix MN02 (Chaos/Clarity/.../War) in training
    aux2_prob: float = 0.4         # mix AX02 (Assimilation/Diversity/Unity) in training
    max_turns: int = 200
    seed: int = 0
    device: str = "cpu"
    save_dir: str | None = None
    # Optional warm-start: path to a snapshot file ({"model": state_dict, ...}).
    # Same format the PPO and AZ trainers both write, so an AZ checkpoint
    # can be handed to PPO (and vice versa) as long as the model
    # architecture and encoder shape match the current code.
    init_ckpt: str | None = None


def _make_config(rng: random.Random, base: TrainConfig) -> GameConfig:
    return GameConfig(
        include_expansion=rng.random() < base.expansion_prob,
        include_main2=rng.random() < base.main2_prob,
        include_aux2=rng.random() < base.aux2_prob,
        seed=rng.randint(0, 2**31 - 1),
        max_turns=base.max_turns,
    )


# ---------------------------------------------------------------------------
# Opponent pool with Prioritized Fictitious Self-Play (PFSP)
# ---------------------------------------------------------------------------


@dataclass
class _PoolMember:
    """One entry in the opponent pool: an agent, a stable display name, and
    a rolling window of recent game outcomes against the trainee."""
    agent: object                         # RandomAgent | GreedyAgent | NNAgent
    name: str
    # results[i] = 1 if the trainee beat this opponent, 0 otherwise.
    # Bounded length = TrainConfig.pfsp_window.
    results: deque = field(default_factory=deque)


class OpponentPool:
    """League pool that samples opponents weighted by current weakness.

    Per-opponent we track a rolling-window win rate WR (trainee's win rate
    vs that opponent). Sampling weight is `max(min_weight, (1 - WR)^p)`,
    so opponents the trainee currently loses to are over-represented while
    everyone still gets some non-zero share. This is the same recipe
    AlphaStar's league used to crack training plateaus.

    Eviction policy: when the pool exceeds `max_size`, we drop the
    member with the **highest trainee WR** (most beaten = least useful
    as a training opponent). Anchors (random / greedy) are exempt
    because their roles in the pool are stability-of-eval, not depth.
    """

    def __init__(self, max_size: int, p: float, min_weight: float, window: int) -> None:
        self.max_size = max_size
        self.p = p
        self.min_weight = min_weight
        self.window = window
        self.members: list[_PoolMember] = []
        # Anchors get exempted from eviction so absolute Elo stays
        # interpretable across the run.
        self._anchor_names: set[str] = set()

    def add(self, agent: object, name: str, *, is_anchor: bool = False) -> str | None:
        """Add `agent` to the pool, evicting the most-beaten non-anchor
        if over capacity. Returns the name of the evicted member (or None)."""
        self.members.append(_PoolMember(
            agent=agent, name=name,
            results=deque(maxlen=self.window),
        ))
        if is_anchor:
            self._anchor_names.add(name)
        evicted: str | None = None
        if len(self.members) > self.max_size:
            # Find the non-anchor with the highest WR (most beaten).
            candidates = [(i, self._wr(m)) for i, m in enumerate(self.members)
                          if m.name not in self._anchor_names]
            if candidates:
                idx = max(candidates, key=lambda t: t[1])[0]
                evicted = self.members[idx].name
                self.members.pop(idx)
        return evicted

    def sample(self, rng: random.Random) -> _PoolMember:
        if not self.members:
            raise RuntimeError("pool is empty")
        weights = [self._weight(m) for m in self.members]
        total = sum(weights)
        if total <= 0:
            # Degenerate (would only happen with all-zero floors + WR=1.0
            # everywhere) — fall back to uniform.
            return rng.choice(self.members)
        r = rng.random() * total
        acc = 0.0
        for m, w in zip(self.members, weights):
            acc += w
            if r <= acc:
                return m
        return self.members[-1]

    def record(self, member: _PoolMember, trainee_won: bool) -> None:
        member.results.append(1 if trainee_won else 0)

    def _wr(self, m: _PoolMember) -> float:
        if not m.results:
            # New entrant: treat as 50/50 so it gets a moderate weight
            # right away rather than being maxed out.
            return 0.5
        return sum(m.results) / len(m.results)

    def _weight(self, m: _PoolMember) -> float:
        wr = self._wr(m)
        return max(self.min_weight, (1.0 - wr) ** self.p)

    def __len__(self) -> int:
        return len(self.members)

    def summary(self) -> list[tuple[str, float, int]]:
        """List of (name, trainee_wr, n_games) for logging."""
        return [(m.name, self._wr(m), len(m.results)) for m in self.members]


def _opp_hand_multi_hot(game: Game, opp_seat: int) -> np.ndarray:
    """Build a multi-hot of the opponent's hand-card def_ids.

    Used as the supervised target for the UNREAL aux head that predicts
    hidden-information state. The agent's own encoder never sees opp's
    hand contents (only counts), so this is genuinely held-out signal.
    """
    # vocab layout matches PolicyValueNet.lookup_card: index 0 = PAD,
    # 1 = HIDDEN, def_id d → token (d + 2). We use the same token id
    # space here so the aux head's output is comparable to the model's
    # internal card vocabulary.
    target = np.zeros(CARD_VOCAB_SIZE, dtype=np.float32)
    for c in game.state.players[opp_seat].hand:
        tok = c.def_id + 2
        if 0 <= tok < CARD_VOCAB_SIZE:
            target[tok] = 1.0
    return target


def play_episode(
    model: PolicyValueNet,
    opponent,
    cfg: TrainConfig,
    rng: random.Random,
    device: torch.device,
) -> tuple[list[StepRecord], int | None, int]:
    """Play one game. Returns (records from agent's seat, winner, agent_seat)."""
    agent_seat = rng.randint(0, 1)
    opp_seat = 1 - agent_seat
    records: list[StepRecord] = []
    agent = NNAgent(model, device=device, stochastic=True, record=records)
    cfg_game = _make_config(rng, cfg)
    game = Game(cfg_game)
    game.start()
    agents = (agent, opponent) if agent_seat == 0 else (opponent, agent)
    # Track shaped rewards from the agent's perspective.
    prev_compiled = (0, 0)
    while not game.is_over():
        who = game.decider()
        legal = game.legal_actions()
        if not legal:
            break
        n_records_before = len(records)
        action = agents[who].choose(game, legal)
        game.step(action)
        # Was this an agent decision? If so, record the shaped reward
        # corresponding to the post-step state.
        if who == agent_seat and len(records) > n_records_before:
            new_compiled = (
                sum(game.state.players[0].compiled),
                sum(game.state.players[1].compiled),
            )
            shaping = (
                (new_compiled[agent_seat] - prev_compiled[agent_seat])
                - (new_compiled[1 - agent_seat] - prev_compiled[1 - agent_seat])
            )
            records[-1].reward = float(shaping)
            records[-1].done = game.is_over()
            prev_compiled = new_compiled
            # Aux ground truth: opp's hidden hand contents *at decision time*.
            # Captured after step() since play_card / refresh effects mutate
            # the same hand on this turn; the post-step state is what the
            # bot's next observation would see, which is the closest match
            # to "what's the opp holding now" the encoder needs to predict.
            records[-1].aux_opp_hand_multi_hot = _opp_hand_multi_hot(game, opp_seat)

    # Terminal credit on the last agent record + back-fill final compile
    # margin onto every captured record. The margin label is the SAME for
    # every record in the episode (it's a per-episode regression target,
    # like a value-function bootstrap from the truly terminal state).
    if records:
        w = game.state.winner
        if w is not None:
            terminal = 1.0 if w == agent_seat else -1.0
        else:
            terminal = 0.0
        records[-1].reward += terminal
        records[-1].done = True
        final_margin = float(
            sum(game.state.players[agent_seat].compiled)
            - sum(game.state.players[opp_seat].compiled)
        )
        for r in records:
            r.aux_compile_margin = final_margin
    return records, game.state.winner, agent_seat


def evaluate(
    model: PolicyValueNet, opponent, n_games: int, device: torch.device, *,
    expansion_prob: float = 0.5, main2_prob: float = 0.4, aux2_prob: float = 0.4,
    seed: int = 0,
) -> dict:
    """Stochastic-policy win-rate vs `opponent`. Alternates seats.

    Eval samples from the policy (rather than argmaxing) because the
    deployed agent will also sample at inference — we're targeting a
    mixed Nash equilibrium, not a deterministic policy. Argmax WR would
    over-state the trainee against an opponent that exploits its
    determinism.
    """
    rng = random.Random(seed)
    agent = NNAgent(model, device=device, stochastic=True)
    wins = {0: 0, 1: 0, None: 0}
    agent_wins = 0
    for i in range(n_games):
        agent_seat = i % 2
        cfg = GameConfig(
            include_expansion=rng.random() < expansion_prob,
            include_main2=rng.random() < main2_prob,
            include_aux2=rng.random() < aux2_prob,
            seed=rng.randint(0, 2**31 - 1),
        )
        g = play_game(
            agent0=agent if agent_seat == 0 else opponent,
            agent1=opponent if agent_seat == 0 else agent,
            config=cfg,
        )
        w = g.state.winner
        wins[w] = wins.get(w, 0) + 1
        if w == agent_seat:
            agent_wins += 1
    return {
        "win_rate": agent_wins / n_games,
        "games": n_games,
        "by_winner": wins,
    }


def ppo_update(
    model: PolicyValueNet,
    optimiser: torch.optim.Optimizer,
    batch: Batch,
    cfg: TrainConfig,
    rng: np.random.Generator,
) -> dict:
    """One PPO update: K epochs of minibatched clipped-surrogate + value MSE.

    Uses the Schulman K3 approx-KL estimator
        kl ≈ E[(r - 1) - log r]
    which is non-negative and lower variance than the K1 estimator
        E[log(old) - log(new)] = -E[log r].
    When `cfg.target_kl` is set, we break out of the epoch loop the moment
    a minibatch's KL exceeds the threshold — standard PPO trust-region trick.
    """
    stats = {
        "pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
        "aux_opp_hand_loss": 0.0, "aux_margin_loss": 0.0,
        "approx_kl": 0.0, "n": 0, "stopped_at_epoch": cfg.ppo_epochs,
    }
    # Per-class entropy bookkeeping (summed numerator + summed weight per
    # class, divided to a mean at the end).
    ent_by_class_sum = [0.0] * N_ACTION_CLASSES
    ent_by_class_n = [0] * N_ACTION_CLASSES
    adv = batch.advantage
    if adv.numel() > 1:
        batch.advantage = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Per-class entropy multipliers as a device tensor — sized once per call.
    class_mult = torch.tensor(
        cfg.per_class_entropy_mult, dtype=torch.float32,
        device=batch.action_idx.device,
    )
    aux_enabled = (cfg.c_aux_opp_hand > 0.0) or (cfg.c_aux_margin > 0.0)

    stop_early = False
    for epoch in range(cfg.ppo_epochs):
        for mb in minibatches(batch, cfg.batch_size, rng):
            if aux_enabled:
                logits, value, aux = model(
                    mb.state, mb.action_raw, mb.action_card_ids, mb.action_proto_ids,
                    mb.action_extra_card_ids, mb.action_mask, return_aux=True,
                )
            else:
                logits, value = model(
                    mb.state, mb.action_raw, mb.action_card_ids, mb.action_proto_ids,
                    mb.action_extra_card_ids, mb.action_mask,
                )
            log_probs = F.log_softmax(logits, dim=-1)
            new_logp = log_probs.gather(1, mb.action_idx.unsqueeze(-1)).squeeze(-1)
            log_ratio = new_logp - mb.old_log_prob
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pg_loss = -torch.min(ratio * mb.advantage, clipped * mb.advantage).mean()
            v_loss = F.mse_loss(value, mb.ret)
            probs = torch.softmax(logits, dim=-1)
            per_step_entropy = -(
                probs * log_probs * mb.action_mask.float()
            ).sum(dim=-1)                                            # [B]
            # Per-class mean entropy — each class contributes proportionally
            # to its multiplier, NOT to how often it appears in the batch.
            # Previously this was `(per_step_entropy * weight_per_step).mean()`
            # which silently weighted DRAFT (~5% of transitions) by its batch
            # share — making a "3.0 mult" actually contribute 3.0 × 5% = 15%
            # of the entropy term, not the intended 3×. The fix is to mean
            # within each class first, then weighted-sum across classes.
            # Normalized by sum-of-multipliers so the absolute magnitude of
            # the entropy term stays comparable to the all-1-mult case
            # (i.e. c_entropy keeps its previous meaning).
            weighted_sum = torch.zeros((), device=per_step_entropy.device)
            total_weight = 0.0
            for cls in range(N_ACTION_CLASSES):
                mask = (mb.action_class == cls)
                if mask.any():
                    cls_mean_entropy = per_step_entropy[mask].mean()
                    weighted_sum = weighted_sum + class_mult[cls] * cls_mean_entropy
                    total_weight += float(cfg.per_class_entropy_mult[cls])
            if total_weight > 0.0:
                entropy_weighted = weighted_sum / total_weight
            else:
                entropy_weighted = per_step_entropy.mean()
            # Unweighted entropy for logging — keeps the metric comparable
            # to historical runs.
            entropy_unweighted = per_step_entropy.mean()

            # UNREAL-style auxiliary losses on the shared trunk.
            aux_oh_loss = torch.tensor(0.0, device=logits.device)
            aux_m_loss = torch.tensor(0.0, device=logits.device)
            if aux_enabled:
                # Opp-hand multi-label classification. Skip rows where the
                # collector didn't fill in a target (all-zero label) so we
                # don't push the head toward "opp hand is empty."
                tgt = mb.aux_opp_hand
                has_target = (tgt.sum(dim=-1) > 0).float().unsqueeze(-1)   # [B, 1]
                if cfg.c_aux_opp_hand > 0.0:
                    bce = F.binary_cross_entropy_with_logits(
                        aux["opp_hand_logits"], tgt, reduction="none",
                    )
                    aux_oh_loss = (bce * has_target).sum() / (has_target.sum() * tgt.shape[-1] + 1e-8)
                if cfg.c_aux_margin > 0.0:
                    aux_m_loss = F.smooth_l1_loss(aux["margin"], mb.aux_compile_margin)

            loss = (
                pg_loss
                + cfg.c_value * v_loss
                - cfg.c_entropy * entropy_weighted
                + cfg.c_aux_opp_hand * aux_oh_loss
                + cfg.c_aux_margin * aux_m_loss
            )

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()

            with torch.no_grad():
                # K3 estimator: non-negative, lower variance.
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                # Per-class entropy: bucket the per-step entropies by
                # action_class and accumulate sum + count.
                for cls in range(N_ACTION_CLASSES):
                    mask = (mb.action_class == cls)
                    if mask.any():
                        ent_by_class_sum[cls] += float(per_step_entropy[mask].sum().item())
                        ent_by_class_n[cls] += int(mask.sum().item())

            stats["pg_loss"] += float(pg_loss.item())
            stats["v_loss"] += float(v_loss.item())
            stats["entropy"] += float(entropy_unweighted.item())
            stats["aux_opp_hand_loss"] += float(aux_oh_loss.item())
            stats["aux_margin_loss"] += float(aux_m_loss.item())
            stats["approx_kl"] += float(approx_kl)
            stats["n"] += 1

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                stop_early = True
                break
        if stop_early:
            stats["stopped_at_epoch"] = epoch + 1
            break

    if stats["n"]:
        for k in (
            "pg_loss", "v_loss", "entropy",
            "aux_opp_hand_loss", "aux_margin_loss", "approx_kl",
        ):
            stats[k] /= stats["n"]
    # Per-class entropy means for diagnostics (NaN where unobserved).
    stats["entropy_by_class"] = {
        ACTION_CLASS_NAMES[cls]: (
            ent_by_class_sum[cls] / ent_by_class_n[cls]
            if ent_by_class_n[cls] > 0 else float("nan")
        )
        for cls in range(N_ACTION_CLASSES)
    }
    stats["count_by_class"] = {
        ACTION_CLASS_NAMES[cls]: ent_by_class_n[cls]
        for cls in range(N_ACTION_CLASSES)
    }
    return stats


class _ConfigWithEntropy:
    """Thin proxy that exposes the same fields as TrainConfig but with a
    per-iter c_entropy override. Lets ppo_update stay pure (no mutation of
    the user's TrainConfig)."""

    def __init__(self, base: TrainConfig, c_entropy: float) -> None:
        self._base = base
        self._c_entropy = c_entropy

    def __getattr__(self, name: str):
        if name == "c_entropy":
            return self._c_entropy
        return getattr(self._base, name)


def _adapt_c_entropy(cfg: TrainConfig, current: float, measured_entropy: float) -> float:
    """Move c_entropy toward keeping policy entropy in [floor, ceiling].

    If entropy < floor: bump c_entropy up so next iter pushes harder on
    the entropy term. If entropy > ceiling: ease it back down. Stays put
    inside the band. Clamped to [c_entropy_min, c_entropy_max] for
    numerical safety.
    """
    if cfg.entropy_floor is None:
        return current
    if measured_entropy < cfg.entropy_floor:
        current *= cfg.c_entropy_step
    elif measured_entropy > cfg.entropy_ceiling:
        current /= cfg.c_entropy_step
    return max(cfg.c_entropy_min, min(cfg.c_entropy_max, current))


def _resolve_device(name: str) -> torch.device:
    if name in ("auto",):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: TrainConfig | None = None) -> PolicyValueNet:
    cfg = cfg or TrainConfig()
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    model = PolicyValueNet().to(device)
    if cfg.init_ckpt:
        state = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)
        # strict=False because models trained before the UNREAL aux heads
        # were added (PR adding `aux_opp_hand_head` + `aux_margin_head`)
        # have no entries for those parameter names. The aux heads stay
        # at their (small) random init and start learning from scratch.
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            print(
                f"  ckpt-load: missing={list(missing)[:4]} "
                f"(showing up to 4 of {len(missing)})  "
                f"unexpected={list(unexpected)[:4]} (of {len(unexpected)})"
            )
        src_iter = state.get("iter")
        src_wrg = state.get("wr_greedy")
        print(
            f"Hot-started PPO from {cfg.init_ckpt}"
            + (f" (source iter={src_iter}" if src_iter is not None else " (")
            + (f", wr_greedy={src_wrg:.2f}" if src_wrg is not None else "")
            + ")"
        )
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Adaptive entropy coefficient — we let `cfg.c_entropy` be the starting
    # point and scale it toward the floor/ceiling band each iter. (Held in
    # a local var so the cfg object stays immutable for reproducibility.)
    c_entropy = cfg.c_entropy

    # Opponent pool starts with light baselines (anchors are exempt from
    # eviction); snapshots get added every snapshot_every.
    pool = OpponentPool(
        max_size=cfg.max_pool_size,
        p=cfg.pfsp_p,
        min_weight=cfg.pfsp_min_weight,
        window=cfg.pfsp_window,
    )
    pool.add(RandomAgent(seed=1), "random", is_anchor=True)
    pool.add(GreedyAgent(seed=2), "greedy", is_anchor=True)
    # When hot-starting, seed the pool with a frozen copy of the source so
    # PFSP has something stronger than random/greedy to weight against.
    # Without this the first iters of PPO sample mostly weak opponents
    # and the policy regresses toward beating-random tactics.
    if cfg.init_ckpt:
        frozen = copy.deepcopy(model).eval()
        # Stochastic pool opponents — matches inference behavior and is
        # the right target for a mixed-equilibrium policy. See the eval
        # docstring above for the same reasoning.
        pool.add(
            NNAgent(frozen, device=device, stochastic=True),
            name=Path(cfg.init_ckpt).stem,
        )

    save_dir = Path(cfg.save_dir) if cfg.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl" if save_dir is not None else None
    # Fresh file per run — caller controls run dirs (typically timestamped).
    if metrics_path is not None and metrics_path.exists():
        metrics_path.unlink()

    for it in range(1, cfg.iters + 1):
        t0 = time.perf_counter()
        all_records: list[StepRecord] = []
        n_wins = 0
        for _ in range(cfg.games_per_iter):
            opp_member = pool.sample(rng)
            records, winner, seat = play_episode(model, opp_member.agent, cfg, rng, device)
            compute_gae(records, gamma=cfg.gamma, lam=cfg.lam, last_value=0.0)
            all_records.extend(records)
            trainee_won = winner == seat
            pool.record(opp_member, trainee_won)
            if trainee_won:
                n_wins += 1
        rollout_wr = n_wins / max(1, cfg.games_per_iter)
        if not all_records:
            print(f"[iter {it}] no records, skipping update")
            continue

        # Drive c_entropy toward the [floor, ceiling] band BEFORE the PPO
        # update so the regulariser used this iter reflects what we want.
        # The decision is based on the entropy we measured at the *end* of
        # the previous iter (kept in stats); for the first iter we leave
        # the configured start value alone.
        cfg_for_update = _ConfigWithEntropy(cfg, c_entropy)
        batch = stack_batch(all_records, device)
        stats = ppo_update(model, optimiser, batch, cfg_for_update, np_rng)
        c_entropy = _adapt_c_entropy(cfg, c_entropy, stats["entropy"])
        dt = time.perf_counter() - t0

        # Per-class entropy line (only show classes that appeared this iter
        # to keep the log readable).
        ent_by_class = stats.get("entropy_by_class", {})
        count_by_class = stats.get("count_by_class", {})
        ent_parts = [
            f"{name}={ent_by_class[name]:.2f}"
            for name in ACTION_CLASS_NAMES
            if count_by_class.get(name, 0) > 0
        ]
        msg = (
            f"[iter {it:4d}] games={cfg.games_per_iter} trans={len(all_records)} "
            f"rollout_wr={rollout_wr:.2f} "
            f"pg={stats['pg_loss']:.3f} v={stats['v_loss']:.3f} "
            f"ent={stats['entropy']:.3f} c_ent={c_entropy:.4f} kl={stats['approx_kl']:.4f} "
            f"aux_oh={stats['aux_opp_hand_loss']:.3f} aux_m={stats['aux_margin_loss']:.3f} "
            f"ent_by_cls=[{' '.join(ent_parts)}] "
            f"stop@ep={stats['stopped_at_epoch']} dt={dt:.1f}s"
        )
        print(msg)

        # Build the per-iter metrics record. Eval fields get filled in below
        # when this iter happens to be a snapshot iter; otherwise they stay null.
        record = {
            "iter": it,
            "games": cfg.games_per_iter,
            "transitions": len(all_records),
            "rollout_wr": rollout_wr,
            "pg_loss": stats["pg_loss"],
            "v_loss": stats["v_loss"],
            "entropy": stats["entropy"],
            # Per-action-type diagnostics — needed to spot mode collapse
            # before it fully sets in (DRAFT entropy crashing toward 0 is
            # the early signal we missed in the 20260523-123850 run).
            "entropy_by_class": stats.get("entropy_by_class", {}),
            "count_by_class": stats.get("count_by_class", {}),
            "aux_opp_hand_loss": stats.get("aux_opp_hand_loss", 0.0),
            "aux_margin_loss": stats.get("aux_margin_loss", 0.0),
            "c_entropy": c_entropy,
            "approx_kl": stats["approx_kl"],
            "stopped_at_epoch": stats["stopped_at_epoch"],
            "dt": dt,
            "pool_size": len(pool),
            "wr_random": None,
            "wr_greedy": None,
            "snapshot_path": None,
            "pool_grew": False,
        }

        if it % cfg.snapshot_every == 0:
            evals = {
                "random": evaluate(model, RandomAgent(seed=11), cfg.eval_games, device,
                                   expansion_prob=cfg.expansion_prob,
                                   main2_prob=cfg.main2_prob, aux2_prob=cfg.aux2_prob),
                "greedy": evaluate(model, GreedyAgent(seed=12), cfg.eval_games, device,
                                   expansion_prob=cfg.expansion_prob,
                                   main2_prob=cfg.main2_prob, aux2_prob=cfg.aux2_prob),
            }
            wr_random = evals["random"]["win_rate"]
            wr_greedy = evals["greedy"]["win_rate"]
            print(
                f"[iter {it:4d}] eval: vs random={wr_random:.2f} "
                f"vs greedy={wr_greedy:.2f}"
            )
            record["wr_random"] = wr_random
            record["wr_greedy"] = wr_greedy
            # Always checkpoint to disk so we have a recoverable history.
            if save_dir is not None:
                ckpt_path = save_dir / f"snapshot_{it:05d}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "iter": it,
                        "wr_random": wr_random,
                        "wr_greedy": wr_greedy,
                    },
                    ckpt_path,
                )
                print(f"[iter {it:4d}] saved {ckpt_path}")
                record["snapshot_path"] = str(ckpt_path)
            if wr_random >= cfg.pool_threshold_wr_random:
                frozen = copy.deepcopy(model).eval()
                # Stochastic — target mixed equilibrium, not argmax-exploitable
                # prior selves. See eval docstring above.
                evicted = pool.add(
                    NNAgent(frozen, device=device, stochastic=True),
                    name=f"iter_{it:05d}",
                )
                record["pool_grew"] = True
                record["pool_size"] = len(pool)
                print(
                    f"[iter {it:4d}] added iter_{it:05d} to pool "
                    f"(wr_random={wr_random:.2f} ≥ {cfg.pool_threshold_wr_random}"
                    f"{f', evicted {evicted}' if evicted else ''})"
                )
                # Periodic PFSP debug — show the per-opponent rolling WR so
                # we can see which historical snapshots are currently
                # giving the trainee trouble.
                rows = pool.summary()
                rows.sort(key=lambda t: t[1])  # worst (lowest WR) first
                top = [f"{n}={wr:.2f}({k})" for n, wr, k in rows[:6]]
                print(f"[iter {it:4d}] PFSP toughest: " + " · ".join(top))
            else:
                print(
                    f"[iter {it:4d}] pool unchanged "
                    f"(wr_random={wr_random:.2f} < {cfg.pool_threshold_wr_random})"
                )

        if metrics_path is not None:
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
    return model
