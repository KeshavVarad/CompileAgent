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
from .agent import NNAgent, StepRecord
from .buffer import Batch, compute_gae, minibatches, stack_batch
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


def play_episode(
    model: PolicyValueNet,
    opponent,
    cfg: TrainConfig,
    rng: random.Random,
    device: torch.device,
) -> tuple[list[StepRecord], int | None, int]:
    """Play one game. Returns (records from agent's seat, winner, agent_seat)."""
    agent_seat = rng.randint(0, 1)
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

    # Terminal credit on the last agent record.
    if records:
        w = game.state.winner
        if w is not None:
            terminal = 1.0 if w == agent_seat else -1.0
        else:
            terminal = 0.0
        records[-1].reward += terminal
        records[-1].done = True
    return records, game.state.winner, agent_seat


def evaluate(
    model: PolicyValueNet, opponent, n_games: int, device: torch.device, *,
    expansion_prob: float = 0.5, main2_prob: float = 0.4, aux2_prob: float = 0.4,
    seed: int = 0,
) -> dict:
    """Inference-mode (argmax) win-rate vs `opponent`. Alternates seats."""
    rng = random.Random(seed)
    agent = NNAgent(model, device=device, stochastic=False)
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
        "approx_kl": 0.0, "n": 0, "stopped_at_epoch": cfg.ppo_epochs,
    }
    adv = batch.advantage
    if adv.numel() > 1:
        batch.advantage = (adv - adv.mean()) / (adv.std() + 1e-8)

    stop_early = False
    for epoch in range(cfg.ppo_epochs):
        for mb in minibatches(batch, cfg.batch_size, rng):
            logits, value = model(
                mb.state, mb.action_raw, mb.action_card_ids, mb.action_proto_ids, mb.action_mask,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            new_logp = log_probs.gather(1, mb.action_idx.unsqueeze(-1)).squeeze(-1)
            log_ratio = new_logp - mb.old_log_prob
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pg_loss = -torch.min(ratio * mb.advantage, clipped * mb.advantage).mean()
            v_loss = F.mse_loss(value, mb.ret)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs * mb.action_mask.float()).sum(dim=-1).mean()
            loss = pg_loss + cfg.c_value * v_loss - cfg.c_entropy * entropy

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()

            with torch.no_grad():
                # K3 estimator: non-negative, lower variance.
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

            stats["pg_loss"] += float(pg_loss.item())
            stats["v_loss"] += float(v_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["approx_kl"] += float(approx_kl)
            stats["n"] += 1

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                stop_early = True
                break
        if stop_early:
            stats["stopped_at_epoch"] = epoch + 1
            break

    if stats["n"]:
        for k in ("pg_loss", "v_loss", "entropy", "approx_kl"):
            stats[k] /= stats["n"]
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

        msg = (
            f"[iter {it:4d}] games={cfg.games_per_iter} trans={len(all_records)} "
            f"rollout_wr={rollout_wr:.2f} "
            f"pg={stats['pg_loss']:.3f} v={stats['v_loss']:.3f} "
            f"ent={stats['entropy']:.3f} c_ent={c_entropy:.4f} kl={stats['approx_kl']:.4f} "
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
                evicted = pool.add(
                    NNAgent(frozen, device=device, stochastic=False),
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
