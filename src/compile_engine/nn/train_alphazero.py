"""AlphaZero-style training loop for Compile.

Compared to `train.py` (PPO), this trainer:

  * Selects every labeler-seat decision via MCTS during self-play. The
    policy is trained on the search-refined visit distribution; the value
    head is trained on the actual game outcome from the labeler's seat.
  * Skips search on cheap states (single-legal, mid-effect, or where the
    policy is already very confident) to bound wall-clock — these states
    play through with policy argmax and are not added to the buffer.
  * Reuses the PPO trainer's opponent pool (PFSP), config sampler, and
    eval helper unchanged. Snapshots land on disk in the same format as
    PPO snapshots so the existing eval + ONNX export pipelines work.

The loop is laptop-tractable on MPS: with the default knobs (16 games/iter,
4 determinizations × 32 sims, 0.85 skip threshold) one iter takes about
2-3 minutes including SGD. Hot-start from a PPO checkpoint so the first
few iters don't waste compute relearning policy basics.

Loss:
    L = MSE(v_pred, outcome) + c_policy * CE(pi_search, pi_pred)
where outcome ∈ {+1, -1, 0} is the *actual* game result from the labeler's
seat (one sample per state), and pi_search is the MCTS visit-count target
from `MCTSAgent.choose_with_target`.
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..agents import GreedyAgent, RandomAgent
from ..game import Game
from .agent import NNAgent
from .encoder import MAX_ACTIONS, encode_actions, encode_state
from .mcts import MCTSAgent, MCTSConfig, _mid_effect, _policy_and_value
from .model import PolicyValueNet

# Reused from the PPO trainer to avoid duplicating PFSP / config sampling
# / eval logic. These are stable enough that pinning to them is fine.
from .train import (
    OpponentPool,
    _make_config,
    _resolve_device,
    evaluate,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AZConfig:
    iters: int = 200
    games_per_iter: int = 16
    # Replay buffer size — number of recent samples to mix into each
    # update. Keeps the policy from forgetting earlier positions, and
    # cushions the noise of per-iter sample variance. ~16 games * ~12
    # labeled states = ~190 samples / iter; 4-iter buffer = ~760 samples.
    buffer_iters: int = 4
    # SGD pass: epochs over the replay buffer per iter.
    sgd_epochs: int = 2
    batch_size: int = 64
    lr: float = 5e-5  # Conservative; we're fine-tuning from a PPO ckpt.
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    c_policy: float = 1.0  # Standard AlphaZero weight on policy CE.
    # MCTS knobs — kept small for compute. Each (dets * sims) leaf
    # evaluation is one NN forward pass. With dets=4 sims=32 batch=8,
    # one searched move is ~0.7s on MPS.
    mcts_dets: int = 4
    mcts_sims: int = 32
    mcts_batch: int = 8
    mcts_c_puct: float = 1.25
    mcts_root_top_k: int = 5
    mcts_root_min_visits: int = 3
    # Gumbel root selection (Danihelka et al. 2022). Replaces UCB + top-k
    # + Dirichlet noise at the root with Gumbel-Top-k + completed-Q
    # selection. The improved-policy target is guaranteed-improving by
    # construction — addresses the "policy collapses onto safe actions"
    # failure mode observed at low sim budgets.
    use_gumbel_root: bool = True
    gumbel_n_candidates: int = 8
    # Skip search when the policy is already confident. 0.85 means we
    # only search when the top action's probability is below 85% — keeps
    # search budget focused on real decisions. Set to 0 to search every
    # non-trivial state.
    skip_top_prob: float = 0.85
    # PFSP pool + opponent sampling (same as PPO trainer).
    max_pool_size: int = 16
    pfsp_p: float = 2.0
    pfsp_min_weight: float = 0.04
    pfsp_window: int = 80
    # Hot-start checkpoint. Strongly recommended — random-init AlphaZero
    # from scratch would need many more iters than a laptop run can
    # afford. Leave None to start from random weights.
    init_ckpt: str | None = None
    # Game-config sampler (same defaults as PPO trainer, plus a draft-
    # subset knob for protocol diversity — see GameConfig.draft_pool_size).
    expansion_prob: float = 0.5
    main2_prob: float = 0.4
    aux2_prob: float = 0.4
    # When set, every training game draws a random `draft_pool_size`
    # protocol subset. 9 mimics the competitive Compile draft format
    # and forces engagement with the ~18 protocols Sparkv2 never picks.
    # Note: pool size must be ≤ 12 to stay valid in the worst case
    # (only MN01 enabled = 12 protocols available); the per-game
    # expansion probs above usually give 12-30 available. We always
    # enable all sets when draft_pool_size is set so the subset is
    # uniformly sampled across the full 30.
    draft_pool_size: int | None = 9
    max_turns: int = 200
    snapshot_every: int = 10
    eval_games: int = 60
    # Pool admission gate. After a snapshot iter, evaluate the new
    # snapshot against the current "best" NN pool member; only add it
    # to the PFSP pool if its WR exceeds this threshold. Prevents the
    # pool from being poisoned by mid-collapse checkpoints (the failure
    # mode we observed in iter-30/40 of the earlier run, where each
    # snapshot dragged the next one further off-trajectory).
    #
    # Set to 0.0 to disable (always add — original behaviour).
    pool_admission_wr: float = 0.55
    pool_admission_games: int = 30  # quick eval; tradeoff variance vs time
    seed: int = 0
    device: str = "mps"
    save_dir: str | None = None


# ---------------------------------------------------------------------------
# Sample buffer
# ---------------------------------------------------------------------------


@dataclass
class _Sample:
    """One labeled state from self-play. Field shapes mirror what the
    model's forward pass expects so we can stack and feed in one go."""
    state: dict[str, np.ndarray]                # state encoding
    action_raw: np.ndarray                      # [MAX_ACTIONS, F_raw]
    action_card_ids: np.ndarray                 # [MAX_ACTIONS]
    action_proto_ids: np.ndarray                # [MAX_ACTIONS]
    action_extra_card_ids: np.ndarray           # [MAX_ACTIONS] — A5 soon-covered
    action_mask: np.ndarray                     # [MAX_ACTIONS] bool
    pi_target: np.ndarray                       # [MAX_ACTIONS], sums to 1 over legal prefix
    outcome: float                              # +1 / -1 / 0 from labeler's POV


def _stack(samples: list[_Sample], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack a list of samples into batched tensors on `device`."""
    state_keys = list(samples[0].state.keys())
    state = {
        k: torch.from_numpy(np.stack([s.state[k] for s in samples])).to(device)
        for k in state_keys
    }
    return {
        "state": state,
        "action_raw": torch.from_numpy(np.stack([s.action_raw for s in samples])).to(device),
        "action_card_ids": torch.from_numpy(np.stack([s.action_card_ids for s in samples])).to(device),
        "action_proto_ids": torch.from_numpy(np.stack([s.action_proto_ids for s in samples])).to(device),
        "action_extra_card_ids": torch.from_numpy(np.stack([s.action_extra_card_ids for s in samples])).to(device),
        "action_mask": torch.from_numpy(np.stack([s.action_mask for s in samples])).to(device).bool(),
        "pi_target": torch.from_numpy(np.stack([s.pi_target for s in samples])).to(device),
        "outcome": torch.from_numpy(
            np.asarray([s.outcome for s in samples], dtype=np.float32)
        ).to(device),
    }


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------


def _best_nn_pool_member(pool: OpponentPool):
    """Return the strongest NN pool member by rolling-window WR (lowest
    trainee WR = toughest opponent = current 'best'). Skips anchors
    (random/greedy) since they're stylistic baselines, not strength
    benchmarks. Returns None if no NN member exists yet."""
    nn_members = [
        m for m in pool.members
        if m.name not in pool._anchor_names
        and isinstance(m.agent, NNAgent)
    ]
    if not nn_members:
        return None
    # Rolling-window WR of the trainee against each member. Lower trainee
    # WR = stronger opponent. If results are empty (fresh member), default
    # to 0.5 (neutral) so freshly-added members still get picked.
    def wr(m):
        if not m.results:
            return 0.5
        return sum(m.results) / len(m.results)
    return min(nn_members, key=wr)


def _build_mcts(model: PolicyValueNet, device: torch.device, cfg: AZConfig, seed: int) -> MCTSAgent:
    return MCTSAgent(
        model=model,
        device=device,
        cfg=MCTSConfig(
            n_determinizations=cfg.mcts_dets,
            sims_per_determinization=cfg.mcts_sims,
            c_puct=cfg.mcts_c_puct,
            # Gumbel root replaces root_top_k + root_min_visits + Dirichlet
            # noise. When enabled, those knobs are ignored at the root.
            use_gumbel_root=cfg.use_gumbel_root,
            gumbel_n_candidates=cfg.gumbel_n_candidates,
            root_top_k=0 if cfg.use_gumbel_root else cfg.mcts_root_top_k,
            root_min_visits_per_action=0 if cfg.use_gumbel_root else cfg.mcts_root_min_visits,
            batch_size=cfg.mcts_batch,
            # Skip-when-confident is handled in our outer loop; we don't
            # also want MCTSAgent doing it internally because we want
            # control over which states are LABELED.
            skip_search_top_prob=0.0,
        ),
        seed=seed,
    )


def play_self_game(
    model: PolicyValueNet,
    opponent,
    cfg: AZConfig,
    rng: random.Random,
    device: torch.device,
    mcts: MCTSAgent,
) -> tuple[list[_Sample], int | None, int, dict]:
    """Play one game and return (samples, winner, trainee_seat, stats).

    `samples` contains one entry per *searched* labeler decision. Opponent
    decisions, forced/mid-effect states, and skip-when-confident states
    do not produce samples — they play through but don't train. The
    outcome from the trainee's POV (+1/-1/0) is stamped onto every
    sample in the trajectory after the game ends.
    """
    trainee_seat = rng.randint(0, 1)
    opp_seat = 1 - trainee_seat
    seat_agents = [None, None]
    seat_agents[trainee_seat] = NNAgent(model, device=device, stochastic=True)
    seat_agents[opp_seat] = opponent

    game_cfg = _make_config(rng, _make_train_config_proxy(cfg))
    if cfg.draft_pool_size is not None:
        # _make_config doesn't know about draft_pool_size; patch it in.
        game_cfg.draft_pool_size = cfg.draft_pool_size
    game = Game(game_cfg)
    game.start()

    pending: list[_Sample] = []  # outcome stamped after game ends
    n_searched = 0
    n_skipped = 0
    n_opp = 0
    while not game.is_over():
        who = game.decider()
        legal = game.legal_actions()
        if not legal:
            break

        if who != trainee_seat:
            # Opponent move — no search, no sample.
            action = seat_agents[who].choose(game, legal)
            game.step(action)
            n_opp += 1
            continue

        # Trainee's move. Three classes:
        if len(legal) == 1 or _mid_effect(game):
            # No search possible (single action or mid-effect generator).
            action = seat_agents[who].choose(game, legal)
            game.step(action)
            continue

        # Peek at policy confidence; skip search if too confident.
        probs, _ = _policy_and_value(model, game, legal, device)
        n_p = min(len(legal), len(probs))
        top_prob = float(probs[:n_p].max())
        if cfg.skip_top_prob > 0 and top_prob >= cfg.skip_top_prob:
            n_skipped += 1
            action = legal[int(np.argmax(probs[:n_p]))]
            game.step(action)
            continue

        # Search-labeled decision. Sample the action from the target
        # distribution (mixed strategy) — Compile is an imperfect-info
        # stochastic game, so the optimal policy is generally a mixed
        # Nash equilibrium. Deterministic argmax play is exploitable
        # and starves training data of trajectory diversity. The
        # *label* (target distribution) is still the Gumbel-improved
        # policy; only the action we step the game with is sampled.
        action, target_over_legal = mcts.choose_with_target(
            game, legal, tau=1.0, return_value=False,
            sample_action=True, sample_temperature=1.0,
        )
        n_searched += 1

        # Snapshot the state + actions for training. Encode against the
        # trainee's perspective (the engine's decider, which is `who`).
        state = encode_state(game, who)
        raw, card_ids, proto_ids, extra_card_ids, mask = encode_actions(game, legal, who)

        pi_padded = np.zeros(MAX_ACTIONS, dtype=np.float32)
        n_t = min(MAX_ACTIONS, len(target_over_legal))
        pi_padded[:n_t] = target_over_legal[:n_t]
        s = pi_padded.sum()
        if s > 0:
            pi_padded /= s

        pending.append(_Sample(
            state=state,
            action_raw=raw,
            action_card_ids=card_ids,
            action_proto_ids=proto_ids,
            action_extra_card_ids=extra_card_ids,
            action_mask=mask,
            pi_target=pi_padded,
            outcome=0.0,  # filled in below
        ))
        game.step(action)

    # Stamp outcome into all pending samples. +1 for trainee win,
    # -1 for loss, 0 for draw/timeout.
    winner = game.state.winner
    if winner is None:
        z = 0.0
    elif winner == trainee_seat:
        z = 1.0
    else:
        z = -1.0
    for sample in pending:
        sample.outcome = z

    stats = {
        "searched": n_searched,
        "skipped": n_skipped,
        "opp_moves": n_opp,
        "turns": game.state.turn,
    }
    return pending, winner, trainee_seat, stats


# Tiny shim so we can reuse `_make_config(rng, TrainConfig)`. The PPO
# trainer's helper only reads four attrs; we add draft_pool_size on top
# so AZ games inherit the diversity knob.
class _make_train_config_proxy:
    __slots__ = ("expansion_prob", "main2_prob", "aux2_prob", "max_turns")
    def __init__(self, cfg: AZConfig):
        # When subset draft is on, force all expansion sets enabled so
        # the subset is sampled from the full 30 protocols (otherwise
        # the pool would sometimes only have 12 to pick from, which is
        # close to draft_pool_size=9 and largely defeats the point).
        force_all = cfg.draft_pool_size is not None
        self.expansion_prob = 1.0 if force_all else cfg.expansion_prob
        self.main2_prob = 1.0 if force_all else cfg.main2_prob
        self.aux2_prob = 1.0 if force_all else cfg.aux2_prob
        self.max_turns = cfg.max_turns


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def az_update(
    model: PolicyValueNet,
    optimiser: torch.optim.Optimizer,
    samples: list[_Sample],
    cfg: AZConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict:
    """Run `cfg.sgd_epochs` passes of minibatched joint CE + MSE over `samples`."""
    if not samples:
        return {"pol_loss": 0.0, "val_loss": 0.0, "n": 0, "kl_to_target": 0.0}

    batch = _stack(samples, device)
    n = len(samples)
    indices = np.arange(n)

    sum_pol = 0.0
    sum_val = 0.0
    sum_kl = 0.0
    n_batches = 0

    for _epoch in range(cfg.sgd_epochs):
        rng.shuffle(indices)
        for start in range(0, n, cfg.batch_size):
            idx = indices[start : start + cfg.batch_size]
            idx_t = torch.from_numpy(idx).to(device)

            sb = {k: v[idx_t] for k, v in batch["state"].items()}
            logits, value = model(
                sb,
                batch["action_raw"][idx_t],
                batch["action_card_ids"][idx_t],
                batch["action_proto_ids"][idx_t],
                batch["action_extra_card_ids"][idx_t],
                batch["action_mask"][idx_t],
            )

            log_probs = F.log_softmax(logits, dim=-1)
            pi_t = batch["pi_target"][idx_t]
            mask = batch["action_mask"][idx_t]
            ce_per_slot = torch.where(mask, pi_t * log_probs, torch.zeros_like(pi_t))
            pol_loss = -ce_per_slot.sum(dim=-1).mean()

            z = batch["outcome"][idx_t]
            val_loss = F.mse_loss(value.squeeze(-1), z)

            loss = cfg.c_policy * pol_loss + val_loss

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()

            with torch.no_grad():
                kl = (pi_t * (torch.log(pi_t + 1e-12) - log_probs)).where(
                    mask, torch.zeros_like(pi_t)
                ).sum(dim=-1).mean()

            sum_pol += float(pol_loss.item())
            sum_val += float(val_loss.item())
            sum_kl += float(kl.item())
            n_batches += 1

    return {
        "pol_loss": sum_pol / max(1, n_batches),
        "val_loss": sum_val / max(1, n_batches),
        "kl_to_target": sum_kl / max(1, n_batches),
        "n": n_batches,
    }


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------


def train(cfg: AZConfig | None = None) -> PolicyValueNet:
    cfg = cfg or AZConfig()
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    model = PolicyValueNet().to(device)
    if cfg.init_ckpt:
        state = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"Hot-started from {cfg.init_ckpt}")
    model.train()

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    pool = OpponentPool(
        max_size=cfg.max_pool_size, p=cfg.pfsp_p,
        min_weight=cfg.pfsp_min_weight, window=cfg.pfsp_window,
    )
    pool.add(RandomAgent(seed=1), "random", is_anchor=True)
    pool.add(GreedyAgent(seed=2), "greedy", is_anchor=True)
    # If hot-starting, seed the pool with a frozen copy of the source so
    # PFSP has something stronger than Random/Greedy to weight against.
    if cfg.init_ckpt:
        frozen = copy.deepcopy(model).eval()
        pool.add(
            NNAgent(frozen, device=device, stochastic=True),
            name=Path(cfg.init_ckpt).stem,
        )

    save_dir = Path(cfg.save_dir) if cfg.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl" if save_dir is not None else None
    if metrics_path is not None and metrics_path.exists():
        metrics_path.unlink()

    # Sliding replay buffer: deque of per-iter sample lists.
    buf: deque[list[_Sample]] = deque(maxlen=cfg.buffer_iters)

    mcts = _build_mcts(model, device, cfg, seed=cfg.seed)

    for it in range(1, cfg.iters + 1):
        t0 = time.perf_counter()
        # The MCTS holds a reference to `model` so re-use across iters is
        # fine. Just bump its RNG seed so root-noise is fresh per iter.
        mcts.rng = np.random.default_rng(cfg.seed * 1000 + it)

        iter_samples: list[_Sample] = []
        n_wins = 0
        n_games = 0
        agg_stats = {"searched": 0, "skipped": 0, "opp_moves": 0, "turns": 0}
        for _ in range(cfg.games_per_iter):
            member = pool.sample(rng)
            samples, winner, trainee_seat, stats = play_self_game(
                model, member.agent, cfg, rng, device, mcts,
            )
            iter_samples.extend(samples)
            trainee_won = winner == trainee_seat
            pool.record(member, trainee_won)
            n_games += 1
            if trainee_won:
                n_wins += 1
            for k in agg_stats:
                agg_stats[k] += stats[k]

        buf.append(iter_samples)
        # Concatenate the whole buffer for the SGD pass.
        training_set: list[_Sample] = []
        for s_list in buf:
            training_set.extend(s_list)

        stats = az_update(model, optimiser, training_set, cfg, device, np_rng)
        dt = time.perf_counter() - t0

        rollout_wr = n_wins / max(1, n_games)
        print(
            f"[az iter {it:4d}] games={n_games} samples={len(iter_samples)}/"
            f"{len(training_set)}buf rollout_wr={rollout_wr:.2f} "
            f"pol={stats['pol_loss']:.3f} val={stats['val_loss']:.3f} "
            f"kl_to_target={stats['kl_to_target']:.3f} "
            f"searched={agg_stats['searched']} skipped={agg_stats['skipped']} "
            f"dt={dt:.1f}s"
        )

        record = {
            "iter": it,
            "games": n_games,
            "iter_samples": len(iter_samples),
            "buffer_samples": len(training_set),
            "rollout_wr": rollout_wr,
            "pol_loss": stats["pol_loss"],
            "val_loss": stats["val_loss"],
            "kl_to_target": stats["kl_to_target"],
            "searched": agg_stats["searched"],
            "skipped": agg_stats["skipped"],
            "opp_moves": agg_stats["opp_moves"],
            "avg_turns": agg_stats["turns"] / max(1, n_games),
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
            print(f"[az iter {it:4d}] eval: vs random={wr_random:.2f} vs greedy={wr_greedy:.2f}")
            record["wr_random"] = wr_random
            record["wr_greedy"] = wr_greedy

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
                print(f"[az iter {it:4d}] saved {ckpt_path}")
                record["snapshot_path"] = str(ckpt_path)

            # Pool admission gate: only add if the new snapshot beats the
            # current best NN pool member at >= cfg.pool_admission_wr. The
            # PFSP weighting + un-gated admission used to be enough on
            # its own, but our earlier run showed that mid-collapse
            # snapshots can poison the pool — each new addition pushes
            # the model further off-trajectory because PFSP weights
            # toward the most-recent (broken) opponent.
            frozen = copy.deepcopy(model).eval()
            candidate_agent = NNAgent(frozen, device=device, stochastic=True)
            best_member = _best_nn_pool_member(pool)
            admit = True
            if cfg.pool_admission_wr > 0 and best_member is not None:
                # Quick eval against the strongest current pool member.
                wr_vs_best = evaluate(
                    model, best_member.agent, cfg.pool_admission_games, device,
                    expansion_prob=cfg.expansion_prob,
                    main2_prob=cfg.main2_prob, aux2_prob=cfg.aux2_prob,
                    seed=cfg.seed + 7919 * it,
                )["win_rate"]
                admit = wr_vs_best >= cfg.pool_admission_wr
                record["pool_admission_wr_vs_best"] = wr_vs_best
                record["pool_admission_best_name"] = best_member.name
                print(
                    f"[az iter {it:4d}] vs best ({best_member.name}): "
                    f"WR={wr_vs_best:.2f} "
                    f"({'ADMIT' if admit else 'REJECT'} at threshold {cfg.pool_admission_wr:.2f})"
                )
            if admit:
                evicted = pool.add(
                    candidate_agent,
                    name=f"az_iter_{it:05d}",
                )
                record["pool_grew"] = True
                record["pool_size"] = len(pool)
                print(
                    f"[az iter {it:4d}] added az_iter_{it:05d} to pool"
                    f"{f' (evicted {evicted})' if evicted else ''}"
                )
            else:
                record["pool_grew"] = False
                print(f"[az iter {it:4d}] rejected az_iter_{it:05d} — not added to pool")

        if metrics_path is not None:
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="AlphaZero-style trainer for Compile")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games-per-iter", type=int, default=16)
    ap.add_argument("--buffer-iters", type=int, default=4)
    ap.add_argument("--sgd-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--c-policy", type=float, default=1.0)
    ap.add_argument("--mcts-dets", type=int, default=4)
    ap.add_argument("--mcts-sims", type=int, default=32)
    ap.add_argument("--mcts-batch", type=int, default=8)
    ap.add_argument("--skip-top-prob", type=float, default=0.85)
    ap.add_argument("--use-gumbel-root", type=int, default=1,
                    help="1 = Gumbel root selection (default), 0 = vanilla UCB+top-k")
    ap.add_argument("--gumbel-n-candidates", type=int, default=8,
                    help="top-m root actions kept under Gumbel selection")
    ap.add_argument("--pool-admission-wr", type=float, default=0.55,
                    help="only add a snapshot to the PFSP pool if it beats the "
                         "current best NN member at this WR. 0 = always admit.")
    ap.add_argument("--pool-admission-games", type=int, default=30)
    ap.add_argument("--draft-pool-size", type=int, default=9,
                    help="size of the per-game protocol subset (None = full enabled pool). "
                         "9 matches the competitive Compile draft and forces protocol "
                         "diversity. Set to 0 to disable (use full pool).")
    ap.add_argument("--init-ckpt", default=None,
                    help="hot-start from this PPO checkpoint (recommended)")
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    cfg = AZConfig(
        iters=args.iters,
        games_per_iter=args.games_per_iter,
        buffer_iters=args.buffer_iters,
        sgd_epochs=args.sgd_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        c_policy=args.c_policy,
        mcts_dets=args.mcts_dets,
        mcts_sims=args.mcts_sims,
        mcts_batch=args.mcts_batch,
        skip_top_prob=args.skip_top_prob,
        use_gumbel_root=bool(args.use_gumbel_root),
        gumbel_n_candidates=args.gumbel_n_candidates,
        pool_admission_wr=args.pool_admission_wr,
        pool_admission_games=args.pool_admission_games,
        draft_pool_size=(None if args.draft_pool_size <= 0 else args.draft_pool_size),
        init_ckpt=args.init_ckpt,
        snapshot_every=args.snapshot_every,
        eval_games=args.eval_games,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
    )
    train(cfg)


if __name__ == "__main__":
    _cli()
