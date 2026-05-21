"""MCTS diagnostic: does search beat policy argmax, and where do they
disagree?

Runs three head-to-heads against the same opponent (default: Greedy):
  1. MCTS vs opp        — measures search win-rate
  2. Policy-argmax vs opp  — measures policy-only win-rate (baseline)
  3. At every MCTS move, we also compute what policy argmax would have
     picked. Log the agreement rate and the rank (under the policy) of
     the action MCTS actually chose.

Read the output as a decision tree:
  - High agreement + WR ≈ baseline  → search is mostly a no-op; not worth
    the compute. Either drop the agent or fix determinization so it
    explores meaningfully different lines.
  - Low agreement + WR ≈ baseline   → search is shuffling between
    near-equivalent moves; noise. Lower c_puct or increase sims.
  - Low agreement + WR > baseline   → search is doing real work; invest
    in tree reuse / better priors.
  - Low agreement + WR < baseline   → search is making bad moves. Look
    at the rank distribution: if MCTS is consistently picking the
    policy's 5th+ choice, the value head is misleading the search.

Usage:
    python scripts/eval/mcts_diagnostic.py \\
        --ckpt runs/20260520-224058/snapshot_00330.pt \\
        --opponent greedy \\
        --games 20 \\
        --dets 4 --sims 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from compile_engine import Game, GameConfig  # noqa: E402
from compile_engine.actions import Action  # noqa: E402
from compile_engine.cards import load_card_defs  # noqa: E402
from compile_engine.nn.agent import NNAgent  # noqa: E402
from compile_engine.nn.mcts import (  # noqa: E402
    MCTSAgent,
    MCTSConfig,
    _mid_effect,
    _policy_and_value,
)

from _lib import (  # noqa: E402
    DEFAULT_MAX_TURNS,
    OpponentSpec,
    build_agent,
    load_model_from_ckpt,
    resolve_device,
)


@dataclass
class MoveDiag:
    """One MCTS top-level decision, with policy comparison."""
    turn: int
    n_legal: int
    chose_idx: int           # index into legal of the MCTS pick
    policy_argmax_idx: int   # what policy would have picked
    mcts_rank_under_policy: int  # 1 = MCTS picked the same as policy argmax
    chose_prob: float        # policy prob mass on the MCTS pick
    top_prob: float          # policy prob mass on its argmax action


class InstrumentedMCTSAgent:
    """Wraps MCTSAgent so we can log policy-disagreement at every search.
    Mid-effect (CHOOSE_TARGET) decisions and forced 1-legal-action steps
    are *not* logged — they don't run search."""

    def __init__(self, mcts: MCTSAgent) -> None:
        self.mcts = mcts
        self.moves: list[MoveDiag] = []

    def choose(self, game: Game, legal: list[Action]) -> Action:
        if len(legal) <= 1 or _mid_effect(game):
            return self.mcts.choose(game, legal)
        probs, _ = _policy_and_value(self.mcts.model, game, legal, self.mcts.device)
        n = min(len(legal), len(probs))
        policy_argmax = int(np.argmax(probs[:n]))
        # rank of every action under the policy (best=1).
        order = np.argsort(-probs[:n])
        rank_of = {int(i): r + 1 for r, i in enumerate(order)}
        # Now run search and record what it picked.
        chosen = self.mcts.choose(game, legal)
        chose_idx = legal.index(chosen)
        self.moves.append(
            MoveDiag(
                turn=game.state.turn,
                n_legal=len(legal),
                chose_idx=chose_idx,
                policy_argmax_idx=policy_argmax,
                mcts_rank_under_policy=rank_of.get(chose_idx, len(legal)),
                chose_prob=float(probs[chose_idx]) if chose_idx < n else 0.0,
                top_prob=float(probs[policy_argmax]),
            )
        )
        return chosen


@dataclass
class HeadToHead:
    label: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total: int = 0
    seconds: float = 0.0

    def add(self, primary_won: bool | None, dt: float) -> None:
        self.total += 1
        self.seconds += dt
        if primary_won is True:
            self.wins += 1
        elif primary_won is False:
            self.losses += 1
        else:
            self.draws += 1

    def wr(self) -> float:
        return self.wins / max(1, self.total)


def _run_game(
    primary,
    opponent,
    primary_seat: int,
    cfg: GameConfig,
    defs,
) -> tuple[bool | None, int, float]:
    """Run one game with `primary` in `primary_seat`. Returns
    (primary_won, turns, seconds)."""
    game = Game(cfg, defs=defs)
    game.start()
    agents = [None, None]
    agents[primary_seat] = primary
    agents[1 - primary_seat] = opponent
    t0 = time.perf_counter()
    while not game.is_over():
        who = game.decider()
        legal = game.legal_actions()
        if not legal:
            break
        action = agents[who].choose(game, legal)
        game.step(action)
    dt = time.perf_counter() - t0
    w = game.state.winner
    if w is None:
        return None, game.state.turn, dt
    return (w == primary_seat), game.state.turn, dt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to .pt snapshot")
    p.add_argument("--opponent", default="greedy", help="random|greedy|<path.pt>")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--dets", type=int, default=4, help="determinizations per move")
    p.add_argument("--sims", type=int, default=25, help="sims per determinization")
    p.add_argument("--c-puct", type=float, default=1.25)
    p.add_argument("--dirichlet-eps", type=float, default=0.0,
                   help="root-noise mixing weight (0=disabled, AZ used 0.25)")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3,
                   help="Dirichlet concentration (lower=more peaked noise)")
    p.add_argument("--root-top-k", type=int, default=0,
                   help="prune root to top-k actions by prior + flatten priors (0=disabled)")
    p.add_argument("--root-min-visits", type=int, default=0,
                   help="round-robin floor: each root action gets ≥N sims before PUCT")
    p.add_argument("--batch-size", type=int, default=1,
                   help="leaf batch size for forward passes (1=disabled)")
    p.add_argument("--skip-top-prob", type=float, default=0.0,
                   help="bypass search when policy top_prob >= threshold (0=disabled)")
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None, help="optional: write summary JSON")
    args = p.parse_args()

    device = resolve_device(args.device)
    model = load_model_from_ckpt(args.ckpt, device)
    opp_spec = OpponentSpec.parse(args.opponent)
    defs = load_card_defs()

    # Both primaries use the same model — only choose() policy differs.
    nn = NNAgent(model, device=device, stochastic=False)
    mcts_cfg = MCTSConfig(
        n_determinizations=args.dets,
        sims_per_determinization=args.sims,
        c_puct=args.c_puct,
        dirichlet_eps=args.dirichlet_eps,
        dirichlet_alpha=args.dirichlet_alpha,
        root_top_k=args.root_top_k,
        root_min_visits_per_action=args.root_min_visits,
        batch_size=args.batch_size,
        skip_search_top_prob=args.skip_top_prob,
    )
    mcts_inner = MCTSAgent(model=model, device=device, cfg=mcts_cfg, seed=args.seed)
    mcts = InstrumentedMCTSAgent(mcts_inner)

    h2h_mcts = HeadToHead("MCTS vs " + opp_spec.name)
    h2h_nn = HeadToHead("Policy-argmax vs " + opp_spec.name)

    # Match seeds and seats across both head-to-heads for fair comparison.
    base_seed = args.seed
    for i in range(args.games):
        game_seed = base_seed + i
        primary_seat = i % 2  # alternate seats
        cfg = GameConfig(
            include_expansion=True,
            include_main2=True,
            include_aux2=True,
            seed=game_seed,
            max_turns=args.max_turns,
        )
        opp = build_agent(opp_spec, device, seed=game_seed + 10_000)
        won, turns, dt = _run_game(mcts, opp, primary_seat, cfg, defs)
        h2h_mcts.add(won, dt)

        # Re-run with policy-argmax as primary on the SAME config + seat.
        opp2 = build_agent(opp_spec, device, seed=game_seed + 10_000)
        cfg2 = GameConfig(
            include_expansion=cfg.include_expansion,
            include_main2=cfg.include_main2,
            include_aux2=cfg.include_aux2,
            seed=cfg.seed,
            max_turns=cfg.max_turns,
        )
        won_nn, turns_nn, dt_nn = _run_game(nn, opp2, primary_seat, cfg2, defs)
        h2h_nn.add(won_nn, dt_nn)
        print(
            f"  game {i:2d}  seat={primary_seat}  seed={game_seed:8d}  "
            f"MCTS={'W' if won is True else 'L' if won is False else '-'} ({dt:5.1f}s, {turns}t)  "
            f"NN={'W' if won_nn is True else 'L' if won_nn is False else '-'} ({dt_nn:5.1f}s, {turns_nn}t)"
        )

    # Agreement stats from the instrumented MCTS.
    n_moves = len(mcts.moves)
    if n_moves == 0:
        print("\n[warning] MCTS made zero searched decisions across all games.")
    n_agree = sum(1 for m in mcts.moves if m.mcts_rank_under_policy == 1)
    rank_counts = Counter(m.mcts_rank_under_policy for m in mcts.moves)
    avg_legal = sum(m.n_legal for m in mcts.moves) / max(1, n_moves)
    avg_chose_prob = sum(m.chose_prob for m in mcts.moves) / max(1, n_moves)

    print()
    print("=" * 70)
    print(f"Model:    {args.ckpt}")
    print(f"Opponent: {opp_spec.name}")
    print(
        f"Settings: dets={args.dets} sims/det={args.sims} games={args.games} "
        f"c_puct={args.c_puct} dir_eps={args.dirichlet_eps} dir_alpha={args.dirichlet_alpha} "
        f"top_k={args.root_top_k} min_visits={args.root_min_visits} "
        f"batch={args.batch_size} skip_top_prob={args.skip_top_prob}"
    )
    print("-" * 70)
    print(f"  {h2h_mcts.label:36s} {h2h_mcts.wins:>2d}/{h2h_mcts.total} = {h2h_mcts.wr():.2%}  ({h2h_mcts.seconds/max(1,h2h_mcts.total):.2f}s/game)")
    print(f"  {h2h_nn.label:36s} {h2h_nn.wins:>2d}/{h2h_nn.total} = {h2h_nn.wr():.2%}  ({h2h_nn.seconds/max(1,h2h_nn.total):.2f}s/game)")
    print(f"  delta WR (MCTS - policy):            {(h2h_mcts.wr() - h2h_nn.wr()):+.2%}")
    print()
    print(f"  searched decisions:      {n_moves}")
    print(f"  avg legal actions/move:  {avg_legal:.1f}")
    print(f"  agreement with policy:   {n_agree}/{n_moves} = {n_agree/max(1,n_moves):.2%}")
    print(f"  avg policy-prob on MCTS pick: {avg_chose_prob:.3f}")
    print(f"  rank-under-policy histogram:")
    for rank in sorted(rank_counts):
        bar = "█" * int(40 * rank_counts[rank] / max(1, n_moves))
        print(f"    rank {rank:>2d}: {rank_counts[rank]:>4d}  {bar}")
    # Confidence buckets: how often does the policy already commit hard?
    # Decisions with top_prob > 0.9 are "obvious" to the policy — MCTS
    # can't reasonably disagree there. Decisions with top_prob < 0.5 are
    # where search would actually have room to differ.
    buckets = {"≥0.99": 0, "0.90–0.99": 0, "0.50–0.90": 0, "0.20–0.50": 0, "<0.20": 0}
    disagree_by_bucket: Counter[str] = Counter()
    for m in mcts.moves:
        if m.top_prob >= 0.99:
            b = "≥0.99"
        elif m.top_prob >= 0.90:
            b = "0.90–0.99"
        elif m.top_prob >= 0.50:
            b = "0.50–0.90"
        elif m.top_prob >= 0.20:
            b = "0.20–0.50"
        else:
            b = "<0.20"
        buckets[b] = buckets.get(b, 0) + 1
        if m.mcts_rank_under_policy != 1:
            disagree_by_bucket[b] += 1
    print()
    print("  policy-confidence buckets (top_prob on argmax action):")
    for b, n in buckets.items():
        dis = disagree_by_bucket.get(b, 0)
        bar = "█" * int(40 * n / max(1, n_moves))
        print(f"    top_prob {b:>9}: {n:>4d} moves, {dis} disagreements  {bar}")
    print("=" * 70)

    summary = {
        "ckpt": args.ckpt,
        "opponent": opp_spec.name,
        "games": args.games,
        "dets": args.dets,
        "sims": args.sims,
        "c_puct": args.c_puct,
        "mcts": {
            "wins": h2h_mcts.wins,
            "total": h2h_mcts.total,
            "win_rate": h2h_mcts.wr(),
            "avg_seconds": h2h_mcts.seconds / max(1, h2h_mcts.total),
        },
        "policy_argmax": {
            "wins": h2h_nn.wins,
            "total": h2h_nn.total,
            "win_rate": h2h_nn.wr(),
            "avg_seconds": h2h_nn.seconds / max(1, h2h_nn.total),
        },
        "agreement": {
            "searched_decisions": n_moves,
            "n_agree_with_policy": n_agree,
            "agreement_rate": n_agree / max(1, n_moves),
            "avg_legal_actions": avg_legal,
            "avg_policy_prob_on_pick": avg_chose_prob,
            "rank_histogram": dict(rank_counts),
        },
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
