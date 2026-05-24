"""Counterfactual draft analysis: measure WR when the NN agent is BLOCKED
from drafting specific protocols.

If WR drops a lot when a protocol is blocked, that protocol was load-bearing
for the agent's win condition. If WR is roughly stable, the protocol was a
selection bias (model picks it because it's slightly above average, but can
win without it).

Specifically targets the question raised by docs/STRATEGY_THESIS_sparkv4.md:
*why does Spark v4 draft Darkness 98% of the time, and is it a bug?*

Usage:
    python scripts/eval/counterfactual_draft.py \
        --model runs/.../snapshot_00040.pt \
        --opp greedy --games 200 \
        --block Darkness Plague Love \
        --out /tmp/counterfactual.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from compile_engine.actions import Action, ActionType  # noqa: E402
from compile_engine.cards import load_card_defs  # noqa: E402
from compile_engine.game import Game  # noqa: E402
from compile_engine.nn.agent import NNAgent  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    DEFAULT_AUX2_PROB, DEFAULT_EXPANSION_PROB, DEFAULT_MAIN2_PROB,
    OpponentSpec, build_agent, load_model_from_ckpt, make_game_config, resolve_device,
)


class BlockDraftWrapper:
    """Wraps an Agent so that DRAFT_PROTOCOL actions whose `protocol` is in
    `blocked` are filtered out of legal_actions before the inner agent
    chooses. The inner agent's policy renormalises over what's left.

    For non-DRAFT phases this is a pass-through.
    """

    def __init__(self, inner, blocked: set[str]) -> None:
        self.inner = inner
        self.blocked = blocked

    def choose(self, game, legal: list[Action]) -> Action:
        if legal and legal[0].type is ActionType.DRAFT_PROTOCOL:
            filtered = [a for a in legal if a.protocol not in self.blocked]
            if filtered:  # never starve the agent of all options
                legal = filtered
        return self.inner.choose(game, legal)


def play_one(*, agent, opp, agent_seat: int, rng, defs, cfg):
    agents = (agent, opp) if agent_seat == 0 else (opp, agent)
    g = Game(cfg, defs=defs)
    g.start()
    drafted_for_agent: list[str] = []
    while not g.is_over():
        who = g.decider()
        legal = g.legal_actions()
        if not legal:
            break
        a = agents[who].choose(g, legal)
        if (who == agent_seat
                and a.type is ActionType.DRAFT_PROTOCOL
                and a.protocol):
            drafted_for_agent.append(a.protocol)
        g.step(a)
    st = g.state
    return {
        "winner": st.winner,
        "agent_seat": agent_seat,
        "turns": st.turn,
        "agent_protocols": drafted_for_agent,
    }


def run_condition(label, *, agent, opp, n_games, seed, defs, ep, mp, ap):
    """Run n_games of agent vs opp under one condition. Returns (wr, list)."""
    rng = random.Random(seed)
    games = []
    t0 = time.perf_counter()
    for i in range(n_games):
        cfg = make_game_config(
            rng, expansion_prob=ep, main2_prob=mp, aux2_prob=ap,
            max_turns=200,
        )
        seat = i % 2
        games.append(play_one(
            agent=agent, opp=opp, agent_seat=seat, rng=rng, defs=defs, cfg=cfg,
        ))
    wins = sum(1 for g in games if g["winner"] == g["agent_seat"])
    wr = wins / len(games)
    print(f"  [{label:<25}] n={n_games:<4} WR={wr:.3f}  ({time.perf_counter()-t0:.1f}s)")
    return wr, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--opp", default="greedy")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--block", nargs="+", default=["Darkness", "Plague", "Love"],
                    help="protocols to block (one condition per protocol, "
                         "plus baseline + block-all)")
    ap.add_argument("--out", default="/tmp/counterfactual.json")
    ap.add_argument("--expansion-prob", type=float, default=DEFAULT_EXPANSION_PROB)
    ap.add_argument("--main2-prob", type=float, default=DEFAULT_MAIN2_PROB)
    ap.add_argument("--aux2-prob", type=float, default=DEFAULT_AUX2_PROB)
    args = ap.parse_args()

    device = resolve_device(args.device)
    defs = load_card_defs()
    model = load_model_from_ckpt(args.model, device)

    # The base NN agent. Stochastic = True (matches eval / training setup).
    base_agent = NNAgent(model, device=device, stochastic=True)
    opp_spec = OpponentSpec.parse(args.opp)
    opp_agent = build_agent(opp_spec, device, seed=args.seed + 1, stochastic=True)

    results = {}

    # 1) Baseline: nothing blocked.
    wr, games = run_condition("baseline", agent=base_agent, opp=opp_agent,
                              n_games=args.games, seed=args.seed,
                              defs=defs,
                              ep=args.expansion_prob, mp=args.main2_prob, ap=args.aux2_prob)
    results["baseline"] = {"wr": wr, "blocked": [], "n": args.games}

    # 2) One condition per blocked protocol.
    for proto in args.block:
        wrapped = BlockDraftWrapper(base_agent, blocked={proto})
        wr, _ = run_condition(f"block_{proto}",
                              agent=wrapped, opp=opp_agent,
                              n_games=args.games, seed=args.seed,
                              defs=defs,
                              ep=args.expansion_prob, mp=args.main2_prob, ap=args.aux2_prob)
        results[f"block_{proto}"] = {"wr": wr, "blocked": [proto], "n": args.games}

    # 3) Block ALL three top protocols simultaneously (strongest counterfactual).
    if len(args.block) >= 2:
        wrapped = BlockDraftWrapper(base_agent, blocked=set(args.block))
        wr, _ = run_condition(f"block_all", agent=wrapped, opp=opp_agent,
                              n_games=args.games, seed=args.seed,
                              defs=defs,
                              ep=args.expansion_prob, mp=args.main2_prob, ap=args.aux2_prob)
        results["block_all"] = {"wr": wr, "blocked": list(args.block), "n": args.games}

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")

    # Print a comparison table.
    base = results["baseline"]["wr"]
    print(f"\n{'condition':<30} {'WR':>6} {'delta':>8}")
    print("-" * 48)
    print(f"{'baseline':<30} {base:>6.3f} {'(ref)':>8}")
    for k, v in results.items():
        if k == "baseline": continue
        d = v["wr"] - base
        sig = "***" if abs(d) > 0.10 else "**" if abs(d) > 0.05 else ""
        print(f"{k:<30} {v['wr']:>6.3f} {d:>+8.3f} {sig}")


if __name__ == "__main__":
    main()
