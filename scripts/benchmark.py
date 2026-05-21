"""Throughput benchmark for the Compile engine.

Measures finished games per second and average decisions per game, with the
random policy across both expansion-on and expansion-off configurations.
This is the headline number to track as we optimize the engine.

Usage:
    python scripts/benchmark.py [--games N] [--include-expansion]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from compile_engine import GameConfig
from compile_engine.agents import RandomAgent
from compile_engine.cards import load_card_defs
from compile_engine.env import play_game


def run(n_games: int, include_expansion: bool, seed: int = 0) -> dict:
    defs = load_card_defs()
    turns = []
    winners = {0: 0, 1: 0, None: 0}
    t0 = time.perf_counter()
    for i in range(n_games):
        cfg = GameConfig(include_expansion=include_expansion, seed=seed + i, max_turns=300)
        g = play_game(
            agent0=RandomAgent(seed=seed + i),
            agent1=RandomAgent(seed=seed + i + 100_000),
            config=cfg,
            card_defs=defs,
        )
        turns.append(g.state.turn)
        winners[g.state.winner] = winners.get(g.state.winner, 0) + 1
    elapsed = time.perf_counter() - t0
    return {
        "games": n_games,
        "elapsed_sec": elapsed,
        "games_per_sec": n_games / elapsed,
        "mean_turns": statistics.mean(turns),
        "median_turns": statistics.median(turns),
        "winners": winners,
        "include_expansion": include_expansion,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--include-expansion", action="store_true")
    ap.add_argument("--both", action="store_true", help="Run both base and expansion modes.")
    ap.add_argument("--workers", type=int, default=1, help=">1 = multiprocess.")
    ap.add_argument("--mix-prob", type=float, default=None,
                    help="Probability of include_expansion per game (mix mode).")
    args = ap.parse_args()

    if args.workers > 1 or args.mix_prob is not None:
        from compile_engine.env import parallel_random_rollouts
        t0 = time.perf_counter()
        results = parallel_random_rollouts(
            args.games,
            workers=args.workers,
            include_expansion=None if args.mix_prob is not None else args.include_expansion,
            expansion_sample_prob=args.mix_prob,
        )
        elapsed = time.perf_counter() - t0
        turns = [r["turns"] for r in results]
        winners = {0: 0, 1: 0, None: 0}
        for r in results:
            winners[r["winner"]] = winners.get(r["winner"], 0) + 1
        exp_on = sum(1 for r in results if r["include_expansion"])
        print({
            "games": args.games,
            "workers": args.workers,
            "elapsed_sec": elapsed,
            "games_per_sec": args.games / elapsed,
            "mean_turns": statistics.mean(turns),
            "winners": winners,
            "expansion_on_count": exp_on,
        })
        return

    if args.both:
        for exp in (False, True):
            r = run(args.games, include_expansion=exp)
            print(f"[expansion={exp}] {r}")
    else:
        r = run(args.games, include_expansion=args.include_expansion)
        print(r)


if __name__ == "__main__":
    main()
