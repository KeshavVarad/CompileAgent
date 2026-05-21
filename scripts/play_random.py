"""Play a single random-vs-random game and print the state each turn.

Useful as a sanity check / human-readable trace.

Usage:
    python scripts/play_random.py [--seed N] [--include-expansion] [--quiet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from compile_engine import Game, GameConfig
from compile_engine.agents import RandomAgent
from compile_engine.env import _render_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-expansion", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = GameConfig(
        include_expansion=args.include_expansion, seed=args.seed, max_turns=200,
    )
    game = Game(cfg)
    game.start()
    agents = (RandomAgent(seed=args.seed + 1), RandomAgent(seed=args.seed + 2))

    last_turn = -1
    while not game.is_over():
        who = game.decider()
        legal = game.legal_actions()
        if not legal:
            break
        action = agents[who].choose(game, legal)
        if not args.quiet and game.state.turn != last_turn:
            print(_render_text(game))
            last_turn = game.state.turn
        game.step(action)
    print("=" * 60)
    print(_render_text(game))


if __name__ == "__main__":
    main()
