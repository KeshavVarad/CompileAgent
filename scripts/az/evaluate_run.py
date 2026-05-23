"""Evaluate a snapshot against random + greedy and characterize its play.

Beyond raw win rate, captures action-type distribution and game-shape
metrics (turn length, compiles, refreshes, face-up vs face-down play
ratio) so we can see *what* the agent is actually doing — not just
whether it's winning.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/az/evaluate_run.py \
        runs/<TS>-az-fresh/snapshot_00070.pt --games 30
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

from compile_engine.actions import Action, ActionType
from compile_engine.agents import GreedyAgent, RandomAgent
from compile_engine.cards import load_card_defs
from compile_engine.env import play_game
from compile_engine.nn.agent import NNAgent
from compile_engine.nn.model import PolicyValueNet
from compile_engine.state import GameConfig


def load_snapshot(path: Path, device: str) -> PolicyValueNet:
    state = torch.load(str(path), map_location=device, weights_only=False)
    m = PolicyValueNet().to(device)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def label(t: ActionType) -> str:
    return {
        ActionType.DRAFT_PROTOCOL: "draft",
        ActionType.PLAY_FACE_UP:   "play_up",
        ActionType.PLAY_FACE_DOWN: "play_down",
        ActionType.REFRESH:        "refresh",
        ActionType.COMPILE_LINE:   "compile",
        ActionType.DISCARD_CARD:   "discard",
        ActionType.SHIFT_OWN_CARD: "shift",
        ActionType.CHOOSE_TARGET:  "choose",
        ActionType.SKIP_OPTIONAL:  "skip",
        ActionType.NOOP:           "noop",
    }[t]


def play_one(*, agent0, agent1, seed: int, max_turns: int = 200):
    """Replicates env.play_game but captures (seat, action) per step."""
    from compile_engine.game import Game
    defs = load_card_defs()
    game = Game(GameConfig(seed=seed, max_turns=max_turns), defs=defs)
    game.start()
    agents = (agent0, agent1)
    history: list[tuple[int, Action]] = []
    while not game.is_over():
        who = game.decider()
        legal = game.legal_actions()
        if not legal:
            break
        a = agents[who].choose(game, legal)
        history.append((who, a))
        game.step(a)
    return game, history


def summarize(games_played, results, label_acc, side_to_eval):
    n = len(results)
    if n == 0:
        return
    wins = sum(1 for r in results if r["winner"] == side_to_eval)
    losses = sum(1 for r in results if r["winner"] is not None and r["winner"] != side_to_eval)
    draws = n - wins - losses
    avg_turns = sum(r["turns"] for r in results) / n
    avg_compiles = sum(r["nn_compiles"] for r in results) / n
    avg_refresh = sum(r["nn_refreshes"] for r in results) / n
    avg_face_up = sum(r["nn_face_up"] for r in results) / n
    avg_face_dn = sum(r["nn_face_down"] for r in results) / n
    avg_decisions = sum(r["nn_decisions"] for r in results) / n
    print(f"  games: {n}, wins: {wins}, losses: {losses}, draws: {draws}, "
          f"WR={wins/n:.2%}")
    print(f"  avg game length: {avg_turns:.1f} turns")
    print(f"  per game, NN agent: {avg_decisions:.1f} decisions / "
          f"{avg_face_up:.2f} face-up plays / {avg_face_dn:.2f} face-down plays / "
          f"{avg_refresh:.2f} refreshes / {avg_compiles:.2f} compiles")
    print(f"  action mix (NN side only):")
    total = sum(label_acc.values())
    for k, v in label_acc.most_common():
        print(f"    {k:>10s}  {v:>5d}  ({v/total:.1%})")


def run_matchup(*, model, device, opponent, opp_name, nn_seat: int, games: int, seed: int):
    print(f"\n=== NN ({Path(model.__class__.__module__).name if False else 'snapshot'}) "
          f"as P{nn_seat} vs {opp_name} ===")
    label_acc: Counter[str] = Counter()
    results = []
    for i in range(games):
        nn = NNAgent(model, device=device, stochastic=False)  # argmax for eval
        agents = [nn, opponent] if nn_seat == 0 else [opponent, nn]
        g, hist = play_one(agent0=agents[0], agent1=agents[1], seed=seed + i)
        # count actions on NN's seat
        nn_compiles = sum(1 for s, a in hist if s == nn_seat and a.type == ActionType.COMPILE_LINE)
        nn_refresh  = sum(1 for s, a in hist if s == nn_seat and a.type == ActionType.REFRESH)
        nn_face_up  = sum(1 for s, a in hist if s == nn_seat and a.type == ActionType.PLAY_FACE_UP)
        nn_face_dn  = sum(1 for s, a in hist if s == nn_seat and a.type == ActionType.PLAY_FACE_DOWN)
        nn_decisions = sum(1 for s, _ in hist if s == nn_seat)
        for s, a in hist:
            if s == nn_seat:
                label_acc[label(a.type)] += 1
        results.append({
            "winner": g.state.winner,
            "turns": g.state.turn,
            "nn_compiles": nn_compiles,
            "nn_refreshes": nn_refresh,
            "nn_face_up": nn_face_up,
            "nn_face_down": nn_face_dn,
            "nn_decisions": nn_decisions,
        })
    summarize(games, results, label_acc, side_to_eval=nn_seat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--games", type=int, default=20,
                    help="games per side per opponent (so total = 4*games)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model = load_snapshot(args.snapshot, args.device)
    print(f"Loaded {args.snapshot.name}")

    # Two opponents x two seats = 4 matchups, all using fresh agents each game
    for opp_factory, name in [
        (lambda s: RandomAgent(seed=s), "random"),
        (lambda s: GreedyAgent(seed=s), "greedy"),
    ]:
        for nn_seat in (0, 1):
            # Use a fresh opponent instance per matchup so the seed scheme
            # is deterministic but the opponent's internal rng resets.
            opp = opp_factory(args.seed + 1000 * nn_seat + (0 if name == "random" else 100))
            run_matchup(model=model, device=args.device, opponent=opp,
                        opp_name=name, nn_seat=nn_seat, games=args.games,
                        seed=args.seed + 10_000 * (1 if name == "greedy" else 0))


if __name__ == "__main__":
    main()
