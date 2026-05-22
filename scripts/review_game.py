"""Replay a single random-seeded game with a trained NN agent and print the
eval bar + a blunder list.

Usage:
    python scripts/review_game.py --ckpt runs/v1/snapshot_00050.pt --seed 0 \\
        --opp greedy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from compile_engine import Game, GameConfig
from compile_engine.agents import GreedyAgent, RandomAgent
from compile_engine.nn.agent import NNAgent
from compile_engine.nn.encoder import encode_actions, encode_state
from compile_engine.nn.model import PolicyValueNet
from compile_engine.nn.train import _resolve_device


def _value_only(model: PolicyValueNet, game: Game, perspective: int, device) -> float:
    """One forward pass to get V(s) from perspective's POV. Sigmoid-scaled."""
    state = encode_state(game, perspective)
    # We need an action set to invoke forward; use the real legal actions.
    legal = game.legal_actions()
    if not legal:
        return 0.0
    raw, card_ids, proto_ids, extra_card_ids, mask = encode_actions(game, legal, perspective)
    import numpy as np
    s = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in state.items()}
    ar = torch.from_numpy(raw).unsqueeze(0).to(device)
    ac = torch.from_numpy(card_ids).unsqueeze(0).to(device)
    ap = torch.from_numpy(proto_ids).unsqueeze(0).to(device)
    ae = torch.from_numpy(extra_card_ids).unsqueeze(0).to(device)
    am = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        _, value = model(s, ar, ac, ap, ae, am)
    return float(value[0].item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opp", type=str, default="greedy", choices=["random", "greedy"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--agent-seat", type=int, default=0, choices=[0, 1])
    ap.add_argument("--blunder-threshold", type=float, default=0.15,
                    help="Δ value drops larger than this are flagged as blunders")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    model = PolicyValueNet().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    cfg = GameConfig(seed=args.seed, max_turns=200)
    game = Game(cfg)
    game.start()
    agent = NNAgent(model, device=device, stochastic=False)
    opp = RandomAgent(seed=args.seed + 1) if args.opp == "random" else GreedyAgent(seed=args.seed + 1)
    agents = (agent, opp) if args.agent_seat == 0 else (opp, agent)

    trace = []
    blunders = []
    while not game.is_over():
        who = game.decider()
        legal = game.legal_actions()
        if not legal:
            break
        # Eval from agent's perspective at start of decision.
        v_before = _value_only(model, game, args.agent_seat, device)
        action = agents[who].choose(game, legal)
        game.step(action)
        v_after = _value_only(model, game, args.agent_seat, device) if not game.is_over() else (
            1.0 if game.state.winner == args.agent_seat else (-1.0 if game.state.winner is not None else 0.0)
        )
        win_prob = (v_after + 1.0) / 2.0
        trace.append({
            "turn": game.state.turn,
            "decider": who,
            "v": v_after,
            "p_win_agent": win_prob,
            "action": str(action),
        })
        delta = v_after - v_before
        if who == args.agent_seat and delta < -args.blunder_threshold:
            blunders.append({
                "turn": game.state.turn,
                "action": str(action),
                "v_before": v_before,
                "v_after": v_after,
                "delta": delta,
            })

    print(f"Game finished. Winner: P{game.state.winner}. Agent seat: P{args.agent_seat}.")
    print()
    print("Eval trace (agent perspective):")
    print("turn | decider | V       | P(win agent) | action")
    print("-----+---------+---------+--------------+-----------------------------")
    for row in trace:
        print(
            f"{row['turn']:>4} | P{row['decider']}      | {row['v']:+.2f}   | "
            f"{row['p_win_agent']:.2%}        | {row['action']}"
        )
    if blunders:
        print()
        print(f"Agent blunders (Δv < -{args.blunder_threshold}):")
        for b in blunders:
            print(
                f"  turn {b['turn']}: {b['action']} "
                f"({b['v_before']:+.2f} → {b['v_after']:+.2f}, Δ={b['delta']:+.2f})"
            )


if __name__ == "__main__":
    main()
