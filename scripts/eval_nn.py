"""Evaluate a trained NN agent against baselines.

Usage:
    python scripts/eval_nn.py --ckpt runs/v1/snapshot_00050.pt --games 400 --opp greedy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from compile_engine.agents import GreedyAgent, RandomAgent
from compile_engine.nn.model import PolicyValueNet
from compile_engine.nn.train import _resolve_device, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--opp", type=str, default="greedy", choices=["random", "greedy"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--expansion-prob", type=float, default=0.5)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    model = PolicyValueNet().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    opp = RandomAgent(seed=42) if args.opp == "random" else GreedyAgent(seed=42)
    result = evaluate(model, opp, args.games, device, expansion_prob=args.expansion_prob)
    print(result)


if __name__ == "__main__":
    main()
