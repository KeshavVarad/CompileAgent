"""Run an all-pairs round-robin between snapshots, compute Elo, dump ratings.

Plays N games per ordered pair (a, b) with a in seat 0, b in seat 1. To keep
seat-bias out of the rating we run both orderings, so each unordered pair
gets 2N games. Random and Greedy are seeded with reference Elos (anchors)
so the ladder is comparable across runs.

Usage:
    python scripts/eval/ladder.py \
        --snapshots runs/latest/snapshot_00010.pt runs/latest/snapshot_00100.pt \
        --games 30 \
        --out runs/latest/eval/ladder.json
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from compile_engine.cards import load_card_defs  # noqa: E402
from compile_engine.env import play_game  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    DEFAULT_AUX2_PROB,
    DEFAULT_EXPANSION_PROB,
    DEFAULT_MAIN2_PROB,
    OpponentSpec,
    build_agent,
    make_game_config,
    resolve_device,
    write_json,
)


# Anchor ratings for the two baselines so absolute Elo is comparable across
# evaluation runs. Random ≈ 1000, Greedy ≈ 1200 is a rough convention; if a
# snapshot can't beat random we expect it to sit below 1000.
ANCHOR_ELO = {"random": 1000.0, "greedy": 1200.0}
INITIAL_ELO = 1500.0
K_FACTOR = 32.0


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def update_elo(r_a: float, r_b: float, score_a: float, *, k: float = K_FACTOR) -> tuple[float, float]:
    """Standard Elo update. score_a in {1, 0.5, 0}."""
    e_a = _expected(r_a, r_b)
    r_a_new = r_a + k * (score_a - e_a)
    r_b_new = r_b + k * ((1 - score_a) - (1 - e_a))
    return r_a_new, r_b_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="+", required=True,
                    help="list of snapshot .pt paths to include in the ladder")
    ap.add_argument("--include-baselines", action="store_true", default=True,
                    help="also include random + greedy as ladder participants (default True)")
    ap.add_argument("--games", type=int, default=30,
                    help="games per ordered pair (so each unordered pair plays 2N games)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--expansion-prob", type=float, default=DEFAULT_EXPANSION_PROB)
    ap.add_argument("--main2-prob", type=float, default=DEFAULT_MAIN2_PROB)
    ap.add_argument("--aux2-prob", type=float, default=DEFAULT_AUX2_PROB)
    args = ap.parse_args()

    device = resolve_device(args.device)
    defs = load_card_defs()

    # Build the roster of participants.
    specs: list[OpponentSpec] = []
    if args.include_baselines:
        specs.append(OpponentSpec(name="random", kind="random"))
        specs.append(OpponentSpec(name="greedy", kind="greedy"))
    for s in args.snapshots:
        specs.append(OpponentSpec.parse(s))

    # Instantiate each agent exactly once (lazy NN load is expensive).
    agents = {s.name: build_agent(s, device, seed=args.seed + i) for i, s in enumerate(specs)}
    elos: dict[str, float] = {
        s.name: ANCHOR_ELO[s.name] if s.name in ANCHOR_ELO else INITIAL_ELO
        for s in specs
    }
    wr_matrix: dict[str, dict[str, dict[str, int]]] = {
        a.name: {b.name: {"wins": 0, "losses": 0, "draws": 0, "n": 0} for b in specs}
        for a in specs
    }

    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    pairs = list(itertools.permutations(specs, 2))
    total = len(pairs) * args.games
    done = 0
    for a, b in pairs:
        for _ in range(args.games):
            cfg = make_game_config(
                rng,
                expansion_prob=args.expansion_prob,
                main2_prob=args.main2_prob,
                aux2_prob=args.aux2_prob,
            )
            g = play_game(agent0=agents[a.name], agent1=agents[b.name], config=cfg, card_defs=defs)
            w = g.state.winner
            if w == 0:
                score_a = 1.0
                wr_matrix[a.name][b.name]["wins"] += 1
                wr_matrix[b.name][a.name]["losses"] += 1
            elif w == 1:
                score_a = 0.0
                wr_matrix[a.name][b.name]["losses"] += 1
                wr_matrix[b.name][a.name]["wins"] += 1
            else:
                score_a = 0.5
                wr_matrix[a.name][b.name]["draws"] += 1
                wr_matrix[b.name][a.name]["draws"] += 1
            wr_matrix[a.name][b.name]["n"] += 1
            wr_matrix[b.name][a.name]["n"] += 1
            # Freeze the anchors so they don't drift; only update the rest.
            r_a, r_b = elos[a.name], elos[b.name]
            r_a_new, r_b_new = update_elo(r_a, r_b, score_a)
            if a.name not in ANCHOR_ELO:
                elos[a.name] = r_a_new
            if b.name not in ANCHOR_ELO:
                elos[b.name] = r_b_new
            done += 1
        print(f"  [{done:4d}/{total}] {a.name} vs {b.name}: "
              f"elo[{a.name}]={elos[a.name]:.0f} elo[{b.name}]={elos[b.name]:.0f}  "
              f"dt={time.perf_counter()-t0:.1f}s")

    # Final ranking
    ranking = sorted(elos.items(), key=lambda kv: kv[1], reverse=True)

    write_json(Path(args.out), {
        "k_factor": K_FACTOR,
        "initial_elo": INITIAL_ELO,
        "anchors": ANCHOR_ELO,
        "games_per_pair": args.games,
        "elos": elos,
        "ranking": [{"name": n, "elo": round(e, 1)} for n, e in ranking],
        "win_loss_matrix": wr_matrix,
    })
    print(f"wrote {args.out}")
    print("ranking:")
    for n, e in ranking:
        print(f"  {n:<24} {e:7.1f}")


if __name__ == "__main__":
    main()
