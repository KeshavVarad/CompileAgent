"""Aggregate per-game telemetry JSONL into a metrics.json file.

Reads one or more JSONL files produced by collect.py (one per opponent
matchup) and emits a single metrics.json combining all results — keyed by
opponent name. The shapes are stable and consumed by card.py / compare.py.

Usage:
    python scripts/eval/metrics.py \
        --in runs/latest/eval/snapshot_00100/ \
        --out runs/latest/eval/snapshot_00100/metrics.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import read_jsonl, write_json  # noqa: E402


# Tunable: action that exceeds this share of all actions = mode-collapse flag.
MODE_COLLAPSE_THRESHOLD = 0.60
# Protocol drafted in >= this share of games where it was available = strong bias.
PROTOCOL_BIAS_THRESHOLD = 0.80
# Heuristic adherence thresholds for green/yellow/red bullets.
COMPILE_WHEN_POSSIBLE_GOOD = 0.85
COMPILE_WHEN_POSSIBLE_BAD = 0.50
REFRESH_WITH_FULL_HAND_GOOD = 0.05   # lower is better
REFRESH_WITH_FULL_HAND_BAD = 0.20


def _bucket_turn(turn: int) -> str:
    if turn <= 10:
        return "early"
    if turn <= 30:
        return "mid"
    return "late"


def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


def aggregate_matchup(games: list[dict]) -> dict:
    """Compute the metrics for a single (agent, opponent) matchup."""
    n = len(games)
    wins = sum(1 for g in games if g["winner"] == g["agent_seat"])
    draws = sum(1 for g in games if g["winner"] is None)
    losses = n - wins - draws
    seat0 = [g for g in games if g["agent_seat"] == 0]
    seat1 = [g for g in games if g["agent_seat"] == 1]

    won_games = [g for g in games if g["winner"] == g["agent_seat"]]
    lost_games = [g for g in games if g["winner"] is not None and g["winner"] != g["agent_seat"]]

    compile_margins = []
    for g in games:
        seat = g["agent_seat"]
        mine = g["compiles_p0"] if seat == 0 else g["compiles_p1"]
        theirs = g["compiles_p1"] if seat == 0 else g["compiles_p0"]
        compile_margins.append(mine - theirs)

    # --- Action distribution ---------------------------------------------
    action_counts: Counter[str] = Counter()
    action_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    face_up_plays = 0
    face_down_plays = 0

    # --- Heuristic adherence ----------------------------------------------
    compileable_seen = 0
    compileable_taken = 0
    refresh_total = 0
    refresh_with_full_hand = 0

    # --- Card preference --------------------------------------------------
    card_plays: Counter[str] = Counter()

    # --- Protocol drafts --------------------------------------------------
    protocol_pick_counts: Counter[str] = Counter()
    protocol_pick_indices: dict[str, list[int]] = defaultdict(list)
    # We approximate "available" by what was actually drafted in the game's
    # pool — we don't reliably know the un-picked pool from telemetry alone.
    # For pick-frequency we just normalise by # of games.

    for g in games:
        seat = g["agent_seat"]
        # Track which draft picks were the agent's.
        picker_order: list[int] = g.get("draft_picker_order", [])
        agent_pick_idx_in_their_picks = 0
        for step_i, who in enumerate(picker_order):
            if who != seat:
                continue
            # Match step_i to the protocol the agent ended up with.
            # st.players[seat].protocols was built in pick order, so we use it.
            seat_protos = g["protocols_p0"] if seat == 0 else g["protocols_p1"]
            if agent_pick_idx_in_their_picks < len(seat_protos):
                proto = seat_protos[agent_pick_idx_in_their_picks]
                protocol_pick_counts[proto] += 1
                # Pick "position" in the agent's own 1..3 picking sequence.
                protocol_pick_indices[proto].append(agent_pick_idx_in_their_picks + 1)
                agent_pick_idx_in_their_picks += 1

        for d in g["decisions"]:
            at = d["action_type"]
            action_counts[at] += 1
            action_by_bucket[_bucket_turn(d["turn"])][at] += 1
            if at == "PLAY_FACE_UP":
                face_up_plays += 1
            elif at == "PLAY_FACE_DOWN":
                face_down_plays += 1
            if d.get("played_key"):
                card_plays[d["played_key"]] += 1
            if d.get("compile_was_legal"):
                compileable_seen += 1
                if at == "COMPILE_LINE":
                    compileable_taken += 1
            if at == "REFRESH":
                refresh_total += 1
                if d.get("hand_size_before", 0) >= 5:
                    refresh_with_full_hand += 1

    total_actions = sum(action_counts.values())
    action_dist = {k: _safe_div(v, total_actions) for k, v in action_counts.items()}
    action_by_bucket_dist = {
        bucket: {k: _safe_div(v, sum(cnts.values())) for k, v in cnts.items()}
        for bucket, cnts in action_by_bucket.items()
    }

    proto_pick_freq = {p: _safe_div(c, n) for p, c in protocol_pick_counts.items()}
    proto_avg_pick_idx = {
        p: round(sum(idxs) / len(idxs), 2)
        for p, idxs in protocol_pick_indices.items()
    }

    # --- Loss-mode buckets ------------------------------------------------
    loss_modes = {"blowout": 0, "close": 0, "timeout": 0, "recompiled_against": 0}
    for g in lost_games:
        seat = g["agent_seat"]
        mine = g["compiles_p0"] if seat == 0 else g["compiles_p1"]
        theirs = g["compiles_p1"] if seat == 0 else g["compiles_p0"]
        margin = theirs - mine
        # categorise (a single game can match multiple buckets — counted once)
        if g.get("timeout"):
            loss_modes["timeout"] += 1
        elif margin >= 2:
            loss_modes["blowout"] += 1
        elif g.get("recompiled"):
            loss_modes["recompiled_against"] += 1
        else:
            loss_modes["close"] += 1

    return {
        "n_games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": _safe_div(wins, n),
        "draw_rate": _safe_div(draws, n),
        "win_rate_seat0": _safe_div(
            sum(1 for g in seat0 if g["winner"] == 0), len(seat0)
        ),
        "win_rate_seat1": _safe_div(
            sum(1 for g in seat1 if g["winner"] == 1), len(seat1)
        ),
        "avg_turns": _safe_div(sum(g["turns"] for g in games), n),
        "avg_turns_won": _safe_div(sum(g["turns"] for g in won_games), len(won_games)),
        "avg_turns_lost": _safe_div(sum(g["turns"] for g in lost_games), len(lost_games)),
        "avg_compile_margin": _safe_div(sum(compile_margins), n),
        "timeout_rate": _safe_div(sum(1 for g in games if g.get("timeout")), n),
        "recompile_rate": _safe_div(sum(1 for g in games if g.get("recompiled")), n),
        "action_distribution": action_dist,
        "action_by_turn": action_by_bucket_dist,
        "face_down_ratio": _safe_div(face_down_plays, face_up_plays + face_down_plays),
        "protocol_pick_frequency": proto_pick_freq,
        "protocol_avg_pick_position": proto_avg_pick_idx,
        "top_cards_played": [
            [k, c] for k, c in card_plays.most_common(10)
        ],
        "heuristics": {
            "compiled_when_possible_rate": _safe_div(compileable_taken, compileable_seen),
            "compileable_decision_count": compileable_seen,
            "refreshed_with_full_hand_rate": _safe_div(refresh_with_full_hand, refresh_total),
            "refresh_decision_count": refresh_total,
        },
        "loss_modes": loss_modes,
    }


def derive_flags(matchup_metrics: dict[str, dict]) -> list[str]:
    """Cross-cutting warnings/observations for the model card header."""
    flags: list[str] = []
    # Mode-collapse: any single action exceeds threshold (averaged across opponents)
    all_action_counts: Counter[str] = Counter()
    total = 0
    for m in matchup_metrics.values():
        for k, v in m["action_distribution"].items():
            all_action_counts[k] += v * m["n_games"]
            total += 0  # we'll renormalise per-matchup below instead
    if matchup_metrics:
        # Use the largest matchup as a single rough signal.
        biggest = max(matchup_metrics.values(), key=lambda m: m["n_games"])
        for k, v in biggest["action_distribution"].items():
            if v >= MODE_COLLAPSE_THRESHOLD:
                flags.append(
                    f"action `{k}` dominates ({v:.0%} of all decisions vs `{list(matchup_metrics)[0]}`)"
                )

    # Heuristic adherence
    for opp, m in matchup_metrics.items():
        h = m["heuristics"]
        if h["compileable_decision_count"] >= 20:
            r = h["compiled_when_possible_rate"]
            if r < COMPILE_WHEN_POSSIBLE_BAD:
                flags.append(
                    f"misses compiles vs {opp}: {r:.0%} of compileable opportunities taken"
                )
        if h["refresh_decision_count"] >= 10:
            r = h["refreshed_with_full_hand_rate"]
            if r > REFRESH_WITH_FULL_HAND_BAD:
                flags.append(
                    f"wasteful refreshes vs {opp}: {r:.0%} of refreshes done with ≥5 cards in hand"
                )
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=str, required=True,
                    help="directory containing one JSONL per opponent (vs_X.jsonl)")
    ap.add_argument("--out", type=str, required=True,
                    help="output metrics.json path")
    ap.add_argument("--model", type=str, default=None,
                    help="display label for the evaluated model (e.g. 'snapshot_00100')")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"input directory not found: {in_dir}")

    matchups: dict[str, dict] = {}
    for jsonl_path in sorted(in_dir.glob("vs_*.jsonl")):
        opp = jsonl_path.stem[len("vs_"):]
        games = read_jsonl(jsonl_path)
        if not games:
            continue
        matchups[opp] = aggregate_matchup(games)

    if not matchups:
        raise SystemExit(f"no vs_*.jsonl found in {in_dir}")

    out = {
        "model": args.model or in_dir.name,
        "matchups": matchups,
        "flags": derive_flags(matchups),
    }
    out_path = Path(args.out)
    write_json(out_path, out)
    print(f"wrote {out_path}  ({len(matchups)} matchups)")
    for opp, m in matchups.items():
        print(f"  vs {opp:<14} n={m['n_games']:<4} wr={m['win_rate']:.2f}  "
              f"avg_turns={m['avg_turns']:.1f}  margin={m['avg_compile_margin']:.2f}")


if __name__ == "__main__":
    main()
