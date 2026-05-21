"""Diff two model-card JSON bundles.

Highlights what changed snapshot-to-snapshot: deltas in win-rate, action
distribution, protocol prefs, Elo. Designed for terminal use; prints a
short summary, doesn't write files.

Usage:
    python scripts/eval/compare.py \
        --a runs/latest/eval/snapshot_00050/model_card.json \
        --b runs/latest/eval/snapshot_00100/model_card.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Don't print noise — only surface deltas above these absolute thresholds.
WR_DELTA_FLOOR = 0.03
ACTION_DELTA_FLOOR = 0.03
PROTO_DELTA_FLOOR = 0.05
ELO_DELTA_FLOOR = 25.0


def _arrow(d: float) -> str:
    if d > 0:
        return "▲"
    if d < 0:
        return "▼"
    return "="


def _label_a_b(a: dict, b: dict) -> tuple[str, str]:
    return a["metrics"].get("model", "A"), b["metrics"].get("model", "B")


def compare_strength(a: dict, b: dict) -> list[str]:
    out: list[str] = []
    a_m = a["metrics"]["matchups"]
    b_m = b["metrics"]["matchups"]
    shared = sorted(set(a_m) & set(b_m))
    for opp in shared:
        wa, wb = a_m[opp]["win_rate"], b_m[opp]["win_rate"]
        d = wb - wa
        if abs(d) >= WR_DELTA_FLOOR:
            out.append(f"  vs {opp:<14} wr {wa:.2f} → {wb:.2f}  {_arrow(d)}{abs(d):.2f}")
    return out


def compare_actions(a: dict, b: dict) -> list[str]:
    """Per opponent, find action-distribution shifts."""
    out: list[str] = []
    a_m = a["metrics"]["matchups"]
    b_m = b["metrics"]["matchups"]
    shared = sorted(set(a_m) & set(b_m))
    for opp in shared:
        a_dist = a_m[opp]["action_distribution"]
        b_dist = b_m[opp]["action_distribution"]
        all_keys = sorted(set(a_dist) | set(b_dist))
        deltas = []
        for k in all_keys:
            d = b_dist.get(k, 0.0) - a_dist.get(k, 0.0)
            if abs(d) >= ACTION_DELTA_FLOOR:
                deltas.append((k, d))
        if deltas:
            out.append(f"  vs {opp}:")
            for k, d in sorted(deltas, key=lambda kv: -abs(kv[1])):
                out.append(f"    {k:<18} {_arrow(d)}{abs(d)*100:4.1f}pp")
    return out


def compare_protocols(a: dict, b: dict) -> list[str]:
    """Protocol-pick preference shift (use first shared matchup)."""
    a_m = a["metrics"]["matchups"]
    b_m = b["metrics"]["matchups"]
    shared = sorted(set(a_m) & set(b_m))
    if not shared:
        return []
    opp = shared[0]
    a_proto = a_m[opp]["protocol_pick_frequency"]
    b_proto = b_m[opp]["protocol_pick_frequency"]
    keys = sorted(set(a_proto) | set(b_proto))
    out: list[str] = []
    for k in keys:
        d = b_proto.get(k, 0.0) - a_proto.get(k, 0.0)
        if abs(d) >= PROTO_DELTA_FLOOR:
            out.append(f"  {k:<14} {a_proto.get(k, 0.0):.2f} → {b_proto.get(k, 0.0):.2f}  "
                       f"{_arrow(d)}{abs(d):.2f}")
    return out


def compare_elo(a: dict, b: dict) -> list[str]:
    out: list[str] = []
    a_l = a.get("ladder")
    b_l = b.get("ladder")
    if not (a_l and b_l):
        return out
    a_elo = a_l.get("elos", {})
    b_elo = b_l.get("elos", {})
    for k in sorted(set(a_elo) & set(b_elo)):
        d = b_elo[k] - a_elo[k]
        if abs(d) >= ELO_DELTA_FLOOR:
            out.append(f"  {k:<24} {a_elo[k]:.0f} → {b_elo[k]:.0f}  {_arrow(d)}{abs(d):.0f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=str, required=True, help="baseline model_card.json")
    ap.add_argument("--b", type=str, required=True, help="newer model_card.json")
    args = ap.parse_args()

    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    a_name, b_name = _label_a_b(a, b)
    print(f"diff: {a_name}  →  {b_name}")
    print()

    def _section(title: str, body: list[str]) -> None:
        if not body:
            return
        print(title)
        for line in body:
            print(line)
        print()

    _section("Strength (win-rate shifts):", compare_strength(a, b))
    _section("Behaviour (action distribution shifts):", compare_actions(a, b))
    _section("Protocol drafts:", compare_protocols(a, b))
    _section("Ladder Elo:", compare_elo(a, b))


if __name__ == "__main__":
    main()
