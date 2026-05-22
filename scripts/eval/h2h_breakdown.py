"""Where does the new model win? Fast diagnostic over h2h jsonl files.

Given one or more h2h jsonl files (output of `collect.py --model X --opp Y`),
slice the agent's win rate by various game-level features to localize where
the win-rate delta vs the opponent comes from. No engine replay or model
inference — just aggregates over the per-decision records that `collect.py`
already wrote.

Slices reported:
  - Outcome breakdown (overall, per-seat, by config).
  - Drafted protocols (per-protocol WR when the agent drafted it).
  - Game characteristics (length quartiles, compile counts, recompiles,
    timeouts).
  - Agent action mix when winning vs losing (proxy for stylistic shifts).

Usage:
  python scripts/eval/h2h_breakdown.py file1.jsonl [file2.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def outcome(rec: dict) -> str:
    w = rec.get("winner")
    if w is None:
        return "draw"
    return "win" if w == rec["agent_seat"] else "loss"


def fmt_wr(wins: int, total: int) -> str:
    if total == 0:
        return "  -   (n=0)"
    p = wins / total
    se = math.sqrt(p * (1 - p) / total) if total > 1 else 0.5
    return f"{p:5.1%}  ±{1.96 * se:4.1%}  (n={total})"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def by_key(recs: list[dict], keyfn) -> dict[object, dict[str, int]]:
    out: dict[object, dict[str, int]] = defaultdict(
        lambda: {"win": 0, "loss": 0, "draw": 0, "n": 0}
    )
    for r in recs:
        k = keyfn(r)
        if k is None:
            continue
        o = outcome(r)
        out[k][o] += 1
        out[k]["n"] += 1
    return out


def print_slice(title: str, buckets: dict, sort_by: str = "n", top: int = 20) -> None:
    print(f"\n{title}")
    items = sorted(buckets.items(), key=lambda kv: -kv[1][sort_by])
    for k, b in items[:top]:
        print(f"  {str(k):<30}  WR {fmt_wr(b['win'], b['n'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="h2h jsonl files to combine")
    args = ap.parse_args()

    recs: list[dict] = []
    for f in args.files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))

    n = len(recs)
    if n == 0:
        print("no records")
        return 1
    opp_name = recs[0].get("opponent_name", "?")
    model_label = "agent (distilled)"
    print(f"Loaded {n} h2h games — {model_label} vs {opp_name}")
    results = [outcome(r) for r in recs]
    w, l, d = results.count("win"), results.count("loss"), results.count("draw")
    print(f"Overall: {w}-{l}-{d}  WR {fmt_wr(w, n)}")

    section("By seat")
    print_slice("agent_seat", by_key(recs, lambda r: r["agent_seat"]))

    section("By config")
    print_slice("include_expansion", by_key(recs, lambda r: r["include_expansion"]))
    print_slice("include_main2", by_key(recs, lambda r: r["include_main2"]))
    print_slice("include_aux2", by_key(recs, lambda r: r["include_aux2"]))

    section("By drafted protocol (agent side)")
    proto_buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"win": 0, "loss": 0, "draw": 0, "n": 0}
    )
    for r in recs:
        seat = r["agent_seat"]
        protos = r[f"protocols_p{seat}"]
        for p in protos:
            o = outcome(r)
            proto_buckets[p][o] += 1
            proto_buckets[p]["n"] += 1
    print_slice("protocol (sorted by n)", proto_buckets)

    section("By compile pacing")
    # Quartile of turns-to-finish
    turns = [r["turns"] for r in recs]
    q1, q2, q3 = (statistics.quantiles(turns, n=4) if len(turns) >= 4 else (turns[0], turns[0], turns[0]))
    def turn_bucket(r):
        t = r["turns"]
        if t < q1: return f"q1 (<{q1:.0f})"
        if t < q2: return f"q2 (<{q2:.0f})"
        if t < q3: return f"q3 (<{q3:.0f})"
        return f"q4 (≥{q3:.0f})"
    print_slice("game length", by_key(recs, turn_bucket))

    # Recompiles
    print_slice("recompiled", by_key(recs, lambda r: r["recompiled"]))
    print_slice("timeout", by_key(recs, lambda r: r["timeout"]))

    section("Compile counts (agent vs opp)")
    seat_pair_buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"win": 0, "loss": 0, "draw": 0, "n": 0}
    )
    for r in recs:
        seat = r["agent_seat"]
        my_c = r[f"compiles_p{seat}"]
        op_c = r[f"compiles_p{1 - seat}"]
        k = f"agent={my_c} opp={op_c}"
        seat_pair_buckets[k][outcome(r)] += 1
        seat_pair_buckets[k]["n"] += 1
    print_slice("(agent_compiles, opp_compiles)", seat_pair_buckets, top=10)

    section("Agent action mix (per-decision) — wins vs losses")
    # The decisions field captures ALL decisions in the game from both
    # seats (DRAFT_PROTOCOL is by definition the drafter's pick;
    # PLAY/REFRESH/COMPILE happen when it's that seat's turn). We
    # can't always tell who acted from the record alone — but for
    # action-type aggregates, comparing the mix in wins vs losses
    # still surfaces stylistic shifts.
    mix_w: Counter[str] = Counter()
    mix_l: Counter[str] = Counter()
    for r in recs:
        o = outcome(r)
        if o not in ("win", "loss"):
            continue
        tgt = mix_w if o == "win" else mix_l
        for d in r.get("decisions", []):
            tgt[d["action_type"]] += 1
    total_w = sum(mix_w.values())
    total_l = sum(mix_l.values())
    print(f"  (decisions in wins: {total_w}, in losses: {total_l})")
    print(f"  {'action_type':<22}  {'wins %':>8}  {'losses %':>9}  {'delta':>7}")
    for a in sorted(set(mix_w) | set(mix_l)):
        wp = mix_w[a] / max(1, total_w)
        lp = mix_l[a] / max(1, total_l)
        print(f"  {a:<22}  {wp:8.1%}  {lp:9.1%}  {wp - lp:+7.1%}")

    section("Face-up vs face-down play ratio in wins vs losses")
    for label, results_set in [("wins", "win"), ("losses", "loss")]:
        fu = 0
        fd = 0
        for r in recs:
            if outcome(r) != results_set:
                continue
            for d in r.get("decisions", []):
                if d["action_type"] == "PLAY_FACE_UP":
                    fu += 1
                elif d["action_type"] == "PLAY_FACE_DOWN":
                    fd += 1
        total = fu + fd
        if total > 0:
            print(f"  {label:<8}  face-up {fu/total:.1%} ({fu}/{total})")

    section("Per-decision card usage in wins vs losses (top deltas)")
    cards_w: Counter[str] = Counter()
    cards_l: Counter[str] = Counter()
    for r in recs:
        o = outcome(r)
        if o not in ("win", "loss"):
            continue
        tgt = cards_w if o == "win" else cards_l
        for d in r.get("decisions", []):
            if d.get("face_up") is True and d.get("played_key"):
                tgt[d["played_key"]] += 1
    cw_total = sum(cards_w.values())
    cl_total = sum(cards_l.values())
    # Top by absolute frequency-delta (normalized by total face-up plays)
    keys = set(cards_w) | set(cards_l)
    rows = []
    for k in keys:
        wp = cards_w[k] / max(1, cw_total)
        lp = cards_l[k] / max(1, cl_total)
        rows.append((k, cards_w[k], cards_l[k], wp - lp))
    rows.sort(key=lambda x: -abs(x[3]))
    print(f"  {'card_key':<22}  {'wins #':>7}  {'losses #':>9}  {'delta%':>7}")
    for k, cw, cl, dlt in rows[:15]:
        print(f"  {k:<22}  {cw:7d}  {cl:9d}  {dlt:+7.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
