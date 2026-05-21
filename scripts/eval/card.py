"""Render a model card (Markdown + JSON) from metrics + (optional) ladder.

The Markdown form is what a future webapp route would render as the bot's
bio. The JSON is the same data in machine-readable form for diffing /
historical tracking.

Usage:
    python scripts/eval/card.py \
        --metrics runs/latest/eval/snapshot_00100/metrics.json \
        --ladder runs/latest/eval/ladder.json \
        --out runs/latest/eval/snapshot_00100/model_card.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import write_json  # noqa: E402
from metrics import (  # noqa: E402
    COMPILE_WHEN_POSSIBLE_BAD,
    COMPILE_WHEN_POSSIBLE_GOOD,
    REFRESH_WITH_FULL_HAND_BAD,
    REFRESH_WITH_FULL_HAND_GOOD,
)


def _flag(value: float, *, good: float, bad: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= good:
            return "🟢"
        if value <= bad:
            return "🔴"
        return "🟡"
    else:
        if value <= good:
            return "🟢"
        if value >= bad:
            return "🔴"
        return "🟡"


def _format_pct(x: float) -> str:
    return f"{x*100:.0f}%"


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    return out


def render_md(metrics: dict, ladder: dict | None) -> str:
    model = metrics.get("model", "unknown")
    matchups = metrics["matchups"]

    lines: list[str] = []
    lines.append(f"# Model card · {model}")
    lines.append("")

    # ---- Header / flags --------------------------------------------------
    if metrics.get("flags"):
        lines.append("## ⚠️ Notable flags")
        for f in metrics["flags"]:
            lines.append(f"- {f}")
        lines.append("")

    # ---- Strength --------------------------------------------------------
    lines.append("## Strength")
    str_rows = []
    for opp, m in matchups.items():
        str_rows.append([
            opp,
            str(m["n_games"]),
            f"{m['win_rate']:.2f}",
            f"{m['draw_rate']:.2f}",
            f"{m['avg_turns']:.1f}",
            f"{m['avg_compile_margin']:+.2f}",
            f"{m['timeout_rate']:.2f}",
        ])
    lines.append(_format_table(
        ["opponent", "n", "wr", "draw", "avg turns", "compile margin", "timeout"],
        str_rows,
    ))

    if ladder:
        rank = ladder.get("ranking", [])
        if rank:
            lines.append("### Ladder Elo")
            elo_rows = [[r["name"], f"{r['elo']:.0f}"] for r in rank]
            lines.append(_format_table(["model", "elo"], elo_rows))

    # ---- Behavioural fingerprint -----------------------------------------
    lines.append("## Behaviour")
    # Use the largest matchup as the primary view.
    primary_opp = max(matchups, key=lambda k: matchups[k]["n_games"])
    m = matchups[primary_opp]
    lines.append(f"_(action distribution from {m['n_games']} games vs **{primary_opp}**)_")
    lines.append("")
    action_rows = sorted(
        m["action_distribution"].items(), key=lambda kv: -kv[1]
    )
    lines.append(_format_table(
        ["action", "share"],
        [[k, _format_pct(v)] for k, v in action_rows],
    ))

    # Face-down ratio
    lines.append(f"- **Face-down ratio** (face-down / total plays): "
                 f"{m['face_down_ratio']:.2f}")
    lines.append("")

    # Protocol prefs
    if m.get("protocol_pick_frequency"):
        proto_rows = sorted(
            m["protocol_pick_frequency"].items(), key=lambda kv: -kv[1]
        )
        lines.append("### Protocol drafts")
        lines.append(_format_table(
            ["protocol", "pick freq", "avg pick pos (1-3)"],
            [
                [p, _format_pct(f), str(m["protocol_avg_pick_position"].get(p, "—"))]
                for p, f in proto_rows
            ],
        ))

    # Top cards
    if m.get("top_cards_played"):
        lines.append("### Top cards played")
        card_rows = [[k, str(c)] for k, c in m["top_cards_played"]]
        lines.append(_format_table(["card", "times played"], card_rows))

    # ---- Heuristic adherence --------------------------------------------
    lines.append("## Heuristic adherence")
    heur_rows = []
    for opp, mm in matchups.items():
        h = mm["heuristics"]
        cwp = h["compiled_when_possible_rate"]
        rwf = h["refreshed_with_full_hand_rate"]
        cwp_flag = _flag(cwp, good=COMPILE_WHEN_POSSIBLE_GOOD, bad=COMPILE_WHEN_POSSIBLE_BAD)
        rwf_flag = _flag(
            rwf, good=REFRESH_WITH_FULL_HAND_GOOD, bad=REFRESH_WITH_FULL_HAND_BAD,
            higher_is_better=False,
        )
        heur_rows.append([
            opp,
            f"{cwp_flag} {cwp:.2f} ({h['compileable_decision_count']} chances)",
            f"{rwf_flag} {rwf:.2f} ({h['refresh_decision_count']} refreshes)",
        ])
    lines.append(_format_table(
        ["opponent", "compile-when-possible", "wasteful-refresh"],
        heur_rows,
    ))

    # ---- Weak spots / loss modes ----------------------------------------
    lines.append("## Weak spots")
    loss_rows = []
    for opp, mm in matchups.items():
        lm = mm["loss_modes"]
        total_losses = sum(lm.values())
        if total_losses == 0:
            loss_rows.append([opp, "—", "—", "—", "—"])
            continue
        loss_rows.append([
            opp,
            f"{lm['close']}/{total_losses}",
            f"{lm['blowout']}/{total_losses}",
            f"{lm['timeout']}/{total_losses}",
            f"{lm['recompiled_against']}/{total_losses}",
        ])
    lines.append(_format_table(
        ["opponent", "close losses", "blowouts", "timeouts", "recompiled against"],
        loss_rows,
    ))

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True)
    ap.add_argument("--ladder", type=str, default=None,
                    help="optional ladder ratings.json to fold in")
    ap.add_argument("--out", type=str, required=True,
                    help="output model_card.md path; a sibling .json is also written")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    ladder = json.loads(Path(args.ladder).read_text()) if args.ladder else None

    md = render_md(metrics, ladder)
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    # Companion JSON: the same data minus rendering, for diffing.
    out_json = out_md.with_suffix(".json")
    bundle = {"metrics": metrics}
    if ladder is not None:
        bundle["ladder"] = ladder
    write_json(out_json, bundle)
    print(f"wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
