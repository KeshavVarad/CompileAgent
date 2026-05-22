"""Playstyle fingerprint for one or more Compile checkpoints.

Computes a fixed set of behavioural metrics for each checkpoint and
renders them side-by-side so different models are easy to compare and
characterise. Also writes a radar chart for a quick visual.

Metrics:
  * Face-up ratio                — face-up plays / (face-up + face-down)
  * Refresh rate                 — REFRESH actions / decisions
  * Compile aggression           — fraction of games with ≥3 compiles
  * Avg game length (turns)
  * Avg compiles per game (agent)
  * Recompile rate               — games where agent re-compiled a line
  * Hand-empty refresh           — fraction of refreshes when hand was empty
  * Mean time-to-first-compile   — turn number of first compile
  * Mid-effect rate              — CHOOSE_TARGET / decisions

Output: markdown table + JSON + radar PNG normalised across the input
checkpoints.

Usage:
    python scripts/analysis/style_fingerprint.py runs/.../snapshot_00100.pt
    python scripts/analysis/style_fingerprint.py \\
        runs/.../snapshot_00100.pt runs/.../snapshot_00200.pt runs/.../snapshot_00500.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _lib import (  # noqa: E402
    REPO,
    analysis_dir,
    ensure_collected,
    outcome,
    read_games,
    summary,
    write_json,
    write_md,
)


METRICS = [
    "face_up_ratio",
    "refresh_rate",
    "compile_aggression",
    "avg_game_length",
    "avg_compiles_agent",
    "recompile_rate",
    "hand_empty_refresh",
    "first_compile_turn",
    "mid_effect_rate",
    "vs_greedy_wr",
]


def fingerprint(recs: list[dict]) -> dict[str, float]:
    """Compute the metric vector for a single jsonl."""
    if not recs:
        return {m: 0.0 for m in METRICS}

    n_games = len(recs)
    action_counts: Counter[str] = Counter()
    face_up = 0
    face_down = 0
    refresh_empty_hand = 0
    refresh_total = 0
    first_compile_turns: list[int] = []
    compiles_per_game: list[int] = []
    recompile_games = 0
    aggressive_compile_games = 0

    for r in recs:
        seat = r["agent_seat"]
        agent_compiles = r.get(f"compiles_p{seat}", 0)
        compiles_per_game.append(agent_compiles)
        if agent_compiles >= 3:
            aggressive_compile_games += 1
        if r.get("recompiled"):
            recompile_games += 1

        first_compile = None
        for d in r.get("decisions", []):
            t = d.get("action_type")
            action_counts[t] += 1
            if t == "PLAY_FACE_UP":
                face_up += 1
            elif t == "PLAY_FACE_DOWN":
                face_down += 1
            elif t == "REFRESH":
                refresh_total += 1
                if d.get("hand_size_before") == 0:
                    refresh_empty_hand += 1
            elif t == "COMPILE_LINE" and first_compile is None:
                first_compile = d.get("turn", 0)
        if first_compile is not None:
            first_compile_turns.append(first_compile)

    total_play = face_up + face_down
    total_decisions = sum(action_counts.values())
    g_sum = summary(recs)

    return {
        "face_up_ratio": face_up / max(1, total_play),
        "refresh_rate": refresh_total / max(1, total_decisions),
        "compile_aggression": aggressive_compile_games / n_games,
        "avg_game_length": sum(r["turns"] for r in recs) / n_games,
        "avg_compiles_agent": sum(compiles_per_game) / n_games,
        "recompile_rate": recompile_games / n_games,
        "hand_empty_refresh": refresh_empty_hand / max(1, refresh_total),
        "first_compile_turn": (
            sum(first_compile_turns) / max(1, len(first_compile_turns))
        ),
        "mid_effect_rate": action_counts.get("CHOOSE_TARGET", 0) / max(1, total_decisions),
        "vs_greedy_wr": g_sum.wr,
    }


def plot_radar(fingerprints: dict[str, dict[str, float]], out_path: Path) -> None:
    """Normalise each metric across the checkpoints to [0, 1] and draw
    a radar chart. Metrics with monotone "more is better" interpretation
    stay as-is; "less is better" ones (e.g. game length) get inverted."""
    INVERT = {"avg_game_length", "first_compile_turn"}
    keys = METRICS
    # Min-max normalize per metric.
    vals = np.array([[fp[k] for k in keys] for fp in fingerprints.values()])
    lo = vals.min(axis=0)
    hi = vals.max(axis=0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    norm = (vals - lo) / span
    for i, k in enumerate(keys):
        if k in INVERT:
            norm[:, i] = 1.0 - norm[:, i]

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    cmap = plt.cm.tab10.colors
    for i, (name, _fp) in enumerate(fingerprints.items()):
        vals_i = norm[i].tolist() + [norm[i][0]]
        ax.plot(angles, vals_i, label=name, linewidth=1.5, color=cmap[i % len(cmap)])
        ax.fill(angles, vals_i, alpha=0.10, color=cmap[i % len(cmap)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["", "", ""], fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_title("Playstyle fingerprint (min-max normalised across runs)", fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05), fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_report(fingerprints: dict[str, dict[str, float]], out_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Playstyle Fingerprint\n")
    lines.append(f"Checkpoints compared: {len(fingerprints)}")
    lines.append("")
    lines.append("## Metric definitions\n")
    lines.append("| metric | meaning |")
    lines.append("|---|---|")
    lines.append("| `face_up_ratio` | face-up plays / (face-up + face-down) |")
    lines.append("| `refresh_rate` | REFRESH actions per decision |")
    lines.append("| `compile_aggression` | games where agent compiled ≥ 3 times |")
    lines.append("| `avg_game_length` | average turns to end |")
    lines.append("| `avg_compiles_agent` | agent's compile count per game |")
    lines.append("| `recompile_rate` | games where agent re-compiled |")
    lines.append("| `hand_empty_refresh` | refreshes when hand was empty (forced rate) |")
    lines.append("| `first_compile_turn` | mean turn of agent's first compile |")
    lines.append("| `mid_effect_rate` | CHOOSE_TARGET share of decisions |")
    lines.append("| `vs_greedy_wr` | win rate vs Greedy (for ground-truth comparison) |")
    lines.append("")
    lines.append("## Side-by-side\n")
    names = list(fingerprints.keys())
    header = "| metric | " + " | ".join(names) + " |"
    sep = "|---|" + "|".join(["---:"] * len(names)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in METRICS:
        row = [m]
        for n in names:
            v = fingerprints[n][m]
            if m in ("avg_game_length", "first_compile_turn", "avg_compiles_agent"):
                row.append(f"{v:.2f}")
            else:
                row.append(f"{v:.1%}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"![radar chart](style_radar.png)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="one or more checkpoints to compare")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None,
                    help="output dir. Defaults to analysis/<stem of first ckpt>")
    args = ap.parse_args()

    fingerprints: dict[str, dict[str, float]] = {}
    for ckpt_str in args.ckpts:
        ckpt = Path(ckpt_str)
        d = analysis_dir(ckpt)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "vs_greedy.jsonl"
        print(f"[{ckpt.stem}] ensuring vs_greedy.jsonl ({args.games} games)")
        ensure_collected(ckpt, "greedy", path,
                         games=args.games, seed=args.seed, device=args.device)
        recs = read_games(path)
        fingerprints[ckpt.stem] = fingerprint(recs)

    out_dir = Path(args.out) if args.out else analysis_dir(Path(args.ckpts[0]))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_radar(fingerprints, out_dir / "style_radar.png")
    report = render_report(fingerprints, out_dir / "style_fingerprint.md")
    write_md(out_dir / "style_fingerprint.md", report)
    write_json(out_dir / "style_fingerprint.json", fingerprints)
    print(f"\nWrote {out_dir / 'style_fingerprint.md'}")
    print(f"      {out_dir / 'style_radar.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
