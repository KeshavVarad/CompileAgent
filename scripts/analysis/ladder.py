"""Polished ladder report for a set of Compile checkpoints.

Wraps `scripts/eval/ladder.py` (which does the actual round-robin + Elo
computation) and renders a shareable markdown report on top of its
JSON output. Use this when you want a single shareable artifact showing
how a set of checkpoints rank.

Usage:
    python scripts/analysis/ladder.py \\
        --snapshots runs/.../snapshot_00010.pt runs/.../snapshot_00100.pt ... \\
        --games 30 \\
        --label "AZ training progression"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _lib import write_json, write_md  # noqa: E402


def run_eval_ladder(snapshots: list[Path], games: int, out_json: Path,
                    seed: int = 0, device: str = "mps") -> dict:
    """Call scripts/eval/ladder.py with the given snapshots; return its JSON."""
    cmd = [
        ".venv/bin/python", "scripts/eval/ladder.py",
        "--snapshots", *[str(s) for s in snapshots],
        "--games", str(games),
        "--seed", str(seed),
        "--device", device,
        "--out", str(out_json),
    ]
    subprocess.run(
        cmd, cwd=REPO,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )
    return json.loads(out_json.read_text())


def plot_elo_curve(ladder: dict, out_path: Path, label: str = "") -> None:
    """Plot Elo by snapshot stem (assumes stems sort naturally, which
    they do for snapshot_NNNNN format)."""
    rows = sorted(
        [
            (name, r["rating"], r["games"])
            for name, r in ladder["ratings"].items()
            if name.startswith("snapshot_") or name.startswith("az_")
        ],
        key=lambda x: x[0],
    )
    if not rows:
        return
    names = [r[0] for r in rows]
    elos = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(rows)), 4))
    ax.plot(range(len(rows)), elos, marker="o", color="#1f77b4")
    ax.axhline(1000, color="#999", linestyle="--", label="Random=1000")
    ax.axhline(1200, color="#666", linestyle="--", label="Greedy=1200")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Elo")
    ax.set_title(f"Strength progression  {label}".strip(), fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_report(ladder: dict, label: str) -> str:
    lines: list[str] = []
    lines.append(f"# Ladder: {label or '(unlabeled)'}\n")

    ratings = sorted(ladder["ratings"].items(), key=lambda x: -x[1]["rating"])
    lines.append("## Final Elo ranking\n")
    lines.append("| rank | name | Elo | games | record |")
    lines.append("|---:|---|---:|---:|---|")
    for i, (name, r) in enumerate(ratings, 1):
        rec = f"{r.get('wins', 0)}-{r.get('losses', 0)}-{r.get('draws', 0)}"
        lines.append(f"| {i} | {name} | {r['rating']:.1f} | {r['games']} | {rec} |")
    lines.append("")

    if "matchups" in ladder:
        lines.append("## Head-to-head WR (row beats column)\n")
        names = [n for n, _ in ratings]
        lines.append("| " + " | ".join(["model"] + names) + " |")
        lines.append("|" + "|".join(["---"] * (len(names) + 1)) + "|")
        for n in names:
            cells = [n]
            for m in names:
                if n == m:
                    cells.append("—")
                else:
                    k = f"{n} vs {m}"
                    if k in ladder["matchups"]:
                        wr = ladder["matchups"][k]
                        cells.append(f"{wr:.0%}")
                    else:
                        cells.append("")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("![Elo curve](elo_curve.png)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None,
                    help="output dir; default = analysis/ladder_<first stem>")
    args = ap.parse_args()

    snapshots = [Path(s) for s in args.snapshots]
    out_dir = Path(args.out) if args.out else (REPO / "analysis" /
                                                f"ladder_{snapshots[0].stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "ladder.json"

    print(f"Running round-robin: {len(snapshots)} snapshots × {args.games} games each pair")
    ladder = run_eval_ladder(snapshots, args.games, out_json,
                             seed=args.seed, device=args.device)

    plot_elo_curve(ladder, out_dir / "elo_curve.png", label=args.label)
    report = render_report(ladder, args.label)
    write_md(out_dir / "ladder.md", report)
    print(f"\nWrote {out_dir / 'ladder.md'}")
    print(f"      {out_dir / 'elo_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
