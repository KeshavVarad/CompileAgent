"""Master 'strategic discoveries' report for a Compile checkpoint.

Runs the full analysis suite (protocol meta, style fingerprint, optional
decision diff vs a baseline) and assembles the outputs into a single
top-level markdown report linking to each section. This is the artifact
you'd publish on GitHub or paste into a blog post.

Usage:
    # Bare report for one checkpoint
    python scripts/analysis/discoveries.py runs/.../snapshot_00500.pt

    # Compare against a baseline checkpoint
    python scripts/analysis/discoveries.py runs/.../snapshot_00500.pt \\
        --baseline runs/.../snapshot_00100.pt

    # Include a ladder of multiple checkpoints
    python scripts/analysis/discoveries.py runs/.../snapshot_00500.pt \\
        --ladder runs/.../snapshot_00100.pt runs/.../snapshot_00300.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import REPO, analysis_dir, write_md  # noqa: E402


def run(name: str, cmd: list[str], cwd: Path = REPO) -> bool:
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    result = subprocess.run(
        cmd, cwd=cwd,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    if result.returncode != 0:
        print(f"  WARN: {name} exited {result.returncode}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--baseline", default=None,
                    help="optional baseline checkpoint for decision-diff")
    ap.add_argument("--ladder", nargs="+", default=None,
                    help="optional additional snapshots to include in a ladder")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--ladder-games", type=int, default=30)
    ap.add_argument("--diff-games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out_dir = analysis_dir(ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Protocol meta.
    run("protocol_meta",
        [".venv/bin/python", "scripts/analysis/protocol_meta.py",
         str(ckpt),
         "--games", str(args.games),
         "--seed", str(args.seed),
         "--device", args.device])

    # 2. Style fingerprint (just this checkpoint; for comparison use baseline).
    fp_args = [str(ckpt)]
    if args.baseline:
        fp_args.append(args.baseline)
    run("style_fingerprint",
        [".venv/bin/python", "scripts/analysis/style_fingerprint.py",
         *fp_args,
         "--games", str(args.games),
         "--seed", str(args.seed),
         "--device", args.device])

    # 3. Decision diff (if baseline given).
    diff_section: str | None = None
    if args.baseline:
        ok = run("decision_diff",
                 [".venv/bin/python", "scripts/analysis/decision_diff.py",
                  "--a", str(ckpt),
                  "--b", args.baseline,
                  "--games", str(args.diff_games),
                  "--seed", str(args.seed),
                  "--device", args.device])
        if ok:
            diff_section = f"decision_diff_vs_{Path(args.baseline).stem}/decision_diff.md"

    # 4. Ladder (if extra snapshots given).
    ladder_section: str | None = None
    if args.ladder:
        snapshots = [str(ckpt), *args.ladder]
        ok = run("ladder",
                 [".venv/bin/python", "scripts/analysis/ladder.py",
                  "--snapshots", *snapshots,
                  "--games", str(args.ladder_games),
                  "--label", f"around {ckpt.stem}",
                  "--seed", str(args.seed),
                  "--device", args.device,
                  "--out", str(out_dir / "ladder")])
        if ok:
            ladder_section = "ladder/ladder.md"

    # 5. Assemble master report.
    lines: list[str] = []
    lines.append(f"# Strategic Discoveries — `{ckpt.stem}`\n")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")
    lines.append(f"**Checkpoint:** `{ckpt}`")
    if args.baseline:
        lines.append(f"**Baseline:** `{args.baseline}`")
    lines.append("")
    lines.append("This report bundles the standard strategic-analysis suite "
                 "for a single Compile checkpoint. Each section links to a "
                 "self-contained markdown file with details, tables, and "
                 "charts.\n")

    lines.append("## Sections\n")
    lines.append("- [Protocol meta](protocol_meta.md) — pick frequency, "
                 "per-protocol WR, top synergies, pair heatmap")
    lines.append("- [Playstyle fingerprint](style_fingerprint.md) — "
                 "face-up ratio, refresh rate, compile aggression, radar chart")
    if diff_section:
        lines.append(f"- [Decision diff vs baseline]({diff_section}) — "
                     "where the new model picks different actions than the baseline")
    if ladder_section:
        lines.append(f"- [Ladder]({ladder_section}) — Elo ranking across the "
                     "tracked checkpoints")
    lines.append("")
    lines.append("## How to reproduce\n")
    lines.append("```")
    lines.append(" ".join([
        ".venv/bin/python", "scripts/analysis/discoveries.py",
        str(ckpt),
        *(["--baseline", args.baseline] if args.baseline else []),
        *(["--ladder", *args.ladder] if args.ladder else []),
        "--games", str(args.games),
    ]))
    lines.append("```")
    lines.append("")
    lines.append(
        f"Eval volume: {args.games} games per opponent for headline stats "
        f"({args.diff_games} for decision-diff, "
        f"{args.ladder_games} per pair for ladder)."
    )
    lines.append("")

    write_md(out_dir / "discoveries.md", "\n".join(lines))
    print(f"\n=== Done ===")
    print(f"Master report: {out_dir / 'discoveries.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
