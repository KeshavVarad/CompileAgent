"""Run the strategic-shift breakdown on an AZ training checkpoint.

For a given snapshot, plays 100 games vs Greedy and 100 games vs the
source checkpoint (the AZ run's hot-start), then summarises the strategy
shift relative to the source: vs-greedy WR, head-to-head WR vs source,
and per-protocol pick deltas.

Output lands at `<ckpt_dir>/diagnostic_<stem>/{vs_greedy.jsonl,
vs_source.jsonl, report.md}`. Skips work if `report.md` already exists,
so it's safe to call repeatedly (e.g. from a watcher loop).

Usage:
    python scripts/az/diagnose.py runs/.../snapshot_00020.pt \\
        --source runs/latest/distill/.../snapshot_00500_distilled.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def collect(model: Path, opp: str, out: Path, *, games: int = 100, seed: int = 0,
            device: str = "mps") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ".venv/bin/python", "scripts/eval/collect.py",
            "--model", str(model), "--opp", opp,
            "--games", str(games), "--seed", str(seed),
            "--device", device,
            "--out", str(out),
        ],
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        check=True,
    )


def outcome(r: dict) -> str:
    w = r.get("winner")
    if w is None:
        return "draw"
    return "win" if w == r["agent_seat"] else "loss"


def summarise(jsonl: Path) -> dict:
    """Return summary stats + per-protocol pick counter from a jsonl."""
    recs = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    n = len(recs)
    if n == 0:
        return {"n_games": 0}
    results = [outcome(r) for r in recs]
    w = results.count("win"); l = results.count("loss"); d = results.count("draw")
    seat0 = [outcome(r) for r in recs if r["agent_seat"] == 0]
    seat1 = [outcome(r) for r in recs if r["agent_seat"] == 1]
    picks: Counter[str] = Counter()
    for r in recs:
        for p in r[f"protocols_p{r['agent_seat']}"]:
            picks[p] += 1
    return {
        "n_games": n,
        "wr": w / n,
        "wins": w, "losses": l, "draws": d,
        "wr_seat0": seat0.count("win") / max(1, len(seat0)),
        "wr_seat1": seat1.count("win") / max(1, len(seat1)),
        "picks": dict(picks),
        "n_picks": sum(picks.values()),
    }


def render_report(
    ckpt: Path,
    source: Path | None,
    vs_greedy: dict,
    vs_source: dict | None,
    source_vs_greedy: dict | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Diagnostic: {ckpt.name}\n")
    lines.append(f"- **Checkpoint**: `{ckpt}`")
    if source:
        lines.append(f"- **Source (baseline)**: `{source}`")
    lines.append("")
    lines.append("## Headline results\n")

    def fmt_wr(s: dict) -> str:
        return f"{s['wr']:.1%}  ({s['wins']}-{s['losses']}-{s['draws']}, n={s['n_games']})"

    lines.append(f"- **vs Greedy**: {fmt_wr(vs_greedy)}")
    lines.append(f"  - seat 0: {vs_greedy['wr_seat0']:.1%}   seat 1: {vs_greedy['wr_seat1']:.1%}")
    if vs_source:
        lines.append(f"- **vs Source**: {fmt_wr(vs_source)}")
        lines.append(f"  - seat 0: {vs_source['wr_seat0']:.1%}   seat 1: {vs_source['wr_seat1']:.1%}")
    if source_vs_greedy:
        lines.append(f"- **(Source vs Greedy, for comparison)**: {fmt_wr(source_vs_greedy)}")
    lines.append("")

    # Protocol picks side by side.
    lines.append("## Protocol pick distribution (vs Greedy)\n")
    az_picks = vs_greedy["picks"]
    n_az = max(1, vs_greedy["n_picks"])
    src_picks = source_vs_greedy["picks"] if source_vs_greedy else {}
    n_src = max(1, source_vs_greedy["n_picks"]) if source_vs_greedy else 1

    all_protos = sorted(set(az_picks) | set(src_picks),
                        key=lambda p: -(az_picks.get(p, 0) + src_picks.get(p, 0)))

    lines.append("| protocol | ckpt % | source % | Δ |")
    lines.append("|---|---:|---:|---:|")
    for p in all_protos:
        a = az_picks.get(p, 0) / n_az * 100
        s = src_picks.get(p, 0) / n_src * 100
        if max(a, s) < 0.5:
            continue
        lines.append(f"| {p} | {a:.1f}% | {s:.1f}% | {a - s:+.1f} |")

    lines.append("")
    lines.append(f"- Unique protocols drafted: **{len(az_picks)}** (source: {len(src_picks)})")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="AZ snapshot to diagnose")
    ap.add_argument("--source", required=True, help="source checkpoint for comparison")
    ap.add_argument("--source-vs-greedy",
                    help="optional: pre-existing source-vs-greedy jsonl to avoid re-running it")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if report.md exists")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    source = Path(args.source)
    out_dir = ckpt.parent / f"diagnostic_{ckpt.stem}"
    report_path = out_dir / "report.md"

    if report_path.exists() and not args.force:
        print(f"[skip] {report_path} exists; pass --force to re-run")
        print(report_path.read_text())
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    vs_greedy_path = out_dir / "vs_greedy.jsonl"
    vs_source_path = out_dir / "vs_source.jsonl"

    if not vs_greedy_path.exists():
        print(f"[1/3] collect {ckpt.stem} vs greedy ({args.games} games)")
        collect(ckpt, "greedy", vs_greedy_path,
                games=args.games, seed=args.seed, device=args.device)
    else:
        print(f"[1/3] reuse {vs_greedy_path}")

    if not vs_source_path.exists():
        print(f"[2/3] collect {ckpt.stem} vs source ({args.games} games)")
        collect(ckpt, str(source), vs_source_path,
                games=args.games, seed=args.seed, device=args.device)
    else:
        print(f"[2/3] reuse {vs_source_path}")

    src_vs_g_path: Path | None
    if args.source_vs_greedy:
        src_vs_g_path = Path(args.source_vs_greedy)
    else:
        # Cache source-vs-greedy under the source dir so subsequent
        # diagnose runs reuse it.
        src_vs_g_path = source.parent / f"diagnostic_{source.stem}__vs_greedy.jsonl"
        if not src_vs_g_path.exists():
            print(f"[3/3] collect source vs greedy ({args.games} games) → cached")
            collect(source, "greedy", src_vs_g_path,
                    games=args.games, seed=args.seed, device=args.device)
        else:
            print(f"[3/3] reuse {src_vs_g_path}")

    vs_greedy = summarise(vs_greedy_path)
    vs_source = summarise(vs_source_path)
    src_vs_greedy = summarise(src_vs_g_path) if src_vs_g_path and src_vs_g_path.exists() else None

    report = render_report(ckpt, source, vs_greedy, vs_source, src_vs_greedy)
    report_path.write_text(report)
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
