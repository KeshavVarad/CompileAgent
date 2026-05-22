"""Protocol-level meta analysis of a Compile checkpoint.

Given a checkpoint, plays N games vs Greedy + Random, then reports:
  * Per-protocol pick frequency (how often it drafted each)
  * Per-protocol WR (with Wilson CIs)
  * Per-pair WR (which protocol pairs perform above/below average)
  * Top protocol pairs sorted by lift above average WR

Produces a markdown report + JSON sidecar + PNG heatmap of pair WRs.

Usage:
    python scripts/analysis/protocol_meta.py runs/.../snapshot_00100.pt
    python scripts/analysis/protocol_meta.py runs/.../snapshot_00100.pt --games 300
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
    fmt_wr,
    per_pair_wr,
    per_protocol_picks,
    per_protocol_wr,
    read_games,
    summary,
    wilson_ci,
    write_json,
    write_md,
)


def plot_pair_heatmap(pair_wr: dict, out_path: Path, *, min_n: int = 3) -> int:
    """Draw a square heatmap of pair WRs. Cells with fewer than `min_n`
    games are masked to grey. Returns the number of cells plotted."""
    # Get all protocols appearing in any pair.
    protos: set[str] = set()
    for (a, b) in pair_wr:
        protos.add(a); protos.add(b)
    if len(protos) < 2:
        return 0
    protos_sorted = sorted(protos)

    mat = np.full((len(protos_sorted), len(protos_sorted)), np.nan)
    n_mat = np.zeros_like(mat)
    for (a, b), (w, n) in pair_wr.items():
        if n < min_n:
            continue
        i = protos_sorted.index(a); j = protos_sorted.index(b)
        mat[i, j] = w / n
        mat[j, i] = w / n
        n_mat[i, j] = n
        n_mat[j, i] = n

    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(protos_sorted)),
                                     max(6, 0.3 * len(protos_sorted))))
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#222")
    im = ax.imshow(masked, vmin=0.3, vmax=0.85, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(protos_sorted)))
    ax.set_yticks(range(len(protos_sorted)))
    ax.set_xticklabels(protos_sorted, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(protos_sorted, fontsize=8)
    ax.set_title(f"Protocol-pair WR  (min n={min_n})", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="win rate")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return int(np.count_nonzero(~np.isnan(mat)))


def render_report(
    ckpt: Path,
    vs_greedy_recs: list[dict],
    vs_random_recs: list[dict],
    n_pair_cells: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# Protocol Meta — `{ckpt.stem}`\n")
    lines.append(f"**Checkpoint:** `{ckpt}`")
    lines.append("")

    g_sum = summary(vs_greedy_recs)
    r_sum = summary(vs_random_recs)
    lines.append("## Headline\n")
    lines.append(
        f"- vs Greedy: **{g_sum.wr:.1%}**  "
        f"({g_sum.wins}-{g_sum.losses}-{g_sum.draws}, n={g_sum.n_games})"
    )
    lines.append(
        f"- vs Random: **{r_sum.wr:.1%}**  "
        f"({r_sum.wins}-{r_sum.losses}-{r_sum.draws}, n={r_sum.n_games})"
    )
    lines.append("")

    # Per-protocol against greedy (greedy is the more informative opponent).
    picks = per_protocol_picks(vs_greedy_recs)
    proto_wr = per_protocol_wr(vs_greedy_recs)
    total_picks = sum(picks.values())

    lines.append("## Protocol pick frequency + WR (vs Greedy)\n")
    lines.append("Sorted by pick rate. WR is win-rate of games where the agent drafted that protocol.\n")
    lines.append("| protocol | pick rate | drafted (n) | WR ± 95% CI |")
    lines.append("|---|---:|---:|---:|")
    for p, n_p in picks.most_common():
        wins, n = proto_wr[p]
        lines.append(
            f"| {p} | {n_p / total_picks:.1%} | {n_p} | {fmt_wr(wins, n)} |"
        )
    lines.append("")

    # Pair WRs — sorted by lift.
    pair_wr = per_pair_wr(vs_greedy_recs)
    baseline_wr = g_sum.wr
    pair_rows: list[tuple[tuple[str, str], int, int, float]] = []
    for (a, b), (w, n) in pair_wr.items():
        if n < 5:
            continue
        pair_rows.append(((a, b), w, n, w / n - baseline_wr))
    pair_rows.sort(key=lambda x: -x[3])

    lines.append("## Top synergistic protocol pairs\n")
    lines.append(f"Pairs with at least 5 games. Lift = pair WR − baseline WR ({baseline_wr:.1%}).\n")
    lines.append("| pair | n | pair WR | lift |")
    lines.append("|---|---:|---:|---:|")
    for (a, b), w, n, lift in pair_rows[:15]:
        lines.append(f"| {a} + {b} | {n} | {w / n:.1%} | {lift:+.1%} |")
    lines.append("")

    if pair_rows:
        lines.append("## Worst protocol pairs\n")
        lines.append("Same gate (n≥5). Lift below baseline = weak pairing.\n")
        lines.append("| pair | n | pair WR | lift |")
        lines.append("|---|---:|---:|---:|")
        for (a, b), w, n, lift in pair_rows[-10:][::-1]:
            lines.append(f"| {a} + {b} | {n} | {w / n:.1%} | {lift:+.1%} |")
        lines.append("")

    lines.append("## Diversity stats\n")
    lines.append(f"- Unique protocols drafted: **{len(picks)}** / 30")
    if picks:
        most = picks.most_common(1)[0]
        lines.append(f"- Most-picked: **{most[0]}** ({most[1] / total_picks:.1%})")
        # Concentration (Herfindahl-Hirschman over protocols).
        hhi = sum((c / total_picks) ** 2 for c in picks.values())
        lines.append(f"- Pick concentration (HHI): **{hhi:.3f}**  (0=uniform across 30, 0.033=baseline; 1=monomania)")
    lines.append(f"- Pair heatmap: `pair_heatmap.png` ({n_pair_cells} cells with n≥3)")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out_dir = analysis_dir(ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)

    vs_greedy_path = out_dir / "vs_greedy.jsonl"
    vs_random_path = out_dir / "vs_random.jsonl"

    print(f"[1/3] eval vs greedy ({args.games} games)")
    ensure_collected(ckpt, "greedy", vs_greedy_path,
                     games=args.games, seed=args.seed, device=args.device)
    print(f"[2/3] eval vs random ({args.games} games)")
    ensure_collected(ckpt, "random", vs_random_path,
                     games=args.games, seed=args.seed, device=args.device)

    vs_greedy = read_games(vs_greedy_path)
    vs_random = read_games(vs_random_path)

    print(f"[3/3] render report + heatmap")
    pair_wr = per_pair_wr(vs_greedy)
    n_cells = plot_pair_heatmap(pair_wr, out_dir / "pair_heatmap.png")

    report = render_report(ckpt, vs_greedy, vs_random, n_cells)
    write_md(out_dir / "protocol_meta.md", report)

    # Sidecar JSON for downstream tooling.
    sidecar = {
        "ckpt": str(ckpt),
        "n_games_per_opp": args.games,
        "vs_greedy_wr": summary(vs_greedy).wr,
        "vs_random_wr": summary(vs_random).wr,
        "protocol_picks": dict(per_protocol_picks(vs_greedy)),
        "protocol_wr": {p: list(v) for p, v in per_protocol_wr(vs_greedy).items()},
        "pair_wr": {f"{a}|{b}": list(v) for (a, b), v in pair_wr.items()},
    }
    write_json(out_dir / "protocol_meta.json", sidecar)
    print(f"\nWrote {out_dir / 'protocol_meta.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
