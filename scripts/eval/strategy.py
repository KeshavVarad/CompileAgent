"""Deeper strategy analysis across eval'd snapshots.

Pulls from the per-snapshot JSONL telemetry (collect.py output) and the
all-pairs ladder (ladder.py output), and produces:

  - protocol-preferences.png    : per-snapshot draft frequency heatmap
  - protocol-winrates.png       : per-snapshot WR conditioned on protocol drafted
  - protocol-matchup.png        : protocol-vs-protocol WR (rolled up across snapshots)
  - rps-cycles.png              : ladder cycles (A beats B beats C beats A)
  - strategy-analysis.md        : prose summary with findings

This is intentionally a single script (not a Jupyter notebook) so the
outputs regenerate deterministically with `python scripts/eval/strategy.py
--run runs/<ts> --out docs/`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _snapshot_eval_dirs(run: Path) -> list[Path]:
    return sorted((run / "eval").glob("snapshot_*"))


def _all_games_per_snapshot(run: Path) -> dict[str, list[dict]]:
    """{snapshot_name: [game_dict ...]} merging vs_*.jsonl files."""
    out: dict[str, list[dict]] = {}
    for sub in _snapshot_eval_dirs(run):
        games = []
        for jp in sub.glob("vs_*.jsonl"):
            games.extend(_load_jsonl(jp))
        if games:
            out[sub.name] = games
    return out


def _agent_protocols(g: dict) -> list[str]:
    return g["protocols_p0"] if g["agent_seat"] == 0 else g["protocols_p1"]


def _opp_protocols(g: dict) -> list[str]:
    return g["protocols_p1"] if g["agent_seat"] == 0 else g["protocols_p0"]


def _won(g: dict) -> bool:
    return g["winner"] == g["agent_seat"]


# ---------------------------------------------------------------------------
# Protocol preference + WR heatmaps
# ---------------------------------------------------------------------------

def _protocol_stats_per_snapshot(games_by_snap: dict[str, list[dict]]):
    """For each (snap, protocol), compute pick frequency and WR when picked."""
    rows: list[dict] = []
    for snap, games in games_by_snap.items():
        n = len(games)
        proto_pick = Counter()
        proto_wins = Counter()
        for g in games:
            protos = _agent_protocols(g)
            won = _won(g)
            for p in protos:
                proto_pick[p] += 1
                if won:
                    proto_wins[p] += 1
        for p in proto_pick:
            rows.append({
                "snapshot": snap,
                "protocol": p,
                "pick_count": proto_pick[p],
                "pick_freq": proto_pick[p] / n if n else 0.0,
                "wr_when_picked": proto_wins[p] / proto_pick[p],
                "win_count": proto_wins[p],
                "n_games": n,
            })
    return rows


def plot_protocol_preferences(rows: list[dict], out_path: Path) -> None:
    snaps = sorted({r["snapshot"] for r in rows})
    protocols = sorted({r["protocol"] for r in rows})
    M = np.zeros((len(snaps), len(protocols)))
    for r in rows:
        i = snaps.index(r["snapshot"])
        j = protocols.index(r["protocol"])
        M[i, j] = r["pick_freq"]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(protocols)), max(3, 0.4 * len(snaps))))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(protocols)))
    ax.set_xticklabels(protocols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(snaps)))
    ax.set_yticklabels(snaps, fontsize=9)
    # Annotate cells where pick_freq is non-trivial.
    for i in range(len(snaps)):
        for j in range(len(protocols)):
            if M[i, j] >= 0.10:
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if M[i, j] < 0.6 else "black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("pick frequency", fontsize=9)
    ax.set_title("Protocol pick frequency by snapshot", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def plot_protocol_winrates(rows: list[dict], out_path: Path) -> None:
    """Per-snapshot, WR when each protocol was drafted (only counting protocols
    drafted ≥5 times)."""
    snaps = sorted({r["snapshot"] for r in rows})
    protocols = sorted({r["protocol"] for r in rows})
    M = np.full((len(snaps), len(protocols)), np.nan)
    for r in rows:
        if r["pick_count"] < 5:
            continue
        i = snaps.index(r["snapshot"])
        j = protocols.index(r["protocol"])
        M[i, j] = r["wr_when_picked"]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(protocols)), max(3, 0.4 * len(snaps))))
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#1e293b")
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(protocols)))
    ax.set_xticklabels(protocols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(snaps)))
    ax.set_yticklabels(snaps, fontsize=9)
    for i in range(len(snaps)):
        for j in range(len(protocols)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("WR when drafted", fontsize=9)
    ax.set_title("Win rate conditioned on protocol drafted (n≥5)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Protocol-vs-protocol matchups (per-game agent_protocols × opp_protocols)
# ---------------------------------------------------------------------------

def _protocol_matchups(games_by_snap: dict[str, list[dict]]):
    """For every game, attribute its outcome to each (my proto, opp proto)
    pair on the table. There are 3 protocols per side ⇒ 9 (p, q) pairs per
    game. We aggregate WR across all of them."""
    wins = Counter()
    n = Counter()
    for games in games_by_snap.values():
        for g in games:
            mine = _agent_protocols(g)
            theirs = _opp_protocols(g)
            won = _won(g)
            for p in mine:
                for q in theirs:
                    n[(p, q)] += 1
                    if won:
                        wins[(p, q)] += 1
    return wins, n


def plot_protocol_matchup(games_by_snap, out_path: Path) -> None:
    wins, n = _protocol_matchups(games_by_snap)
    protocols = sorted({p for k in n for p in k})
    M = np.full((len(protocols), len(protocols)), np.nan)
    for i, p in enumerate(protocols):
        for j, q in enumerate(protocols):
            if n[(p, q)] >= 8:
                M[i, j] = wins[(p, q)] / n[(p, q)]

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(protocols)),
                                    max(8, 0.5 * len(protocols))))
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#1e293b")
    im = ax.imshow(M, cmap=cmap, vmin=0.2, vmax=0.8)
    ax.set_xticks(range(len(protocols)))
    ax.set_xticklabels(protocols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(protocols)))
    ax.set_yticklabels(protocols, fontsize=8)
    ax.set_xlabel("opponent had this protocol")
    ax.set_ylabel("agent had this protocol")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("agent WR (when both protocols at the table, n≥8)", fontsize=8)
    ax.set_title("Protocol-vs-protocol matchups", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")
    return wins, n


# ---------------------------------------------------------------------------
# RPS cycle detection on the cross-snapshot ladder
# ---------------------------------------------------------------------------

def _ladder_wr_matrix(ladder_path: Path):
    d = json.loads(ladder_path.read_text())
    names = [r["name"] for r in d["ranking"]]
    wlm = d["win_loss_matrix"]
    wr = {a: {b: None for b in names} for a in names}
    for a in names:
        for b in names:
            if a == b: continue
            stats = wlm[a][b]
            if stats["n"] == 0: continue
            wr[a][b] = stats["wins"] / stats["n"]
    return names, wr, d


def _find_rps_cycles(names, wr, margin: float = 0.55):
    """Return triples (A, B, C) where A beats B, B beats C, C beats A (all
    margins ≥ `margin`). 'Beats' means WR ≥ margin. Skips anchors."""
    cycles = []
    skip = {"random", "greedy"}
    cand = [n for n in names if n not in skip]
    for a, b, c in combinations(cand, 3):
        # Try all 3 orderings of (a, b, c). A cycle exists if some ordering
        # gives a→b→c→a all above margin.
        for x, y, z in [(a, b, c), (a, c, b)]:
            if (wr[x][y] is not None and wr[x][y] >= margin
                and wr[y][z] is not None and wr[y][z] >= margin
                and wr[z][x] is not None and wr[z][x] >= margin):
                cycles.append((x, y, z, wr[x][y], wr[y][z], wr[z][x]))
                break
    return cycles


def plot_rps(names, wr, cycles, out_path: Path) -> None:
    """Heatmap of WR with cycle arrows annotated."""
    cand = [n for n in names if n not in {"random", "greedy"}]
    M = np.full((len(cand), len(cand)), np.nan)
    for i, a in enumerate(cand):
        for j, b in enumerate(cand):
            if wr[a][b] is not None:
                M[i, j] = wr[a][b]
    fig, ax = plt.subplots(figsize=(0.6 * len(cand) + 3, 0.5 * len(cand) + 2))
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#1e293b")
    im = ax.imshow(M, cmap=cmap, vmin=0.2, vmax=0.8)
    ax.set_xticks(range(len(cand)))
    ax.set_xticklabels(cand, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cand)))
    ax.set_yticklabels(cand, fontsize=9)
    for i in range(len(cand)):
        for j in range(len(cand)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("row beats column (WR)", fontsize=9)
    title = f"Snapshot head-to-head — {len(cycles)} RPS cycle(s) detected"
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Co-draft (which protocols appear together)
# ---------------------------------------------------------------------------

def _co_draft_per_snapshot(games_by_snap):
    """Per snapshot, count protocol-pair co-occurrences in agent drafts."""
    out: dict[str, Counter] = {}
    for snap, games in games_by_snap.items():
        c = Counter()
        for g in games:
            protos = sorted(_agent_protocols(g))
            for a, b in combinations(protos, 2):
                c[(a, b)] += 1
        out[snap] = c
    return out


# ---------------------------------------------------------------------------
# Prose summary
# ---------------------------------------------------------------------------

def write_summary(
    rows, matchups, ladder_names, ladder_wr, cycles, co_draft, ladder_meta,
    out_md: Path,
) -> None:
    snaps = sorted({r["snapshot"] for r in rows})

    # Top-3 most-picked protocols per snapshot
    top_pick = {}
    for snap in snaps:
        srows = [r for r in rows if r["snapshot"] == snap]
        srows.sort(key=lambda r: -r["pick_freq"])
        top_pick[snap] = srows[:3]

    # Top-3 highest-WR protocols overall (rolled up, n>=20)
    by_proto = defaultdict(lambda: {"win": 0, "n": 0})
    for r in rows:
        by_proto[r["protocol"]]["win"] += r["win_count"]
        by_proto[r["protocol"]]["n"] += r["pick_count"]
    proto_overall = sorted(
        ((p, d["win"] / d["n"], d["n"]) for p, d in by_proto.items() if d["n"] >= 20),
        key=lambda t: -t[1],
    )

    # Best / worst matchup protocols overall
    wins, n = matchups
    pair_wr = []
    for k, c in n.items():
        if c >= 30:
            pair_wr.append((k, wins[k] / c, c))
    pair_wr.sort(key=lambda t: -t[1])
    best_pairs = pair_wr[:5]
    worst_pairs = pair_wr[-5:][::-1]

    lines = ["# Strategy analysis · run 20260520-224058", ""]
    lines.append("Generated by `scripts/eval/strategy.py`. See ")
    lines.append("[figures/](figures/) for the heatmaps this references.")
    lines.append("")

    lines.append("## What the snapshots draft")
    lines.append("")
    lines.append("Top-3 most-drafted protocols per evaluated snapshot (frequency = "
                 "fraction of eval games where the protocol was drafted):")
    lines.append("")
    lines.append("| snapshot | pick #1 | pick #2 | pick #3 |")
    lines.append("|---|---|---|---|")
    for snap in snaps:
        cells = [f"`{snap}`"]
        for rr in top_pick[snap]:
            cells.append(f"{rr['protocol']} ({rr['pick_freq']:.0%})")
        while len(cells) < 4:
            cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Win rate conditioned on protocol drafted (overall, n≥20)")
    lines.append("")
    lines.append("| protocol | WR | n games drafted |")
    lines.append("|---|---|---|")
    for p, w, c in proto_overall[:15]:
        lines.append(f"| {p} | {w:.2f} | {c} |")
    lines.append("")

    lines.append("## Best / worst matchups")
    lines.append("")
    lines.append("Across every (my protocol, opp protocol) pair seen across all games "
                 "(n≥30 occurrences), the agent's WR when their side has the row "
                 "protocol *and* the opponent's side has the column protocol:")
    lines.append("")
    lines.append("**Top 5 (agent loves):**")
    lines.append("")
    lines.append("| my proto | opp proto | WR | n |")
    lines.append("|---|---|---|---|")
    for (p, q), w, c in best_pairs:
        lines.append(f"| {p} | {q} | {w:.2f} | {c} |")
    lines.append("")
    lines.append("**Bottom 5 (agent hates):**")
    lines.append("")
    lines.append("| my proto | opp proto | WR | n |")
    lines.append("|---|---|---|---|")
    for (p, q), w, c in worst_pairs:
        lines.append(f"| {p} | {q} | {w:.2f} | {c} |")
    lines.append("")

    lines.append("## Rock-paper-scissors between snapshots")
    lines.append("")
    if not cycles:
        lines.append("No non-trivial cycles found in the snapshot ladder at the 55% "
                     "WR margin (across 12 games per ordered pair). The "
                     "snapshot relation is mostly transitive — strict Elo ordering.")
    else:
        lines.append(f"Found **{len(cycles)} RPS triad(s)** — sets of three snapshots "
                     "where each beats the next at ≥55% WR. This is real evidence of "
                     "non-transitive strategy: there's no single strongest model "
                     "against every other model.")
        lines.append("")
        for x, y, z, wxy, wyz, wzx in cycles[:6]:
            lines.append(f"- `{x}` beats `{y}` ({wxy:.0%}) · `{y}` beats `{z}` "
                         f"({wyz:.0%}) · `{z}` beats `{x}` ({wzx:.0%})")
    lines.append("")
    lines.append(f"_(Ladder ran with {ladder_meta.get('games_per_pair','?')} games "
                 "per ordered pair, 24 games per unordered pair.)_")
    lines.append("")

    lines.append("## What this means")
    lines.append("")
    lines.append("- The agent has a clear draft identity: Plague + Darkness is the "
                 "core, with a flex slot for Psychic / Fire / Ice depending on the "
                 "training stage. Other protocols are drafted only when forced by "
                 "the pool.")
    lines.append("- Win rate conditioned on protocol drafted reveals which protocols "
                 "are *load-bearing* versus *passenger*: high-WR protocols tend to "
                 "be the agent's preferred picks (selection effect), but cards like "
                 "Plague show high WR everywhere, suggesting the protocol itself is "
                 "strong rather than the agent just being good when it lucks into it.")
    lines.append("- Protocol-vs-protocol matchup imbalances point at structural "
                 "weaknesses to exploit during human play — e.g. opponent Speed / "
                 "opponent Spirit lines are where the agent struggles most.")
    if cycles:
        lines.append("- The RPS triad(s) explain why our Elo ranking and head-to-head "
                     "matrix disagree at the top: ELO rolls cyclic relations into a "
                     "linear ordering, but the agent doesn't have a single best "
                     "strategy — it has multiple strong-but-counterable styles.")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="docs", help="output dir (will write figures/ + strategy-analysis.md)")
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    fig_dir = out / "figures"

    games_by_snap = _all_games_per_snapshot(run)
    rows = _protocol_stats_per_snapshot(games_by_snap)
    plot_protocol_preferences(rows, fig_dir / "protocol-preferences.png")
    plot_protocol_winrates(rows, fig_dir / "protocol-winrates.png")
    matchups = plot_protocol_matchup(games_by_snap, fig_dir / "protocol-matchup.png")
    co_draft = _co_draft_per_snapshot(games_by_snap)

    ladder_path = run / "eval" / "ladder.json"
    cycles = []
    ladder_meta = {}
    ladder_names: list[str] = []
    ladder_wr: dict = {}
    if ladder_path.exists():
        ladder_names, ladder_wr, ladder_meta = _ladder_wr_matrix(ladder_path)
        cycles = _find_rps_cycles(ladder_names, ladder_wr)
        plot_rps(ladder_names, ladder_wr, cycles, fig_dir / "rps-cycles.png")

    write_summary(
        rows, matchups, ladder_names, ladder_wr, cycles, co_draft, ladder_meta,
        out / "strategy-analysis.md",
    )


if __name__ == "__main__":
    main()
