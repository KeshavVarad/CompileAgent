"""Render training-progress and eval figures from a training run + its
eval/ directory. Output PNGs suitable for embedding in the README.

The plots are intentionally simple so they don't depend on seaborn / fancy
themes — pure matplotlib so anyone with a vanilla install can reproduce.

Usage:
    python scripts/eval/plot.py --run runs/20260520-224058 --out docs/figures/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Visual constants — keep the palette small + readable on dark and light README.
RANDOM_COLOR = "#94a3b8"   # slate-400
GREEDY_COLOR = "#fb923c"   # orange-400
NN_COLOR = "#34d399"       # emerald-400
LINE_W = 2.0


def _read_metrics_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_eval_curve_from_log(path: Path) -> dict[int, dict[str, float]]:
    """Parse `[iter NNN] eval: vs random=X vs greedy=Y` lines out of the
    train.log. Returns {iter: {"random": ..., "greedy": ...}}."""
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    rx = re.compile(r"\[iter\s+(\d+)\] eval: vs random=([\d.]+) vs greedy=([\d.]+)")
    for line in path.read_text().splitlines():
        m = rx.search(line)
        if m:
            out[int(m.group(1))] = {
                "random": float(m.group(2)),
                "greedy": float(m.group(3)),
            }
    return out


def plot_training_progress(run_dir: Path, out_path: Path) -> None:
    """Combined loss + win-rate figure. Two stacked subplots sharing the x-axis.

    Top: policy gradient loss, value loss, entropy over PPO iterations.
    Bottom: training-log eval win-rate vs random and greedy at snapshot iters.
    """
    metrics = _read_metrics_jsonl(run_dir / "metrics.jsonl")
    eval_curve = _read_eval_curve_from_log(run_dir / "train.log")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.5, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    if metrics:
        it = np.array([r["iter"] for r in metrics])
        pg = np.array([r["pg_loss"] for r in metrics])
        v = np.array([r["v_loss"] for r in metrics])
        ent = np.array([r["entropy"] for r in metrics])
        ax_top.plot(it, _smooth(pg, 5), label="PG loss",
                    color="#60a5fa", linewidth=LINE_W)
        ax_top.plot(it, _smooth(v, 5), label="Value loss",
                    color="#f87171", linewidth=LINE_W)
        ax_top.plot(it, _smooth(ent, 5), label="Entropy",
                    color="#a78bfa", linewidth=LINE_W)
        ax_top.axhline(0, color="#475569", linewidth=0.5, linestyle="--")
        ax_top.set_ylabel("loss / entropy")
        ax_top.legend(loc="upper right", framealpha=0.9, fontsize=9)
        ax_top.grid(alpha=0.2)

    if eval_curve:
        eit = sorted(eval_curve)
        wr_r = [eval_curve[i]["random"] for i in eit]
        wr_g = [eval_curve[i]["greedy"] for i in eit]
        ax_bot.plot(eit, wr_r, "o-", label="vs random",
                    color=RANDOM_COLOR, linewidth=LINE_W, markersize=4)
        ax_bot.plot(eit, wr_g, "o-", label="vs greedy",
                    color=GREEDY_COLOR, linewidth=LINE_W, markersize=4)
        ax_bot.axhline(0.5, color="#475569", linewidth=0.5, linestyle="--")
        ax_bot.set_ylim(0.0, 1.0)
        ax_bot.set_ylabel("win rate")
        ax_bot.set_xlabel("PPO iteration")
        ax_bot.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax_bot.grid(alpha=0.2)

    plt.suptitle(f"Training progress — {run_dir.name}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def plot_eval_sweep(run_dir: Path, out_path: Path) -> None:
    """Per-snapshot win-rate + compile margin from our eval pipeline (not the
    training log). Compares vs random + greedy on equal footing.
    """
    eval_dir = run_dir / "eval"
    points = []
    for sub in sorted(eval_dir.glob("snapshot_*")):
        mpath = sub / "metrics.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        it = int(sub.name.split("_")[1])
        matchups = m.get("matchups", {})
        points.append({
            "iter": it,
            "wr_random": matchups.get("random", {}).get("win_rate"),
            "wr_greedy": matchups.get("greedy", {}).get("win_rate"),
            "margin_greedy": matchups.get("greedy", {}).get("avg_compile_margin"),
        })
    if not points:
        print(f"no eval data in {eval_dir}")
        return

    its = [p["iter"] for p in points]
    wr_r = [p["wr_random"] for p in points]
    wr_g = [p["wr_greedy"] for p in points]
    margin = [p["margin_greedy"] for p in points]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    ax1.plot(its, wr_r, "o-", label="vs random",
             color=RANDOM_COLOR, linewidth=LINE_W, markersize=6)
    ax1.plot(its, wr_g, "o-", label="vs greedy",
             color=GREEDY_COLOR, linewidth=LINE_W, markersize=6)
    ax1.axhline(0.5, color="#475569", linewidth=0.5, linestyle="--")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("win rate (50 games)")
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax1.grid(alpha=0.2)

    ax2.plot(its, margin, "s-", label="avg compile margin vs greedy",
             color=NN_COLOR, linewidth=LINE_W, markersize=5)
    ax2.axhline(0, color="#475569", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("compiles\n(me − opp)")
    ax2.set_xlabel("PPO iteration")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax2.grid(alpha=0.2)

    plt.suptitle(f"Per-snapshot evaluation — {run_dir.name}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def plot_ladder(ladder_path: Path, out_path: Path) -> None:
    """Bar chart of Elo ratings from a ladder run, sorted high → low.
    Anchors (random/greedy) are tinted distinctively."""
    if not ladder_path.exists():
        print(f"no ladder.json at {ladder_path}")
        return
    d = json.loads(ladder_path.read_text())
    ranking = d.get("ranking", [])
    if not ranking:
        return
    anchors = set(d.get("anchors", {}).keys())
    names = [r["name"] for r in ranking]
    elos = [r["elo"] for r in ranking]
    colors = [RANDOM_COLOR if n in anchors else NN_COLOR for n in names]

    fig, ax = plt.subplots(figsize=(9.5, max(3, 0.45 * len(names))))
    yp = np.arange(len(names))
    ax.barh(yp, elos, color=colors, edgecolor="#1e293b", linewidth=0.5)
    for y, (n, e) in enumerate(zip(names, elos)):
        ax.text(e + 8, y, f"{e:.0f}", va="center", fontsize=9)
    ax.set_yticks(yp)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"Elo (anchored at random={d['anchors']['random']:.0f}, "
                  f"greedy={d['anchors']['greedy']:.0f})")
    ax.grid(alpha=0.2, axis="x")
    plt.title(f"Ladder — {d.get('games_per_pair', '?')} games per ordered pair", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def _smooth(values: Iterable[float], window: int) -> np.ndarray:
    """Simple centered moving average that handles short windows gracefully."""
    arr = np.asarray(list(values), dtype=float)
    if window <= 1 or arr.size < window:
        return arr
    pad = window // 2
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    out = np.empty_like(arr)
    out[: pad] = arr[: pad]
    out[pad: pad + smoothed.size] = smoothed
    out[pad + smoothed.size:] = arr[pad + smoothed.size:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to runs/<timestamp>/")
    ap.add_argument("--out", required=True, help="output directory for the PNGs")
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_progress(run_dir, out_dir / "training-progress.png")
    plot_eval_sweep(run_dir, out_dir / "eval-sweep.png")
    plot_ladder(run_dir / "eval" / "ladder.json", out_dir / "ladder.png")


if __name__ == "__main__":
    main()
