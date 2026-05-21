"""Plot learning curves from one (or more) runs' metrics.jsonl files.

Usage:
    python scripts/plot_metrics.py runs/latest
    python scripts/plot_metrics.py runs/baseline runs/with-keywords --output curves.png

Each panel uses metrics.jsonl as the data source. If matplotlib isn't
installed, suggests how to install it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.jsonl in {run_dir}")
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="One or more runs/ subdirs.")
    ap.add_argument("--output", default=None, help="Save plot to file instead of showing it.")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed. Install with:\n"
            "  source .venv/bin/activate && pip install matplotlib\n"
            "Or read the jsonl files directly — each line is a JSON object."
        )
        sys.exit(1)

    runs = [(Path(d).name, load_metrics(Path(d).resolve())) for d in args.run_dirs]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ---------------- Panel 1: eval win-rate vs random / greedy ----------------
    ax = axes[0, 0]
    for name, rs in runs:
        wr_r = [(r["iter"], r["wr_random"]) for r in rs if r.get("wr_random") is not None]
        wr_g = [(r["iter"], r["wr_greedy"]) for r in rs if r.get("wr_greedy") is not None]
        if wr_r:
            ax.plot([x[0] for x in wr_r], [x[1] for x in wr_r], "o-", label=f"{name}: vs random")
        if wr_g:
            ax.plot([x[0] for x in wr_g], [x[1] for x in wr_g], "s-", label=f"{name}: vs greedy")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("iter"); ax.set_ylabel("win rate")
    ax.set_title("Eval win rate (snapshot iters)")
    ax.set_ylim(0, 1.0); ax.legend(); ax.grid(alpha=0.3)

    # ---------------- Panel 2: approx KL + entropy -----------------------------
    ax = axes[0, 1]
    for name, rs in runs:
        iters = [r["iter"] for r in rs]
        ax.plot(iters, [r["approx_kl"] for r in rs], label=f"{name}: approx_kl")
        ax.plot(iters, [r["entropy"] for r in rs], "--", label=f"{name}: entropy")
    ax.set_xlabel("iter")
    ax.set_title("Approx-KL & entropy"); ax.legend(); ax.grid(alpha=0.3)

    # ---------------- Panel 3: pg / value loss ---------------------------------
    ax = axes[1, 0]
    for name, rs in runs:
        iters = [r["iter"] for r in rs]
        ax.plot(iters, [r["pg_loss"] for r in rs], label=f"{name}: pg_loss")
        ax.plot(iters, [r["v_loss"] for r in rs], "--", label=f"{name}: v_loss")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("iter")
    ax.set_title("Policy-gradient & value-MSE losses"); ax.legend(); ax.grid(alpha=0.3)

    # ---------------- Panel 4: rollout WR + PPO early-stop epoch ---------------
    ax = axes[1, 1]
    for name, rs in runs:
        iters = [r["iter"] for r in rs]
        ax.plot(iters, [r["rollout_wr"] for r in rs], "o", alpha=0.5, label=f"{name}: rollout_wr")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_ylabel("rollout WR"); ax.set_ylim(0, 1.0)
    ax2 = ax.twinx()
    for name, rs in runs:
        iters = [r["iter"] for r in rs]
        ax2.plot(iters, [r["stopped_at_epoch"] for r in rs], "x", alpha=0.5,
                 label=f"{name}: stop_at_ep")
    ax2.set_ylabel("PPO epochs completed"); ax2.set_ylim(0, 5)
    ax.set_xlabel("iter")
    ax.set_title("Rollout WR (left) & PPO early-stop epoch (right)")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=110)
        print(f"saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
