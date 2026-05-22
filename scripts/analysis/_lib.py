"""Shared helpers for the strategic-analysis suite.

These scripts process per-game JSONL telemetry produced by
`scripts/eval/collect.py`. They never call the model themselves — they
just slice and aggregate the recorded game data. This keeps them fast,
reproducible, and runnable without GPU.

Conventions used across the suite:
  - Output dir for a checkpoint: `analysis/<ckpt_stem>/`
  - Each analysis writes a markdown + JSON sidecar + optional PNG chart
  - Confidence intervals on win-rate use Wilson normal approximation,
    which is well-behaved at small n unlike naive normal.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a Bernoulli WR estimate. Returns (p, lo, hi)."""
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = wins / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (p, (centre - margin) / denom, (centre + margin) / denom)


def outcome(rec: dict) -> str:
    w = rec.get("winner")
    if w is None:
        return "draw"
    return "win" if w == rec["agent_seat"] else "loss"


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_games(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def write_md(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n")


# ---------------------------------------------------------------------------
# Eval invocation (lazy — only re-runs when jsonl missing)
# ---------------------------------------------------------------------------


def ensure_collected(
    model: Path, opp: str, out: Path,
    *, games: int = 200, seed: int = 0, device: str = "mps",
) -> Path:
    """Run scripts/eval/collect.py if `out` doesn't already exist."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import os
    subprocess.run(
        [
            ".venv/bin/python", "scripts/eval/collect.py",
            "--model", str(model), "--opp", opp,
            "--games", str(games), "--seed", str(seed),
            "--device", device,
            "--out", str(out),
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )
    return out


def analysis_dir(ckpt: Path) -> Path:
    """Canonical output dir for a checkpoint's analysis artifacts."""
    return REPO / "analysis" / ckpt.stem


# ---------------------------------------------------------------------------
# Common slices
# ---------------------------------------------------------------------------


def agent_protocols(rec: dict) -> list[str]:
    return rec[f"protocols_p{rec['agent_seat']}"]


def opp_protocols(rec: dict) -> list[str]:
    return rec[f"protocols_p{1 - rec['agent_seat']}"]


@dataclass
class GameSummary:
    n_games: int
    wins: int
    losses: int
    draws: int
    @property
    def wr(self) -> float:
        return self.wins / max(1, self.n_games)


def summary(recs: list[dict]) -> GameSummary:
    results = [outcome(r) for r in recs]
    return GameSummary(
        n_games=len(recs),
        wins=results.count("win"),
        losses=results.count("loss"),
        draws=results.count("draw"),
    )


def per_protocol_picks(recs: list[dict]) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in recs:
        for p in agent_protocols(r):
            c[p] += 1
    return c


def per_protocol_wr(recs: list[dict]) -> dict[str, tuple[int, int]]:
    """For each protocol the agent drafted, return (wins, total)."""
    out: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))  # type: ignore[assignment]
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [wins, n]
    for r in recs:
        won = outcome(r) == "win"
        for p in agent_protocols(r):
            counts[p][1] += 1
            if won:
                counts[p][0] += 1
    return {p: (v[0], v[1]) for p, v in counts.items()}


def per_pair_wr(recs: list[dict]) -> dict[tuple[str, str], tuple[int, int]]:
    """For each unordered pair of protocols the agent drafted together,
    return (wins, total). Pairs are 3-choose-2 = 3 per game."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in recs:
        protos = sorted(agent_protocols(r))
        won = outcome(r) == "win"
        for i in range(len(protos)):
            for j in range(i + 1, len(protos)):
                k = (protos[i], protos[j])
                counts[k][1] += 1
                if won:
                    counts[k][0] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def fmt_wr(wins: int, total: int) -> str:
    if total == 0:
        return "  -"
    p, lo, hi = wilson_ci(wins, total)
    return f"{p:5.1%} ± {(hi - lo) / 2:4.1%}  (n={total})"


def fmt_pct(x: float) -> str:
    return f"{x:5.1%}"
