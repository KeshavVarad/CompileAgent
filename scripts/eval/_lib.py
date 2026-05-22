"""Shared helpers for the model-card eval pipeline.

Keeps the per-script CLIs small: agent loading, opponent-spec parsing, and
the training-style GameConfig sampler all live here.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from compile_engine.agents import GreedyAgent, RandomAgent  # noqa: E402
from compile_engine.nn.agent import NNAgent  # noqa: E402
from compile_engine.nn.model import PolicyValueNet  # noqa: E402
from compile_engine.nn.train import _resolve_device  # noqa: E402
from compile_engine.state import GameConfig  # noqa: E402


# Default training distribution — matches scripts/train_nn.py defaults so eval
# numbers are comparable to training rollout_wr / eval lines in train.log.
DEFAULT_EXPANSION_PROB = 0.5
DEFAULT_MAIN2_PROB = 0.4
DEFAULT_AUX2_PROB = 0.4
DEFAULT_MAX_TURNS = 200


@dataclass(frozen=True)
class OpponentSpec:
    """Specifies an opponent agent. Either a baseline name ('random'/'greedy')
    or a snapshot checkpoint path."""
    name: str           # display name used in JSON/markdown
    kind: str           # 'random' | 'greedy' | 'snapshot'
    ckpt_path: str | None = None

    @classmethod
    def parse(cls, raw: str) -> "OpponentSpec":
        if raw == "random":
            return cls(name="random", kind="random")
        if raw == "greedy":
            return cls(name="greedy", kind="greedy")
        p = Path(raw)
        if not p.exists():
            raise ValueError(f"opponent must be 'random', 'greedy', or a .pt path; got {raw!r}")
        # Use the snapshot filename stem as display name (e.g. snapshot_00100)
        return cls(name=p.stem, kind="snapshot", ckpt_path=str(p))


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> PolicyValueNet:
    """Load a PolicyValueNet from a snapshot. Always returns the model in
    eval() mode."""
    model = PolicyValueNet().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def build_agent(spec: OpponentSpec, device: torch.device, *, seed: int = 0,
                stochastic: bool = True):
    """Instantiate an Agent from a spec. `stochastic=True` (default) makes
    NN agents sample from the policy distribution — appropriate for
    measuring mixed-Nash performance in an imperfect-information game
    like Compile. Set False for argmax/deterministic play (useful for
    reproducibility-critical comparisons)."""
    if spec.kind == "random":
        return RandomAgent(seed=seed)
    if spec.kind == "greedy":
        return GreedyAgent(seed=seed)
    if spec.kind == "snapshot":
        assert spec.ckpt_path is not None
        m = load_model_from_ckpt(spec.ckpt_path, device)
        return NNAgent(m, device=device, stochastic=stochastic)
    raise ValueError(f"unknown opponent kind: {spec.kind}")


def make_game_config(
    rng: random.Random,
    *,
    expansion_prob: float = DEFAULT_EXPANSION_PROB,
    main2_prob: float = DEFAULT_MAIN2_PROB,
    aux2_prob: float = DEFAULT_AUX2_PROB,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> GameConfig:
    """Sample a GameConfig matching the training distribution."""
    return GameConfig(
        include_expansion=rng.random() < expansion_prob,
        include_main2=rng.random() < main2_prob,
        include_aux2=rng.random() < aux2_prob,
        seed=rng.randint(0, 2**31 - 1),
        max_turns=max_turns,
    )


def resolve_device(name: str) -> torch.device:
    return _resolve_device(name)


# ---------------------------------------------------------------------------
# JSONL telemetry I/O — shared by collect.py and metrics.py.
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """One decision made by the evaluated agent during a game."""
    turn: int
    phase: str
    action_type: str
    hand_index: int | None = None
    line_index: int | None = None
    choice_index: int | None = None
    protocol: str | None = None
    played_def_id: int | None = None
    played_key: str | None = None       # "MN01:Speed:0"
    face_up: bool | None = None         # for PLAY_FACE_UP / PLAY_FACE_DOWN
    compile_was_legal: bool = False     # COMPILE_LINE was in legal_actions
    hand_size_before: int = 0
    deck_size_before: int = 0


@dataclass
class GameSummary:
    """Per-game summary written as one JSONL line."""
    game_index: int
    seed: int
    include_expansion: bool
    include_main2: bool
    include_aux2: bool
    agent_seat: int
    opponent_name: str
    protocols_p0: list[str]
    protocols_p1: list[str]
    draft_picker_order: list[int]       # seat that picked at each draft step
    turns: int
    winner: int | None
    timeout: bool                       # game hit max_turns
    compiles_p0: int
    compiles_p1: int
    recompiled: bool                    # any single line was compiled >once
    decisions: list[DecisionRecord] = field(default_factory=list)


def write_jsonl(path: Path, rows: Iterable[GameSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False))
