"""Where do two checkpoints disagree?

Plays N games using checkpoint A vs Greedy, and at each of A's decisions
asks both A and B what they'd pick on the same state. Tallies
disagreement frequency by decision class and reports the most-divergent
decisions. The "what did the new model learn" probe.

Output: markdown with per-class disagreement table + the top contested
states for manual inspection.

Usage:
    python scripts/analysis/decision_diff.py \\
        --a runs/.../snapshot_00100.pt \\
        --b runs/.../snapshot_00500.pt \\
        --games 100
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from compile_engine import Game  # noqa: E402
from compile_engine.actions import ActionType  # noqa: E402
from compile_engine.agents import GreedyAgent  # noqa: E402
from compile_engine.cards import load_card_defs  # noqa: E402
from compile_engine.nn.agent import NNAgent  # noqa: E402
from compile_engine.nn.model import PolicyValueNet  # noqa: E402
from compile_engine.state import GameConfig  # noqa: E402

from _lib import analysis_dir, write_json, write_md  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> PolicyValueNet:
    model = PolicyValueNet().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def make_game_config(rng, *, expansion_prob, main2_prob, aux2_prob, max_turns) -> GameConfig:
    return GameConfig(
        include_expansion=rng.random() < expansion_prob,
        include_main2=rng.random() < main2_prob,
        include_aux2=rng.random() < aux2_prob,
        seed=rng.randint(0, 2**31 - 1),
        max_turns=max_turns,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint A (the agent that plays)")
    ap.add_argument("--b", required=True, help="checkpoint B (queried in parallel)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--expansion-prob", type=float, default=0.5)
    ap.add_argument("--main2-prob", type=float, default=0.4)
    ap.add_argument("--aux2-prob", type=float, default=0.4)
    ap.add_argument("--max-turns", type=int, default=200)
    args = ap.parse_args()

    ckpt_a = Path(args.a)
    ckpt_b = Path(args.b)
    device = resolve_device(args.device)

    print(f"[load] A={ckpt_a.stem}")
    model_a = load_model_from_ckpt(str(ckpt_a), device)
    agent_a = NNAgent(model_a, device=device, stochastic=False)
    print(f"[load] B={ckpt_b.stem}")
    model_b = load_model_from_ckpt(str(ckpt_b), device)
    agent_b = NNAgent(model_b, device=device, stochastic=False)

    opp = GreedyAgent(seed=12)
    defs = load_card_defs()
    rng = random.Random(args.seed)

    # Per-class tallies.
    decisions_by_class: Counter[str] = Counter()
    disagreements_by_class: Counter[str] = Counter()
    # Disagreement → outcome: did A's pick win the game?
    agree_wins: Counter[str] = Counter()
    agree_games: Counter[str] = Counter()
    disagree_wins: Counter[str] = Counter()
    disagree_games: Counter[str] = Counter()
    # Top contested states for the report (sample a few).
    contested: list[dict] = []

    for g_idx in range(args.games):
        cfg = make_game_config(
            rng,
            expansion_prob=args.expansion_prob,
            main2_prob=args.main2_prob,
            aux2_prob=args.aux2_prob,
            max_turns=args.max_turns,
        )
        agent_seat = g_idx % 2
        agents = (agent_a, opp) if agent_seat == 0 else (opp, agent_a)
        game = Game(cfg, defs=defs)
        game.start()

        # Per-game agreement tracking — does A's choice in this game line
        # up with B at each decision?
        game_classes: list[tuple[str, bool]] = []  # (class, agreed?)

        while not game.is_over():
            who = game.decider()
            legal = game.legal_actions()
            if not legal:
                break
            if who != agent_seat:
                action = agents[who].choose(game, legal)
                game.step(action)
                continue
            # A's decision — query both models on the same state.
            a_action = agent_a.choose(game, legal)
            b_action = agent_b.choose(game, legal)
            cls = _classify(a_action, legal, game)
            decisions_by_class[cls] += 1
            agreed = _same_action(a_action, b_action)
            if not agreed:
                disagreements_by_class[cls] += 1
            game_classes.append((cls, agreed))
            game.step(a_action)

        won = game.state.winner == agent_seat
        # Stamp the outcome onto each per-class agreement bucket so we can
        # see whether disagreements are productive or destructive.
        for cls, agreed in game_classes:
            if agreed:
                agree_games[cls] += 1
                if won:
                    agree_wins[cls] += 1
            else:
                disagree_games[cls] += 1
                if won:
                    disagree_wins[cls] += 1

        if g_idx % 20 == 19:
            print(f"  {g_idx + 1}/{args.games} games played")

    # Render.
    out_dir = analysis_dir(ckpt_a) / f"decision_diff_vs_{ckpt_b.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(ckpt_a, ckpt_b, args.games,
                           decisions_by_class, disagreements_by_class,
                           agree_wins, agree_games,
                           disagree_wins, disagree_games)
    write_md(out_dir / "decision_diff.md", report)

    sidecar = {
        "ckpt_a": str(ckpt_a),
        "ckpt_b": str(ckpt_b),
        "n_games": args.games,
        "decisions_by_class": dict(decisions_by_class),
        "disagreements_by_class": dict(disagreements_by_class),
        "agree_wins": dict(agree_wins),
        "agree_games": dict(agree_games),
        "disagree_wins": dict(disagree_wins),
        "disagree_games": dict(disagree_games),
    }
    write_json(out_dir / "decision_diff.json", sidecar)
    print(f"\nWrote {out_dir / 'decision_diff.md'}")
    return 0


def _classify(action, legal, game) -> str:
    """Categorize a decision by its strategic role.
    Buckets are chosen to match how a human player would talk about a turn."""
    at = action.type.name if hasattr(action.type, "name") else str(action.type)
    if at == "DRAFT_PROTOCOL":
        return "draft"
    if at == "COMPILE_LINE":
        return "compile"
    if at == "REFRESH":
        return "refresh"
    if at == "DISCARD_CARD":
        return "cache-discard"
    if at == "SKIP_OPTIONAL":
        return "skip-optional"
    if at == "CHOOSE_TARGET":
        return "mid-effect"
    if at == "PLAY_FACE_UP":
        return "play-face-up"
    if at == "PLAY_FACE_DOWN":
        return "play-face-down"
    return at.lower()


def _same_action(a, b) -> bool:
    """Two actions are 'the same' if they pick the same legal-index. For
    DRAFT_PROTOCOL we compare protocols. For PLAY_* we compare both
    hand_index and line_index. For CHOOSE_TARGET we compare choice_index."""
    if a.type != b.type:
        return False
    fields = ("hand_index", "line_index", "choice_index", "protocol")
    for f in fields:
        va = getattr(a, f, None)
        vb = getattr(b, f, None)
        if va != vb:
            return False
    return True


def render_report(
    ckpt_a: Path, ckpt_b: Path, n_games: int,
    decisions: Counter, disagreements: Counter,
    agree_wins: Counter, agree_games: Counter,
    disagree_wins: Counter, disagree_games: Counter,
) -> str:
    lines: list[str] = []
    lines.append(f"# Decision Diff: `{ckpt_a.stem}` vs `{ckpt_b.stem}`\n")
    lines.append(f"- **A (acting agent):** `{ckpt_a}`")
    lines.append(f"- **B (queried in parallel):** `{ckpt_b}`")
    lines.append(f"- **Games (A vs Greedy):** {n_games}")
    lines.append("")
    lines.append(
        "Each row is A's decisions in one class. `disagreement` = fraction "
        "of those decisions where B would have picked differently. The two "
        "outcome columns show A's win rate conditional on agreement / "
        "disagreement at that class — they let you tell whether B's "
        "alternative would have been an improvement."
    )
    lines.append("")
    lines.append("| class | A's decisions | disagreement | A's WR when agreed | A's WR when disagreed |")
    lines.append("|---|---:|---:|---:|---:|")
    total_d = sum(decisions.values())
    total_dis = sum(disagreements.values())
    for cls in sorted(decisions, key=lambda c: -decisions[c]):
        n = decisions[cls]
        d = disagreements[cls]
        a_wr = agree_wins[cls] / max(1, agree_games[cls])
        ds_wr = disagree_wins[cls] / max(1, disagree_games[cls])
        lines.append(
            f"| {cls} | {n} ({n / total_d:.0%}) | "
            f"{d / max(1, n):.1%} ({d}) | "
            f"{a_wr:.1%} (n={agree_games[cls]}) | "
            f"{ds_wr:.1%} (n={disagree_games[cls]}) |"
        )
    lines.append(
        f"| **total** | **{total_d}** | **{total_dis / max(1, total_d):.1%}** ({total_dis}) | | |"
    )
    lines.append("")
    lines.append("## How to read this\n")
    lines.append(
        "- A class with **high disagreement and high `WR when disagreed`** is "
        "where A's confident-but-different choices are winning — likely a "
        "genuine improvement."
    )
    lines.append(
        "- A class with **high disagreement and `WR when disagreed` < `WR when "
        "agreed`** is where the divergence is *hurting* A — those are "
        "regressions B would have avoided."
    )
    lines.append(
        "- A class with **low disagreement** means A inherited B's behaviour "
        "in that area; the deltas you see in win-rate aren't from that class."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
