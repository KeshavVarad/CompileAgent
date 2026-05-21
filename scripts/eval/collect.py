"""Collect per-game telemetry for a single (agent, opponent) matchup.

Plays N deterministic games and writes one JSONL line per game, alternating
which seat the evaluated agent sits in. Used as the upstream of metrics.py
and card.py.

Usage:
    python scripts/eval/collect.py \
        --model runs/latest/snapshot_00100.pt \
        --opp greedy \
        --games 200 \
        --out runs/latest/eval/snapshot_00100/vs_greedy.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from compile_engine.actions import Action, ActionType  # noqa: E402
from compile_engine.cards import load_card_defs  # noqa: E402
from compile_engine.game import Game  # noqa: E402
from compile_engine.nn.agent import NNAgent  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    DEFAULT_AUX2_PROB,
    DEFAULT_EXPANSION_PROB,
    DEFAULT_MAIN2_PROB,
    DecisionRecord,
    GameSummary,
    OpponentSpec,
    build_agent,
    load_model_from_ckpt,
    make_game_config,
    resolve_device,
    write_jsonl,
)


def _record_decision(
    game: Game, action: Action, legal: list[Action], defs,
) -> DecisionRecord:
    """Snapshot the decision context at the moment the agent chose `action`."""
    st = game.state
    seat = game.decider()
    ps = st.players[seat]
    compile_was_legal = any(a.type is ActionType.COMPILE_LINE for a in legal)

    played_def_id: int | None = None
    played_key: str | None = None
    face_up: bool | None = None
    if action.type in (ActionType.PLAY_FACE_UP, ActionType.PLAY_FACE_DOWN, ActionType.DISCARD_CARD):
        if 0 <= action.hand_index < len(ps.hand):
            c = ps.hand[action.hand_index]
            played_def_id = c.def_id
            if 0 <= c.def_id < len(defs):
                played_key = defs[c.def_id].key
        face_up = action.type is ActionType.PLAY_FACE_UP

    return DecisionRecord(
        turn=st.turn,
        phase=st.phase.name,
        action_type=action.type.name,
        hand_index=action.hand_index if action.hand_index >= 0 else None,
        line_index=action.line_index if action.line_index >= 0 else None,
        choice_index=action.choice_index if action.choice_index >= 0 else None,
        protocol=action.protocol or None,
        played_def_id=played_def_id,
        played_key=played_key,
        face_up=face_up,
        compile_was_legal=compile_was_legal,
        hand_size_before=len(ps.hand),
        deck_size_before=len(ps.deck),
    )


def play_one(
    game_index: int,
    *,
    agent,
    opp,
    opp_name: str,
    rng: random.Random,
    defs,
    expansion_prob: float,
    main2_prob: float,
    aux2_prob: float,
    max_turns: int,
) -> GameSummary:
    """Play one game and return the telemetry summary."""
    cfg = make_game_config(
        rng,
        expansion_prob=expansion_prob,
        main2_prob=main2_prob,
        aux2_prob=aux2_prob,
        max_turns=max_turns,
    )
    # Alternate seats by parity so agent plays seat 0 half the time.
    agent_seat = game_index % 2
    agents = (agent, opp) if agent_seat == 0 else (opp, agent)

    game = Game(cfg, defs=defs)
    game.start()

    decisions: list[DecisionRecord] = []
    # Track per-line compile counts to detect recompiles.
    compile_count_p0 = [0, 0, 0]
    compile_count_p1 = [0, 0, 0]
    draft_picker_order: list[int] = []

    while not game.is_over():
        who = game.decider()
        legal = game.legal_actions()
        if not legal:
            break
        # Draft picker order (engine alternates per its schedule).
        st = game.state
        if st.phase.name == "DRAFT":
            draft_picker_order.append(who)
        action = agents[who].choose(game, legal)
        if who == agent_seat:
            decisions.append(_record_decision(game, action, legal, defs))
        if action.type is ActionType.COMPILE_LINE and 0 <= action.line_index < 3:
            if who == 0:
                compile_count_p0[action.line_index] += 1
            else:
                compile_count_p1[action.line_index] += 1
        game.step(action)

    st = game.state
    # Reached max_turns = game ended via the turn-cap, not a compile win.
    # The engine's `turn_cap_resolution` policy still picks a winner (leader),
    # but we tag the game as a timeout so metrics can bucket it separately.
    timeout = st.turn >= cfg.max_turns
    return GameSummary(
        game_index=game_index,
        seed=cfg.seed,
        include_expansion=cfg.include_expansion,
        include_main2=cfg.include_main2,
        include_aux2=cfg.include_aux2,
        agent_seat=agent_seat,
        opponent_name=opp_name,
        protocols_p0=list(st.players[0].protocols),
        protocols_p1=list(st.players[1].protocols),
        draft_picker_order=draft_picker_order,
        turns=st.turn,
        winner=st.winner,
        timeout=timeout,
        compiles_p0=sum(compile_count_p0),
        compiles_p1=sum(compile_count_p1),
        recompiled=any(c > 1 for c in compile_count_p0 + compile_count_p1),
        decisions=decisions,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="path to evaluated snapshot .pt")
    ap.add_argument("--opp", type=str, required=True,
                    help="'random', 'greedy', or a path to a snapshot .pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, required=True,
                    help="output JSONL path; one row per game")
    ap.add_argument("--expansion-prob", type=float, default=DEFAULT_EXPANSION_PROB)
    ap.add_argument("--main2-prob", type=float, default=DEFAULT_MAIN2_PROB)
    ap.add_argument("--aux2-prob", type=float, default=DEFAULT_AUX2_PROB)
    ap.add_argument("--max-turns", type=int, default=200)
    args = ap.parse_args()

    device = resolve_device(args.device)
    defs = load_card_defs()

    model = load_model_from_ckpt(args.model, device)
    agent = NNAgent(model, device=device, stochastic=False)
    opp_spec = OpponentSpec.parse(args.opp)
    opp = build_agent(opp_spec, device, seed=args.seed + 1)

    rng = random.Random(args.seed)
    rows: list[GameSummary] = []
    t0 = time.perf_counter()
    for i in range(args.games):
        summary = play_one(
            i,
            agent=agent,
            opp=opp,
            opp_name=opp_spec.name,
            rng=rng,
            defs=defs,
            expansion_prob=args.expansion_prob,
            main2_prob=args.main2_prob,
            aux2_prob=args.aux2_prob,
            max_turns=args.max_turns,
        )
        rows.append(summary)
        if (i + 1) % 50 == 0:
            wr = sum(1 for r in rows if r.winner == r.agent_seat) / len(rows)
            print(f"  [{i+1:4d}/{args.games}] running wr={wr:.2f} dt={time.perf_counter()-t0:.1f}s")

    out_path = Path(args.out)
    write_jsonl(out_path, rows)
    wr = sum(1 for r in rows if r.winner == r.agent_seat) / max(1, len(rows))
    print(
        f"wrote {len(rows)} games to {out_path}  wr_vs_{opp_spec.name}={wr:.3f}  "
        f"({time.perf_counter()-t0:.1f}s)"
    )


if __name__ == "__main__":
    main()
