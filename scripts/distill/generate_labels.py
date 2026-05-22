"""Generate MCTS-labeled training data for policy distillation.

Runs self-play with the policy (stochastic, both seats), and at every
top-level decision where the policy is *not* already confident, runs
MCTS to produce a soft target distribution. Skip-when-confident filters
out states where MCTS would just reproduce the policy argmax — those
add no learning signal and are pure compute waste.

Target formulation lives in `MCTSAgent.choose_with_target`:
    target(a) = softmax(log(prior(a)) + tau * Q_search(a))

That's the Gumbel-AlphaZero "completed Q" target — at low sim budgets
it's a less-noisy training signal than raw visit counts because every
sim refines Q whereas only the *last* visit changes which action would
be argmax-by-visits.

Output is a single `.pt` file holding stacked tensors that
`scripts/distill/train.py` consumes directly.

Usage:
    python scripts/distill/generate_labels.py \\
        --ckpt runs/latest/snapshot_00500.pt \\
        --games 50 --skip-top-prob 0.9 --tau 1.0 \\
        --out runs/latest/distill/labels.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from compile_engine import Game, GameConfig  # noqa: E402
from compile_engine.agents import GreedyAgent, RandomAgent  # noqa: E402
from compile_engine.cards import load_card_defs  # noqa: E402
from compile_engine.nn.agent import NNAgent  # noqa: E402
from compile_engine.nn.encoder import MAX_ACTIONS, encode_actions, encode_state  # noqa: E402
from compile_engine.nn.mcts import MCTSAgent, MCTSConfig, _mid_effect, _policy_and_value  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "eval"))
from _lib import DEFAULT_MAX_TURNS, load_model_from_ckpt, resolve_device  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="output .pt file for labels")
    p.add_argument("--games", type=int, default=50)
    # MCTS config
    p.add_argument("--dets", type=int, default=8)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--c-puct", type=float, default=1.25)
    p.add_argument("--root-top-k", type=int, default=5)
    p.add_argument("--root-min-visits", type=int, default=3)
    # Distillation knobs
    p.add_argument("--skip-top-prob", type=float, default=0.9,
                   help="don't label states where policy top_prob >= this")
    p.add_argument("--tau", type=float, default=1.0,
                   help="temperature on Q in target = softmax(log(prior) + tau*Q)")
    # Opponent mix. Per game, sample one opponent type from this
    # distribution and label only the policy-under-training seat. The
    # default (1.0/0/0) preserves the original pure-self-play behavior.
    p.add_argument("--mix-self", type=float, default=1.0,
                   help="fraction of games to label with self (NNAgent) as the opponent")
    p.add_argument("--mix-greedy", type=float, default=0.0,
                   help="fraction of games with Greedy as the opponent")
    p.add_argument("--mix-random", type=float, default=0.0,
                   help="fraction of games with Random as the opponent")
    # Engine config
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--include-expansion", action="store_true", default=True)
    p.add_argument("--include-main2", action="store_true", default=True)
    p.add_argument("--include-aux2", action="store_true", default=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = resolve_device(args.device)
    model = load_model_from_ckpt(args.ckpt, device)
    defs = load_card_defs()

    # Normalize opponent mix.
    mix_weights = np.array(
        [args.mix_self, args.mix_greedy, args.mix_random], dtype=np.float64
    )
    if mix_weights.sum() <= 0:
        raise SystemExit("--mix-* values must sum to > 0")
    mix_weights /= mix_weights.sum()
    opp_kinds = ["self", "greedy", "random"]
    mix_rng = np.random.default_rng(args.seed)
    print(
        f"Opponent mix: self={mix_weights[0]:.2f} "
        f"greedy={mix_weights[1]:.2f} random={mix_weights[2]:.2f}"
    )

    mcts = MCTSAgent(
        model=model,
        device=device,
        cfg=MCTSConfig(
            n_determinizations=args.dets,
            sims_per_determinization=args.sims,
            c_puct=args.c_puct,
            root_top_k=args.root_top_k,
            root_min_visits_per_action=args.root_min_visits,
            batch_size=args.batch_size,
            # Don't pass skip_top_prob here — we filter upstream so we
            # have control over which states are LABELED vs which are
            # just played through.
            skip_search_top_prob=0.0,
        ),
        seed=args.seed,
    )

    # Buffers for stacked output.
    state_buf: dict[str, list[np.ndarray]] = {}
    raws: list[np.ndarray] = []
    cards: list[np.ndarray] = []
    protos: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    value_targets: list[float] = []  # MCTS root V per labeled state, in [-1, 1]

    n_labeled = 0
    n_skipped_confident = 0
    n_decisions = 0
    n_mid_effect = 0
    games_per_opp: Counter[str] = Counter()
    labels_per_opp: Counter[str] = Counter()
    t0 = time.perf_counter()

    for g_idx in range(args.games):
        # Pick the labeler seat and sample an opponent type for this game.
        # Labels come ONLY from the labeler's turns, so the state
        # distribution we train on matches the on-policy distribution
        # against this opponent — fixing the train/eval mismatch when the
        # eval opponent differs from the self-play partner.
        labeler_seat = int(mix_rng.integers(0, 2))
        opp_seat = 1 - labeler_seat
        opp_kind = opp_kinds[int(mix_rng.choice(3, p=mix_weights))]
        games_per_opp[opp_kind] += 1
        opp_base_seed = args.seed + g_idx * 1000 + 17
        if opp_kind == "self":
            opp_agent: object = NNAgent(model, device=device, stochastic=True)
        elif opp_kind == "greedy":
            opp_agent = GreedyAgent(seed=opp_base_seed)
        else:  # "random"
            opp_agent = RandomAgent(seed=opp_base_seed)
        labeler_agent = NNAgent(model, device=device, stochastic=True)
        seat_agents: list[object] = [None, None]  # type: ignore[list-item]
        seat_agents[labeler_seat] = labeler_agent
        seat_agents[opp_seat] = opp_agent

        cfg = GameConfig(
            include_expansion=args.include_expansion,
            include_main2=args.include_main2,
            include_aux2=args.include_aux2,
            seed=args.seed + g_idx,
            max_turns=args.max_turns,
        )
        game = Game(cfg, defs=defs)
        game.start()
        labels_this_game = 0
        while not game.is_over():
            who = game.decider()
            legal = game.legal_actions()
            if not legal:
                break
            n_decisions += 1
            # Opponent's turn — just step their action, never label.
            if who != labeler_seat:
                action = seat_agents[who].choose(game, legal)  # type: ignore[attr-defined]
                game.step(action)
                continue
            # Labeler's turn — three classes of decision:
            #   1. Forced (1 legal) or mid-effect → just step, don't label.
            #   2. Policy confident → step with policy argmax, don't label.
            #   3. Real decision → MCTS-label, then step with MCTS pick.
            if len(legal) == 1 or _mid_effect(game):
                if _mid_effect(game):
                    n_mid_effect += 1
                action = labeler_agent.choose(game, legal)
                game.step(action)
                continue
            # Class 2 or 3: peek at policy confidence first.
            probs, _ = _policy_and_value(model, game, legal, device)
            n_p = min(len(legal), len(probs))
            top_prob = float(probs[:n_p].max())
            if top_prob >= args.skip_top_prob:
                n_skipped_confident += 1
                # Step with policy argmax — same as inference would do
                # given the skip-when-confident gate.
                action = legal[int(np.argmax(probs[:n_p]))]
                game.step(action)
                continue

            # Class 3: real label.
            action, target_over_legal, v_root = mcts.choose_with_target(
                game, legal, tau=args.tau, return_value=True,
            )
            perspective = who
            state = encode_state(game, perspective)
            raw, card_ids, proto_ids, mask = encode_actions(game, legal, perspective)
            # Pad target to MAX_ACTIONS so we can stack.
            target_padded = np.zeros(MAX_ACTIONS, dtype=np.float32)
            n_t = min(MAX_ACTIONS, len(target_over_legal))
            target_padded[:n_t] = target_over_legal[:n_t]
            # Renormalize after potential truncation (paranoia — MAX_ACTIONS
            # is 32 and we've never seen a state with >32 legal actions).
            s = target_padded.sum()
            if s > 0:
                target_padded /= s

            for k, v in state.items():
                state_buf.setdefault(k, []).append(v)
            raws.append(raw)
            cards.append(card_ids)
            protos.append(proto_ids)
            masks.append(mask)
            targets.append(target_padded)
            value_targets.append(v_root)
            n_labeled += 1
            labels_this_game += 1

            game.step(action)

        labels_per_opp[opp_kind] += labels_this_game
        elapsed = time.perf_counter() - t0
        winner = game.state.winner
        print(
            f"  game {g_idx:3d}  opp={opp_kind:<6} labeler=s{labeler_seat}  "
            f"winner={winner}  turns={game.state.turn:3d}  "
            f"labeled={n_labeled:5d}  skipped_conf={n_skipped_confident:5d}  "
            f"mid_effect={n_mid_effect:5d}  elapsed={elapsed:6.1f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 70)
    print(f"Total decisions:        {n_decisions}")
    print(f"  forced/mid-effect:    {n_mid_effect} (no label, no MCTS)")
    print(f"  skipped (confident):  {n_skipped_confident} (no label, no MCTS)")
    print(f"  labeled (searched):   {n_labeled}")
    print(f"  fraction labeled:     {n_labeled / max(1, n_decisions):.1%}")
    print(f"Elapsed:                {elapsed:.1f}s ({elapsed / max(1, n_labeled):.2f}s/label)")
    print()
    print("Per-opponent breakdown:")
    for k in opp_kinds:
        print(
            f"  {k:<6}  games={games_per_opp[k]:3d}  labels={labels_per_opp[k]:5d}"
        )

    if n_labeled == 0:
        print("[error] no labeled states produced; aborting save.")
        return 1

    # Stack everything into tensors and save.
    stacked_state = {k: np.stack(v) for k, v in state_buf.items()}
    payload = {
        "state": stacked_state,
        "action_raw": np.stack(raws),
        "action_card_ids": np.stack(cards),
        "action_proto_ids": np.stack(protos),
        "action_mask": np.stack(masks),
        "target": np.stack(targets),
        "value_target": np.asarray(value_targets, dtype=np.float32),
        "meta": {
            "n_labeled": n_labeled,
            "n_skipped_confident": n_skipped_confident,
            "n_mid_effect": n_mid_effect,
            "n_decisions": n_decisions,
            "ckpt": args.ckpt,
            "tau": args.tau,
            "skip_top_prob": args.skip_top_prob,
            "mcts_cfg": vars(mcts.cfg),
            "opp_mix": {
                "self": float(mix_weights[0]),
                "greedy": float(mix_weights[1]),
                "random": float(mix_weights[2]),
            },
            "games_per_opp": dict(games_per_opp),
            "labels_per_opp": dict(labels_per_opp),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"\nWrote {n_labeled} labels to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
