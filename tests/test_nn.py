"""Smoke tests for the NN agent.

Cover:
 - Encoder output shapes are as designed.
 - Model forward returns finite logits + value ∈ [-1, +1].
 - NNAgent.choose returns a legal action.
 - One PPO training iteration runs without error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from compile_engine import Game, GameConfig
from compile_engine.actions import Action, ActionType
from compile_engine.agents import RandomAgent
from compile_engine.nn import NNAgent, PolicyValueNet
from compile_engine.nn.encoder import (
    MAX_ACTIONS,
    MAX_HAND,
    MAX_STACK,
    NUM_CARDS,
    encode_actions,
    encode_state,
)
from compile_engine.nn.train import TrainConfig, train


def _make_game(seed: int = 0) -> Game:
    g = Game(GameConfig(seed=seed, max_turns=200))
    g.set_predetermined_draft([
        ["Speed", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    return g


def test_encode_state_shapes():
    g = _make_game()
    obs = encode_state(g, perspective=0)
    assert obs["field_tokens"].shape == (3, 2, MAX_STACK)
    assert obs["field_meta"].shape == (3, 2, 3)
    assert obs["protocols"].shape == (2, 3, 2)
    assert obs["hand_tokens"].shape == (MAX_HAND,)
    assert obs["trash"].shape == (2, NUM_CARDS)
    assert obs["line_vals"].shape == (3, 2)
    assert obs["scalars"].shape == (8,)  # +1 for draft_pick_idx_norm
    assert obs["phase"].shape == (9,)


def test_encode_actions_shapes_and_mask():
    g = _make_game()
    legal = g.legal_actions()
    raw, card_ids, proto_ids, mask = encode_actions(g, legal, perspective=0)
    assert raw.shape == (MAX_ACTIONS, raw.shape[1])
    assert card_ids.shape == (MAX_ACTIONS,)
    assert proto_ids.shape == (MAX_ACTIONS,)
    assert mask.shape == (MAX_ACTIONS,)
    assert int(mask.sum()) == len(legal)
    # Padded rows should be zero
    assert (raw[len(legal):] == 0).all()


def test_model_forward_value_bounded():
    g = _make_game()
    obs = encode_state(g, 0)
    legal = g.legal_actions()
    raw, card_ids, proto_ids, mask = encode_actions(g, legal, 0)
    model = PolicyValueNet()
    state = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
    ar = torch.from_numpy(raw).unsqueeze(0)
    ac = torch.from_numpy(card_ids).unsqueeze(0)
    ap = torch.from_numpy(proto_ids).unsqueeze(0)
    am = torch.from_numpy(mask).unsqueeze(0)
    logits, value = model(state, ar, ac, ap, am)
    assert logits.shape == (1, MAX_ACTIONS)
    assert value.shape == (1,)
    assert torch.isfinite(value).all()
    assert (value >= -1).all() and (value <= 1).all()
    # Masked entries should have -1e9 logits
    assert (logits[0, ~am[0]] < -1e8).all()


def test_nn_agent_returns_legal_action():
    g = _make_game()
    model = PolicyValueNet()
    agent = NNAgent(model, device="cpu", stochastic=True)
    legal = g.legal_actions()
    chosen = agent.choose(g, legal)
    assert chosen in legal


def test_card_static_features_present_in_model():
    """The model's card_static buffer should have the expected layout:
    rows 0/1 (PAD/HIDDEN) all zero, every real card row has nonzero
    value one-hot and matching keyword bits."""
    from compile_engine.cards import (
        CARD_STATIC_FEATS_DIM, KEYWORD_VOCAB, NUM_KEYWORDS, VALUE_RANGE,
        load_card_defs,
    )
    model = PolicyValueNet()
    static = model.card_static
    # Vocab is 180 cards + 2 sentinel slots (PAD, HIDDEN).
    assert static.shape == (182, CARD_STATIC_FEATS_DIM)
    # PAD and HIDDEN slots stay zero
    assert (static[0] == 0).all()
    assert (static[1] == 0).all()
    # Pick a known card: Fire 0 has keywords {covered, draw, flip} and value 0.
    defs = load_card_defs()
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    row = static[fire_0.def_id + 2]
    for kw in ("covered", "draw", "flip"):
        idx = KEYWORD_VOCAB.index(kw)
        assert row[idx].item() == 1.0, f"keyword {kw} flag missing"
    # value=0 → first slot of the value one-hot
    assert row[NUM_KEYWORDS + 0].item() == 1.0
    # Fire 0 has middle and bottom text but no top — text-presence flags reflect that
    text_base = NUM_KEYWORDS + VALUE_RANGE
    assert row[text_base + 0].item() == 0.0  # has_top
    assert row[text_base + 1].item() == 1.0  # has_middle
    assert row[text_base + 2].item() == 1.0  # has_bottom


def test_draft_pick_idx_norm_advances_during_draft():
    """The draft_pick_idx_norm scalar should monotonically increase as picks
    happen, then sit at 1.0 once the draft completes."""
    from compile_engine import Game, GameConfig
    from compile_engine.nn.encoder import encode_state
    g = Game(GameConfig(seed=1))
    g.start()
    seen = []
    while True:
        legal = g.legal_actions()
        a = legal[0]
        if a.type is not ActionType.DRAFT_PROTOCOL:
            break
        obs = encode_state(g, perspective=g.decider())
        seen.append(obs["scalars"][-1])  # last scalar = draft_pick_idx_norm
        g.step(a)
    # 6 draft picks → 6 increasing values, all in [0, 1).
    assert len(seen) == 6
    assert all(seen[i] <= seen[i + 1] for i in range(5))
    assert seen[0] == 0.0
    # After the last pick we're out of the draft phase.
    obs_post = encode_state(g, perspective=g.decider())
    assert obs_post["scalars"][-1] == 1.0


def test_one_ppo_iter_end_to_end():
    cfg = TrainConfig(
        iters=1,
        games_per_iter=2,
        ppo_epochs=1,
        batch_size=32,
        snapshot_every=999,
        device="cpu",
        eval_games=4,
        max_turns=80,
    )
    model = train(cfg)
    # Sanity check the trained model still runs and returns a legal action.
    g = _make_game(seed=99)
    agent = NNAgent(model, device="cpu", stochastic=False)
    legal = g.legal_actions()
    assert agent.choose(g, legal) in legal


def test_metrics_jsonl_well_formed(tmp_path):
    """A short training run writes one valid JSON object per iter to
    metrics.jsonl, with eval fields populated on snapshot iters."""
    cfg = TrainConfig(
        iters=3,
        games_per_iter=2,
        ppo_epochs=1,
        batch_size=32,
        snapshot_every=3,
        device="cpu",
        eval_games=4,
        max_turns=60,
        save_dir=str(tmp_path),
    )
    train(cfg)
    metrics_path = tmp_path / "metrics.jsonl"
    assert metrics_path.exists()
    lines = [l for l in metrics_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    import json as _json
    records = [_json.loads(l) for l in lines]
    required_keys = {
        "iter", "games", "transitions", "rollout_wr", "pg_loss", "v_loss",
        "entropy", "approx_kl", "stopped_at_epoch", "dt", "pool_size",
        "wr_random", "wr_greedy", "snapshot_path", "pool_grew",
    }
    for r in records:
        assert required_keys.issubset(r.keys()), f"missing keys: {required_keys - r.keys()}"
    # Iters 1 and 2 should have null evals; iter 3 (snapshot) should have populated.
    assert records[0]["wr_random"] is None and records[1]["wr_random"] is None
    assert records[2]["wr_random"] is not None
    assert records[2]["snapshot_path"] is not None


def test_luck_decisions_are_exposed_to_agent_with_rich_encoding():
    """Luck 0 (state-a-number) and Luck 3 (state-a-protocol) must yield real
    agent choices, and the encoder must populate richer features for them:
      - state-a-number: stated_value scalar slot is filled (0..1)
      - state-a-protocol: proto_ids entry is non-zero (protocol embedding live)
    Confirms the agent has access to the same actions a human would, with
    sufficient features to actually learn to play them well."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.nn.encoder import (
        encode_actions, encode_state, _action_target_meta,
    )
    from compile_engine.state import CardInst, Phase

    defs = load_card_defs()
    luck_0 = next(d for d in defs if d.key == "MN02:Luck:0")
    luck_3 = next(d for d in defs if d.key == "MN02:Luck:3")

    # --- Luck 0: stating a number 0..6 ---
    g = Game(GameConfig(seed=1, include_main2=True))
    g.set_predetermined_draft([
        ["Luck", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.players[0].hand = [
        CardInst(inst_id=9001, def_id=luck_0.def_id, owner=0, face_up=False),
    ]
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=9100 + i, def_id=luck_0.def_id, owner=0, face_up=False)
        for i in range(6)
    ]
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []

    # Play Luck 0 face-up into line 0 (Luck protocol).
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))

    # The agent is now prompted to state a number 0-6 → 7 CHOOSE_TARGET actions.
    legal = g.legal_actions()
    choose_actions = [a for a in legal if a.type is ActionType.CHOOSE_TARGET]
    assert len(choose_actions) == 7, f"expected 7 number choices, got {len(choose_actions)}"
    # Each CHOOSE_TARGET should carry a unique stated_value scalar in the
    # encoder's raw features (the last slot).
    raw, _card_ids, _proto_ids, mask = encode_actions(g, legal, perspective=0)
    stated_values = sorted({float(raw[i][-1]) for i in range(len(legal)) if mask[i]})
    # We expect at least 7 distinct numbers in [0, 1] for the 7 options.
    assert sum(1 for v in stated_values if 0.0 <= v <= 1.0) >= 7

    # --- Luck 3: stating a protocol ---
    g2 = Game(GameConfig(seed=2, include_main2=True))
    g2.set_predetermined_draft([
        ["Luck", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    g2.state.lines = [type(g2.state.lines[0])() for _ in range(3)]
    g2.state.players[0].hand = [
        CardInst(inst_id=9201, def_id=luck_3.def_id, owner=0, face_up=False),
    ]
    g2.state.players[1].hand = []
    g2.state.players[1].deck = [
        CardInst(inst_id=9300 + i, def_id=luck_3.def_id, owner=1, face_up=False)
        for i in range(6)
    ]
    g2.state.current_player = 0
    g2.state.phase = Phase.ACTION
    g2.state.scratch["_engine"] = g2
    g2._pending = []

    g2.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))

    legal2 = g2.legal_actions()
    choose2 = [a for a in legal2 if a.type is ActionType.CHOOSE_TARGET]
    # 30 protocols (12+3+12+3) across all sets.
    assert len(choose2) == 30
    raw2, card_ids2, proto_ids2, mask2 = encode_actions(g2, legal2, perspective=0)
    # Each CHOOSE_TARGET option should have a non-zero proto_id (matching one
    # of the 30 protocols), so the protocol embedding kicks in.
    proto_ids_set = {int(proto_ids2[i]) for i in range(len(legal2)) if mask2[i]}
    proto_ids_set.discard(0)
    assert len(proto_ids_set) == 30, f"expected 30 distinct protocol ids; got {len(proto_ids_set)}"


def test_play_episode_records_have_filled_rewards():
    """A finished episode should have at least one record with non-zero
    terminal reward + done=True."""
    from compile_engine.nn.train import _resolve_device, play_episode
    import random as _r
    cfg = TrainConfig(max_turns=80)
    rng = _r.Random(0)
    device = torch.device("cpu")
    model = PolicyValueNet()
    records, winner, seat = play_episode(model, RandomAgent(seed=1), cfg, rng, device)
    assert records, "expected at least one agent decision"
    last = records[-1]
    assert last.done
    # Some non-zero terminal credit should land somewhere in the trajectory
    # when there is a winner. With max_turns=80 + the new 180-card pool the
    # last record may carry shaping that partially cancels the terminal, so
    # we just check the SUM is in the right ballpark instead of the last
    # record alone.
    if winner is not None:
        total = sum(r.reward for r in records)
        assert abs(total) >= 0.5, f"expected non-trivial reward signal, got {total}"
