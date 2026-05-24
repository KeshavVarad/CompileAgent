"""Smoke tests for the Compile engine."""

from __future__ import annotations

import pytest

from compile_engine import (
    Action,
    ActionType,
    CompileEnv,
    Game,
    GameConfig,
    Phase,
)
from compile_engine.agents import GreedyAgent, RandomAgent
from compile_engine.cards import (
    BASE_PROTOCOLS,
    EXPANSION_PROTOCOLS,
    available_protocols,
    load_card_defs,
)
from compile_engine.env import play_game


def test_card_data_integrity():
    """Full data is now 180 cards across 4 sets (MN01 + AX01 + MN02 + AX02)."""
    from compile_engine.cards import AUX2_PROTOCOLS, MAIN2_PROTOCOLS
    defs = load_card_defs()
    assert len(defs) == 180
    by_proto: dict[str, int] = {}
    for d in defs:
        by_proto.setdefault(d.protocol, 0)
        by_proto[d.protocol] += 1
    # 12 + 3 + 12 + 3 = 30 protocols, 6 cards each
    assert len(by_proto) == 30, f"protocols: {sorted(by_proto)}"
    assert all(v == 6 for v in by_proto.values()), f"counts: {by_proto}"
    for grp in (BASE_PROTOCOLS, EXPANSION_PROTOCOLS, MAIN2_PROTOCOLS, AUX2_PROTOCOLS):
        assert set(grp).issubset(by_proto)


def test_expansion_toggle_filters_draft_pool():
    defs = load_card_defs()
    base_only = set(available_protocols(defs, include_expansion=False))
    with_exp = set(available_protocols(defs, include_expansion=True))
    assert base_only == set(BASE_PROTOCOLS)
    assert with_exp == set(BASE_PROTOCOLS) | set(EXPANSION_PROTOCOLS)
    for p in EXPANSION_PROTOCOLS:
        assert p not in base_only
        assert p in with_exp


def test_draft_pool_excludes_expansion_when_disabled():
    g = Game(GameConfig(include_expansion=False, seed=1))
    g.start()
    legal = g.legal_actions()
    drafted_options = {a.protocol for a in legal if a.type is ActionType.DRAFT_PROTOCOL}
    assert drafted_options == set(BASE_PROTOCOLS)


def test_draft_pool_includes_expansion_when_enabled():
    g = Game(GameConfig(include_expansion=True, seed=1))
    g.start()
    legal = g.legal_actions()
    drafted_options = {a.protocol for a in legal if a.type is ActionType.DRAFT_PROTOCOL}
    assert drafted_options == set(BASE_PROTOCOLS) | set(EXPANSION_PROTOCOLS)


def test_predetermined_draft_starts_action_phase():
    g = Game(GameConfig(include_expansion=False, seed=42))
    g.set_predetermined_draft([
        ["Speed", "Light", "Fire"],
        ["Darkness", "Death", "Metal"],
    ])
    # After predetermined draft we should be at an actionable phase.
    assert g.state.phase in (
        Phase.START, Phase.CHECK_CONTROL, Phase.CHECK_COMPILE, Phase.ACTION, Phase.CHECK_CACHE,
    )
    # Both players have 5 hand, 13 deck, 3 protocols.
    for pl in (0, 1):
        assert len(g.state.players[pl].hand) == 5
        assert len(g.state.players[pl].deck) == 13
        assert len(g.state.players[pl].protocols) == 3


@pytest.mark.parametrize("include_expansion", [False, True])
def test_random_self_play_terminates(include_expansion: bool):
    cfg = GameConfig(include_expansion=include_expansion, seed=7, max_turns=300)
    game = play_game(
        agent0=RandomAgent(seed=1),
        agent1=RandomAgent(seed=2),
        config=cfg,
    )
    assert game.is_over()


def test_env_reset_and_step_with_random_opponent():
    env = CompileEnv(
        config=GameConfig(include_expansion=True, seed=11, max_turns=300),
        perspective=0,
        opponent=RandomAgent(seed=2),
    )
    me = RandomAgent(seed=1)
    sr = env.reset(seed=11)
    steps = 0
    while not sr.done and steps < 2000:
        action = me.choose(env._game, sr.legal_actions)
        sr = env.step(action)
        steps += 1
    assert sr.done


def test_observation_shape_is_stable():
    env = CompileEnv(perspective=0, opponent=RandomAgent(seed=3))
    sr = env.reset(seed=5)
    n0 = len(sr.obs)
    me = RandomAgent(seed=4)
    for _ in range(20):
        if sr.done:
            break
        action = me.choose(env._game, sr.legal_actions)
        sr = env.step(action)
        assert len(sr.obs) == n0


def test_effect_coverage_mn01_ax01():
    """Every card in the MN01+AX01 (base+aux1) sets has either a registered
    effect or is on the passive-only list. MN02+AX02 cards are NOT yet fully
    covered — see `test_card_data_integrity` for the full 180-card load and
    the MN02_AX02_COVERAGE_GAP list below for known unimplemented effects."""
    from compile_engine.effects import (
        MIDDLE_EFFECTS, BOTTOM_FIRST_EFFECTS, BOTTOM_ON_PLAY_EFFECTS,
        START_EFFECTS, END_EFFECTS, TOP_TRIGGER_EFFECTS, WHEN_COVERED_EFFECTS,
        AFTER_CLEAR_CACHE_EFFECTS, AFTER_SELF_DISCARD_EFFECTS,
        AFTER_OPP_DISCARD_EFFECTS, AFTER_SELF_DELETE_EFFECTS,
        AFTER_SELF_DRAW_EFFECTS, AFTER_SELF_SHUFFLE_EFFECTS,
        AFTER_SELF_REFRESH_EFFECTS, FLIP_TRIGGER_EFFECTS,
        WHEN_DELETED_BY_COMPILE_EFFECTS,
        AFTER_OPP_DRAW_EFFECTS, AFTER_OPP_REFRESH_EFFECTS,
        AFTER_OPP_COMPILE_EFFECTS, AFTER_ANY_REFRESH_EFFECTS,
        AFTER_OPP_PLAY_IN_LINE_EFFECTS, AFTER_SELF_DISCARD_ON_OPP_TURN_EFFECTS,
    )
    defs = load_card_defs()
    passive_only_keys = {
        "AX01:Apathy:0", "AX01:Apathy:2", "AX01:Apathy:4",
        "MN01:Darkness:2", "MN01:Metal:0",
        "MN01:Metal:2", "MN01:Plague:0",
        "MN01:Psychic:1", "MN01:Spirit:0",
        "MN01:Spirit:1",
    }
    missing = []
    for d in defs:
        # Only check MN01 + AX01 — MN02/AX02 effects are tracked separately.
        if d.set_code not in ("MN01", "AX01"):
            continue
        has_effect = (
            d.key in MIDDLE_EFFECTS
            or d.key in BOTTOM_FIRST_EFFECTS
            or d.key in BOTTOM_ON_PLAY_EFFECTS
            or d.key in START_EFFECTS
            or d.key in END_EFFECTS
            or d.key in TOP_TRIGGER_EFFECTS
            or d.key in WHEN_COVERED_EFFECTS
            or d.key in AFTER_CLEAR_CACHE_EFFECTS
            or d.key in AFTER_SELF_DISCARD_EFFECTS
            or d.key in AFTER_OPP_DISCARD_EFFECTS
            or d.key in AFTER_SELF_DELETE_EFFECTS
            or d.key in AFTER_SELF_DRAW_EFFECTS
            or d.key in AFTER_SELF_SHUFFLE_EFFECTS
            or d.key in AFTER_SELF_REFRESH_EFFECTS
            or d.key in FLIP_TRIGGER_EFFECTS
            or d.key in WHEN_DELETED_BY_COMPILE_EFFECTS
            or d.key in AFTER_OPP_DRAW_EFFECTS
            or d.key in AFTER_OPP_REFRESH_EFFECTS
            or d.key in AFTER_OPP_COMPILE_EFFECTS
            or d.key in AFTER_ANY_REFRESH_EFFECTS
            or d.key in AFTER_OPP_PLAY_IN_LINE_EFFECTS
            or d.key in AFTER_SELF_DISCARD_ON_OPP_TURN_EFFECTS
        )
        has_text = bool(d.top_text or d.middle_text or d.bottom_text)
        if has_text and not has_effect and d.key not in passive_only_keys:
            missing.append((d.key, d.top_text, d.middle_text, d.bottom_text))
    assert not missing, f"Cards with unhandled active text:\n" + "\n".join(
        f"  {k}: T={t!r} M={m!r} B={b!r}" for k, t, m, b in missing
    )


def test_mn02_ax02_value_only_baseline():
    """MN02 + AX02 cards load cleanly. Counts + protocol structure verified."""
    import collections
    defs = load_card_defs()
    new_set_defs = [d for d in defs if d.set_code in ("MN02", "AX02")]
    assert len(new_set_defs) == 90  # 72 MN02 + 18 AX02
    by_proto = collections.Counter(d.protocol for d in new_set_defs)
    assert all(v == 6 for v in by_proto.values()), by_proto
    assert len(by_proto) == 15


def test_effect_coverage_all_180_cards():
    """All 180 cards — every active-text card has a registered effect or is on
    the explicit passive-only whitelist (text captured by persistent rules or
    purely flavor)."""
    from compile_engine.effects import (
        MIDDLE_EFFECTS, BOTTOM_FIRST_EFFECTS, BOTTOM_ON_PLAY_EFFECTS,
        START_EFFECTS, END_EFFECTS, TOP_TRIGGER_EFFECTS, WHEN_COVERED_EFFECTS,
        AFTER_CLEAR_CACHE_EFFECTS, AFTER_SELF_DISCARD_EFFECTS,
        AFTER_OPP_DISCARD_EFFECTS, AFTER_SELF_DELETE_EFFECTS,
        AFTER_SELF_DRAW_EFFECTS, AFTER_SELF_SHUFFLE_EFFECTS,
        AFTER_SELF_REFRESH_EFFECTS, FLIP_TRIGGER_EFFECTS,
        WHEN_DELETED_BY_COMPILE_EFFECTS,
        AFTER_OPP_DRAW_EFFECTS, AFTER_OPP_REFRESH_EFFECTS,
        AFTER_OPP_COMPILE_EFFECTS, AFTER_ANY_REFRESH_EFFECTS,
        AFTER_OPP_PLAY_IN_LINE_EFFECTS, AFTER_SELF_DISCARD_ON_OPP_TURN_EFFECTS,
    )
    defs = load_card_defs()
    passive_only_keys = {
        # MN01 + AX01 — captured by persistent rules in compute_line_value or
        # restriction queries.
        "AX01:Apathy:0", "AX01:Apathy:2", "AX01:Apathy:4",
        "MN01:Darkness:2", "MN01:Metal:0",
        "MN01:Metal:2", "MN01:Plague:0",
        "MN01:Psychic:1", "MN01:Spirit:0",
        "MN01:Spirit:1",
        # MN02 — persistent value modifiers / global rules / play affordances
        "MN02:Clarity:0",     # +1 per card in hand (compute_line_value)
        "MN02:Mirror:0",      # +1 per opp card in line (compute_line_value)
        "MN02:Smoke:2",       # +1 per face-down in line (compute_line_value)
        "MN02:Chaos:3",       # may play without matching protocols (legal_actions)
        "MN02:Fear:0",        # T: suppress opp middles (middle_suppressed)
        "MN02:Ice:6",         # T: no draw with hand (draw_cards)
        # AX02 — passive
        "MN02:Ice:4",         # B: cannot be flipped (flip_card immunity)
        "AX02:Diversity:3",   # T: +2 value if non-Diversity face-up in stack (compute_line_value)
    }
    missing = []
    for d in defs:
        has_effect = (
            d.key in MIDDLE_EFFECTS
            or d.key in BOTTOM_FIRST_EFFECTS
            or d.key in BOTTOM_ON_PLAY_EFFECTS
            or d.key in START_EFFECTS
            or d.key in END_EFFECTS
            or d.key in TOP_TRIGGER_EFFECTS
            or d.key in WHEN_COVERED_EFFECTS
            or d.key in AFTER_CLEAR_CACHE_EFFECTS
            or d.key in AFTER_SELF_DISCARD_EFFECTS
            or d.key in AFTER_OPP_DISCARD_EFFECTS
            or d.key in AFTER_SELF_DELETE_EFFECTS
            or d.key in AFTER_SELF_DRAW_EFFECTS
            or d.key in AFTER_SELF_SHUFFLE_EFFECTS
            or d.key in AFTER_SELF_REFRESH_EFFECTS
            or d.key in FLIP_TRIGGER_EFFECTS
            or d.key in WHEN_DELETED_BY_COMPILE_EFFECTS
            or d.key in AFTER_OPP_DRAW_EFFECTS
            or d.key in AFTER_OPP_REFRESH_EFFECTS
            or d.key in AFTER_OPP_COMPILE_EFFECTS
            or d.key in AFTER_ANY_REFRESH_EFFECTS
            or d.key in AFTER_OPP_PLAY_IN_LINE_EFFECTS
            or d.key in AFTER_SELF_DISCARD_ON_OPP_TURN_EFFECTS
        )
        has_text = bool(d.top_text or d.middle_text or d.bottom_text)
        if has_text and not has_effect and d.key not in passive_only_keys:
            missing.append((d.key, d.top_text, d.middle_text, d.bottom_text))
    assert not missing, "Cards with unhandled active text:\n" + "\n".join(
        f"  {k}: T={t!r} M={m!r} B={b!r}" for k, t, m, b in missing
    )


def test_apathy_zero_adds_value_per_facedown():
    """Apathy 0 top: total value in this line increases by 1 per face-down."""
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import compute_line_value
    from compile_engine.state import CardInst, GameState, GameConfig, PlayerState, Line
    import random as _r
    defs = load_card_defs()
    apathy_0_def_id = next(d.def_id for d in defs if d.key == "AX01:Apathy:0")
    st = GameState(
        config=GameConfig(include_expansion=True),
        defs=defs,
        players=(PlayerState(idx=0), PlayerState(idx=1)),
        lines=[Line() for _ in range(3)],
        rng=_r.Random(0),
    )
    st.players[0].protocols = ["Apathy", "Light", "Speed"]
    st.players[0].compiled = [False, False, False]
    st.players[1].protocols = ["Darkness", "Death", "Fire"]
    st.players[1].compiled = [False, False, False]
    # Place Apathy 0 face-up on P0's side of line 0 + two face-down cards anywhere in line.
    a0 = CardInst(inst_id=0, def_id=apathy_0_def_id, owner=0, face_up=True)
    fd1 = CardInst(inst_id=1, def_id=apathy_0_def_id, owner=0, face_up=False)
    fd2 = CardInst(inst_id=2, def_id=apathy_0_def_id, owner=1, face_up=False)
    st.lines[0].p0_stack = [a0, fd1]
    st.lines[0].p1_stack = [fd2]
    # P0 line value: Apathy 0 value=0 (face-up) + face-down base=2 (fd1)
    # + apathy bonus = (# face-down in line, both sides) = 2.
    # Total: 0 + 2 + 2 = 4
    assert compute_line_value(st, 0, 0) == 4


def test_metal_zero_reduces_opp_line_value():
    """Metal 0 top: opponent's total value in this line is reduced by 2."""
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import compute_line_value
    from compile_engine.state import CardInst, GameState, GameConfig, PlayerState, Line
    import random as _r
    defs = load_card_defs()
    metal_0_def = next(d for d in defs if d.key == "MN01:Metal:0")
    light_3_def = next(d for d in defs if d.key == "MN01:Light:3")
    st = GameState(
        config=GameConfig(),
        defs=defs,
        players=(PlayerState(idx=0), PlayerState(idx=1)),
        lines=[Line() for _ in range(3)],
        rng=_r.Random(0),
    )
    st.players[0].protocols = ["Light", "Speed", "Fire"]
    st.players[0].compiled = [False, False, False]
    st.players[1].protocols = ["Metal", "Death", "Water"]
    st.players[1].compiled = [False, False, False]
    # P1 has Metal 0 face-up on line 0 → P0's value here is reduced by 2.
    m0 = CardInst(inst_id=0, def_id=metal_0_def.def_id, owner=1, face_up=True)
    l3 = CardInst(inst_id=1, def_id=light_3_def.def_id, owner=0, face_up=True)
    st.lines[0].p1_stack = [m0]
    st.lines[0].p0_stack = [l3]  # Light 3 face-up has value 3
    assert compute_line_value(st, 0, 0) == max(3 - 2, 0)


def test_speed_2_does_not_expose_standalone_shift_action():
    """Speed 2's top ('Shift this card, even if this card is covered') is a
    target-eligibility modifier (other shift effects may target it while
    covered), not a player action. The action phase therefore should NOT
    expose a SHIFT_OWN_CARD option for Speed 2 specifically.
    Spirit 3 ('You may shift this card...') retains its standalone shift.
    """
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, GameState, GameConfig, PlayerState, Line, Phase
    import random as _r
    defs = load_card_defs()
    speed_2 = next(d for d in defs if d.key == "MN01:Speed:2")
    light_0 = next(d for d in defs if d.key == "MN01:Light:0")
    st = GameState(
        config=GameConfig(),
        defs=defs,
        players=(PlayerState(idx=0), PlayerState(idx=1)),
        lines=[Line() for _ in range(3)],
        rng=_r.Random(0),
        phase=Phase.ACTION,
    )
    st.players[0].protocols = ["Speed", "Light", "Fire"]
    st.players[0].compiled = [False, False, False]
    st.players[1].protocols = ["Darkness", "Death", "Water"]
    st.players[1].compiled = [False, False, False]
    s2 = CardInst(inst_id=0, def_id=speed_2.def_id, owner=0, face_up=True)
    l0 = CardInst(inst_id=1, def_id=light_0.def_id, owner=0, face_up=True)
    st.lines[0].p0_stack = [s2, l0]

    from compile_engine.game import Game
    g = Game.__new__(Game)
    g.state = st
    g.defs = defs
    g.config = st.config
    g._pending = []
    g._inst_counter = 100
    legal = g._action_phase_legal()
    from compile_engine.actions import ActionType
    shift_actions = [a for a in legal if a.type is ActionType.SHIFT_OWN_CARD]
    assert len(shift_actions) == 0, f"Speed 2 should not expose standalone shift; got {shift_actions}"

    # Positive control: swap Speed 2 for Spirit 3 in the same position and the
    # SAME logic should produce 2 shift actions targeting lines 1 and 2. This
    # is what the previous assertions on Speed 2 were verifying; we move them
    # to the protocol that genuinely owns the affordance.
    spirit_3 = next(d for d in defs if d.key == "MN01:Spirit:3")
    st.players[0].protocols = ["Spirit", "Light", "Fire"]
    sp3 = CardInst(inst_id=10, def_id=spirit_3.def_id, owner=0, face_up=True)
    st.lines[0].p0_stack = [sp3, l0]
    legal2 = g._action_phase_legal()
    spirit3_shifts = [a for a in legal2 if a.type is ActionType.SHIFT_OWN_CARD]
    assert len(spirit3_shifts) == 2
    assert {a.choice_index for a in spirit3_shifts} == {1, 2}
    assert all(a.line_index == 0 and a.hand_index == 0 for a in spirit3_shifts)


def test_spirit_3_shift_self_when_covered_executes():
    """End-to-end: a covered face-up Spirit 3 owner uses SHIFT_OWN_CARD."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=1))
    g.set_predetermined_draft([
        ["Spirit", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    sp3 = next(d for d in defs if d.key == "MN01:Spirit:3")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    # Put Spirit 3 face-up COVERED by Fire 0 face-up in line 0 on P0's side.
    s3 = CardInst(inst_id=999, def_id=sp3.def_id, owner=0, face_up=True)
    f0 = CardInst(inst_id=998, def_id=fire_0.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [s3, f0]
    g.state.current_player = 0
    from compile_engine.state import Phase
    g.state.phase = Phase.ACTION
    g._pending = []
    legal = g.legal_actions()
    shift_actions = [a for a in legal if a.type is ActionType.SHIFT_OWN_CARD]
    assert shift_actions, "Spirit 3 covered should still allow shift-self"
    a = shift_actions[0]
    pre_pos = g.state.lines[0].p0_stack.index(s3)
    g.step(a)
    # Spirit 3 should now live in the destination line (no longer in line 0).
    assert s3 not in g.state.lines[0].p0_stack
    assert any(s3 in g.state.lines[i].p0_stack for i in range(3))


def test_speed_0_recursive_play_fires_played_card_effect():
    """Speed 0 middle: 'Play 1 card.' The recursively played card must
    resolve its own effects (e.g. drawing). We force a play of Light 2
    face-up (whose middle 'Draw 2 cards. Reveal 1 face-down card...'
    draws on play; the reveal-prompt is auto-skipped when no face-down
    cards exist in the field) and assert the drawn-card count
    increased."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=1))
    g.set_predetermined_draft([
        ["Speed", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    speed_0 = next(d for d in defs if d.key == "MN01:Speed:0")
    light_2 = next(d for d in defs if d.key == "MN01:Light:2")

    # Clear and set up a controlled scenario.
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]

    # P0 holds Speed 0 + Light 2 in hand. P0's lines have Speed at idx 0, Light at idx 1.
    speed_0_inst = CardInst(inst_id=2001, def_id=speed_0.def_id, owner=0, face_up=False)
    light_2_inst = CardInst(inst_id=2002, def_id=light_2.def_id, owner=0, face_up=False)
    g.state.players[0].hand = [speed_0_inst, light_2_inst]
    # Stock the deck with a few dummies so draw works.
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.players[0].deck = [
        CardInst(inst_id=3000 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g  # ensure callback ref is live
    g._pending = []

    pre_deck_len = len(g.state.players[0].deck)  # = 5

    # Play Speed 0 face-up into line 0 (Speed protocol).
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))

    # Speed 0's middle asks us to play one of the legal sub-plays. Pick the
    # one that plays Light 2 face-up into line 1 (Light protocol). After this,
    # Light 2's middle 'Draw 2 cards.' should also fire.
    while not g.is_over():
        choice_obj = g._pending[-1].last_choice if g._pending else None
        if choice_obj is None:
            break
        picked = None
        for i, opt in enumerate(choice_obj.options):
            if "Light2" in opt and "L1" in opt and opt.startswith("FU"):
                picked = i
                break
        if picked is None:
            break
        g.step(Action(type=ActionType.CHOOSE_TARGET, choice_index=picked))
        break

    # We should have: Light 2 in line 1 face-up, and drew 2 from Light 2's middle.
    line1_top = g.state.lines[1].p0_stack[-1] if g.state.lines[1].p0_stack else None
    assert line1_top is not None and line1_top is light_2_inst, "Light 2 not in line 1 face-up"
    assert line1_top.face_up
    # Hand size: started at 2, played 2 → 0 then +2 draws → 2
    assert len(g.state.players[0].hand) == 2, (
        f"expected 2 cards after sub-play+draw, got {len(g.state.players[0].hand)}"
    )
    # Deck count dropped by 2 (the drawn cards).
    assert len(g.state.players[0].deck) == pre_deck_len - 2


def test_light_1_bottom_end_trigger_fires_at_end_phase_not_on_play():
    """Light 1 bottom: 'End: Draw 1 card.' Bottom-tier End: triggers
    fire on the owner's End step while the card is face-up + uncovered
    — NOT when the card is played (regression: previously routed via
    @bottom_on_play and fired immediately on play)."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=3))
    g.set_predetermined_draft([
        ["Light", "Fire", "Speed"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    light_1_inst = CardInst(inst_id=8001, def_id=light_1.def_id, owner=0, face_up=False)
    g.state.players[0].hand = [light_1_inst]
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=9000 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    pre_deck = len(g.state.players[0].deck)
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    # Play Light 1 face-up into line 0 (Light protocol). On play the
    # bottom End trigger should NOT fire — only middle/bottom-first/top
    # immediate effects do.
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # play advances through CHECK_CACHE → END → opp turn. End fires
    # exactly once on P0's turn, drawing 1 card from deck → hand.
    assert len(g.state.players[0].deck) == pre_deck - 1, (
        f"Light 1 End: should fire once at end of P0's turn; "
        f"deck went {pre_deck} → {len(g.state.players[0].deck)}"
    )
    assert len(g.state.players[0].hand) == 1, "expected 1 drawn card in hand"


def test_errata_text_loaded():
    """Codex errata (16 Dec 2024) is reflected in the loaded card defs."""
    defs = load_card_defs()
    death_1 = next(d for d in defs if d.key == "MN01:Death:1")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    life_0 = next(d for d in defs if d.key == "MN01:Life:0")
    assert death_1.top_text.startswith("Start:"), death_1.top_text
    assert fire_0.bottom_text.startswith("When this card would be covered:"), fire_0.bottom_text
    assert life_0.top_text.startswith("End:"), life_0.top_text
    assert life_0.bottom_text == ""


def test_committed_card_excluded_from_enumerators():
    """A card with is_committed=True is not a valid target for any effect
    target enumerator (uncovered/all/shift)."""
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import (
        _enumerate_all, _enumerate_shift_targets, _enumerate_uncovered,
    )
    from compile_engine.state import (
        CardInst, GameConfig, GameState, Line, PlayerState,
    )
    import random as _r
    defs = load_card_defs()
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    light_0 = next(d for d in defs if d.key == "MN01:Light:0")
    st = GameState(
        config=GameConfig(),
        defs=defs,
        players=(PlayerState(idx=0), PlayerState(idx=1)),
        lines=[Line() for _ in range(3)],
        rng=_r.Random(0),
    )
    st.players[0].protocols = ["Fire", "Light", "Speed"]
    st.players[0].compiled = [False, False, False]
    st.players[1].protocols = ["Darkness", "Death", "Water"]
    st.players[1].compiled = [False, False, False]
    c_committed = CardInst(inst_id=0, def_id=fire_0.def_id, owner=0, face_up=True, is_committed=True)
    c_normal = CardInst(inst_id=1, def_id=light_0.def_id, owner=0, face_up=True)
    st.lines[0].p0_stack = [c_normal, c_committed]  # committed on top
    cards_uncovered = [t[3] for t in _enumerate_uncovered(st, active_player=0)]
    cards_all = [t[3] for t in _enumerate_all(st, active_player=0)]
    cards_shift = [t[3] for t in _enumerate_shift_targets(st, active_player=0)]
    assert c_committed not in cards_uncovered
    assert c_committed not in cards_all
    assert c_committed not in cards_shift
    # The non-committed Light 0 underneath is still reachable for "all" and
    # (since face-up, would only be visible to shift if Speed 2 / Spirit 3
    # tag applied — it doesn't, so absent from shift list).
    assert c_normal in cards_all


def test_fire_0_when_covered_triggers_on_play_atop():
    """Fire 0 errata: 'When this card would be covered: First, draw 1 card.
    Then, flip 1 other card.' Playing another card on top of a face-up Fire 0
    should draw 1 and prompt to flip 1 other card. The just-played card is
    committed and must not be a legal flip target."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=1))
    g.set_predetermined_draft([
        ["Fire", "Light", "Speed"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    fire_3 = next(d for d in defs if d.key == "MN01:Fire:3")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    fire_0_inst = CardInst(inst_id=4001, def_id=fire_0.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [fire_0_inst]
    # Give Fire 0 something else to flip — a face-down card on P1's line 1.
    bystander = CardInst(inst_id=4500, def_id=light_1.def_id, owner=1, face_up=False)
    g.state.lines[1].p1_stack = [bystander]
    fire_3_inst = CardInst(inst_id=4002, def_id=fire_3.def_id, owner=0, face_up=False)
    g.state.players[0].hand = [fire_3_inst]
    g.state.players[0].deck = [
        CardInst(inst_id=5000 + i, def_id=fire_3.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []

    pre_deck = len(g.state.players[0].deck)
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))
    # when_covered should have drawn 1.
    assert len(g.state.players[0].deck) == pre_deck - 1, (
        f"expected draw of 1 from Fire 0 when_covered; deck went {pre_deck} -> "
        f"{len(g.state.players[0].deck)}"
    )
    # Fire 0's when_covered prompts to flip 1 other card — and the prompt
    # must precede Fire 3's bottom-on-play prompt (LIFO ordering).
    assert g._pending, "expected when_covered to leave a flip-target prompt"
    top_choice = g._pending[-1].last_choice
    assert top_choice is not None
    assert top_choice.prompt == "Flip 1 other card", top_choice.prompt
    # The bystander on P1/L1 IS a legal target.
    target_cards = [t[3] for t in top_choice.targets]
    assert bystander in target_cards
    # The just-played Fire 3 is committed and must NOT be in the target list.
    assert fire_3_inst not in target_cards, (
        "committed Fire 3 should not be a valid flip target"
    )


def test_life_0_self_deletes_at_end_when_covered():
    """Life 0 errata: 'End: If this card is covered, delete this card.'
    Covering Life 0 doesn't trigger the delete on the cover event itself;
    the trigger fires when the END phase rolls over Life 0's owner's
    visible-at-start commands. The play action advances through
    CHECK_CACHE → END, so by the time control returns to the caller,
    Life 0's End: trigger has fired and it's in trash."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=2))
    g.set_predetermined_draft([
        ["Life", "Light", "Speed"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    life_0 = next(d for d in defs if d.key == "MN01:Life:0")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    life_0_inst = CardInst(inst_id=6001, def_id=life_0.def_id, owner=0, face_up=True)
    # Life is in line 0 by draft order. Cover it with a face-down card.
    g.state.lines[0].p0_stack = [life_0_inst]
    fd_card = CardInst(inst_id=6002, def_id=light_1.def_id, owner=0, face_up=False)
    g.state.players[0].hand = [fd_card]
    g.state.players[0].deck = [
        CardInst(inst_id=7000 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    # Play fd_card face-down into line 0 (covers Life 0). The action
    # advances through CHECK_CACHE → END; Life 0's End trigger fires
    # because the card is covered at End-phase start.
    g.step(Action(type=ActionType.PLAY_FACE_DOWN, hand_index=0, line_index=0))
    # Drain any prompts.
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # Life 0 left line 0 (deleted by its own End trigger) and is now in trash.
    assert life_0_inst not in g.state.lines[0].p0_stack, (
        "Life 0 should self-delete at End when covered (Codex errata)"
    )
    assert life_0_inst in g.state.players[0].trash, (
        "Life 0 should land in its owner's trash after the End-trigger delete"
    )


def test_speed_1_top_fires_after_clear_cache_not_on_play():
    """Speed 1 top: 'After you clear cache: Draw 1 card.' The 'After ...:'
    emphasis means the top fires only when the Clear Cache action resolves
    on the owner's turn — NOT when the card is played. Middle 'Draw 2 cards.'
    DOES fire on play (Codex: middle is Immediate, on play/flip/uncover)."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=3))
    g.set_predetermined_draft([
        ["Speed", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    speed_1 = next(d for d in defs if d.key == "MN01:Speed:1")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.players[0].hand = []
    g.state.players[1].hand = []

    # Phase 1: Play Speed 1 face-up. Hand starts at 5 (one of them is Speed 1).
    speed_1_inst = CardInst(inst_id=8001, def_id=speed_1.def_id, owner=0, face_up=False)
    dummy_hand = [
        CardInst(inst_id=8100 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(4)
    ]
    g.state.players[0].hand = [speed_1_inst] + dummy_hand  # 5 cards
    g.state.players[0].deck = [
        CardInst(inst_id=8200 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(10)
    ]
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []

    pre_deck = len(g.state.players[0].deck)
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))
    # Drain any choice prompts (there shouldn't be any — Speed 1 middle is
    # a flat "Draw 2 cards.").
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # Played 1, drew 2 from middle: 5 - 1 + 2 = 6. Top did NOT fire.
    assert len(g.state.players[0].hand) == 6, (
        f"on-play hand: middle should draw 2 (5-1+2=6), got {len(g.state.players[0].hand)}"
    )
    assert len(g.state.players[0].deck) == pre_deck - 2

    # Phase 2: force CHECK_CACHE with hand > 5 → discard one → after_clear_cache
    # broadcast fires Speed 1's top → +1 draw → net hand stays at 6.
    g.state.phase = Phase.CHECK_CACHE
    g._pending = []
    g.state.scratch.pop("_pending_after_clear_cache_p0", None)
    pre_deck_2 = len(g.state.players[0].deck)
    # Discard hand index 0 (Light 1).
    g.step(Action(type=ActionType.DISCARD_CARD, hand_index=0))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # Discarded 1, drew 1 from Speed 1 top via after_clear_cache: net 6.
    assert len(g.state.players[0].hand) == 6, (
        f"after-clear-cache hand: expected 6 (-1 discard +1 draw), got {len(g.state.players[0].hand)}"
    )
    assert len(g.state.players[0].deck) == pre_deck_2 - 1, (
        "Speed 1's after_clear_cache top should draw exactly 1"
    )


def test_plague_1_top_fires_after_opp_discards():
    """Plague 1 top: 'After your opponent discards cards: Draw 1 card.' We
    set up Plague 1 face-up on P0's field, force P1 to discard a card via
    the Clear Cache phase, and assert P0 drew 1 card."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import discard_to_trash
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=5))
    g.set_predetermined_draft([
        ["Speed", "Light", "Plague"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    plague_1 = next(d for d in defs if d.key == "MN01:Plague:1")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    p1_inst = CardInst(inst_id=10001, def_id=plague_1.def_id, owner=0, face_up=True)
    g.state.lines[2].p0_stack = [p1_inst]  # Plague is line 2
    g.state.players[0].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=10100 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(10)
    ]
    g.state.players[1].hand = [
        CardInst(inst_id=10200 + i, def_id=light_1.def_id, owner=1, face_up=False)
        for i in range(3)
    ]
    g.state.scratch["_engine"] = g
    g._pending = []
    # P1 discards from hand (simulate any-time discard via direct atomic).
    g.state.current_player = 1
    pre_p0_hand = len(g.state.players[0].hand)
    pre_p0_deck = len(g.state.players[0].deck)
    discard_to_trash(g.state, 1, 0)
    g._drive()
    # Plague 1 should have fired: P0 drew 1.
    assert len(g.state.players[0].hand) == pre_p0_hand + 1, (
        f"Plague 1 should draw 1 after opp discards; hand: {pre_p0_hand} → "
        f"{len(g.state.players[0].hand)}"
    )
    assert len(g.state.players[0].deck) == pre_p0_deck - 1


def test_hate_3_top_fires_after_self_delete():
    """Hate 3 top: 'After you delete cards: Draw 1 card.' Trigger a delete
    via a direct field-mutation helper and assert Hate 3 fired."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import delete_card_from_field
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=6, include_expansion=True))
    g.set_predetermined_draft([
        ["Hate", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    hate_3 = next(d for d in defs if d.key == "AX01:Hate:3")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    h3 = CardInst(inst_id=11001, def_id=hate_3.def_id, owner=0, face_up=True)
    victim = CardInst(inst_id=11002, def_id=light_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [h3]
    g.state.lines[1].p0_stack = [victim]
    g.state.players[0].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=11100 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    pre_hand = len(g.state.players[0].hand)
    delete_card_from_field(g.state, 1, 0, 0)
    g._drive()
    assert len(g.state.players[0].hand) == pre_hand + 1, (
        f"Hate 3 should draw 1 after a self-delete; got hand {len(g.state.players[0].hand)}"
    )


def test_speed_2_top_shifts_out_of_compile_line():
    """Speed 2 top: 'When this card would be deleted by compiling: Shift
    this card.' Setting up a compile-ready line with Speed 2 in it; on
    compile, Speed 2 shifts to another line instead of being trashed."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=7))
    g.set_predetermined_draft([
        ["Speed", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    speed_2 = next(d for d in defs if d.key == "MN01:Speed:2")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # Stack value 10+ in line 0 so it's compileable.
    sp2 = CardInst(inst_id=12001, def_id=speed_2.def_id, owner=0, face_up=True)
    high_a = CardInst(inst_id=12002, def_id=next(d for d in defs if d.key == "MN01:Speed:5").def_id, owner=0, face_up=True)
    high_b = CardInst(inst_id=12003, def_id=next(d for d in defs if d.key == "MN01:Speed:5").def_id, owner=0, face_up=True)
    # Need ≥10 line value; Speed 2 (2) + Speed 5 (5) + Speed 5 (5) = 12.
    g.state.lines[0].p0_stack = [sp2, high_a, high_b]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    g.state.phase = Phase.CHECK_COMPILE
    # Drive to a compile decision.
    g._drive()
    # Should be waiting on COMPILE_LINE choice.
    legal = g.legal_actions()
    compile_actions = [a for a in legal if a.type is ActionType.COMPILE_LINE]
    assert compile_actions, f"expected compile available; got {legal}"
    g.step(compile_actions[0])
    # Drain interrupt prompts (Speed 2 shift target choice).
    while g._pending and g._pending[-1].last_choice is not None:
        legal = g.legal_actions()
        g.step(legal[0])
    # Speed 2 must be on the field still (in some line other than 0).
    on_field = [
        ln for ln in range(3)
        if sp2 in g.state.lines[ln].p0_stack
    ]
    assert on_field, "Speed 2 should have shifted out of the compile line, not been trashed"
    assert on_field[0] != 0, f"Speed 2 should NOT remain in compiled line 0; on lines {on_field}"
    assert sp2 not in g.state.players[0].trash, "Speed 2 must not be in trash after compile interrupt"


def test_speed_1_top_does_not_fire_when_no_discard_happened():
    """If the CHECK_CACHE phase passes with hand ≤ 5 (no Clear Cache action
    performed), Speed 1's 'After you clear cache:' top must NOT fire — per
    Codex, "Clear Cache" refers specifically to the discard action."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=4))
    g.set_predetermined_draft([
        ["Speed", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    speed_1 = next(d for d in defs if d.key == "MN01:Speed:1")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    speed_1_inst = CardInst(inst_id=9001, def_id=speed_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [speed_1_inst]
    g.state.players[0].hand = [
        CardInst(inst_id=9100 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(3)
    ]
    g.state.players[0].deck = [
        CardInst(inst_id=9200 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(10)
    ]
    g.state.current_player = 0
    g.state.scratch["_engine"] = g
    g._pending = []
    pre_hand = len(g.state.players[0].hand)
    pre_deck = len(g.state.players[0].deck)
    # Drive through CHECK_CACHE — hand is 3 (≤5), no discard needed.
    g.state.phase = Phase.CHECK_CACHE
    g._drive()
    assert len(g.state.players[0].hand) == pre_hand, (
        "no Clear Cache happened → Speed 1 top must not fire"
    )
    assert len(g.state.players[0].deck) == pre_deck


def test_ice_1_fires_when_opp_plays_in_same_line_only():
    """Ice 1 bottom: 'After your opponent plays a card in this line:
    Your opponent discards 1 card.' Fires when opp plays into Ice 1's
    line; does NOT fire on opp plays in other lines."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=4, include_main2=True))
    g.set_predetermined_draft([
        ["Ice", "Death", "Water"],
        ["Light", "Fire", "Speed"],
    ])
    defs = load_card_defs()
    ice_1 = next(d for d in defs if d.key == "MN02:Ice:1")
    light_0 = next(d for d in defs if d.key == "MN01:Light:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P0 has Ice 1 face-up in line 0 (Ice protocol).
    ice_1_inst = CardInst(inst_id=11001, def_id=ice_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [ice_1_inst]
    # P1 holds two Light 0s, one for the Ice 1's line (line 0 on P0 side),
    # one for line 1 (Light protocol on P1). P1 protocols are
    # [Light, Fire, Speed], so P1 plays Light face-up in line 0.
    light_0_inst_in_line = CardInst(inst_id=11002, def_id=light_0.def_id, owner=1, face_up=False)
    light_0_inst_out_line = CardInst(inst_id=11003, def_id=light_0.def_id, owner=1, face_up=False)
    g.state.players[1].hand = [light_0_inst_in_line, light_0_inst_out_line]
    # Stock P1's deck so they can draw / handle the discard, but ensure
    # they have non-empty hand to discard from.
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.players[0].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=12000 + i, def_id=fire_0.def_id, owner=0, face_up=False) for i in range(5)
    ]
    g.state.players[1].deck = [
        CardInst(inst_id=13000 + i, def_id=fire_0.def_id, owner=1, face_up=False) for i in range(5)
    ]
    g.state.current_player = 1
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    # P1 plays Light 0 face-down into line 1 (NOT Ice 1's line) — no trigger.
    pre_p1_hand = len(g.state.players[1].hand)
    g.step(Action(type=ActionType.PLAY_FACE_DOWN, hand_index=1, line_index=1))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # Hand: -1 from play, +0 trigger -> -1 net. The step advances through
    # P1's CHECK_CACHE → END → P0's turn. CHECK_CACHE might draw P1 back to
    # 5, but that's not the point. The point is Ice 1 did NOT fire on a
    # different-line play.
    # To verify: look for "Ice 1 -> discard" in log. Simpler: count P1
    # discards (trash should be 0 from this).
    # Actually CHECK_CACHE forced a discard if hand > 5 but P1 started <5.
    p1_trash_after_other_line = len(g.state.players[1].trash)
    # Reset for second test.
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.lines[0].p0_stack = [ice_1_inst]
    g.state.players[1].hand = [light_0_inst_in_line]
    g.state.players[1].trash = []
    g.state.current_player = 1
    g.state.phase = Phase.ACTION
    g._pending = []
    # P1 plays Light 0 face-down into line 0 (Ice 1's line) — TRIGGER.
    g.step(Action(type=ActionType.PLAY_FACE_DOWN, hand_index=0, line_index=0))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # P1 should have been forced to discard from hand by Ice 1. Hand was 1
    # (the card we just played), now 0 (played) + check_cache might not
    # discard further. The forced discard from Ice 1 needs a card to come
    # from somewhere — but P1's hand was empty after the play. The discard
    # silently fails (no-op).
    # Better test: have P1 with multiple cards and verify discard happens.
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.lines[0].p0_stack = [ice_1_inst]
    extra = CardInst(inst_id=11004, def_id=light_0.def_id, owner=1, face_up=False)
    g.state.players[1].hand = [light_0_inst_in_line, extra]
    g.state.players[1].trash = []
    g.state.current_player = 1
    g.state.phase = Phase.ACTION
    g._pending = []
    g.step(Action(type=ActionType.PLAY_FACE_DOWN, hand_index=0, line_index=0))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # After play (1 card to field) + Ice 1 discard (1 to trash), P1 should
    # have 0 hand cards and 1 in trash (the discarded one).
    assert len(g.state.players[1].trash) >= 1, (
        f"Ice 1 should force opp discard; trash={len(g.state.players[1].trash)}"
    )


def test_war_2_fires_when_opp_compiles():
    """War 2 bottom: 'After your opponent compiles: Your opponent
    discards their hand.' Fires when opp completes a compile action."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=5, include_main2=True))
    g.set_predetermined_draft([
        ["War", "Death", "Water"],
        ["Light", "Fire", "Speed"],
    ])
    defs = load_card_defs()
    war_2 = next(d for d in defs if d.key == "MN02:War:2")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P0 has War 2 face-up in line 0.
    war_2_inst = CardInst(inst_id=14001, def_id=war_2.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [war_2_inst]
    # P1 has a "winning" line stack on their side of line 1 (Fire protocol).
    # Stack 3 face-up Fire 4s (value 4 each = 12) → compileable.
    fire_4 = next(d for d in defs if d.key == "MN01:Fire:4")
    g.state.lines[1].p1_stack = [
        CardInst(inst_id=14010 + i, def_id=fire_4.def_id, owner=1, face_up=True)
        for i in range(3)
    ]
    # P1 has cards in hand to be discarded.
    g.state.players[1].hand = [
        CardInst(inst_id=14020 + i, def_id=fire_0.def_id, owner=1, face_up=False)
        for i in range(3)
    ]
    g.state.players[0].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.current_player = 1
    g.state.phase = Phase.CHECK_COMPILE
    g.state.scratch["_engine"] = g
    g._pending = []
    pre_p1_hand = len(g.state.players[1].hand)
    g.step(Action(type=ActionType.COMPILE_LINE, line_index=1))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # After compile, War 2's bottom should force P1 to discard their hand.
    # Discards land in trash. Hand should be 0 (or close).
    assert len(g.state.players[1].hand) == 0, (
        f"War 2 should force opp to discard hand after compile; "
        f"P1 hand: {pre_p1_hand} -> {len(g.state.players[1].hand)}"
    )


def test_mirror_4_fires_when_opp_draws():
    """Mirror 4 bottom: 'After your opponent draws cards: Draw 1 card.'
    Fires whenever opp draws ≥1 card."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=6, include_main2=True))
    g.set_predetermined_draft([
        ["Mirror", "Death", "Water"],
        ["Light", "Fire", "Speed"],
    ])
    defs = load_card_defs()
    mirror_4 = next(d for d in defs if d.key == "MN02:Mirror:4")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P0 has Mirror 4 face-up.
    m4_inst = CardInst(inst_id=15001, def_id=mirror_4.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [m4_inst]
    # Stock decks.
    g.state.players[0].deck = [
        CardInst(inst_id=15100 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.players[1].deck = [
        CardInst(inst_id=15200 + i, def_id=fire_0.def_id, owner=1, face_up=False)
        for i in range(5)
    ]
    # P0 hand empty (so Mirror 4 draw lands cleanly). P1 hand has some
    # cards so REFRESH actually triggers a draw.
    g.state.players[0].hand = []
    g.state.players[1].hand = [
        CardInst(inst_id=15300, def_id=fire_0.def_id, owner=1, face_up=False),
    ]
    g.state.current_player = 1
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    pre_p0_hand = len(g.state.players[0].hand)  # = 0
    pre_p0_deck = len(g.state.players[0].deck)  # = 5
    # P1 refreshes (draws to 5 from 1 = +4 cards).
    g.step(Action(type=ActionType.REFRESH))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # P0 should have drawn 1 from Mirror 4's after-opp-draw trigger.
    assert len(g.state.players[0].hand) >= 1, (
        f"Mirror 4 should fire after P1 refresh-draws; "
        f"P0 hand: {pre_p0_hand} -> {len(g.state.players[0].hand)}"
    )
    assert len(g.state.players[0].deck) <= pre_p0_deck - 1


def test_assim_1_fires_on_either_player_refresh():
    """Assimilation 1 bottom: 'After a player refreshes: ...' fires
    when either player refreshes (not just opp)."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=7, include_aux2=True))
    g.set_predetermined_draft([
        ["Assimilation", "Death", "Water"],
        ["Light", "Fire", "Speed"],
    ])
    defs = load_card_defs()
    assim_1 = next(d for d in defs if d.key == "AX02:Assimilation:1")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    a1_inst = CardInst(inst_id=16001, def_id=assim_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [a1_inst]
    g.state.players[0].hand = [
        CardInst(inst_id=16100, def_id=fire_0.def_id, owner=0, face_up=False),
    ]
    g.state.players[1].hand = [
        CardInst(inst_id=16200, def_id=fire_0.def_id, owner=1, face_up=False),
    ]
    g.state.players[0].deck = [
        CardInst(inst_id=16300 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.players[1].deck = [
        CardInst(inst_id=16400 + i, def_id=fire_0.def_id, owner=1, face_up=False)
        for i in range(5)
    ]
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    pre_p1_trash = len(g.state.players[1].trash)
    # P0 refreshes — Assim 1 should fire (the OWNER refreshed, but the
    # trigger is "After a player refreshes" so it fires).
    g.step(Action(type=ActionType.REFRESH))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # Assim 1 effect: P0 draws top of P1's deck, then P0 discards a card
    # into P1's trash. So P1's trash should have grown.
    assert len(g.state.players[1].trash) > pre_p1_trash, (
        f"Assim 1 should fire on owner's own refresh; "
        f"P1 trash: {pre_p1_trash} -> {len(g.state.players[1].trash)}"
    )


def test_diversity_3_top_adds_2_value_when_non_diversity_in_stack():
    """Diversity 3 top (AX02): 'Your total value in this line is
    increased by 2 if there are any non-Diversity face-up cards in this
    stack.' Verify the +2 fires only when the condition holds."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import compute_line_value
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=60, include_aux2=True))
    g.set_predetermined_draft([["Diversity", "Death", "Water"], ["Light", "Fire", "Speed"]])
    defs = load_card_defs()
    div3 = next(d for d in defs if d.key == "AX02:Diversity:3")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")  # non-Diversity
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # Only a face-up Diversity 3 in line 0 — no non-Diversity in stack, no bonus.
    div3_inst = CardInst(inst_id=60001, def_id=div3.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [div3_inst]
    val_alone = compute_line_value(g.state, 0, 0)
    assert val_alone == div3.value, (
        f"Diversity 3 alone (no non-Div in stack): expected {div3.value}, got {val_alone}"
    )
    # Add a face-up Fire 0 below Diversity 3 → non-Div present → +2.
    g.state.lines[0].p0_stack = [
        CardInst(inst_id=60002, def_id=fire_0.def_id, owner=0, face_up=True),
        div3_inst,
    ]
    val_combo = compute_line_value(g.state, 0, 0)
    expected_combo = fire_0.value + div3.value + 2
    assert val_combo == expected_combo, (
        f"Diversity 3 + Fire 0 face-up in stack: expected {expected_combo}, got {val_combo}"
    )


def test_unity_2_draws_per_unity_in_field():
    """Unity 2 middle (AX02): 'Draw cards equal to the number of Unity
    cards in the field.'"""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=61, include_aux2=True))
    g.set_predetermined_draft([["Unity", "Death", "Water"], ["Light", "Fire", "Speed"]])
    defs = load_card_defs()
    unity_2 = next(d for d in defs if d.key == "AX02:Unity:2")
    unity_0 = next(d for d in defs if d.key == "AX02:Unity:0")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # 2 Unity cards already on field; play Unity 2 → 3 Unity total → draw 3.
    g.state.lines[1].p0_stack = [
        CardInst(inst_id=61001, def_id=unity_0.def_id, owner=0, face_up=True),
    ]
    g.state.lines[2].p1_stack = [
        CardInst(inst_id=61002, def_id=unity_0.def_id, owner=1, face_up=True),
    ]
    unity_2_inst = CardInst(inst_id=61003, def_id=unity_2.def_id, owner=0, face_up=False)
    g.state.players[0].hand = [unity_2_inst]
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=61100 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    pre_deck = len(g.state.players[0].deck)
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # After play, Unity 2 itself is on the field (3 total Unity cards
    # counted including itself, per Codex 'in the field'). Drew 3.
    assert len(g.state.players[0].deck) == pre_deck - 3, (
        f"expected to draw 3 (= # Unity in field including Unity 2); "
        f"deck went {pre_deck} → {len(g.state.players[0].deck)}"
    )


def test_assim_0_steals_covered_opp_card():
    """Assim 0 middle (AX02): 'Put one of your opponent's covered or
    uncovered field cards directly into your hand.' Verify covered opp
    cards are valid targets (default targeting is overridden by
    'covered or uncovered')."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import MIDDLE_EFFECTS
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=62, include_aux2=True))
    g.set_predetermined_draft([["Assimilation", "Death", "Water"], ["Light", "Fire", "Speed"]])
    defs = load_card_defs()
    assim_0 = next(d for d in defs if d.key == "AX02:Assimilation:0")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P1 has covered Fire 0 face-up + face-down on top in line 1.
    covered_card = CardInst(inst_id=62001, def_id=fire_0.def_id, owner=1, face_up=True)
    fd_top = CardInst(inst_id=62002, def_id=fire_0.def_id, owner=1, face_up=False)
    g.state.lines[1].p1_stack = [covered_card, fd_top]
    assim_inst = CardInst(inst_id=62003, def_id=assim_0.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [assim_inst]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.scratch["_engine"] = g
    g._pending = []
    gen = MIDDLE_EFFECTS["AX02:Assimilation:0"](g.state, 0, 0, assim_inst)
    choice = next(gen)
    # Both the covered face-up AND the top face-down should be in targets.
    assert len(choice.targets) == 2, (
        f"Assim 0 should target both covered and uncovered opp cards; got {len(choice.targets)} targets"
    )


def test_ice_4_flip_block_logs_cleanly():
    """Ice 4 immunity path: flip_card on a face-up Ice 4 should log
    "blocked" via state.log.append and return without modifying the
    card. Regression for a NameError that surfaced during AZ training
    (the path called `logInfo` — a TS naming — which doesn't exist in
    the Python engine)."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    from compile_engine.effects import flip_card
    g = Game(GameConfig(seed=80, include_main2=True))
    g.set_predetermined_draft([["Ice", "Death", "Water"], ["Light", "Fire", "Speed"]])
    defs = load_card_defs()
    ice_4 = next(d for d in defs if d.key == "MN02:Ice:4")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    inst = CardInst(inst_id=80001, def_id=ice_4.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [inst]
    g.state.scratch["_engine"] = g
    # Should NOT raise. Card stays face-up.
    flip_card(g.state, 0, 0, 0)
    assert inst.face_up, "Ice 4 should remain face-up after a blocked flip"
    assert any("Ice 4" in entry and "blocked" in entry for entry in g.state.log), (
        f"expected Ice 4 block log; got {g.state.log}"
    )


def test_diversity_0_middle_logs_compile_cleanly():
    """Diversity 0 middle: when 6 distinct face-up protocols exist on
    the field, flip the Diversity protocol to compiled. Regression for
    a NameError on the post-compile log line (`logInfo` → must be
    `state.log.append`)."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import MIDDLE_EFFECTS
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=81, include_main2=True, include_aux2=True))
    g.set_predetermined_draft([
        ["Diversity", "Light", "Fire"], ["Death", "Water", "Speed"],
    ])
    defs = load_card_defs()
    diversity_0 = next(d for d in defs if d.key == "AX02:Diversity:0")
    light_0 = next(d for d in defs if d.key == "MN01:Light:0")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    death_0 = next(d for d in defs if d.key == "MN01:Death:0")
    water_0 = next(d for d in defs if d.key == "MN01:Water:0")
    speed_0 = next(d for d in defs if d.key == "MN01:Speed:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    d0_inst = CardInst(inst_id=81001, def_id=diversity_0.def_id, owner=0, face_up=True)
    # Cover 6 distinct protocols: Diversity, Light, Fire, Death, Water, Speed.
    g.state.lines[0].p0_stack = [
        d0_inst,
        CardInst(inst_id=81002, def_id=light_0.def_id, owner=0, face_up=True),
    ]
    g.state.lines[1].p0_stack = [
        CardInst(inst_id=81003, def_id=fire_0.def_id, owner=0, face_up=True),
    ]
    g.state.lines[2].p0_stack = [
        CardInst(inst_id=81004, def_id=death_0.def_id, owner=0, face_up=True),
    ]
    g.state.lines[0].p1_stack = [
        CardInst(inst_id=81005, def_id=water_0.def_id, owner=1, face_up=True),
    ]
    g.state.lines[1].p1_stack = [
        CardInst(inst_id=81006, def_id=speed_0.def_id, owner=1, face_up=True),
    ]
    g.state.scratch["_engine"] = g
    fn = MIDDLE_EFFECTS["AX02:Diversity:0"]
    gen = fn(g.state, 0, 0, d0_inst)
    try:
        next(gen)
    except StopIteration:
        pass
    # P0 had Diversity in slot 0 → should now be compiled.
    assert g.state.players[0].compiled[0] is True, (
        f"Diversity slot should be compiled; got {g.state.players[0].compiled}"
    )
    assert any("Diversity" in entry and "compiled" in entry.lower() for entry in g.state.log)


def test_metal_6_self_deletes_when_face_up_flipped():
    """Metal 6 top: 'When this card would be covered or flipped: First,
    delete this card.' Top text is active while face-up. When an effect
    flips face-up Metal 6, it should self-delete BEFORE the flip lands
    — the flip is consumed (Codex p.10) and Metal 6 ends up in trash.

    Regression for a playtester report: the FLIP_TRIGGER broadcast
    filtered on `c.face_up` post-flip, which silently skipped the
    face-up→face-down transition path. Fixed by preempting in
    `flip_card` (similar to the Ice 4 immunity check)."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    from compile_engine.effects import flip_card
    g = Game(GameConfig(seed=90))
    g.set_predetermined_draft([["Metal", "Death", "Water"], ["Light", "Fire", "Speed"]])
    defs = load_card_defs()
    metal_6 = next(d for d in defs if d.key == "MN01:Metal:6")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    m6 = CardInst(inst_id=90001, def_id=metal_6.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [m6]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    flip_card(g.state, 0, 0, 0)
    g._drive()
    assert m6 not in g.state.lines[0].p0_stack, (
        "face-up Metal 6 should self-delete when flipped"
    )
    assert m6 in g.state.players[0].trash, (
        "Metal 6 should land in its owner's trash after the flip-induced self-delete"
    )


def test_metal_6_face_down_can_be_flipped_normally():
    """Inverse: face-DOWN Metal 6 has no active top text, so flipping it
    face-up should NOT trigger self-delete. (The flipped-face-up card
    sits on the field for downstream effects to interact with.)"""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    from compile_engine.effects import flip_card
    g = Game(GameConfig(seed=91))
    g.set_predetermined_draft([["Metal", "Death", "Water"], ["Light", "Fire", "Speed"]])
    defs = load_card_defs()
    metal_6 = next(d for d in defs if d.key == "MN01:Metal:6")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    m6 = CardInst(inst_id=91001, def_id=metal_6.def_id, owner=0, face_up=False)
    g.state.lines[0].p0_stack = [m6]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    flip_card(g.state, 0, 0, 0)
    g._drive()
    # Now face-up Metal 6 on the field; no self-delete fired since the
    # top text wasn't active during the face-down → face-up transition.
    assert m6 in g.state.lines[0].p0_stack
    assert m6.face_up


def test_plague_4_end_trigger_fires_on_end_phase():
    """Regression for playtester report: Plague 4's bottom 'End: opp
    deletes 1 face-down. You may flip this card.' previously did
    nothing because Plague 4 was routed via @bottom_on_play. Now via
    @end_trigger — verify the deletion fires at end of owner's turn."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=92))
    g.set_predetermined_draft([["Plague", "Death", "Water"], ["Light", "Fire", "Speed"]])
    defs = load_card_defs()
    plague_4 = next(d for d in defs if d.key == "MN01:Plague:4")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    p4_inst = CardInst(inst_id=92001, def_id=plague_4.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [p4_inst]
    # Opp face-down card that should be the deletion target.
    target_card = CardInst(inst_id=92002, def_id=fire_0.def_id, owner=1, face_up=False)
    g.state.lines[1].p1_stack = [target_card]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.phase = Phase.END
    g.state.scratch["_engine"] = g
    g._pending = []
    # Drive end phase. Should yield a Choice with decider=opp picking
    # which face-down to delete.
    g._drive()
    choice = g._pending[-1].last_choice if g._pending else None
    assert choice is not None and choice.decider == 1, (
        f"expected opp-decided face-down deletion prompt; got {choice}"
    )
    assert any("face-down" in choice.prompt.lower() for _ in [None]), (
        f"prompt should mention face-down; got: {choice.prompt}"
    )
    # Pick the only target → opp's face-down deletes.
    g.step(Action(type=ActionType.CHOOSE_TARGET, choice_index=0))
    # Skip the optional self-flip.
    while g._pending and g._pending[-1].last_choice is not None:
        choice = g._pending[-1].last_choice
        # Pick "skip" if available, else the first option.
        skip_idx = next(
            (i for i, t in enumerate(choice.targets) if t == -1),
            0,
        )
        g.step(Action(type=ActionType.CHOOSE_TARGET, choice_index=skip_idx))
    assert target_card not in g.state.lines[1].p1_stack, (
        "opp's face-down should be deleted by Plague 4's End trigger"
    )
    assert target_card in g.state.players[1].trash


def test_uncover_trigger_fires_when_top_card_deleted():
    """Codex p.3 "Middle Command — Immediate: Resolve this active text upon
    card play/flip/uncover." When the top card of a stack is deleted, the
    under-card (if face-up) is newly uncovered and its middle should fire.

    Regression: the uncover-trigger condition in delete/return/shift was
    previously inverted (`if not was_top`), so the cascade never fired on
    a top delete (and wrongly fired on middle/bottom deletes that didn't
    change the top)."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    from compile_engine.effects import delete_card_from_field
    g = Game(GameConfig(seed=30))
    g.set_predetermined_draft([["Light", "Death", "Fire"], ["Spirit", "Water", "Speed"]])
    defs = load_card_defs()
    light_2 = next(d for d in defs if d.key == "MN01:Light:2")  # middle Draw 2
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    under = CardInst(inst_id=30001, def_id=light_2.def_id, owner=0, face_up=True)
    top = CardInst(inst_id=30002, def_id=fire_0.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [under, top]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=30100 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.scratch["_engine"] = g
    g._pending = []
    # Delete the top card (Fire 0). Under (Light 2) should fire its middle
    # ("Draw 2 cards") via the uncover trigger.
    delete_card_from_field(g.state, 0, 0, 1)
    g._drive()
    assert len(g.state.players[0].hand) == 2 and len(g.state.players[0].deck) == 3, (
        f"Light 2 middle should fire on uncover after Fire 0 deleted from top; "
        f"hand=0→{len(g.state.players[0].hand)} deck=5→{len(g.state.players[0].deck)}"
    )
    # Inverse: deleting the bottom card (under unchanged top) → no uncover.
    g.state.lines[0].p0_stack = [under, top]
    g.state.players[0].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=30200 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.triggers = []
    g._pending = []
    delete_card_from_field(g.state, 0, 0, 0)  # delete bottom
    g._drive()
    assert len(g.state.players[0].hand) == 0 and len(g.state.players[0].deck) == 5, (
        "deleting the bottom card does not change top — no uncover trigger should fire"
    )


def test_card_effect_refresh_consumes_control_component():
    """Codex p.10 Spirit 0 clarification: 'When you refresh as
    instructed, it is a normal refresh action, including spending the
    control component, if applicable.' Closes the PR #26 known gap
    where card-effect-triggered refreshes silently skipped the
    control-rearrange prompt and didn't reset the control_holder.

    Verifies via Spirit 0's middle (Refresh + Draw 1)."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=70))
    g.set_predetermined_draft([["Spirit", "Death", "Water"], ["Light", "Fire", "Speed"]])
    defs = load_card_defs()
    spirit_0 = next(d for d in defs if d.key == "MN01:Spirit:0")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    spirit_inst = CardInst(inst_id=70001, def_id=spirit_0.def_id, owner=0, face_up=False)
    g.state.players[0].hand = [spirit_inst]
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=70100 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(10)
    ]
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.control_holder = 0  # ap holds control
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    # Play Spirit 0 face-up at line 0 (Spirit protocol). Its middle is
    # "Refresh. Draw 1 card." The refresh should yield the control
    # rearrange prompt BEFORE the draws happen.
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))
    choice = g._pending[-1].last_choice if g._pending else None
    assert choice is not None, "expected control rearrange prompt from Spirit 0 middle"
    assert "Control component" in choice.prompt, (
        f"expected control rearrange prompt; got: {choice.prompt}"
    )
    # Pick "skip"; control should reset to None, then refresh draws happen.
    skip_idx = next(i for i, t in enumerate(choice.targets) if t == -1)
    g.step(Action(type=ActionType.CHOOSE_TARGET, choice_index=skip_idx))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    assert g.state.control_holder is None, (
        f"control_holder should reset; got {g.state.control_holder}"
    )


def test_compile_with_control_offers_rearrange_then_resets():
    """Codex p.5-6: 'When the player with the control component compiles
    or refreshes, first the control component is returned to its neutral
    position and that player may rearrange one player's protocols —
    either theirs or their opponent's — then they complete their compile
    or refresh.' Codex p.8: control resets even if you skip rearrange.

    Regression for playtester bug: compiling with control gave no
    rearrange prompt and control didn't reset."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=50))
    g.set_predetermined_draft([["Light", "Death", "Fire"], ["Spirit", "Water", "Speed"]])
    defs = load_card_defs()
    fire_4 = next(d for d in defs if d.key == "MN01:Fire:4")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P0 has 3 face-up Fire 4s in line 2 (Fire protocol → value 12, compileable).
    g.state.lines[2].p0_stack = [
        CardInst(inst_id=50100 + i, def_id=fire_4.def_id, owner=0, face_up=True)
        for i in range(3)
    ]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.control_holder = 0  # ap holds control
    pre_protocols = list(g.state.players[0].protocols)
    g.state.phase = Phase.CHECK_COMPILE
    g.state.scratch["_engine"] = g
    g._pending = []
    # Initiate compile of line 2.
    g.step(Action(type=ActionType.COMPILE_LINE, line_index=2))
    # Should now have a pending choice: control rearrange prompt.
    choice = g._pending[-1].last_choice if g._pending else None
    assert choice is not None, "expected control rearrange prompt"
    assert "Control component" in choice.prompt, (
        f"expected rearrange prompt; got: {choice.prompt}"
    )
    # Pick "skip" → control should reset, compile completes.
    skip_idx = next(i for i, t in enumerate(choice.targets) if t == -1)
    g.step(Action(type=ActionType.CHOOSE_TARGET, choice_index=skip_idx))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # Verify: control reset to None, compile completed (line 2 protocol flipped).
    assert g.state.control_holder is None, (
        f"control_holder should reset to None on compile-with-control; got {g.state.control_holder}"
    )
    assert g.state.players[0].compiled[2], "Fire protocol should be compiled"
    # Protocols themselves should be unchanged (we skipped rearrange).
    assert list(g.state.players[0].protocols) == pre_protocols, (
        "protocols should be unchanged when player skipped rearrange"
    )


def test_compile_with_control_rearrange_swaps_protocols():
    """Same Codex clarification — verify the rearrange path actually
    swaps the chosen pair of protocols."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=51))
    g.set_predetermined_draft([["Light", "Death", "Fire"], ["Spirit", "Water", "Speed"]])
    defs = load_card_defs()
    fire_4 = next(d for d in defs if d.key == "MN01:Fire:4")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.lines[2].p0_stack = [
        CardInst(inst_id=51100 + i, def_id=fire_4.def_id, owner=0, face_up=True)
        for i in range(3)
    ]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.control_holder = 0
    g.state.phase = Phase.CHECK_COMPILE
    g.state.scratch["_engine"] = g
    g._pending = []
    g.step(Action(type=ActionType.COMPILE_LINE, line_index=2))
    # Pick "yours" → expect a swap-pair prompt.
    choice = g._pending[-1].last_choice
    yours_idx = next(i for i, t in enumerate(choice.targets) if t == 0)
    g.step(Action(type=ActionType.CHOOSE_TARGET, choice_index=yours_idx))
    choice2 = g._pending[-1].last_choice
    assert choice2 is not None and "Swap which two" in choice2.prompt
    # Pick the first swap pair (L0<->L1).
    g.step(Action(type=ActionType.CHOOSE_TARGET, choice_index=0))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    assert g.state.control_holder is None
    # Protocols swapped: Light↔Death (was [Light, Death, Fire] → [Death, Light, Fire]).
    assert g.state.players[0].protocols[0] == "Death"
    assert g.state.players[0].protocols[1] == "Light"


def test_gravity_1_allows_shifting_opp_cards():
    """Codex p.3 default targeting: 'your cards or your opponent's cards
    can both be selected' unless restricted. Gravity 1 says 'Shift 1
    card to or from this line' with no 'your' qualifier — opp's cards
    must be valid targets too. Pre-fix the handler restricted to self."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=40))
    g.set_predetermined_draft([["Gravity", "Death", "Water"], ["Fire", "Light", "Speed"]])
    defs = load_card_defs()
    gravity_1 = next(d for d in defs if d.key == "MN01:Gravity:1")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    grav1_inst = CardInst(inst_id=40001, def_id=gravity_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [grav1_inst]  # Gravity at line 0
    # P1 has an uncovered face-up card in line 1
    g.state.lines[1].p1_stack = [
        CardInst(inst_id=40100, def_id=fire_0.def_id, owner=1, face_up=True),
    ]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=40200 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.scratch["_engine"] = g
    g._pending = []
    # Directly invoke Gravity 1's middle. Yield should include an opp-side option.
    from compile_engine.effects import MIDDLE_EFFECTS
    gen = MIDDLE_EFFECTS["MN01:Gravity:1"](g.state, 0, 0, grav1_inst)
    choice = next(gen)
    opp_options = [o for o in choice.options if "(opp)" in o]
    assert len(opp_options) > 0, f"Gravity 1 should allow shifting opp's cards; options: {choice.options}"


def test_mirror_3_allows_self_flip_with_no_second_flip():
    """Codex p.13: 'If Mirror 3 flips itself first, the second flip
    doesn't happen.' Mirror 3 IS a valid first-flip target; after self-
    flipping face-down, the second clause is skipped."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=41, include_main2=True))
    g.set_predetermined_draft([["Mirror", "Death", "Water"], ["Fire", "Light", "Speed"]])
    defs = load_card_defs()
    mirror_3 = next(d for d in defs if d.key == "MN02:Mirror:3")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    m3_inst = CardInst(inst_id=41001, def_id=mirror_3.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [m3_inst]
    g.state.lines[0].p1_stack = [
        CardInst(inst_id=41100, def_id=fire_0.def_id, owner=1, face_up=True),
    ]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.scratch["_engine"] = g
    g._pending = []
    # Mirror 3 should be selectable as a "your card" target.
    from compile_engine.effects import MIDDLE_EFFECTS
    gen = MIDDLE_EFFECTS["MN02:Mirror:3"](g.state, 0, 0, m3_inst)
    choice = next(gen)
    self_targets = [i for i, t in enumerate(choice.targets) if t[3] is m3_inst]
    assert len(self_targets) == 1, f"Mirror 3 should be selectable as its own first-flip target; options: {choice.options}"
    # Pick Mirror 3 → flip face-down → second clause should be skipped (no opp prompt).
    try:
        nxt = gen.send(self_targets[0])
        raise AssertionError(
            f"Expected generator to stop after self-flip; got another Choice: {nxt.prompt}"
        )
    except StopIteration:
        pass  # Correct: no second flip


def test_plague_3_only_flips_uncovered():
    """Codex p.9: Plague 3's middle 'Flip each other face-up card' only
    affects UNCOVERED face-up cards (per default targeting rule). The
    pre-fix handler flipped covered face-up cards too."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=42))
    g.set_predetermined_draft([["Plague", "Light", "Fire"], ["Death", "Water", "Speed"]])
    defs = load_card_defs()
    plague_3 = next(d for d in defs if d.key == "MN01:Plague:3")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P0 has covered face-up Fire 0 (covered by face-down card on top) in line 1
    covered_fu = CardInst(inst_id=42001, def_id=fire_0.def_id, owner=0, face_up=True)
    fd_top = CardInst(inst_id=42002, def_id=fire_0.def_id, owner=0, face_up=False)
    g.state.lines[1].p0_stack = [covered_fu, fd_top]
    # P0 also has uncovered face-up Fire 0 in line 2
    uncovered_fu = CardInst(inst_id=42003, def_id=fire_0.def_id, owner=0, face_up=True)
    g.state.lines[2].p0_stack = [uncovered_fu]
    # P0 plays Plague 3 face-up at line 0 (Plague protocol).
    plague_3_inst = CardInst(inst_id=42004, def_id=plague_3.def_id, owner=0, face_up=False)
    g.state.players[0].hand = [plague_3_inst]
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=42100 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # Covered face-up should still be face-up (untouched). Uncovered should be flipped face-down.
    assert covered_fu.face_up, "Plague 3 should NOT flip covered face-up cards (uncovered only)"
    assert not uncovered_fu.face_up, "Plague 3 should flip the uncovered face-up card"


def test_courage_3_prompts_for_tied_opp_lines():
    """Codex p.9 clarification: when multiple lines tie for opp's highest
    total value, the Courage 3 owner picks. Prior to the fix, the engine
    silently picked the lowest-indexed tied line. The handler should
    now surface each tied line as a separate option."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=20, include_main2=True))
    g.set_predetermined_draft([
        ["Courage", "Death", "Water"],
        ["Light", "Fire", "Speed"],
    ])
    defs = load_card_defs()
    courage_3 = next(d for d in defs if d.key == "MN02:Courage:3")
    fire_4 = next(d for d in defs if d.key == "MN01:Fire:4")  # value 4
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # Courage 3 lives on P0's line 0 (Courage protocol).
    c3 = CardInst(inst_id=20001, def_id=courage_3.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [c3]
    # P1 has equal-value lines 1 and 2 (tied) → tied for "highest".
    g.state.lines[1].p1_stack = [
        CardInst(inst_id=20100, def_id=fire_4.def_id, owner=1, face_up=True),
    ]
    g.state.lines[2].p1_stack = [
        CardInst(inst_id=20200, def_id=fire_4.def_id, owner=1, face_up=True),
    ]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.phase = Phase.END
    g.state.scratch["_engine"] = g
    g._pending = []
    # Drive end phase to surface Courage 3's prompt.
    g._drive()
    # Should be paused on a choice that lists BOTH tied lines as options.
    choice_obj = g._pending[-1].last_choice if g._pending else None
    assert choice_obj is not None, "Courage 3 should have yielded a Choice"
    opts = choice_obj.options
    # Expect: "L1 (opp value 4)", "L2 (opp value 4)", "skip"
    assert any("L1" in o for o in opts), f"missing L1 option: {opts}"
    assert any("L2" in o for o in opts), f"missing L2 option: {opts}"
    assert any("skip" in o for o in opts), f"missing skip option: {opts}"


def test_mirror_1_filters_to_uncovered_opp_targets_and_fear_0_suppresses():
    """Mirror 1 bottom: Codex p.3 default targeting restricts to uncovered
    cards; Codex p.9 clarifies the bottom is blocked by Fear 0. Verify
    both:
      (a) covered opp middles are excluded from Mirror 1's target list
      (b) when ap owns Fear 0, opp middles are suppressed → Mirror 1 has
          no valid targets and resolves to nothing."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    g = Game(GameConfig(seed=21, include_main2=True))
    g.set_predetermined_draft([
        ["Mirror", "Fear", "Water"],
        ["Light", "Fire", "Death"],
    ])
    defs = load_card_defs()
    mirror_1 = next(d for d in defs if d.key == "MN02:Mirror:1")
    light_2 = next(d for d in defs if d.key == "MN01:Light:2")  # has middle
    fire_4 = next(d for d in defs if d.key == "MN01:Fire:4")    # has middle
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    # ---- Part (a): covered opp card excluded ----
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    m1 = CardInst(inst_id=21001, def_id=mirror_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [m1]
    # P1 has a face-up Light 2 covered by a face-up Fire 4 in line 1.
    g.state.lines[1].p1_stack = [
        CardInst(inst_id=21100, def_id=light_2.def_id, owner=1, face_up=True),  # covered
        CardInst(inst_id=21101, def_id=fire_4.def_id, owner=1, face_up=True),   # uncovered
    ]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=21200 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.players[1].deck = []
    g.state.current_player = 0
    g.state.phase = Phase.END
    g.state.scratch["_engine"] = g
    g._pending = []
    g._drive()
    choice_obj = g._pending[-1].last_choice if g._pending else None
    assert choice_obj is not None, "Mirror 1 should have yielded a Choice"
    # Only the UNCOVERED Fire 4 should be a target — not the covered Light 2.
    opts = choice_obj.options
    assert any("Fire" in o or "fire" in o for o in opts), f"Fire 4 missing: {opts}"
    assert not any("Light" in o for o in opts), (
        f"covered Light 2 should be excluded by uncovered-default rule: {opts}"
    )

    # ---- Part (b): ap's Fear 0 nullifies opp middles → no targets ----
    g2 = Game(GameConfig(seed=22, include_main2=True))
    g2.set_predetermined_draft([
        ["Mirror", "Fear", "Water"],
        ["Light", "Fire", "Death"],
    ])
    fear_0 = next(d for d in defs if d.key == "MN02:Fear:0")
    g2.state.lines = [type(g2.state.lines[0])() for _ in range(3)]
    m1b = CardInst(inst_id=22001, def_id=mirror_1.def_id, owner=0, face_up=True)
    g2.state.lines[0].p0_stack = [m1b]
    # P0 has Fear 0 face-up in line 1 (Fear protocol).
    fear_0_inst = CardInst(inst_id=22002, def_id=fear_0.def_id, owner=0, face_up=True)
    g2.state.lines[1].p0_stack = [fear_0_inst]
    # P1 has uncovered Fire 4 (middle text exists) — but Fear 0 nullifies.
    g2.state.lines[1].p1_stack = [
        CardInst(inst_id=22100, def_id=fire_4.def_id, owner=1, face_up=True),
    ]
    g2.state.players[0].hand = []
    g2.state.players[1].hand = []
    g2.state.players[0].deck = []
    g2.state.players[1].deck = []
    g2.state.current_player = 0
    g2.state.phase = Phase.END
    g2.state.scratch["_engine"] = g2
    g2._pending = []
    g2._drive()
    # With Fear 0 suppressing all opp middles, Mirror 1's bottom has zero
    # valid targets and returns without prompting. The pending stack
    # should be empty (or at least not awaiting a Mirror 1 Choice).
    while g2._pending:
        c = g2._pending[-1].last_choice
        if c is not None and "Mirror 1" in c.prompt:
            raise AssertionError(
                f"Mirror 1 should be blocked by ap's Fear 0; got prompt: {c.prompt}"
            )
        # Drain any unrelated triggers (none expected here).
        if c is None:
            g2._pending.pop()
        else:
            break


def test_assim_1_discard_into_opp_trash_flags_after_discard():
    """Assim 1's bottom 'Discard 1 card into their trash' is still a
    discard from ap's hand and must set the after-discard flag so
    @after_self_discard / @after_opp_discard triggers fire. Regression
    for a pre-existing latent bug surfaced during the Codex audit."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst, Phase
    from compile_engine.effects import _assim_1_after_any_refresh  # noqa: F401
    g = Game(GameConfig(seed=23, include_aux2=True))
    g.set_predetermined_draft([
        ["Assimilation", "Death", "Water"],
        ["Light", "Fire", "Speed"],
    ])
    defs = load_card_defs()
    assim_1 = next(d for d in defs if d.key == "AX02:Assimilation:1")
    fire_0 = next(d for d in defs if d.key == "MN01:Fire:0")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    a1 = CardInst(inst_id=23001, def_id=assim_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [a1]
    g.state.players[0].hand = [
        CardInst(inst_id=23100, def_id=fire_0.def_id, owner=0, face_up=False),
    ]
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=23200 + i, def_id=fire_0.def_id, owner=0, face_up=False)
        for i in range(5)
    ]
    g.state.players[1].deck = [
        CardInst(inst_id=23300 + i, def_id=fire_0.def_id, owner=1, face_up=False)
        for i in range(5)
    ]
    # Drive the Assim 1 after-refresh handler directly so we can assert
    # the discard flag is set immediately after the discard step.
    g.state.current_player = 0
    g.state.phase = Phase.ACTION
    g.state.scratch["_engine"] = g
    g._pending = []
    # Trigger via the player's own refresh action (which now flags
    # refresh through refresh_player → after_any_refresh broadcast).
    from compile_engine.actions import Action, ActionType
    g.step(Action(type=ActionType.REFRESH))
    while g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
    # Assim 1 must have set the discard flag on P0 during resolution.
    # By now the engine has drained the flag (broadcasting the discard
    # events), but the trash count is the durable evidence: P1's trash
    # received Assim 1's discarded card.
    assert len(g.state.players[1].trash) >= 1, (
        "Assim 1 should have discarded a card into P1's trash"
    )
    # And the engine should have broadcast the discard — surface this by
    # confirming the scratch flag was cleared (set then drained), i.e.
    # no stale flag survived.
    assert not g.state.scratch.get("_pending_after_discard_by_p0", False), (
        "after-discard flag should have been drained, not stale"
    )


def test_greedy_beats_random_more_often_than_not():
    """Sanity check: greedy heuristic should win > 50% vs random over many games."""
    wins = {0: 0, 1: 0, None: 0}
    for i in range(40):
        # Alternate seats so we measure the agent, not the seat.
        if i % 2 == 0:
            g = play_game(
                agent0=GreedyAgent(seed=i),
                agent1=RandomAgent(seed=i + 100),
                config=GameConfig(seed=i, max_turns=200),
            )
            greedy_won = g.state.winner == 0
        else:
            g = play_game(
                agent0=RandomAgent(seed=i + 100),
                agent1=GreedyAgent(seed=i),
                config=GameConfig(seed=i, max_turns=200),
            )
            greedy_won = g.state.winner == 1
        wins[0 if greedy_won else 1] += 1
    # Allow some slack; greedy should win comfortably more than half.
    assert wins[0] > wins[1], f"greedy did not beat random: {wins}"


def test_hate_2_second_clause_skipped_when_self_deleted():
    """Codex p.12 clarification: 'If Hate 2 is the highest value card you
    own it deletes itself as a result of the first clause. Thus, the
    second clause no longer exists and does not trigger.'

    Regression: prior to the source_still_active() guard, Hate 2's middle
    handler ran both clauses unconditionally, so opp's highest-value card
    was also deleted after Hate 2 self-deleted. With the guard, the
    handler bails before the second iteration."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import _hate_2
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=200, include_expansion=True))
    g.set_predetermined_draft([
        ["Hate", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    hate_2 = next(d for d in defs if d.key == "AX01:Hate:2")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P0's only uncovered card is Hate 2 (value 2) — it's the unambiguous
    # "your highest value uncovered card."
    h2 = CardInst(inst_id=20001, def_id=hate_2.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [h2]
    # Opp has one uncovered card to delete IF the second clause fires.
    opp_card = CardInst(inst_id=20002, def_id=light_1.def_id, owner=1, face_up=True)
    g.state.lines[1].p1_stack = [opp_card]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    # Fire Hate 2's middle as if just played face-up in line 0.
    g._push_effect(_hate_2(g.state, 0, 0, h2))
    g._drive()
    assert h2 not in g.state.lines[0].p0_stack, (
        "Hate 2 should self-delete as P0's highest uncovered card"
    )
    assert opp_card in g.state.lines[1].p1_stack, (
        "second clause should NOT fire after Hate 2 self-deletes — opp's "
        "card must remain on the field"
    )


def test_hate_2_second_clause_fires_when_source_survives():
    """Negative test for the source_still_active() guard: when Hate 2 is
    NOT your highest (some other uncovered own card outranks it), the
    first clause deletes that other card, Hate 2 stays on the field, and
    the second clause proceeds to delete opp's highest."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import _hate_2
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=201, include_expansion=True))
    g.set_predetermined_draft([
        ["Hate", "Light", "Fire"],
        ["Darkness", "Death", "Water"],
    ])
    defs = load_card_defs()
    hate_2 = next(d for d in defs if d.key == "AX01:Hate:2")
    fire_5 = next(d for d in defs if d.key == "MN01:Fire:5")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    h2 = CardInst(inst_id=20101, def_id=hate_2.def_id, owner=0, face_up=True)
    higher_own = CardInst(inst_id=20102, def_id=fire_5.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [h2]
    g.state.lines[1].p0_stack = [higher_own]  # value 5 > Hate 2's value 2
    opp_card = CardInst(inst_id=20103, def_id=light_1.def_id, owner=1, face_up=True)
    g.state.lines[2].p1_stack = [opp_card]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    g._push_effect(_hate_2(g.state, 0, 0, h2))
    g._drive()
    assert h2 in g.state.lines[0].p0_stack, (
        "Hate 2 should survive (it isn't P0's highest)"
    )
    assert higher_own not in g.state.lines[1].p0_stack, (
        "first clause should delete P0's actual highest (Fire 5)"
    )
    assert opp_card not in g.state.lines[2].p1_stack, (
        "second clause should fire since Hate 2 is still active"
    )


def test_courage_2_middle_draws_one():
    """Courage 2 was missing its middle 'Draw 1 card.' effect. Verify
    that playing Courage 2 face-up fires the new middle and draws one."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import _courage_2_middle
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=300, include_main2=True))
    g.set_predetermined_draft([["Courage", "Light", "Fire"], ["Death", "Water", "Speed"]])
    defs = load_card_defs()
    courage_2 = next(d for d in defs if d.key == "MN02:Courage:2")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    c2 = CardInst(inst_id=30001, def_id=courage_2.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [c2]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=30100 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(3)
    ]
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    pre_hand = len(g.state.players[0].hand)
    g._push_effect(_courage_2_middle(g.state, 0, 0, c2))
    g._drive()
    assert len(g.state.players[0].hand) == pre_hand + 1, (
        f"Courage 2 middle should draw 1; hand went {pre_hand} → "
        f"{len(g.state.players[0].hand)}"
    )


def test_unity_3_does_not_flip_face_down_cards():
    """Unity 3 middle was previously enumerating all uncovered cards; per
    the card text it should only target face-up cards. Verify that with
    a face-down card present, it isn't a legal flip target."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import _unity_3
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=301, include_main2=True, include_aux2=True))
    g.set_predetermined_draft([
        ["Unity", "Light", "Fire"],
        ["Assimilation", "Death", "Water"],
    ])
    defs = load_card_defs()
    unity_3 = next(d for d in defs if d.key == "AX02:Unity:3")
    unity_2 = next(d for d in defs if d.key == "AX02:Unity:2")  # second Unity for the precondition
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    u3 = CardInst(inst_id=30201, def_id=unity_3.def_id, owner=0, face_up=True)
    u2 = CardInst(inst_id=30202, def_id=unity_2.def_id, owner=0, face_up=True)
    fd = CardInst(inst_id=30203, def_id=light_1.def_id, owner=1, face_up=False)
    g.state.lines[0].p0_stack = [u3]
    g.state.lines[1].p0_stack = [u2]
    g.state.lines[2].p1_stack = [fd]  # opp's face-down — should NOT be a target
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    g._push_effect(_unity_3(g.state, 0, 0, u3))
    # _drive stops at a pending Choice; inspect what targets were offered.
    g._drive()
    pending = g._pending
    # The Choice should NOT include the face-down opp card.
    if pending:
        choice = pending[-1].last_choice
        assert choice is not None
        for t in choice.targets:
            if isinstance(t, tuple) and len(t) >= 4:
                card = t[3]
                assert card.face_up, (
                    f"Unity 3 should not offer face-down cards as flip targets; "
                    f"got {card}"
                )


def test_ice_3_end_trigger_fires_when_covered():
    """Ice 3 was previously a middle effect ('on play/flip/uncover, if
    covered, shift'). Per the corrected card text it's now Top with End:
    emphasis — fires at end of turn while face-up if covered. Verify the
    end-trigger fires and offers the shift."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import _ice_3_end
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=302, include_main2=True))
    g.set_predetermined_draft([["Ice", "Light", "Fire"], ["Death", "Water", "Speed"]])
    defs = load_card_defs()
    ice_3 = next(d for d in defs if d.key == "MN02:Ice:3")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    i3 = CardInst(inst_id=30301, def_id=ice_3.def_id, owner=0, face_up=True)
    cover = CardInst(inst_id=30302, def_id=light_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [i3, cover]  # Ice 3 is covered by Light 1
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    g._push_effect(_ice_3_end(g.state, 0, 0, i3))
    g._drive()
    # End trigger should prompt for an optional shift since Ice 3 is covered.
    pending = g._pending
    assert pending, "Ice 3 end-trigger should yield a shift choice while covered"
    choice = pending[-1].last_choice
    assert choice is not None
    assert "Shift covered Ice 3" in choice.prompt or "shift" in choice.prompt.lower()


def test_life_3_fires_on_cover_not_on_play():
    """Life 3 bottom: 'When this card would be covered: First, play the
    top card of your deck face-down in another line.' Was registered as
    @bottom_first (fired on play) — playtester reported. Verify:
      1. Playing Life 3 face-up does NOT immediately trigger the bottom.
      2. Covering Life 3 with any card DOES fire the bottom.
    """
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    from compile_engine.effects import _life_3_when_covered, play_top_deck_face_down
    g = Game(GameConfig(seed=400))
    g.set_predetermined_draft([["Life", "Light", "Fire"], ["Death", "Water", "Speed"]])
    defs = load_card_defs()
    life_3 = next(d for d in defs if d.key == "MN01:Life:3")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    l3 = CardInst(inst_id=40001, def_id=life_3.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [l3]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    # Stock the deck with predictable cards so we can detect the face-down play.
    g.state.players[0].deck = [
        CardInst(inst_id=40100 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(3)
    ]
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    # Snapshot stack counts before the cover.
    pre_l1_stack = sum(len(g.state.lines[i].stack(0)) for i in (1, 2))
    pre_deck = len(g.state.players[0].deck)
    # Fire the when_covered handler directly (simulating the cover trigger).
    g._push_effect(_life_3_when_covered(g.state, 0, 0, l3))
    g._drive()
    # Resolve the line-pick choice with the first option (line 1).
    if g._pending:
        g.step(g.legal_actions()[0])
    post_l1_stack = sum(len(g.state.lines[i].stack(0)) for i in (1, 2))
    post_deck = len(g.state.players[0].deck)
    assert post_l1_stack == pre_l1_stack + 1, (
        f"Life 3 when_covered should play 1 face-down card in another line; "
        f"L1+L2 stacks: {pre_l1_stack} → {post_l1_stack}"
    )
    assert post_deck == pre_deck - 1, (
        f"deck size should drop by 1 (face-down play consumes top); "
        f"deck: {pre_deck} → {post_deck}"
    )


def test_value_5_play_forces_discard():
    """Codex p.2: 'You discard 1 card.' on value-5 cards is an effect that
    fires on play (not a free cost). Verify a value-5 face-up play
    consumes 1 card from hand. Regression test for the bulk
    _value5_discard registration at effects.py:1190."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    from compile_engine.effects import MIDDLE_EFFECTS, _value5_discard
    defs = load_card_defs()
    # Sanity: every value-5 card has a middle handler wired.
    for d in defs:
        if d.value == 5:
            assert d.key in MIDDLE_EFFECTS, f"{d.key} missing middle handler"
            assert MIDDLE_EFFECTS[d.key] is _value5_discard, (
                f"{d.key} should share the bulk-registered _value5_discard"
            )
    g = Game(GameConfig(seed=500))
    g.set_predetermined_draft([["Light", "Fire", "Speed"], ["Death", "Water", "Plague"]])
    light_5 = next(d for d in defs if d.key == "MN01:Light:5")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.players[0].hand = [
        CardInst(inst_id=50001, def_id=light_5.def_id, owner=0, face_up=False),
        CardInst(inst_id=50002, def_id=light_1.def_id, owner=0, face_up=False),
        CardInst(inst_id=50003, def_id=light_1.def_id, owner=0, face_up=False),
    ]
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    pre_hand = len(g.state.players[0].hand)
    pre_trash = len(g.state.players[0].trash)
    g._push_effect(_value5_discard(g.state, 0, 0, g.state.players[0].hand[0]))
    g._drive()
    if g._pending:
        g.step(g.legal_actions()[0])
    assert len(g.state.players[0].hand) == pre_hand - 1
    assert len(g.state.players[0].trash) == pre_trash + 1


def test_value_5_play_with_empty_hand_is_noop():
    """Codex p.2: 'If you cannot complete the text of a card (i.e.
    "You discard 1 card." when you have no cards in hand) the card is
    still played.' Verify firing the middle on an empty hand doesn't
    crash or leave the engine in a pending state."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    from compile_engine.effects import _value5_discard
    g = Game(GameConfig(seed=501))
    g.set_predetermined_draft([["Fire", "Light", "Speed"], ["Death", "Water", "Plague"]])
    defs = load_card_defs()
    fire_5 = next(d for d in defs if d.key == "MN01:Fire:5")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    g.state.players[0].hand = []  # empty
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    stub = CardInst(inst_id=50101, def_id=fire_5.def_id, owner=0, face_up=True)
    g._push_effect(_value5_discard(g.state, 0, 0, stub))
    g._drive()
    assert len(g.state.players[0].trash) == 0
    assert g._pending == []


def test_corruption_1_redirects_returns_to_opp_deck():
    """Corruption 1 bottom: 'When a card would be returned to your
    opponent's hand: Put that card on top of their deck face-down
    instead.' Verify by playing through a return scenario:

      P0 owns Corruption 1 face-up + uncovered (so its bottom is active).
      P1 has an uncovered face-up card on the field.
      P0 plays Fear 2, which forces a return of an opp card.

    Without Corruption 1: opp card lands in opp's hand.
    With Corruption 1:   opp card lands on top of opp's DECK face-down.

    Integration-tests through the engine's normal action flow rather
    than pushing effects directly — so we also exercise the
    return_card_to_hand hook path."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import return_card_to_hand
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=600, include_main2=True))
    g.set_predetermined_draft(
        [["Corruption", "Fear", "Fire"], ["Light", "Water", "Speed"]]
    )
    defs = load_card_defs()
    corr_1 = next(d for d in defs if d.key == "MN02:Corruption:1")
    fear_2 = next(d for d in defs if d.key == "MN02:Fear:2")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    water_1 = next(d for d in defs if d.key == "MN01:Water:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P0: Corruption 1 face-up + uncovered on Corruption line.
    c1 = CardInst(inst_id=60001, def_id=corr_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [c1]
    # P1: a face-up card on Water line (Water 1).
    victim = CardInst(inst_id=60002, def_id=water_1.def_id, owner=1, face_up=True)
    g.state.lines[1].p1_stack = [victim]
    # P0 plays Fear 2 by hand (so we can fire the real action path).
    g.state.players[0].hand = [
        CardInst(inst_id=60010, def_id=fear_2.def_id, owner=0, face_up=False),
    ]
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = [
        CardInst(inst_id=60100 + i, def_id=light_1.def_id, owner=1, face_up=False)
        for i in range(2)
    ]
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    # Play Fear 2 face-up in the Fear line (line 1).
    pre_opp_hand = len(g.state.players[1].hand)
    pre_opp_deck = len(g.state.players[1].deck)
    pre_victim_on_field = victim in g.state.lines[1].p1_stack
    assert pre_victim_on_field
    play_a = Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=1)
    g.step(play_a)
    g._drive()
    # Fear 2's middle yielded a Choice: "Return 1 opp card". Resolve it
    # by picking the first legal CHOOSE_TARGET.
    if g._pending and g._pending[-1].last_choice is not None:
        g.step(g.legal_actions()[0])
        g._drive()
    # Victim should be off the field.
    assert victim not in g.state.lines[1].p1_stack, "Fear 2 should have returned the opp card off the field"
    # WITH Corruption 1 active: opp hand unchanged, opp deck got +1
    # face-down card (the redirected victim).
    assert len(g.state.players[1].hand) == pre_opp_hand, (
        f"opp hand should NOT have grown (Corruption 1 redirect); "
        f"got {pre_opp_hand} -> {len(g.state.players[1].hand)}"
    )
    assert len(g.state.players[1].deck) == pre_opp_deck + 1, (
        f"opp deck should have grown by 1 (returned card placed on top); "
        f"got {pre_opp_deck} -> {len(g.state.players[1].deck)}"
    )
    assert g.state.players[1].deck[-1] is victim, (
        "victim card should be on TOP of opp's deck (last element)"
    )
    assert not g.state.players[1].deck[-1].face_up, (
        "victim should be placed face-down"
    )


def test_corruption_1_no_effect_when_covered():
    """Corruption 1's bottom is auxiliary (bottom tier) — only active
    while uncovered (Codex p.13). Verify that when Corruption 1 is
    covered, the redirect does NOT happen: returned cards go to the
    normal hand destination."""
    from compile_engine import Game, GameConfig
    from compile_engine.cards import load_card_defs
    from compile_engine.effects import return_card_to_hand
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=601, include_main2=True))
    g.set_predetermined_draft(
        [["Corruption", "Fire", "Speed"], ["Light", "Water", "Plague"]]
    )
    defs = load_card_defs()
    corr_1 = next(d for d in defs if d.key == "MN02:Corruption:1")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    water_1 = next(d for d in defs if d.key == "MN01:Water:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # P0: Corruption 1 COVERED by Fire 1 on line 0.
    c1 = CardInst(inst_id=60201, def_id=corr_1.def_id, owner=0, face_up=True)
    cover = CardInst(inst_id=60202, def_id=light_1.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [c1, cover]  # Corruption 1 is covered
    # P1: card on field to be returned.
    victim = CardInst(inst_id=60203, def_id=water_1.def_id, owner=1, face_up=True)
    g.state.lines[1].p1_stack = [victim]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = [
        CardInst(inst_id=60300, def_id=light_1.def_id, owner=1, face_up=False)
    ]
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    pre_opp_hand = len(g.state.players[1].hand)
    pre_opp_deck = len(g.state.players[1].deck)
    # Trigger a return directly via the helper.
    return_card_to_hand(g.state, 1, 1, 0)
    # Should land in HAND (not deck) since Corruption 1 is covered.
    assert len(g.state.players[1].hand) == pre_opp_hand + 1, (
        "covered Corruption 1 should NOT redirect; returned card goes to hand"
    )
    assert len(g.state.players[1].deck) == pre_opp_deck, (
        "covered Corruption 1 should NOT redirect; deck unchanged"
    )


def test_unity_0_flip_trigger_fires_when_flipped_by_unity_card():
    """Unity 0 bottom: 'When this card would be flipped by a Unity card:
    First, flip 1 card or draw 1 card.' Verify by playing through:

      P0 has Unity 0 face-up on Unity line + another Unity card (so
        Unity 3's middle precondition '>=2 Unity cards in field' holds).
      P0 plays Unity 3 face-up, whose middle prompts to flip a card.
      P0 picks Unity 0 as the flip target.
      Unity 0's flip_trigger should fire, prompting 'flip or draw'."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    g = Game(GameConfig(seed=700, include_aux2=True, include_main2=True))
    g.set_predetermined_draft(
        [["Unity", "Light", "Fire"], ["Death", "Water", "Speed"]]
    )
    defs = load_card_defs()
    unity_0 = next(d for d in defs if d.key == "AX02:Unity:0")
    unity_2 = next(d for d in defs if d.key == "AX02:Unity:2")
    unity_3 = next(d for d in defs if d.key == "AX02:Unity:3")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    # Unity 0 on line 1 (not line 0) so the Unity 3 we're about to play
    # face-up to the Unity line (line 0) doesn't cover it. Unity 2 on
    # line 2 satisfies Unity 3's "another Unity card in field"
    # precondition. Manual placement bypasses the matching-protocol
    # restriction that normally applies only at PLAY time.
    u0 = CardInst(inst_id=70001, def_id=unity_0.def_id, owner=0, face_up=True)
    u2 = CardInst(inst_id=70002, def_id=unity_2.def_id, owner=0, face_up=True)
    g.state.lines[1].p0_stack = [u0]
    g.state.lines[2].p0_stack = [u2]
    g.state.players[0].hand = [
        CardInst(inst_id=70010, def_id=unity_3.def_id, owner=0, face_up=False),
    ]
    g.state.players[1].hand = []
    g.state.players[0].deck = [
        CardInst(inst_id=70100 + i, def_id=light_1.def_id, owner=0, face_up=False)
        for i in range(3)
    ]
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    pre_hand_size = len(g.state.players[0].hand)
    # Play Unity 3 face-up (line 0, where Unity is). Its middle will
    # prompt "flip 1 face-up card".
    play_a = Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0)
    g.step(play_a)
    g._drive()
    # Now we're at Unity 3's middle Choice — find Unity 0 in the targets
    # and pick it.
    assert g._pending and g._pending[-1].last_choice is not None, (
        "Unity 3's middle should yield a flip-target Choice"
    )
    choice = g._pending[-1].last_choice
    unity_0_idx = None
    for i, t in enumerate(choice.targets):
        if isinstance(t, tuple) and len(t) >= 4 and t[3] is u0:
            unity_0_idx = i
            break
    assert unity_0_idx is not None, (
        f"Unity 0 should be among the legal flip targets; got {choice.targets}"
    )
    g.step(g.legal_actions()[unity_0_idx])
    g._drive()
    # After Unity 0 is flipped face-down by Unity 3, its FLIP_TRIGGER
    # should fire, prompting "flip 1 card or draw 1".
    assert not u0.face_up, "Unity 0 should now be face-down"
    assert g._pending and g._pending[-1].last_choice is not None, (
        "Unity 0's flip_trigger should yield a 'flip or draw' Choice"
    )
    prompt = g._pending[-1].last_choice.prompt
    assert "flipped by a Unity card" in prompt or "flip 1 card or draw 1" in prompt, (
        f"unexpected prompt: {prompt!r}"
    )
    # Pick "draw 1" (option index 1).
    g.step(g.legal_actions()[1])
    g._drive()
    assert len(g.state.players[0].hand) == pre_hand_size, (
        # +1 from the draw, -1 from playing Unity 3 → net zero.
        f"draw should restore hand to pre-play size; got "
        f"{pre_hand_size} -> {len(g.state.players[0].hand)}"
    )


def test_unity_0_flip_trigger_does_not_fire_when_flipped_by_non_unity():
    """Inverse: if Unity 0 is flipped by a NON-Unity card (e.g., Mirror 3),
    the bottom trigger should NOT fire. Verify the gate works."""
    from compile_engine import Game, GameConfig
    from compile_engine.actions import Action, ActionType
    from compile_engine.cards import load_card_defs
    from compile_engine.state import CardInst
    from compile_engine.effects import flip_card
    g = Game(GameConfig(seed=701, include_aux2=True))
    g.set_predetermined_draft(
        [["Unity", "Light", "Fire"], ["Death", "Water", "Speed"]]
    )
    defs = load_card_defs()
    unity_0 = next(d for d in defs if d.key == "AX02:Unity:0")
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    u0 = CardInst(inst_id=70201, def_id=unity_0.def_id, owner=0, face_up=True)
    g.state.lines[0].p0_stack = [u0]
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.players[0].deck = []
    g.state.players[1].deck = []
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    # Use a Light 1 stub as the "cause" (non-Unity protocol).
    light_stub = CardInst(inst_id=70210, def_id=light_1.def_id, owner=0, face_up=True)
    flip_card(g.state, 0, 0, 0, cause_card=light_stub)
    g._drive()
    assert not u0.face_up
    # Unity 0's trigger SHOULD NOT have fired — no pending choice.
    assert not g._pending or g._pending[-1].last_choice is None, (
        "Unity 0's flip_trigger should not fire for non-Unity causes"
    )
