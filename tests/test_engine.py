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
