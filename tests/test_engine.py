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
    )
    defs = load_card_defs()
    passive_only_keys = {
        "AX01:Apathy:0", "AX01:Apathy:2", "AX01:Apathy:4", "AX01:Hate:3",
        "MN01:Darkness:2", "MN01:Death:1", "MN01:Life:0", "MN01:Metal:0",
        "MN01:Metal:2", "MN01:Metal:6", "MN01:Plague:0", "MN01:Plague:1",
        "MN01:Psychic:1", "MN01:Speed:1", "MN01:Speed:2", "MN01:Spirit:0",
        "MN01:Spirit:1", "MN01:Spirit:3",
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
    )
    defs = load_card_defs()
    passive_only_keys = {
        # MN01 + AX01 — captured by persistent rules in compute_line_value or
        # restriction queries.
        "AX01:Apathy:0", "AX01:Apathy:2", "AX01:Apathy:4", "AX01:Hate:3",
        "MN01:Darkness:2", "MN01:Death:1", "MN01:Life:0", "MN01:Metal:0",
        "MN01:Metal:2", "MN01:Metal:6", "MN01:Plague:0", "MN01:Plague:1",
        "MN01:Psychic:1", "MN01:Speed:1", "MN01:Speed:2", "MN01:Spirit:0",
        "MN01:Spirit:1", "MN01:Spirit:3",
        # MN02 — persistent value modifiers / global rules / play affordances
        "MN02:Clarity:0",     # +1 per card in hand (compute_line_value)
        "MN02:Mirror:0",      # +1 per opp card in line (compute_line_value)
        "MN02:Smoke:2",       # +1 per face-down in line (compute_line_value)
        "MN02:Chaos:3",       # may play without matching protocols (legal_actions)
        "MN02:Fear:0",        # T: suppress opp middles (middle_suppressed)
        "MN02:Ice:6",         # T: no draw with hand (draw_cards)
        "AX02:Diversity:6",   # T: continuous self-destruct (field mutation hook)
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
    resolve its own effects (e.g. drawing). We force a play of Light 1
    face-up (whose bottom 'Draw 1 card.' fires on play) and assert the
    drawn-card count increased."""
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
    light_1 = next(d for d in defs if d.key == "MN01:Light:1")

    # Clear and set up a controlled scenario.
    g.state.players[0].hand = []
    g.state.players[1].hand = []
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]

    # P0 holds Speed 0 + Light 1 in hand. P0's lines have Speed at idx 0, Light at idx 1.
    speed_0_inst = CardInst(inst_id=2001, def_id=speed_0.def_id, owner=0, face_up=False)
    light_1_inst = CardInst(inst_id=2002, def_id=light_1.def_id, owner=0, face_up=False)
    g.state.players[0].hand = [speed_0_inst, light_1_inst]
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

    pre_hand_len = len(g.state.players[0].hand)  # = 2
    pre_deck_len = len(g.state.players[0].deck)  # = 5

    # Play Speed 0 face-up into line 0 (Speed protocol).
    g.step(Action(type=ActionType.PLAY_FACE_UP, hand_index=0, line_index=0))

    # Speed 0's middle asks us to play one of the legal sub-plays. Pick the
    # one that plays Light 1 face-up into line 1 (Light protocol). After this,
    # Light 1's bottom 'Draw 1 card.' should also fire.
    while not g.is_over():
        legal = g.legal_actions()
        # Find a CHOOSE_TARGET action whose label corresponds to Light 1 FU L1.
        choice_obj = g._pending[-1].last_choice if g._pending else None
        if choice_obj is None:
            break
        picked = None
        for i, opt in enumerate(choice_obj.options):
            if "Light1" in opt and "L1" in opt and opt.startswith("FU"):
                picked = i
                break
        if picked is None:
            # If we lost the prompt, abort.
            break
        g.step(Action(type=ActionType.CHOOSE_TARGET, choice_index=picked))
        break

    # We should have: hand = 0 (played both), Light 1 in line 1 face-up,
    # and drew 1 from Light 1's bottom effect.
    line1_top = g.state.lines[1].p0_stack[-1] if g.state.lines[1].p0_stack else None
    assert line1_top is not None and line1_top is light_1_inst, "Light 1 not in line 1 face-up"
    assert line1_top.face_up
    # Hand size: started at 2, played 2 → 0 then +1 draw → 1
    assert len(g.state.players[0].hand) == 1, (
        f"expected 1 card after sub-play+draw, got {len(g.state.players[0].hand)}"
    )
    # Deck count dropped by 1 (the drawn card).
    assert len(g.state.players[0].deck) == pre_deck_len - 1


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


def test_life_0_fires_at_end_only():
    """Life 0 errata: 'End: If this card is covered, delete this card.' Cover
    a face-up Life 0 — it should NOT delete on the cover event itself, only
    at the End phase of its owner's next turn."""
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
    # Play fd_card face-down into line 0 (covers Life 0).
    g.step(Action(type=ActionType.PLAY_FACE_DOWN, hand_index=0, line_index=0))
    # After play resolves, drain to the next decision point.
    while g._pending and g._pending[-1].last_choice is not None:
        legal = g.legal_actions()
        g.step(legal[0])
    # Life 0 should STILL be in line 0 (not yet deleted — only fires at End).
    assert life_0_inst in g.state.lines[0].p0_stack, (
        "Life 0 should remain in field after being covered (errata: only at End)"
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
