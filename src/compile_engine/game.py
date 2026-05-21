"""Game orchestrator: drafting, turn flow, effect resolution, win check.

Lifecycle
---------
1. Game(config) -> creates state and starts in DRAFT phase.
2. while not game.is_over():
       agent = agents[game.decider()]
       legal = game.legal_actions()
       action = agent.choose(state_view, legal)
       game.step(action)
3. game.state.winner gives the result (0 or 1; None on draw / hit max_turns).

All randomness flows through `state.rng`, so deterministic seeding produces
reproducible games — required for RL training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generator

from .actions import Action, ActionType, Choice
from .cards import (
    BASE_PROTOCOLS,
    BASE_SET,
    CardDef,
    EXPANSION_PROTOCOLS,
    EXPANSION_SET,
    available_protocols,
    defs_for_protocol,
    load_card_defs,
)
from .effects import (
    EffectGen,
    compute_line_value,
    delete_card_from_field,
    discard_to_trash,
    draw_cards,
    flip_card,
    get_bottom_first_effect,
    get_bottom_on_play_effect,
    get_end_effect,
    get_middle_effect,
    get_start_effect,
    get_top_trigger,
    middle_suppressed,
    opp_must_play_facedown,
    opp_play_blocked_in_line,
    opp_play_facedown_blocked_in_line,
    player_can_compile,
    player_may_play_any_line_faceup,
    player_skips_check_cache,
    uncommit_sentinel,
)
from .state import (
    COMPILE_THRESHOLD,
    HAND_SIZE_LIMIT,
    NUM_LINES,
    NUM_PROTOCOLS_PER_PLAYER,
    STARTING_HAND,
    CardInst,
    GameConfig,
    GameState,
    Line,
    Phase,
    PlayerState,
)


class GameOver(RuntimeError):
    """Raised when an action is attempted on a finished game."""


@dataclass(slots=True)
class _PendingEffect:
    gen: EffectGen
    last_choice: Choice | None = None




class Game:
    """Compile game engine. Drives a single match via legal_actions / step."""

    def __init__(self, config: GameConfig | None = None, *, defs: list[CardDef] | None = None) -> None:
        self.config = config or GameConfig()
        self.defs: list[CardDef] = defs if defs is not None else load_card_defs()
        seed = self.config.seed
        rng = random.Random(seed)
        # Mix include_expansion into config from sample prob if needed.
        if self.config.expansion_sample_prob is not None:
            self.config.include_expansion = rng.random() < self.config.expansion_sample_prob

        self.state = GameState(
            config=self.config,
            defs=self.defs,
            players=(PlayerState(idx=0), PlayerState(idx=1)),
            lines=[Line() for _ in range(NUM_LINES)],
            rng=rng,
        )
        # Draft scaffolding — assemble enabled set list from per-set flags.
        from .cards import AUX2_SET, BASE_SET, EXPANSION_SET, MAIN2_SET
        enabled: list[str] = [BASE_SET]
        if self.config.include_expansion: enabled.append(EXPANSION_SET)
        if self.config.include_main2: enabled.append(MAIN2_SET)
        if self.config.include_aux2: enabled.append(AUX2_SET)
        self._draft_pool: list[str] = available_protocols(
            self.defs, enabled_sets=tuple(enabled),
        )
        rng.shuffle(self._draft_pool)
        # Snake-style draft per rules: P0 picks 1, P1 picks 2, P0 picks 2, P1 picks 1.
        self._draft_schedule: list[int] = [0, 1, 1, 0, 0, 1]
        self._draft_idx: int = 0
        self._first_player: int = 0  # P0 drafted first per rules

        # Effect resolution stack (top of list = currently resolving).
        self._pending: list[_PendingEffect] = []
        self._inst_counter: int = 0
        # Stash a back-reference on state.scratch so effects can call back
        # into the engine for recursive operations like Speed 0's "Play 1 card".
        self.state.scratch["_engine"] = self

        # If config marks deterministic draft, callers can pass picks ahead of time.
        self._predetermined_picks: list[list[str]] | None = None

    def start(self) -> None:
        """Idempotent kick-off; drive into the first decision point."""
        self._drive()

    # ------------------------------------------------------------------ utils

    def _new_inst(self, def_id: int, owner: int) -> CardInst:
        c = CardInst(inst_id=self._inst_counter, def_id=def_id, owner=owner)
        self._inst_counter += 1
        return c

    def set_predetermined_draft(self, picks: list[list[str]]) -> None:
        """picks[player] = list of 3 protocols. Skips the draft phase."""
        from .cards import PROTOCOLS_BY_SET
        assert len(picks) == 2 and all(len(p) == NUM_PROTOCOLS_PER_PLAYER for p in picks)
        all_known: set[str] = set()
        for ps in PROTOCOLS_BY_SET.values():
            all_known.update(ps)
        all_picks = set(picks[0]) | set(picks[1])
        for p in all_picks:
            if p not in self._draft_pool and p not in all_known:
                raise ValueError(f"Unknown protocol: {p}")
            if p not in self._draft_pool:
                raise ValueError(
                    f"Protocol {p} not in enabled draft pool (check include_* flags)"
                )
        self._predetermined_picks = picks
        self._drive()

    def is_over(self) -> bool:
        return self.state.phase is Phase.GAME_OVER

    def decider(self) -> int:
        """Which player is making the current decision."""
        if self._pending:
            top = self._pending[-1]
            if top.last_choice is not None:
                return top.last_choice.decider
        return self.state.current_player

    # ---------------------------------------------------------------- legality

    def legal_actions(self) -> list[Action]:
        st = self.state
        if st.phase is Phase.GAME_OVER:
            return []

        # If an effect is mid-resolution and prompted a choice, return choice actions.
        if self._pending and self._pending[-1].last_choice is not None:
            choice = self._pending[-1].last_choice
            acts = [
                Action(type=ActionType.CHOOSE_TARGET, choice_index=i)
                for i in range(len(choice.options))
            ]
            if choice.optional:
                acts.append(Action(type=ActionType.SKIP_OPTIONAL))
            return acts

        if st.phase is Phase.DRAFT:
            return [
                Action(type=ActionType.DRAFT_PROTOCOL, protocol=p)
                for p in self._draft_pool
            ]

        if st.phase is Phase.CHECK_CACHE:
            ps = st.players[st.current_player]
            return [
                Action(type=ActionType.DISCARD_CARD, hand_index=i)
                for i in range(len(ps.hand))
            ]

        if st.phase is Phase.CHECK_COMPILE:
            ap = st.current_player
            forced = self._compileable_lines(ap)
            return [Action(type=ActionType.COMPILE_LINE, line_index=ln) for ln in forced]

        if st.phase is Phase.ACTION:
            return self._action_phase_legal()

        # All other phases are auto-driven by step().
        return [Action(type=ActionType.NOOP)]

    def _action_phase_legal(self) -> list[Action]:
        st = self.state
        ap = st.current_player
        ps = st.players[ap]
        actions: list[Action] = []
        spirit_1_active = player_may_play_any_line_faceup(st, ap)
        psychic_1_forces_facedown = opp_must_play_facedown(st, ap)
        # Per-line opponent-imposed restrictions on our plays:
        line_blocked = [opp_play_blocked_in_line(st, ln, ap) for ln in range(NUM_LINES)]
        line_facedown_blocked = [
            opp_play_facedown_blocked_in_line(st, ln, ap) for ln in range(NUM_LINES)
        ]
        for hi, c in enumerate(ps.hand):
            d = st.defs[c.def_id]
            # Chaos 3 bottom: "This card may be played without matching protocols."
            # Corruption 0 bottom: "You may play this card in any line on either
            # player's side." — these are play-time affordances on the CARD ITSELF,
            # not persistent rules from cards already in play.
            chaos_3_self = d.protocol == "Chaos" and d.value == 3
            corruption_0_self = d.protocol == "Corruption" and d.value == 0
            unrestricted_faceup = spirit_1_active or chaos_3_self or corruption_0_self
            for ln in range(NUM_LINES):
                if line_blocked[ln]:
                    continue
                if psychic_1_forces_facedown:
                    continue
                if unrestricted_faceup or ps.protocols[ln] == d.protocol:
                    actions.append(Action(
                        type=ActionType.PLAY_FACE_UP, hand_index=hi, line_index=ln,
                    ))
            for ln in range(NUM_LINES):
                if line_blocked[ln] or line_facedown_blocked[ln]:
                    continue
                actions.append(Action(
                    type=ActionType.PLAY_FACE_DOWN, hand_index=hi, line_index=ln,
                ))
            # Corruption 0 may also be played onto the OPPONENT's side. Encode
            # that as PLAY_FACE_UP with line_index in {3,4,5} → opponent lines
            # 0..2. (Same hand_index points at the same hand card.)
            if corruption_0_self:
                for ln_opp in range(NUM_LINES):
                    actions.append(Action(
                        type=ActionType.PLAY_FACE_UP, hand_index=hi,
                        line_index=NUM_LINES + ln_opp,
                    ))
                    actions.append(Action(
                        type=ActionType.PLAY_FACE_DOWN, hand_index=hi,
                        line_index=NUM_LINES + ln_opp,
                    ))
        # Speed 2 / Spirit 3 affordance: while face-up on our side (even when
        # covered), we may optionally shift this card to another line as our
        # Spirit 3 ONLY — its top text is explicitly "You may shift this card,
        # even if this card is covered", which reads as a player-initiated
        # action. Speed 2's text (no "you may") is a target-eligibility
        # modifier handled by `_enumerate_shift_targets`, not a standalone
        # action.
        for ln_src in range(NUM_LINES):
            stack = st.lines[ln_src].stack(ap)
            for pos, c in enumerate(stack):
                if not c.face_up:
                    continue
                d = st.defs[c.def_id]
                if not (d.protocol == "Spirit" and d.value == 3):
                    continue
                for ln_dst in range(NUM_LINES):
                    if ln_dst == ln_src:
                        continue
                    actions.append(Action(
                        type=ActionType.SHIFT_OWN_CARD,
                        line_index=ln_src,
                        hand_index=pos,
                        choice_index=ln_dst,
                    ))

        # Refresh is always available; mandatory when no plays exist.
        if not actions:
            actions = [Action(type=ActionType.REFRESH)]
        else:
            actions.append(Action(type=ActionType.REFRESH))
        return actions

    # ------------------------------------------------------------------ step

    def step(self, action: Action) -> None:
        st = self.state
        if st.phase is Phase.GAME_OVER:
            raise GameOver("Game is already finished")

        # Choice resolution path
        if self._pending and self._pending[-1].last_choice is not None:
            self._resolve_choice(action)
            self._drive()
            return

        if st.phase is Phase.DRAFT:
            self._do_draft_pick(action)
            self._drive()
            return
        if st.phase is Phase.CHECK_CACHE:
            self._do_clear_cache(action)
            self._drive()
            return
        if st.phase is Phase.CHECK_COMPILE:
            self._do_compile(action)
            self._drive()
            return
        if st.phase is Phase.ACTION:
            self._do_action(action)
            self._drive()
            return

        # Otherwise, auto-driving phases consume a NOOP.
        self._drive()

    # ------------------------------------------------------------------ drive

    def _drain_pending(self) -> bool:
        """Resolve queued effects/triggers until either a decision is needed
        (returns True) or the queue empties (returns False). Honors the
        per-turn effect-push budget by dropping in-flight resolution if
        exceeded."""
        st = self.state
        while True:
            if self._budget_exhausted() and (self._pending or st.triggers):
                self._pending.clear()
                st.triggers.clear()
                return False
            while self._pending:
                top = self._pending[-1]
                if top.last_choice is not None:
                    return True
                if st.triggers:
                    self._fire_next_trigger()
                    continue
                try:
                    choice = next(top.gen)
                except StopIteration:
                    try:
                        self._pending.remove(top)
                    except ValueError:
                        pass
                    continue
                top.last_choice = choice
                return True
            if st.triggers:
                self._fire_next_trigger()
                continue
            return False

    def _drive(self) -> None:
        """Advance through auto-phases, trigger queue, and effect resumption
        until a real decision point is reached (or game ends). Phase steps
        that push effects (start/end triggers, after_clear_cache broadcasts)
        must be drained before advancing further, so each phase iteration
        re-runs the drain."""
        st = self.state

        # Drain effects, then any deferred "after X" broadcasts, then drain
        # again. Loop until everything stabilises (or a decision is needed).
        while True:
            if self._drain_pending():
                return
            if not self._drain_pending_after_events():
                break

        # Advance phases.
        while True:
            if st.phase is Phase.GAME_OVER:
                return
            if st.phase is Phase.DRAFT:
                if self._predetermined_picks is not None:
                    self._apply_predetermined_picks()
                # else: caller picks via DRAFT_PROTOCOL actions
                if st.phase is Phase.DRAFT:  # not yet finalized
                    return
                continue

            if st.phase is Phase.START:
                if self._do_start_phase():
                    return  # start-effect needs a decision
                continue
            if st.phase is Phase.CHECK_CONTROL:
                self._do_check_control()
                continue
            if st.phase is Phase.CHECK_COMPILE:
                forced = self._compileable_lines(st.current_player)
                if forced:
                    return  # need player to pick which (usually 1)
                st.phase = Phase.ACTION
                continue
            if st.phase is Phase.ACTION:
                return  # need a player action
            if st.phase is Phase.CHECK_CACHE:
                ps = st.players[st.current_player]
                # Spirit 0 bottom: skip check cache while uncovered. No
                # Clear Cache action occurred → no after_clear_cache.
                if player_skips_check_cache(st, st.current_player):
                    st.phase = Phase.END
                    continue
                if len(ps.hand) <= HAND_SIZE_LIMIT:
                    # Cache phase complete. If we discarded at least once
                    # this phase, fire `After you clear cache:` triggers
                    # (Codex p.6: "after" commands fire when the triggering
                    # effect and its consequences are fully resolved).
                    ap = st.current_player
                    flag_key = f"_pending_after_clear_cache_p{ap}"
                    if st.scratch.pop(flag_key, False):
                        self._broadcast_after_clear_cache(ap)
                        # Drain the broadcast effects before advancing to
                        # END. If a choice is needed, return — drain
                        # resumes on the next step.
                        if self._drain_pending():
                            return
                    st.phase = Phase.END
                    continue
                return  # need DISCARD_CARD
            if st.phase is Phase.END:
                if self._do_end_phase():
                    return  # end-effect needs a decision
                continue
            if st.phase is Phase.RESOLVING_EFFECT:
                # Should not happen here; we drained pending above.
                st.phase = Phase.ACTION
                continue

    # ------------------------------------------------------------------ draft

    def _do_draft_pick(self, action: Action) -> None:
        assert action.type is ActionType.DRAFT_PROTOCOL
        p = action.protocol
        if p not in self._draft_pool:
            raise ValueError(f"Protocol {p} not in draft pool")
        picker = self._draft_schedule[self._draft_idx]
        ps = self.state.players[picker]
        ps.protocols.append(p)
        ps.compiled.append(False)
        self._draft_pool.remove(p)
        self._draft_idx += 1
        if self._draft_idx >= len(self._draft_schedule):
            self._finalize_draft()

    def _apply_predetermined_picks(self) -> None:
        picks = self._predetermined_picks
        assert picks is not None
        for pl in (0, 1):
            for p in picks[pl]:
                self.state.players[pl].protocols.append(p)
                self.state.players[pl].compiled.append(False)
        self._finalize_draft()

    def _finalize_draft(self) -> None:
        st = self.state
        # Build each player's 18-card deck from their 3 drafted protocols.
        for pl in (0, 1):
            ps = st.players[pl]
            deck: list[CardInst] = []
            for proto in ps.protocols:
                for d in defs_for_protocol(self.defs, proto):
                    deck.append(self._new_inst(d.def_id, owner=pl))
            st.rng.shuffle(deck)
            ps.deck = deck
            # Starting hand
            draw_cards(st, pl, STARTING_HAND)
        st.phase = Phase.START
        st.current_player = self._first_player
        st.turn = 1
        st.compiled_this_turn = False

    # ------------------------------------------------------------------ phases

    def _do_start_phase(self) -> bool:
        """Fire Start: triggers on the active player's face-up cards. Top-text
        Start triggers fire on any face-up card; bottom-text Start triggers
        fire only on uncovered face-up cards — the handler decides if it
        applies (we surface every face-up card here)."""
        st = self.state
        ap = st.current_player
        any_pushed = False
        for ln in range(NUM_LINES):
            stack = list(st.lines[ln].stack(ap))
            for c in stack:
                if not c.face_up:
                    continue
                d = st.defs[c.def_id]
                fn = get_start_effect(d)
                if fn is not None:
                    self._push_effect(fn(st, ap, ln, c))
                    any_pushed = True
        st.phase = Phase.CHECK_CONTROL
        return any_pushed

    def _do_check_control(self) -> None:
        st = self.state
        ap = st.current_player
        won_lines = 0
        for ln in range(NUM_LINES):
            ours = compute_line_value(st, ln, ap)
            theirs = compute_line_value(st, ln, 1 - ap)
            if ours > theirs:
                won_lines += 1
        if won_lines >= 2:
            st.control_holder = ap
        st.phase = Phase.CHECK_COMPILE

    def _compileable_lines(self, player: int) -> list[int]:
        if self.state.compiled_this_turn:
            return []
        if not player_can_compile(self.state, player):
            return []
        st = self.state
        out: list[int] = []
        opp = 1 - player
        for ln in range(NUM_LINES):
            ours = compute_line_value(st, ln, player)
            theirs = compute_line_value(st, ln, opp)
            if ours >= COMPILE_THRESHOLD and ours > theirs:
                out.append(ln)
        return out

    def _do_compile(self, action: Action) -> None:
        assert action.type is ActionType.COMPILE_LINE
        st = self.state
        ln = action.line_index
        # Codex p.11: when compiling, "all cards in the line are deleted at
        # the same time". Cards with a `when_deleted_by_compile` interrupt
        # (Speed 2) get to fire before the bulk delete and can shift
        # themselves out, escaping the compile. We collect interrupts now,
        # push them as effects, and queue a finalizer that runs the bulk
        # delete + protocol flip once interrupts resolve.
        from .effects import get_when_deleted_by_compile_effect
        interrupts: list[tuple[int, CardInst, object]] = []
        for pl in (0, 1):
            for c in st.lines[ln].stack(pl):
                if not c.face_up:
                    continue
                d = st.defs[c.def_id]
                fn = get_when_deleted_by_compile_effect(d)
                if fn is not None:
                    interrupts.append((pl, c, fn))

        if not interrupts:
            self._compile_finalize(action)
            return

        ap = st.current_player
        # Push finalizer first (drains last). Then interrupts in reverse so
        # the first listed drains first under LIFO.
        self._push_effect(self._compile_finalizer_gen(action))
        for pl, c, fn in reversed(interrupts):
            self._push_effect(fn(st, ap, ln, c))

    def _compile_finalizer_gen(self, action: Action):
        """Generator wrapper that runs the bulk-delete + protocol flip
        after `when_deleted_by_compile` interrupts (Speed 2 etc.) have
        drained. Sits on `_pending` like any other effect."""
        self._compile_finalize(action)
        if False:
            yield  # marks this as a generator

    def _compile_finalize(self, action: Action) -> None:
        st = self.state
        ap = st.current_player
        ln = action.line_index
        opp = 1 - ap
        # Delete all remaining cards on both sides in this line. Cards that
        # shifted out via a when_deleted_by_compile interrupt (Speed 2) are
        # no longer in the line and survive.
        for c in st.lines[ln].p0_stack:
            c.face_up = True
            st.players[c.owner].trash.append(c)
        for c in st.lines[ln].p1_stack:
            c.face_up = True
            st.players[c.owner].trash.append(c)
        st.lines[ln].p0_stack = []
        st.lines[ln].p1_stack = []
        # Diversity 6 continuous check — compile bulk-trashes both sides so the
        # protocol count can drop.
        from .effects import _check_diversity_6_self_destruct
        _check_diversity_6_self_destruct(st)
        # Compile / Recompile
        if st.players[ap].compiled[ln]:
            # Recompile: instead of flipping, draw top of opponent's deck.
            if st.players[opp].deck:
                stolen = st.players[opp].deck.pop()
                # Ownership changes (per rules): "If a card ever changes
                # ownership, it retains its new ownership until end of game".
                stolen.owner = ap
                stolen.face_up = False
                st.players[ap].hand.append(stolen)
        else:
            st.players[ap].compiled[ln] = True
        st.compiled_this_turn = True
        # Win check
        if st.players[ap].all_compiled():
            st.winner = ap
            st.phase = Phase.GAME_OVER
            return
        # Per rules, compile is the only action this turn; jump to End.
        st.phase = Phase.END

    def _do_action(self, action: Action) -> None:
        st = self.state
        ap = st.current_player
        if action.type is ActionType.REFRESH:
            self._do_refresh(ap)
            st.phase = Phase.CHECK_CACHE
            return
        if action.type is ActionType.PLAY_FACE_UP:
            self._play_card(ap, action.hand_index, action.line_index, face_up=True)
            st.phase = Phase.CHECK_CACHE
            return
        if action.type is ActionType.PLAY_FACE_DOWN:
            self._play_card(ap, action.hand_index, action.line_index, face_up=False)
            st.phase = Phase.CHECK_CACHE
            return
        if action.type is ActionType.SHIFT_OWN_CARD:
            # Speed 2 / Spirit 3 affordance — shift this card even if covered.
            from .effects import shift_card
            src_line = action.line_index
            src_pos = action.hand_index
            dst_line = action.choice_index
            stack = st.lines[src_line].stack(ap)
            # Validate the position points to a face-up Speed 2 / Spirit 3.
            if not (0 <= src_pos < len(stack)):
                raise ValueError("SHIFT_OWN_CARD: src_pos out of range")
            c = stack[src_pos]
            d = st.defs[c.def_id]
            valid = c.face_up and d.protocol == "Spirit" and d.value == 3
            if not valid:
                raise ValueError("SHIFT_OWN_CARD: target is not Spirit 3 face-up")
            shift_card(st, src_line, ap, src_pos, dst_line)
            st.phase = Phase.CHECK_CACHE
            return
        raise ValueError(f"Illegal action in ACTION phase: {action}")

    def play_card_for_effect(self, player: int, hand_index: int, line_index: int, face_up: bool) -> None:
        """Public-ish entry point for card effects (e.g. Speed 0 middle) to
        recursively play a card on `player`'s behalf. The played card's
        enter-play triggers fire normally."""
        self._play_card(player, hand_index, line_index, face_up=face_up)

    def _do_refresh(self, player: int) -> None:
        ps = self.state.players[player]
        need = STARTING_HAND - len(ps.hand)
        if need > 0:
            draw_cards(self.state, player, need)

    def _play_card(self, player: int, hand_index: int, line_index: int, face_up: bool) -> None:
        """Per Compile Codex (16 Dec 2024): the played card is committed
        from placement until placement settles. Orchestration is pushed onto
        the _pending stack in reverse chronological order so LIFO drain
        fires when_covered first, then uncommits, then enter-play effects."""
        st = self.state
        ps = st.players[player]
        if not (0 <= hand_index < len(ps.hand)):
            raise ValueError(f"hand_index {hand_index} out of range")
        c = ps.hand.pop(hand_index)
        d = st.defs[c.def_id]
        # Corruption 0 lets the player play this card on either side. Encoded
        # as line_index in [NUM_LINES, 2*NUM_LINES) → opponent's lines 0..2.
        target_side = player
        actual_line = line_index
        cross_side = (d.protocol == "Corruption" and d.value == 0) and (
            NUM_LINES <= line_index < 2 * NUM_LINES
        )
        if cross_side:
            target_side = 1 - player
            actual_line = line_index - NUM_LINES
        if face_up:
            chaos_3_self = d.protocol == "Chaos" and d.value == 3
            corruption_0_self = d.protocol == "Corruption" and d.value == 0
            target_protos = st.players[target_side].protocols
            allowed = (
                target_protos[actual_line] == d.protocol
                or player_may_play_any_line_faceup(st, player)
                or chaos_3_self
                or corruption_0_self
            )
            if not allowed:
                raise ValueError(
                    f"Face-up play of {d.protocol} must be in matching line"
                )
            c.face_up = True
        else:
            c.face_up = False
        # Ownership transfers if Corruption 0 placed on opponent side.
        if cross_side:
            c.owner = target_side
        target_stack = st.lines[actual_line].stack(target_side)
        soon_covered = target_stack[-1] if target_stack else None
        c.is_committed = True
        target_stack.append(c)
        # Rebind for downstream effect pushes — they need the actual line/side.
        line_index = actual_line
        # Push order (deepest first → drains last):
        #   1. enter-play effects (only if face-up): middle, bottom_first,
        #      bottom_on_play, top_trigger — pushed in this order so LIFO
        #      drains them top → bottom_first → bottom_on_play → middle.
        #   2. uncommit sentinel — clears is_committed before enter-play.
        #   3. when_covered effect (only if applicable) — drains first.
        if face_up:
            # Codex "Middle Command - Immediate: Resolve this active text
            # upon card play/flip/uncover." → middle fires on play. Push
            # middle first so the LIFO drain runs top → bottom_first →
            # bottom_on_play → middle (top resolves first).
            mid_fn = None if middle_suppressed(st, line_index, c) else get_middle_effect(d)
            if mid_fn is not None and d.middle_text:
                self._push_effect(mid_fn(st, player, line_index, c))
            bf_fn = get_bottom_first_effect(d)
            if bf_fn is not None:
                self._push_effect(bf_fn(st, player, line_index, c))
            bp_fn = get_bottom_on_play_effect(d)
            if bp_fn is not None:
                self._push_effect(bp_fn(st, player, line_index, c))
            top_fn = get_top_trigger(d)
            if top_fn is not None:
                self._push_effect(top_fn(st, player, line_index, c))
        # Uncommit sentinel sits above all enter-play effects.
        self._push_effect(uncommit_sentinel(c))
        # When-covered: the card under us just transitioned uncovered →
        # covered. Fire its `@when_covered` hook (Fire 0 errata bottom and
        # similar). Middle is NOT fired here — middle fires on
        # play/flip/uncover per the Codex, not on cover.
        if soon_covered is not None and soon_covered.face_up:
            from .effects import get_when_covered_effect
            d_under = st.defs[soon_covered.def_id]
            wc_fn = get_when_covered_effect(d_under)
            if wc_fn is not None:
                self._push_effect(
                    wc_fn(st, soon_covered.owner, line_index, soon_covered)
                )
        # Diversity 6 continuous check — placing a card may not change the
        # protocol count (it can only stay the same or grow), but Diversity 6
        # itself can enter a sub-3-protocol field and must self-delete.
        from .effects import _check_diversity_6_self_destruct
        _check_diversity_6_self_destruct(st)

    def _push_effect(self, gen) -> None:
        """Push an effect generator if under per-turn depth and total caps.
        Rational play never hits these; random rollouts will."""
        st = self.state
        if (
            len(self._pending) >= self.config.max_effect_stack_depth
            or st.effect_pushes_this_turn >= self.config.max_effect_pushes_per_turn
        ):
            return
        st.effect_pushes_this_turn += 1
        self._pending.append(_PendingEffect(gen=gen, last_choice=None))

    def _budget_exhausted(self) -> bool:
        return (
            self.state.effect_pushes_this_turn
            >= self.config.max_effect_pushes_per_turn
        )

    def _enqueue_face_up_triggers(self, card: CardInst, ap: int, line_idx: int) -> None:
        """Push (in resolution order: top -> bottom-first -> middle) the
        effects that fire when `card` enters face-up at (line, ap). Pushing
        in reverse-resolution order yields LIFO drain that resolves top
        first.

        Rules (Codex 22SEP2025 + rules.txt "Card Anatomy"):
          - Top is PERSISTENT: passive text active while card is face-up.
            Unconditional tops fire as a one-shot here. Tops with emphasis
            ('Start:', 'End:', 'After you clear cache:', 'Flip:', 'When this
            card would be ...:', etc.) must register on the matching
            event-specific decorator, NOT @top_trigger.
          - Middle is IMMEDIATE: resolves on play/flip/uncover.
          - Bottom is AUXILIARY: triggered effects, viable while uncovered.
        """
        st = self.state
        d = st.defs[card.def_id]
        # Resolution order desired: top_trigger -> bottom_first -> middle.
        # Push in reverse so the stack drains in the desired order.
        mid_fn = None if middle_suppressed(st, line_idx, card) else get_middle_effect(d)
        if mid_fn is not None and d.middle_text:
            self._push_effect(mid_fn(st, ap, line_idx, card))
        # Bottom: "First, ..." resolves before middle. Also handle bottom-on-play
        # for cards whose bottom text is not a "First" trigger but still fires
        # when the card enters face-up (e.g. Light 1 bottom: "Draw 1 card.").
        bf_fn = get_bottom_first_effect(d)
        if bf_fn is not None:
            self._push_effect(bf_fn(st, ap, line_idx, card))
        bp_fn = get_bottom_on_play_effect(d)
        if bp_fn is not None:
            self._push_effect(bp_fn(st, ap, line_idx, card))
        # Top one-shot trigger resolves first.
        top_fn = get_top_trigger(d)
        if top_fn is not None:
            self._push_effect(top_fn(st, ap, line_idx, card))

    def _enqueue_enter_play_triggers_skip_middle(self, card: CardInst, ap: int, line_idx: int) -> None:
        """Variant of `_enqueue_face_up_triggers` used by Luck 1, which says
        'flip that card, ignoring its middle command' — so top and bottom
        triggers still fire, but middle does not."""
        st = self.state
        d = st.defs[card.def_id]
        bf_fn = get_bottom_first_effect(d)
        if bf_fn is not None:
            self._push_effect(bf_fn(st, ap, line_idx, card))
        bp_fn = get_bottom_on_play_effect(d)
        if bp_fn is not None:
            self._push_effect(bp_fn(st, ap, line_idx, card))
        top_fn = get_top_trigger(d)
        if top_fn is not None:
            self._push_effect(top_fn(st, ap, line_idx, card))

    def _fire_next_trigger(self) -> None:
        """Drain a queued trigger from state.triggers.

        Trigger kinds:
          - ("uncommit", card): clear is_committed on `card`. No effect push.
          - ("when_covered", line, owner, card): fire the now-covered card's
            registered when_covered effect (e.g. Fire 0 errata bottom).
            Middle is NOT fired here — middle fires on play/flip/uncover,
            not on cover.
          - ("face_up", line, owner, card): fire full enter-play stack
            (top + bottom + middle).
          - ("uncover", line, owner, card): fire middle (the Codex "Middle
            Command - Immediate: Resolve this active text upon card
            play/flip/uncover").
        """
        from .effects import get_when_covered_effect
        st = self.state
        t = st.triggers.pop()  # LIFO
        kind = t[0]
        if kind == "uncommit":
            card = t[1]
            card.is_committed = False
            return
        _, ln, pl, card = t
        if kind == "when_covered":
            # Validate the card is still face-up and still under cover. (Cover
            # check: not the top of its stack.)
            stack = st.lines[ln].stack(pl)
            if card not in stack or not card.face_up:
                return
            if stack.index(card) == len(stack) - 1:
                # Not covered anymore (cover removed before trigger fired).
                return
            d = st.defs[card.def_id]
            fn = get_when_covered_effect(d)
            if fn is None:
                return
            self._push_effect(fn(st, pl, ln, card))
            return
        # face_up / uncover require the card still in field and face-up.
        stack = st.lines[ln].stack(pl)
        if card not in stack or not card.face_up:
            return
        if kind == "face_up":
            self._enqueue_face_up_triggers(card, pl, ln)
            return
        # "uncover": fire middle only.
        d = st.defs[card.def_id]
        if middle_suppressed(st, ln, card):
            return
        fn = get_middle_effect(d)
        if fn is None or not d.middle_text:
            return
        self._push_effect(fn(st, pl, ln, card))

    def _do_clear_cache(self, action: Action) -> None:
        assert action.type is ActionType.DISCARD_CARD
        ap = self.state.current_player
        discard_to_trash(self.state, ap, action.hand_index)
        # Mark that *this* CHECK_CACHE phase actually performed a discard
        # (i.e., a Clear Cache action per Codex p.5: "discard action is
        # called Clear Cache"). The transition out of CHECK_CACHE will
        # broadcast `after_clear_cache` iff this flag is set.
        self.state.scratch[f"_pending_after_clear_cache_p{ap}"] = True
        # Loop back via _drive to recheck size.

    def _broadcast_after_clear_cache(self, ap: int) -> None:
        """Push `@after_clear_cache` effects for every face-up card on
        `ap`'s side of the field. Matches the side-of-field convention
        used by Start/End effects. Multiple cards triggering at once:
        Codex p.4 says the active player chooses the order; we push in
        line-ascending order, which the LIFO drain reverses — line 2's
        effect resolves first, line 0 last."""
        st = self.state
        from .effects import get_after_clear_cache_effect
        for ln in range(NUM_LINES):
            for c in list(st.lines[ln].stack(ap)):
                if not c.face_up:
                    continue
                d = st.defs[c.def_id]
                fn = get_after_clear_cache_effect(d)
                if fn is not None:
                    self._push_effect(fn(st, ap, ln, c))

    def _broadcast_for_side(self, owner: int, getter) -> bool:
        """Iterate face-up cards on `owner`'s side; for each card whose
        def_id resolves via `getter` (one of the get_after_*_effect
        wrappers), push the effect. Returns True iff anything pushed."""
        st = self.state
        pushed = False
        for ln in range(NUM_LINES):
            for c in list(st.lines[ln].stack(owner)):
                if not c.face_up:
                    continue
                d = st.defs[c.def_id]
                fn = getter(d)
                if fn is not None:
                    self._push_effect(fn(st, owner, ln, c))
                    pushed = True
        return pushed

    def _drain_pending_after_events(self) -> bool:
        """Check for deferred "after X" event flags set by atomic helpers
        in effects.py. For each flag, broadcast the corresponding effects
        to both players' fields (the "self" vs "opp" naming is relative
        to the actor — e.g. `after_self_discard` fires on the discarder's
        field; `after_opp_discard` fires on the other side). Returns True
        if any effect was pushed (caller should drain again)."""
        from .effects import (
            get_after_self_discard_effect, get_after_opp_discard_effect,
            get_after_self_delete_effect, get_after_self_draw_effect,
            get_after_self_shuffle_effect, get_after_self_refresh_effect,
            get_flip_trigger_effect,
        )
        st = self.state
        pushed_any = False

        # Discards: one flag per discarder. Fire that player's own
        # `after_self_discard` plus the opponent's `after_opp_discard`.
        for p in (0, 1):
            key = f"_pending_after_discard_by_p{p}"
            if st.scratch.pop(key, False):
                pushed_any |= self._broadcast_for_side(p, get_after_self_discard_effect)
                pushed_any |= self._broadcast_for_side(1 - p, get_after_opp_discard_effect)

        # Draws — only the drawer's side fires `after_self_draw`.
        for p in (0, 1):
            key = f"_pending_after_draw_by_p{p}"
            if st.scratch.pop(key, False):
                pushed_any |= self._broadcast_for_side(p, get_after_self_draw_effect)

        # Deletes — attributed to whoever was current_player at the time.
        for p in (0, 1):
            key = f"_pending_after_delete_by_p{p}"
            if st.scratch.pop(key, False):
                pushed_any |= self._broadcast_for_side(p, get_after_self_delete_effect)

        # Shuffles — fires on shuffler's side.
        for p in (0, 1):
            key = f"_pending_after_shuffle_by_p{p}"
            if st.scratch.pop(key, False):
                pushed_any |= self._broadcast_for_side(p, get_after_self_shuffle_effect)

        # Refresh — fires on refresher's side.
        for p in (0, 1):
            key = f"_pending_after_refresh_by_p{p}"
            if st.scratch.pop(key, False):
                pushed_any |= self._broadcast_for_side(p, get_after_self_refresh_effect)

        # Flip triggers — list of card refs (the cards that flipped).
        flip_list = st.scratch.pop("_pending_flip_cards", None)
        if flip_list:
            for c in flip_list:
                # Locate the card in the field (it may have moved or left).
                for ln in range(NUM_LINES):
                    for pl in (0, 1):
                        s = st.lines[ln].stack(pl)
                        if c in s and c.face_up:
                            d = st.defs[c.def_id]
                            fn = get_flip_trigger_effect(d)
                            if fn is not None:
                                self._push_effect(fn(st, pl, ln, c))
                                pushed_any = True
                            break
        return pushed_any

    def _do_end_phase(self) -> bool:
        st = self.state
        ap = st.current_player
        any_pushed = False
        # Per Codex: End effects on top text fire whenever the card is face-up
        # (top text is persistent under cover). Bottom-text End triggers only
        # while uncovered. We iterate all face-up cards and let each handler
        # decide if its conditions are met.
        for ln in range(NUM_LINES):
            stack = list(st.lines[ln].stack(ap))  # snapshot — effects may mutate
            for c in stack:
                if not c.face_up:
                    continue
                d = st.defs[c.def_id]
                fn = get_end_effect(d)
                if fn is not None:
                    self._push_effect(fn(st, ap, ln, c))
                    any_pushed = True
        if any_pushed:
            # We'll re-enter end phase logic after effects resolve. For
            # simplicity, mark phase to re-enter END so the next _drive
            # invocation continues. But we need to make sure end actions
            # only fire once. Use a flag:
            if not st.scratch.pop("end_resolved", False):
                st.scratch["end_resolved"] = True
                return True
        # Reset end flag for next turn.
        st.scratch.pop("end_resolved", None)
        # Clear "cannot compile next turn" flag for the player whose turn just
        # ended (they served their "no compile" turn).
        st.players[ap].cannot_compile_next_turn = False
        # Pass turn.
        st.current_player = 1 - st.current_player
        st.turn += 1
        st.compiled_this_turn = False
        st.effect_pushes_this_turn = 0
        if st.turn > st.config.max_turns:
            # Resolve the cap per config.
            if st.config.turn_cap_resolution == "leader_wins":
                a = sum(st.players[0].compiled)
                b = sum(st.players[1].compiled)
                if a > b:
                    st.winner = 0
                elif b > a:
                    st.winner = 1
                # else: tie -> winner stays None
            st.phase = Phase.GAME_OVER
            return False
        st.phase = Phase.START
        return False

    # ------------------------------------------------------------------ choice

    def _resolve_choice(self, action: Action) -> None:
        top = self._pending[-1]
        choice = top.last_choice
        assert choice is not None
        if action.type is ActionType.SKIP_OPTIONAL:
            if not choice.optional:
                raise ValueError("Cannot skip non-optional choice")
            top.last_choice = None
            try:
                next_c = top.gen.send(-1)
            except StopIteration:
                try:
                    self._pending.remove(top)
                except ValueError:
                    pass
                return
            top.last_choice = next_c
            return
        if action.type is not ActionType.CHOOSE_TARGET:
            raise ValueError(f"Expected CHOOSE_TARGET, got {action.type}")
        idx = action.choice_index
        if not (0 <= idx < len(choice.options)):
            raise ValueError(f"choice_index {idx} out of range")
        top.last_choice = None
        try:
            next_c = top.gen.send(idx)
        except StopIteration:
            try:
                self._pending.remove(top)
            except ValueError:
                pass
            return
        top.last_choice = next_c
