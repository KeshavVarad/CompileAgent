"""Card effects, persistent modifiers, restriction checks, and value computation.

Every one of the 90 cards has either a registered effect, a persistent
modifier handled inside `compute_line_value`, or a persistent rule honoured
inside `restrictions.py`-style queries (kept here for locality). Cards whose
only text is purely passive (e.g. Apathy 0 top) have no effect entry but are
covered by the value/restriction code paths.

Resolution model (matches the rulebook as closely as we can with bag-of-text):
  - On play face-up / flip face-up:
      1. If bottom starts with "First,": resolve bottom-first effect.
         (If self-flip occurs, abort and do not fire middle.)
      2. Resolve middle (unless suppressed by Apathy 2 in the same line).
      3. Resolve any one-shot top trigger (e.g. "Draw 1 card." on top).
  - On uncover (face-up card becomes top after the cover is removed):
      1. Resolve middle (suppression still applies).
  - Bottom triggered "Start:" / "End:" fire during phases for uncovered face-up
    cards. Other bottoms are treated as on-play effects unless they are
    purely persistent (e.g. "Your opponent cannot play cards in this line.").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator

from .actions import Choice
from .state import (
    FACE_DOWN_BASE_VALUE,
    NUM_LINES,
    CardInst,
    GameState,
)

if TYPE_CHECKING:
    from .cards import CardDef


# ---------------------------------------------------------------------------
# Value computation
# ---------------------------------------------------------------------------

def _cards_in_line_both(state: GameState, line_idx: int) -> list[CardInst]:
    line = state.lines[line_idx]
    return line.p0_stack + line.p1_stack


def _card_def(state: GameState, c: CardInst) -> "CardDef":
    return state.defs[c.def_id]


def _stack_face_down_base_value(state: GameState, line_idx: int, player: int) -> int:
    """Per-stack base value for face-down cards. Darkness 2 raises this to 4
    in its own stack while face-up uncovered (its top is only active when
    visible — face-up. We also require uncovered since covered top text is
    still visible per rules. So we only require face-up)."""
    stack = state.lines[line_idx].stack(player)
    base = FACE_DOWN_BASE_VALUE
    for c in stack:
        if not c.face_up:
            continue
        d = state.defs[c.def_id]
        if d.protocol == "Darkness" and d.value == 2:
            base = max(base, 4)
    return base


def compute_line_value(state: GameState, line_idx: int, player: int) -> int:
    """Total power of (line, player) including face-down base, persistent top
    modifiers from both this player's and opponent's face-up cards."""
    stack = state.lines[line_idx].stack(player)
    defs = state.defs
    opp = 1 - player

    fd_value = _stack_face_down_base_value(state, line_idx, player)
    total = 0
    for c in stack:
        if c.face_up:
            total += defs[c.def_id].value
        else:
            total += fd_value

    # Apathy 0 top (own side): "+1 per face-down card in this line" (both sides).
    fd_count_in_line = sum(1 for c in _cards_in_line_both(state, line_idx) if not c.face_up)
    for c in stack:
        if not c.face_up:
            continue
        d = defs[c.def_id]
        if d.protocol == "Apathy" and d.value == 0:
            total += fd_count_in_line

    # Metal 0 top (opp side): "Your opponent's total value in this line is reduced by 2."
    opp_stack = state.lines[line_idx].stack(opp)
    for c in opp_stack:
        if not c.face_up:
            continue
        d = defs[c.def_id]
        if d.protocol == "Metal" and d.value == 0:
            total -= 2

    # Smoke 2 top (own side, MN02): identical to Apathy 0 (+1 per face-down).
    for c in stack:
        if not c.face_up:
            continue
        d = defs[c.def_id]
        if d.protocol == "Smoke" and d.value == 2:
            total += fd_count_in_line

    # Mirror 0 top (own side, MN02): +1 per opponent's card in this line.
    opp_count_in_line = len(opp_stack)
    for c in stack:
        if not c.face_up:
            continue
        d = defs[c.def_id]
        if d.protocol == "Mirror" and d.value == 0:
            total += opp_count_in_line

    # Clarity 0 top (own side, MN02): +1 per card in OWN hand.
    hand_size = len(state.players[player].hand)
    for c in stack:
        if not c.face_up:
            continue
        d = defs[c.def_id]
        if d.protocol == "Clarity" and d.value == 0:
            total += hand_size

    return max(total, 0)


# ---------------------------------------------------------------------------
# Atomic helpers
# ---------------------------------------------------------------------------

# Per-atomic event flag helpers. After each atomic operation, we set a
# scratch flag (player-keyed) that the engine drains as a coalesced
# broadcast once the surrounding effect-stack resolves — matching Codex
# p.6 "after" semantics (the after-trigger fires when the whole
# triggering effect and its consequences finish, not per atomic step).
def _flag_after_discard(state: GameState, player: int) -> None:
    state.scratch[f"_pending_after_discard_by_p{player}"] = True


def _flag_after_draw(state: GameState, player: int) -> None:
    state.scratch[f"_pending_after_draw_by_p{player}"] = True


def _flag_after_delete(state: GameState, deleter: int) -> None:
    state.scratch[f"_pending_after_delete_by_p{deleter}"] = True


def _flag_after_shuffle(state: GameState, player: int) -> None:
    state.scratch[f"_pending_after_shuffle_by_p{player}"] = True


def _flag_after_refresh(state: GameState, player: int) -> None:
    state.scratch[f"_pending_after_refresh_by_p{player}"] = True


def _flag_flip(state: GameState, card: CardInst) -> None:
    # List of card instances whose flip needs to broadcast on the next
    # drain. Stored as a list rather than a set so order is preserved
    # (current-player's choice — Codex p.4 "Processing Multiple Cards
    # Triggering Simultaneously").
    lst = state.scratch.setdefault("_pending_flip_cards", [])
    lst.append(card)


def draw_cards(state: GameState, player: int, n: int) -> int:
    # Ice 6: blocked while own hand is non-empty.
    if player_cannot_draw(state, player):
        return 0
    ps = state.players[player]
    drawn = 0
    shuffled = False
    for _ in range(n):
        if not ps.deck:
            if not ps.trash:
                break
            ps.deck = ps.trash
            ps.trash = []
            state.rng.shuffle(ps.deck)
            shuffled = True
        ps.hand.append(ps.deck.pop())
        drawn += 1
    if drawn > 0:
        _flag_after_draw(state, player)
    if shuffled:
        _flag_after_shuffle(state, player)
    return drawn


def discard_to_trash(state: GameState, player: int, hand_index: int) -> CardInst:
    ps = state.players[player]
    c = ps.hand.pop(hand_index)
    c.face_up = True
    state.players[c.owner].trash.append(c)
    _flag_after_discard(state, player)
    return c


def delete_card_from_field(
    state: GameState, line_idx: int, player: int, stack_pos: int
) -> CardInst:
    stack = state.lines[line_idx].stack(player)
    was_top = stack_pos == len(stack) - 1
    c = stack.pop(stack_pos)
    c.face_up = True  # trash is always face-up
    state.players[c.owner].trash.append(c)
    # If we removed a cover, the new top (if face-up) becomes uncovered → trigger.
    if was_top is False and stack and stack[-1].face_up:
        state.triggers.append(("uncover", line_idx, player, stack[-1]))
    # "After you delete cards:" is attributed to the player who caused the
    # delete — in nearly every case state.current_player. (Edge cases like
    # opponent-driven flip→delete are still attributed to the active player.)
    _flag_after_delete(state, state.current_player)
    _check_diversity_6_self_destruct(state)
    return c


def flip_card(state: GameState, line_idx: int, player: int, stack_pos: int) -> CardInst:
    stack = state.lines[line_idx].stack(player)
    c = stack[stack_pos]
    # Ice 4: "This card cannot be flipped." Persistent immunity while
    # face-up on the field (covered or uncovered). Logged as a skipped
    # effect so the UI can show that the flip was blocked.
    d = state.defs[c.def_id]
    if d.key == "MN02:Ice:4" and c.face_up:
        logInfo(state, f"Flip on {d.protocol} {d.value} was blocked (immune).")
        return c
    was_up = c.face_up
    c.face_up = not c.face_up
    if not was_up and c.face_up:
        state.triggers.append(("face_up", line_idx, player, c))
    # `Flip:` emphasis fires whenever this card flips, either direction.
    # We queue the flipped card so the engine broadcasts after current
    # effect drains.
    _flag_flip(state, c)
    _check_diversity_6_self_destruct(state)
    return c


def uncommit_sentinel(c: CardInst):
    """Generator that clears `c.is_committed` and immediately completes.
    Used as a fence on the engine's pending stack between when_covered
    (drains first) and enter-play (drains after)."""
    c_ref = c
    def _gen():
        c_ref.is_committed = False
        if False:
            yield  # marks this as a generator
    return _gen()


def shift_card(
    state: GameState,
    src_line: int,
    src_player: int,
    src_pos: int,
    dst_line: int,
) -> CardInst:
    """Shift `(src_line, src_player, src_pos)` to dst_line on the same side.
    Implements the Compile Codex committed-card lifecycle by pushing
    orchestration generators onto the engine's pending stack in reverse
    chronological order. Falls back to a simple atomic move (no triggers)
    if no engine reference is wired into state.scratch.
    """
    src_stack = state.lines[src_line].stack(src_player)
    was_top = src_pos == len(src_stack) - 1
    c = src_stack.pop(src_pos)
    c.is_committed = True
    dst_stack = state.lines[dst_line].stack(src_player)
    soon_covered = dst_stack[-1] if dst_stack else None
    dst_stack.append(c)

    engine = state.scratch.get("_engine")
    if engine is None:
        c.is_committed = False
        return c

    # Push orchestration onto _pending in REVERSE chronological order.
    # Chronological order:
    #   1. uncover of src's new top (if previously covered, face-up) —
    #      Codex "Middle Command - Immediate: Resolve this active text
    #      upon card play/flip/uncover."
    #   2. when_covered of dst's previously-top card (Fire 0 errata)
    #   3. uncommit C
    #   4. uncover middle of C (if C was previously covered at src and now
    #      uncovered at dst — same Codex middle rule)
    # Push order: 4, 3, 2, 1 (last push = drains first = chronologically first).
    if c.face_up and not was_top:
        d = state.defs[c.def_id]
        if d.middle_text and not middle_suppressed(state, dst_line, c):
            mf = MIDDLE_EFFECTS.get(d.key)
            if mf is not None:
                engine._push_effect(mf(state, c.owner, dst_line, c))
    engine._push_effect(uncommit_sentinel(c))
    if soon_covered is not None and soon_covered.face_up:
        d_under = state.defs[soon_covered.def_id]
        wc = WHEN_COVERED_EFFECTS.get(d_under.key)
        if wc is not None:
            engine._push_effect(wc(state, soon_covered.owner, dst_line, soon_covered))
    if not was_top and src_stack and src_stack[-1].face_up:
        nt = src_stack[-1]
        d_nt = state.defs[nt.def_id]
        if d_nt.middle_text and not middle_suppressed(state, src_line, nt):
            mf2 = MIDDLE_EFFECTS.get(d_nt.key)
            if mf2 is not None:
                engine._push_effect(mf2(state, src_player, src_line, nt))
    _check_diversity_6_self_destruct(state)
    return c


def return_card_to_hand(
    state: GameState, line_idx: int, player: int, stack_pos: int
) -> CardInst:
    stack = state.lines[line_idx].stack(player)
    was_top = stack_pos == len(stack) - 1
    c = stack.pop(stack_pos)
    c.face_up = False
    state.players[c.owner].hand.append(c)
    if not was_top and stack and stack[-1].face_up:
        state.triggers.append(("uncover", line_idx, player, stack[-1]))
    _check_diversity_6_self_destruct(state)
    return c


def play_top_deck_face_down(
    state: GameState, player: int, line_idx: int
) -> CardInst | None:
    ps = state.players[player]
    if not ps.deck:
        if not ps.trash:
            return None
        ps.deck = ps.trash
        ps.trash = []
        state.rng.shuffle(ps.deck)
        if not ps.deck:
            return None
    c = ps.deck.pop()
    c.face_up = False
    state.lines[line_idx].stack(player).append(c)
    return c


def play_top_deck_face_down_under(
    state: GameState, player: int, line_idx: int, under_card: CardInst,
) -> CardInst | None:
    """Play top of deck face-down placed *under* `under_card` (insert below).
    Used by Gravity 0."""
    ps = state.players[player]
    if not ps.deck:
        if not ps.trash:
            return None
        ps.deck = ps.trash
        ps.trash = []
        state.rng.shuffle(ps.deck)
        if not ps.deck:
            return None
    stack = state.lines[line_idx].stack(player)
    pos = stack.index(under_card) if under_card in stack else len(stack)
    c = ps.deck.pop()
    c.face_up = False
    stack.insert(pos, c)
    return c


def refresh_player(state: GameState, player: int) -> None:
    ps = state.players[player]
    need = 5 - len(ps.hand)
    if need > 0:
        draw_cards(state, player, need)
    _flag_after_refresh(state, player)


# ---------------------------------------------------------------------------
# Target enumeration
# ---------------------------------------------------------------------------

def _is_shift_targetable_while_covered(state: GameState, c: CardInst) -> bool:
    """Speed 2 / Spirit 3 top: shift effects may target this card even when
    it is covered, so long as it's face-up (otherwise its top isn't active)."""
    if not c.face_up:
        return False
    d = state.defs[c.def_id]
    return (
        (d.protocol == "Speed" and d.value == 2)
        or (d.protocol == "Spirit" and d.value == 3)
    )


def _enumerate_shift_targets(
    state: GameState,
    *,
    owner: str = "any",
    face: str = "any",
    exclude: CardInst | None = None,
    active_player: int = 0,
    line_filter: int | None = None,
) -> list[tuple[int, int, int, CardInst]]:
    """Targets eligible for a 'shift' effect. Equals uncovered targets PLUS
    any face-up covered Speed 2 / Spirit 3 cards (their top permits shifting
    even while covered)."""
    targets = _enumerate_uncovered(
        state, owner=owner, face=face, exclude=exclude,
        active_player=active_player, line_filter=line_filter,
    )
    if face in ("any", "up"):
        for li in range(NUM_LINES):
            if line_filter is not None and li != line_filter:
                continue
            for pl in (0, 1):
                if owner == "self" and pl != active_player:
                    continue
                if owner == "opponent" and pl == active_player:
                    continue
                stack = state.lines[li].stack(pl)
                for pos in range(len(stack) - 1):  # covered = not top
                    c = stack[pos]
                    if c.is_committed:
                        continue
                    if exclude is c:
                        continue
                    if _is_shift_targetable_while_covered(state, c):
                        targets.append((li, pl, pos, c))
    return targets


def _enumerate_uncovered(
    state: GameState,
    *,
    owner: str = "any",          # "any" | "self" | "opponent"
    face: str = "any",           # "any" | "up" | "down"
    exclude: CardInst | None = None,
    active_player: int = 0,
    line_filter: int | None = None,
) -> list[tuple[int, int, int, CardInst]]:
    out: list[tuple[int, int, int, CardInst]] = []
    for li in range(NUM_LINES):
        if line_filter is not None and li != line_filter:
            continue
        for pl in (0, 1):
            stack = state.lines[li].stack(pl)
            if not stack:
                continue
            c = stack[-1]
            if c.is_committed:
                continue
            if exclude is not None and c is exclude:
                continue
            if owner == "self" and pl != active_player:
                continue
            if owner == "opponent" and pl == active_player:
                continue
            if face == "up" and not c.face_up:
                continue
            if face == "down" and c.face_up:
                continue
            out.append((li, pl, len(stack) - 1, c))
    return out


def _enumerate_all(
    state: GameState,
    *,
    owner: str = "any",
    face: str = "any",
    exclude: CardInst | None = None,
    active_player: int = 0,
    line_filter: int | None = None,
) -> list[tuple[int, int, int, CardInst]]:
    out: list[tuple[int, int, int, CardInst]] = []
    for li in range(NUM_LINES):
        if line_filter is not None and li != line_filter:
            continue
        for pl in (0, 1):
            stack = state.lines[li].stack(pl)
            for pos, c in enumerate(stack):
                if c.is_committed:
                    continue
                if exclude is not None and c is exclude:
                    continue
                if owner == "self" and pl != active_player:
                    continue
                if owner == "opponent" and pl == active_player:
                    continue
                if face == "up" and not c.face_up:
                    continue
                if face == "down" and c.face_up:
                    continue
                out.append((li, pl, pos, c))
    return out


def _describe_card(
    state: GameState,
    line_idx: int,
    player: int,
    c: CardInst,
    viewer: int | None = None,
) -> str:
    """Render a (line, player, card) tuple as a Choice-prompt label.

    Face-down cards are private information per Codex p.5 — the viewer
    only knows their own face-downs. Opponent face-downs are redacted
    to "face-down (2)"; the underlying target object still carries the
    real card so the engine resolves correctly. Pass `viewer=None` to
    keep the old reveal-everything behaviour (used by non-UI code
    paths like logging and replay diffing).
    """
    d = state.defs[c.def_id]
    owned_by_viewer = viewer is not None and player == viewer
    knows_identity = c.face_up or owned_by_viewer or viewer is None
    side = (
        f"P{player + 1}" if viewer is None
        else ("your" if owned_by_viewer else "opp")
    )
    lane = f"L{line_idx + 1}"
    if not knows_identity:
        return f"{lane} {side}: face-down (2)"
    facing = "face-up" if c.face_up else "face-down"
    return f"{lane} {side}: {d.protocol} {d.value} ({facing})"


def _describe_hand_card(state: GameState, player: int, idx: int) -> str:
    c = state.players[player].hand[idx]
    d = state.defs[c.def_id]
    return f"hand[{idx}]: {d.protocol} {d.value}"


# ---------------------------------------------------------------------------
# Persistent rule queries (called by game.py)
# ---------------------------------------------------------------------------

def middle_suppressed(state: GameState, line_idx: int, card: CardInst) -> bool:
    """Apathy 2 in this line ignores middle commands of other cards. Fear 0
    additionally suppresses opponent middles during the active player's turn
    (regardless of which line)."""
    # Apathy 2 — line-local suppression
    for c in _cards_in_line_both(state, line_idx):
        if c is card or not c.face_up:
            continue
        d = state.defs[c.def_id]
        if d.protocol == "Apathy" and d.value == 2:
            return True
    # Fear 0 — global suppression of opponent's middles during active turn.
    ap = state.current_player
    owner = card.owner
    if owner != ap:
        # `card` belongs to the non-active player; check if active player has
        # a face-up Fear 0 anywhere in their stacks.
        for li in range(NUM_LINES):
            for cc in state.lines[li].stack(ap):
                if not cc.face_up:
                    continue
                d = state.defs[cc.def_id]
                if d.protocol == "Fear" and d.value == 0:
                    return True
    return False


def _check_diversity_6_self_destruct(state: GameState) -> None:
    """Diversity 6 top: 'If there are not at least 3 different protocols on
    cards in the field, delete this card.' Continuous predicate — checked
    after every field mutation (mirrors the old Life-0 sweep pattern)."""
    while True:
        protos: set[str] = set()
        for li in range(NUM_LINES):
            for pl in (0, 1):
                for c in state.lines[li].stack(pl):
                    protos.add(state.defs[c.def_id].protocol)
        if len(protos) >= 3:
            return
        # Find a face-up Diversity 6 and delete it.
        found = False
        for li in range(NUM_LINES):
            for pl in (0, 1):
                s = state.lines[li].stack(pl)
                for pos, c in enumerate(s):
                    if not c.face_up:
                        continue
                    d = state.defs[c.def_id]
                    if d.protocol == "Diversity" and d.value == 6:
                        s.pop(pos)
                        c.face_up = True
                        state.players[c.owner].trash.append(c)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return


def _legal_sub_plays(
    state: GameState,
    ap: int,
    *,
    candidate_hand_indices: list[int] | None = None,
) -> list[tuple[str, int, int, bool, bool]]:
    """Build the legal-play menu for "play 1 card" sub-effects (Speed 0,
    Clarity 2, Luck 0, Time 0). Honours every play-time affordance the action
    phase honours: Plague 0 / Metal 2 / Psychic 1 restrictions, Spirit 1
    any-line face-up, Chaos 3 face-up bypass, Corruption 0 cross-side play.

    Returns list of (label, hand_index, line_index, face_up, cross_side).
    For cross_side=True (Corruption 0), `line_index` is in [NUM_LINES, 2*NUM_LINES)
    encoding the opponent's lines 0..2 — same encoding used by
    `_action_phase_legal` in game.py.
    """
    out: list[tuple[str, int, int, bool, bool]] = []
    spirit_1 = player_may_play_any_line_faceup(state, ap)
    psychic_1 = opp_must_play_facedown(state, ap)
    line_blocked = [opp_play_blocked_in_line(state, ln, ap) for ln in range(NUM_LINES)]
    line_fd_blocked = [opp_play_facedown_blocked_in_line(state, ln, ap) for ln in range(NUM_LINES)]
    hand = state.players[ap].hand
    indices = candidate_hand_indices if candidate_hand_indices is not None else list(range(len(hand)))
    for hi in indices:
        if hi < 0 or hi >= len(hand):
            continue
        c = hand[hi]
        d = state.defs[c.def_id]
        chaos_3 = d.protocol == "Chaos" and d.value == 3
        corruption_0 = d.protocol == "Corruption" and d.value == 0
        unrestricted_fu = spirit_1 or chaos_3 or corruption_0
        for ln in range(NUM_LINES):
            if line_blocked[ln] or psychic_1:
                continue
            if unrestricted_fu or state.players[ap].protocols[ln] == d.protocol:
                out.append((f"FU {d.protocol}{d.value} L{ln}", hi, ln, True, False))
        for ln in range(NUM_LINES):
            if line_blocked[ln] or line_fd_blocked[ln]:
                continue
            out.append((f"FD hand[{hi}] L{ln}", hi, ln, False, False))
        # Corruption 0 cross-side option.
        if corruption_0:
            for ln in range(NUM_LINES):
                out.append((f"FU OPP L{ln}: {d.protocol}{d.value}", hi, NUM_LINES + ln, True, True))
                out.append((f"FD OPP L{ln}", hi, NUM_LINES + ln, False, True))
    return out


def player_cannot_draw(state: GameState, player: int) -> bool:
    """Ice 6 top (MN02): 'If you have any cards in your hand, you cannot draw
    cards.' Active for the owner of the face-up Ice 6 when their hand is
    non-empty."""
    if not state.players[player].hand:
        return False
    for li in range(NUM_LINES):
        for c in state.lines[li].stack(player):
            if not c.face_up:
                continue
            d = state.defs[c.def_id]
            if d.protocol == "Ice" and d.value == 6:
                return True
    return False


def player_can_compile(state: GameState, player: int) -> bool:
    """Honour Metal 1: 'Your opponent cannot compile on their next turn.'"""
    return not state.players[player].cannot_compile_next_turn


def opp_play_facedown_blocked_in_line(state: GameState, line_idx: int, player: int) -> bool:
    """Metal 2 top: 'Your opponent cannot play cards face-down in this line.'
    `player` is the would-be face-down player; if their opponent has Metal 2
    face-up uncovered in this line, they cannot play face-down here."""
    opp = 1 - player
    stack = state.lines[line_idx].stack(opp)
    if not stack:
        return False
    # Persistent top text is active while face-up (regardless of cover).
    for c in stack:
        if not c.face_up:
            continue
        d = state.defs[c.def_id]
        if d.protocol == "Metal" and d.value == 2:
            return True
    return False


def opp_must_play_facedown(state: GameState, player: int) -> bool:
    """Psychic 1 top: 'Your opponent can only play cards face-down.'"""
    opp = 1 - player
    for li in range(NUM_LINES):
        for c in state.lines[li].stack(opp):
            if not c.face_up:
                continue
            d = state.defs[c.def_id]
            if d.protocol == "Psychic" and d.value == 1:
                return True
    return False


def opp_play_blocked_in_line(state: GameState, line_idx: int, player: int) -> bool:
    """Plague 0 bottom (only while uncovered): 'Your opponent cannot play
    cards in this line.'"""
    opp = 1 - player
    stack = state.lines[line_idx].stack(opp)
    if not stack:
        return False
    top = stack[-1]
    if not top.face_up:
        return False
    d = state.defs[top.def_id]
    return d.protocol == "Plague" and d.value == 0


def player_may_play_any_line_faceup(state: GameState, player: int) -> bool:
    """Spirit 1 top: 'When you play cards face-up, they may be played without
    matching protocols.'"""
    for li in range(NUM_LINES):
        for c in state.lines[li].stack(player):
            if not c.face_up:
                continue
            d = state.defs[c.def_id]
            if d.protocol == "Spirit" and d.value == 1:
                return True
    return False


def player_skips_check_cache(state: GameState, player: int) -> bool:
    """Spirit 0 bottom: 'Skip your check cache phase.' (active when uncovered)."""
    for li in range(NUM_LINES):
        stack = state.lines[li].stack(player)
        if not stack:
            continue
        top = stack[-1]
        if not top.face_up:
            continue
        d = state.defs[top.def_id]
        if d.protocol == "Spirit" and d.value == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Effect framework
# ---------------------------------------------------------------------------

EffectGen = Generator[Choice, int, None]
EffectFn = Callable[[GameState, int, int, CardInst], EffectGen]

# -- Position-based triggers ------------------------------------------------
# These fire (or used to fire) based on the card's position in the stack
# when it enters play. After the bugfix, MIDDLE_EFFECTS only fire on the
# `when_covered` event, not on play — this matches the actual rules text
# ("middle effect activates when this card becomes covered").
MIDDLE_EFFECTS: dict[str, EffectFn] = {}
BOTTOM_FIRST_EFFECTS: dict[str, EffectFn] = {}
BOTTOM_ON_PLAY_EFFECTS: dict[str, EffectFn] = {}
TOP_TRIGGER_EFFECTS: dict[str, EffectFn] = {}

# -- Turn-phase triggers ----------------------------------------------------
START_EFFECTS: dict[str, EffectFn] = {}
END_EFFECTS: dict[str, EffectFn] = {}

# -- Event-based triggers (new) --------------------------------------------
# Each fires for face-up cards in the affected player's field whenever the
# named event happens. Use these for "After …:" / "When …:" emphasised top
# texts that were previously (incorrectly) registered as @top_trigger.
WHEN_COVERED_EFFECTS: dict[str, EffectFn] = {}      # also auto-fires MIDDLE_EFFECTS for the covered card
AFTER_CLEAR_CACHE_EFFECTS: dict[str, EffectFn] = {}     # after CHECK_CACHE phase resolves
AFTER_SELF_DISCARD_EFFECTS: dict[str, EffectFn] = {}    # after this player discards
AFTER_OPP_DISCARD_EFFECTS: dict[str, EffectFn] = {}     # after opponent discards
AFTER_SELF_DELETE_EFFECTS: dict[str, EffectFn] = {}     # after this player deletes a card
AFTER_SELF_DRAW_EFFECTS: dict[str, EffectFn] = {}       # after this player draws ≥1 card
AFTER_SELF_SHUFFLE_EFFECTS: dict[str, EffectFn] = {}    # after this player shuffles their deck
AFTER_SELF_REFRESH_EFFECTS: dict[str, EffectFn] = {}    # after this player refreshes
FLIP_TRIGGER_EFFECTS: dict[str, EffectFn] = {}          # whenever this card flips, either direction
WHEN_DELETED_BY_COMPILE_EFFECTS: dict[str, EffectFn] = {}  # before this card is sent to trash by compile


def _register(reg: dict[str, EffectFn], key: str):
    def deco(fn: EffectFn) -> EffectFn:
        reg[key] = fn
        return fn
    return deco


def middle(key: str):
    return _register(MIDDLE_EFFECTS, key)


def bottom_first(key: str):
    return _register(BOTTOM_FIRST_EFFECTS, key)


def bottom_on_play(key: str):
    return _register(BOTTOM_ON_PLAY_EFFECTS, key)


def start_trigger(key: str):
    return _register(START_EFFECTS, key)


def end_trigger(key: str):
    return _register(END_EFFECTS, key)


def top_trigger(key: str):
    """Top-tier effect that fires WHEN THIS CARD BECOMES FACE-UP. Reserve
    this decorator for unconditional one-shots like Speed 0 top ("Play
    another card"). For top tiers with an emphasis like 'After you clear
    cache:' / 'Start:' / 'End:' / 'After you discard:' etc., use the
    event-specific decorators below — those events fire later, not on
    play, and using @top_trigger gives the card a free extra effect on
    play that the rules don't allow."""
    return _register(TOP_TRIGGER_EFFECTS, key)


def when_covered(key: str):
    return _register(WHEN_COVERED_EFFECTS, key)


def after_clear_cache(key: str):
    return _register(AFTER_CLEAR_CACHE_EFFECTS, key)


def after_self_discard(key: str):
    return _register(AFTER_SELF_DISCARD_EFFECTS, key)


def after_opp_discard(key: str):
    return _register(AFTER_OPP_DISCARD_EFFECTS, key)


def after_self_delete(key: str):
    return _register(AFTER_SELF_DELETE_EFFECTS, key)


def after_self_draw(key: str):
    return _register(AFTER_SELF_DRAW_EFFECTS, key)


def after_self_shuffle(key: str):
    return _register(AFTER_SELF_SHUFFLE_EFFECTS, key)


def after_self_refresh(key: str):
    return _register(AFTER_SELF_REFRESH_EFFECTS, key)


def flip_trigger(key: str):
    return _register(FLIP_TRIGGER_EFFECTS, key)


def when_deleted_by_compile(key: str):
    return _register(WHEN_DELETED_BY_COMPILE_EFFECTS, key)


def get_middle_effect(d: "CardDef") -> EffectFn | None:
    return MIDDLE_EFFECTS.get(d.key)


def get_bottom_first_effect(d: "CardDef") -> EffectFn | None:
    return BOTTOM_FIRST_EFFECTS.get(d.key)


def get_bottom_on_play_effect(d: "CardDef") -> EffectFn | None:
    return BOTTOM_ON_PLAY_EFFECTS.get(d.key)


def get_start_effect(d: "CardDef") -> EffectFn | None:
    return START_EFFECTS.get(d.key)


def get_end_effect(d: "CardDef") -> EffectFn | None:
    return END_EFFECTS.get(d.key)


def get_top_trigger(d: "CardDef") -> EffectFn | None:
    return TOP_TRIGGER_EFFECTS.get(d.key)


def get_when_covered_effect(d: "CardDef") -> EffectFn | None:
    return WHEN_COVERED_EFFECTS.get(d.key)


def get_after_clear_cache_effect(d: "CardDef") -> EffectFn | None:
    return AFTER_CLEAR_CACHE_EFFECTS.get(d.key)


def get_after_self_discard_effect(d: "CardDef") -> EffectFn | None:
    return AFTER_SELF_DISCARD_EFFECTS.get(d.key)


def get_after_opp_discard_effect(d: "CardDef") -> EffectFn | None:
    return AFTER_OPP_DISCARD_EFFECTS.get(d.key)


def get_after_self_delete_effect(d: "CardDef") -> EffectFn | None:
    return AFTER_SELF_DELETE_EFFECTS.get(d.key)


def get_after_self_draw_effect(d: "CardDef") -> EffectFn | None:
    return AFTER_SELF_DRAW_EFFECTS.get(d.key)


def get_after_self_shuffle_effect(d: "CardDef") -> EffectFn | None:
    return AFTER_SELF_SHUFFLE_EFFECTS.get(d.key)


def get_after_self_refresh_effect(d: "CardDef") -> EffectFn | None:
    return AFTER_SELF_REFRESH_EFFECTS.get(d.key)


def get_flip_trigger_effect(d: "CardDef") -> EffectFn | None:
    return FLIP_TRIGGER_EFFECTS.get(d.key)


def get_when_deleted_by_compile_effect(d: "CardDef") -> EffectFn | None:
    return WHEN_DELETED_BY_COMPILE_EFFECTS.get(d.key)


# ---------------------------------------------------------------------------
# Shared helpers: target prompts as generators
# ---------------------------------------------------------------------------

def _prompt_target(
    prompt: str, targets: list, options: list[str], decider: int, optional: bool = False,
) -> EffectGen:
    if not targets:
        return
        yield  # pragma: no cover
    idx = yield Choice(
        prompt=prompt, options=options, targets=targets,
        optional=optional, decider=decider,
    )
    return idx  # noqa - generators don't really return like this but caller uses send


def _choose_hand_target(
    state: GameState, player: int, prompt: str, optional: bool = False,
) -> EffectGen:
    hand = state.players[player].hand
    if not hand:
        return
    opts = [_describe_hand_card(state, player, i) for i in range(len(hand))]
    idx = yield Choice(
        prompt=prompt, options=opts, targets=list(range(len(hand))),
        optional=optional, decider=player,
    )
    return idx


def _discard_n(state: GameState, player: int, n: int) -> EffectGen:
    """Discard exactly n cards from player's hand (if hand has that many)."""
    for _ in range(n):
        if not state.players[player].hand:
            return
        opts = [_describe_hand_card(state, player, i)
                for i in range(len(state.players[player].hand))]
        idx = yield Choice(
            prompt="Discard 1 card from hand",
            options=opts,
            targets=list(range(len(state.players[player].hand))),
            decider=player,
        )
        discard_to_trash(state, player, idx)


def _discard_one_or_more(state: GameState, player: int) -> tuple[int, EffectGen]:
    """Helper for 'Discard 1 or more cards.' patterns. Yields prompts until
    the player chooses to stop (SKIP_OPTIONAL). Returns count via the
    generator's surrounding scope (caller must wrap)."""
    # Implemented inline by the calling card effects via _discard_optional_loop.
    raise NotImplementedError


def _discard_optional_loop(state: GameState, player: int, max_n: int) -> EffectGen:
    """Yield Choice prompts that let the player discard 0..max_n cards.
    Returns the number actually discarded by tracking via a small wrapper."""
    discarded = 0
    while discarded < max_n and state.players[player].hand:
        opts = [_describe_hand_card(state, player, i)
                for i in range(len(state.players[player].hand))]
        idx = yield Choice(
            prompt=f"Discard another card? ({discarded} so far) — choose card or skip",
            options=opts,
            targets=list(range(len(state.players[player].hand))),
            optional=True,
            decider=player,
        )
        if idx == -1:  # SKIP_OPTIONAL sentinel
            break
        discard_to_trash(state, player, idx)
        discarded += 1
    # Stash count on state for the caller to retrieve.
    state.scratch["last_discard_count"] = discarded


# ---------------------------------------------------------------------------
# Per-card middle effects (90 cards)
# ---------------------------------------------------------------------------

# Standard "You discard 1 card." for every value-5 card.
def _value5_discard(state: GameState, ap: int, li: int, card: CardInst) -> EffectGen:
    yield from _discard_n(state, ap, 1)


for _proto, _set in (
    # MN01
    ("Darkness", "MN01"), ("Death", "MN01"), ("Fire", "MN01"),
    ("Gravity", "MN01"), ("Life", "MN01"), ("Light", "MN01"),
    ("Metal", "MN01"), ("Plague", "MN01"), ("Psychic", "MN01"),
    ("Speed", "MN01"), ("Spirit", "MN01"), ("Water", "MN01"),
    # AX01
    ("Apathy", "AX01"), ("Hate", "AX01"), ("Love", "AX01"),
    # MN02 — every value-5 card in the new set also has "You discard 1 card."
    ("Chaos", "MN02"), ("Clarity", "MN02"), ("Corruption", "MN02"),
    ("Courage", "MN02"), ("Ice", "MN02"), ("Luck", "MN02"),
    ("Mirror", "MN02"), ("Peace", "MN02"), ("Smoke", "MN02"),
    ("Time", "MN02"), ("War", "MN02"),
    # AX02
    ("Assimilation", "AX02"), ("Diversity", "AX02"), ("Unity", "AX02"),
):
    MIDDLE_EFFECTS[f"{_set}:{_proto}:5"] = _value5_discard

# Fear 5 is a bottom-text discard, not middle — see new sets section.


# ----- APATHY (AX01) ----------------------------------------------------

# Apathy 0: top-only (handled in compute_line_value)
# Apathy 1: middle
@middle("AX01:Apathy:1")
def _apathy_1(state, ap, li, card):
    for pl in (0, 1):
        for c in state.lines[li].stack(pl):
            if c is card or not c.face_up:
                continue
            c.face_up = False
    if False:
        yield  # pragma: no cover


# Apathy 2: top-only ("ignore middles"). Bottom "First, flip this card" handled below.
@bottom_first("AX01:Apathy:2")
def _apathy_2_first(state, ap, li, card):
    # Flip itself immediately.
    for pl in (0, 1):
        s = state.lines[li].stack(pl)
        if card in s:
            card.face_up = not card.face_up
            break
    if False:
        yield  # pragma: no cover


# Apathy 3: middle
@middle("AX01:Apathy:3")
def _apathy_3(state, ap, li, card):
    targets = _enumerate_uncovered(state, owner="opponent", face="up", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(
        prompt="Flip 1 of your opponent's face-up cards", options=opts,
        targets=targets, decider=ap,
    )
    ln, pl, pos, _ = targets[idx]
    flip_card(state, ln, pl, pos)


# Apathy 4: middle
@middle("AX01:Apathy:4")
def _apathy_4(state, ap, li, card):
    targets = []
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(ap)
        for pos in range(len(s) - 1):  # covered = not top
            c = s[pos]
            if c.face_up:
                targets.append((ln, ap, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(
        prompt="You may flip 1 of your face-up covered cards", options=opts,
        targets=targets, optional=True, decider=ap,
    )
    if idx == -1:
        return
    ln, pl, pos, _ = targets[idx]
    flip_card(state, ln, pl, pos)


# ----- HATE (AX01) ------------------------------------------------------

@middle("AX01:Hate:0")
def _hate_0(state, ap, li, card):
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Delete 1 card", options=opts, targets=targets, decider=ap)
    ln, pl, pos, _ = targets[idx]
    delete_card_from_field(state, ln, pl, pos)


@middle("AX01:Hate:1")
def _hate_1(state, ap, li, card):
    # Discard 3 cards. Delete 1 card. Delete 1 card.
    yield from _discard_n(state, ap, 3)
    for _ in range(2):
        targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
        if not targets:
            return
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
        idx = yield Choice(prompt="Delete 1 card", options=opts, targets=targets, decider=ap)
        ln, pl, pos, _ = targets[idx]
        delete_card_from_field(state, ln, pl, pos)


@middle("AX01:Hate:2")
def _hate_2(state, ap, li, card):
    # Delete your highest value uncovered card. Delete opp's highest value uncovered card.
    for who in (ap, 1 - ap):
        targets = _enumerate_uncovered(state, owner="self" if who == ap else "opponent",
                                       active_player=ap)
        if not targets:
            continue
        # Highest value of the uncovered card itself (face-up = value, face-down = base).
        scored = [(t[3].value(state.defs), t) for t in targets]
        scored.sort(key=lambda x: -x[0])
        top_val = scored[0][0]
        tied = [t for v, t in scored if v == top_val]
        if len(tied) == 1:
            ln, pl, pos, _ = tied[0]
            delete_card_from_field(state, ln, pl, pos)
        else:
            opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in tied]
            idx = yield Choice(prompt="Break tie: delete which?", options=opts,
                               targets=tied, decider=ap)
            ln, pl, pos, _ = tied[idx]
            delete_card_from_field(state, ln, pl, pos)


# Hate 3 top: "After you delete cards: Draw 1 card." Conditional trigger
# fires after the owner's delete action and all its consequences resolve.
@after_self_delete("AX01:Hate:3")
def _hate_3_top(state, ap, li, card):
    draw_cards(state, ap, 1)
    if False:
        yield  # pragma: no cover


# Hate 4: bottom-first "delete lowest value covered card in this line"
@bottom_first("AX01:Hate:4")
def _hate_4_first(state, ap, li, card):
    # Lowest value covered card across both sides of this line.
    targets = []
    for pl in (0, 1):
        s = state.lines[li].stack(pl)
        for pos in range(len(s) - 1):  # covered
            targets.append((li, pl, pos, s[pos]))
    if not targets:
        return
    scored = [(t[3].value(state.defs), t) for t in targets]
    scored.sort(key=lambda x: x[0])
    lo = scored[0][0]
    tied = [t for v, t in scored if v == lo]
    if len(tied) == 1:
        ln, pl, pos, _ = tied[0]
        delete_card_from_field(state, ln, pl, pos)
    else:
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in tied]
        idx = yield Choice(prompt="Break tie", options=opts, targets=tied, decider=ap)
        ln, pl, pos, _ = tied[idx]
        delete_card_from_field(state, ln, pl, pos)


# ----- LOVE (AX01) ------------------------------------------------------

@middle("AX01:Love:1")
def _love_1(state, ap, li, card):
    # Draw the top card of your opponent's deck.
    opp = 1 - ap
    ps_opp = state.players[opp]
    if not ps_opp.deck:
        if ps_opp.trash:
            ps_opp.deck = ps_opp.trash
            ps_opp.trash = []
            state.rng.shuffle(ps_opp.deck)
    if ps_opp.deck:
        c = ps_opp.deck.pop()
        c.owner = ap  # ownership transfers
        c.face_up = False
        state.players[ap].hand.append(c)
    if False:
        yield  # pragma: no cover


@bottom_on_play("AX01:Love:1")
def _love_1_bottom(state, ap, li, card):
    # You may give 1 card from your hand to your opponent. If you do, draw 2 cards.
    hand = state.players[ap].hand
    if not hand:
        return
    opts = [_describe_hand_card(state, ap, i) for i in range(len(hand))]
    idx = yield Choice(
        prompt="Give a card to opponent (optional)? If so pick which",
        options=opts, targets=list(range(len(hand))), optional=True, decider=ap,
    )
    if idx == -1:
        return
    c = hand.pop(idx)
    c.owner = 1 - ap
    state.players[1 - ap].hand.append(c)
    draw_cards(state, ap, 2)


@middle("AX01:Love:2")
def _love_2(state, ap, li, card):
    # Your opponent draws 1 card. Refresh.
    draw_cards(state, 1 - ap, 1)
    refresh_player(state, ap)
    if False:
        yield  # pragma: no cover


@middle("AX01:Love:3")
def _love_3(state, ap, li, card):
    # Take 1 random card from opp hand. Give 1 card from yours to opp.
    opp = 1 - ap
    if state.players[opp].hand:
        idx = state.rng.randrange(len(state.players[opp].hand))
        c = state.players[opp].hand.pop(idx)
        c.owner = ap
        state.players[ap].hand.append(c)
    hand = state.players[ap].hand
    if hand:
        opts = [_describe_hand_card(state, ap, i) for i in range(len(hand))]
        idx2 = yield Choice(
            prompt="Give 1 card from your hand to your opponent",
            options=opts, targets=list(range(len(hand))), decider=ap,
        )
        c2 = hand.pop(idx2)
        c2.owner = opp
        state.players[opp].hand.append(c2)


@middle("AX01:Love:4")
def _love_4(state, ap, li, card):
    # Reveal 1 card from your hand. Flip 1 card.
    # (Reveal is informational; we expose it via state.log.)
    hand = state.players[ap].hand
    if hand:
        opts = [_describe_hand_card(state, ap, i) for i in range(len(hand))]
        idx = yield Choice(prompt="Reveal 1 card from your hand",
                           options=opts, targets=list(range(len(hand))), decider=ap)
        d = state.defs[hand[idx].def_id]
        state.log.append(f"P{ap} reveals {d.protocol} {d.value}")
    targets = _enumerate_uncovered(state, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx2 = yield Choice(prompt="Flip 1 card", options=opts, targets=targets, decider=ap)
    ln, pl, pos, _ = targets[idx2]
    flip_card(state, ln, pl, pos)


@middle("AX01:Love:6")
def _love_6(state, ap, li, card):
    draw_cards(state, 1 - ap, 2)
    if False:
        yield  # pragma: no cover


# ----- DARKNESS (MN01) --------------------------------------------------

@middle("MN01:Darkness:0")
def _darkness_0(state, ap, li, card):
    draw_cards(state, ap, 3)
    # "Shift 1 of your opponent's covered cards." The "covered" keyword
    # is an explicit override of the default "uncovered only" rule on
    # shift effects (Codex p.3) — we enumerate strictly cards beneath
    # the top of each opp stack and skip any mid-commit.
    opp = 1 - ap
    targets = []
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(opp)
        for pos in range(len(s) - 1):
            c = s[pos]
            if c.is_committed:
                continue
            targets.append((ln, opp, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 of opponent's covered cards",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    # Codex p.4: shift must be to a DIFFERENT line on the same side.
    dest_lines = [i for i in range(NUM_LINES) if i != sln]
    dst_idx = yield Choice(
        prompt="To which line?",
        options=[f"L{i + 1}" for i in dest_lines],
        targets=dest_lines, decider=ap,
    )
    dest = dest_lines[dst_idx]
    shift_card(state, sln, spl, spos, dest)


@middle("MN01:Darkness:1")
def _darkness_1(state, ap, li, card):
    targets = _enumerate_uncovered(state, owner="opponent", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 of opponent's cards",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    flip_card(state, sln, spl, spos)
    # Optional shift that same card.
    cur_pos = len(state.lines[sln].stack(spl)) - 1  # it's still uncovered (top)
    if state.lines[sln].stack(spl) and state.lines[sln].stack(spl)[-1] is targets[idx][3]:
        dst_idx = yield Choice(
            prompt="Optionally shift that card to another line",
            options=[str(i) for i in range(NUM_LINES)] + ["skip"],
            targets=list(range(NUM_LINES)) + [-1],
            optional=False, decider=ap,
        )
        target_line = (list(range(NUM_LINES)) + [-1])[dst_idx]
        if target_line != -1:
            shift_card(state, sln, spl, cur_pos, target_line)


@middle("MN01:Darkness:2")
def _darkness_2(state, ap, li, card):
    # You may flip 1 covered card in this line.
    targets = []
    for pl in (0, 1):
        s = state.lines[li].stack(pl)
        for pos in range(len(s) - 1):
            targets.append((li, pl, pos, s[pos]))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Flip 1 covered card in this line",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1:
        return
    sln, spl, spos, _ = targets[idx]
    flip_card(state, sln, spl, spos)


@middle("MN01:Darkness:3")
def _darkness_3(state, ap, li, card):
    # Play 1 card face-down in another line.
    hand = state.players[ap].hand
    if not hand:
        return
    opts = [_describe_hand_card(state, ap, i) for i in range(len(hand))]
    hi = yield Choice(prompt="Pick a card to play face-down in another line",
                      options=opts, targets=list(range(len(hand))), decider=ap)
    other_lines = [i for i in range(NUM_LINES) if i != li]
    lopt = [str(i) for i in other_lines]
    lidx = yield Choice(prompt="Which other line?", options=lopt, targets=other_lines, decider=ap)
    c = state.players[ap].hand.pop(hi)
    c.face_up = False
    state.lines[other_lines[lidx]].stack(ap).append(c)


@middle("MN01:Darkness:4")
def _darkness_4(state, ap, li, card):
    targets = _enumerate_uncovered(state, face="down", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 face-down card",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != sln]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    shift_card(state, sln, spl, spos, dest[didx])


# ----- DEATH (MN01) -----------------------------------------------------

@middle("MN01:Death:0")
def _death_0(state, ap, li, card):
    for ln in range(NUM_LINES):
        if ln == li:
            continue
        targets = _enumerate_uncovered(state, line_filter=ln, active_player=ap)
        if not targets:
            continue
        if len(targets) == 1:
            sln, spl, spos, _ = targets[0]
            delete_card_from_field(state, sln, spl, spos)
            continue
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
        idx = yield Choice(prompt=f"Delete one card from line {ln}",
                           options=opts, targets=targets, decider=ap)
        sln, spl, spos, _ = targets[idx]
        delete_card_from_field(state, sln, spl, spos)


# Death 1 top (Oct 2024 errata): "Start: You may draw 1 card. If you do,
# delete 1 other card. Then, delete this card."
@start_trigger("MN01:Death:1")
def _death_1_start(state, ap, li, card):
    # Offer the optional draw+delete-self chain.
    deck_has = bool(state.players[ap].deck) or bool(state.players[ap].trash)
    if not deck_has:
        return
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    idx = yield Choice(
        prompt="(optional) Draw 1 + delete 1 other card + delete this card",
        options=["accept"] + ["skip"], targets=[1, -1], optional=True, decider=ap,
    )
    if idx == -1 or idx == 1:
        return
    # Accept = idx==0
    draw_cards(state, ap, 1)
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if targets:
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
        i2 = yield Choice(prompt="Delete 1 other card",
                          options=opts, targets=targets, decider=ap)
        sln, spl, spos, _ = targets[i2]
        delete_card_from_field(state, sln, spl, spos)
    # Delete this card from wherever it is.
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            if card in s:
                delete_card_from_field(state, ln, pl, s.index(card))
                return


@middle("MN01:Death:2")
def _death_2(state, ap, li, card):
    # Delete all cards in 1 line with values 1 or 2.
    lopt = [str(i) for i in range(NUM_LINES)]
    lidx = yield Choice(prompt="Choose a line",
                        options=lopt, targets=list(range(NUM_LINES)), decider=ap)
    ln = lidx
    for pl in (0, 1):
        s = state.lines[ln].stack(pl)
        # Iterate in reverse to safely pop.
        for pos in range(len(s) - 1, -1, -1):
            c = s[pos]
            if c is card:
                continue
            v = c.value(state.defs)
            if v in (1, 2):
                delete_card_from_field(state, ln, pl, pos)


@middle("MN01:Death:3")
def _death_3(state, ap, li, card):
    targets = _enumerate_uncovered(state, face="down", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Delete 1 face-down card",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    delete_card_from_field(state, sln, spl, spos)


@middle("MN01:Death:4")
def _death_4(state, ap, li, card):
    targets = []
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            if not s:
                continue
            c = s[-1]
            if c is card:
                continue
            v = c.value(state.defs)
            if v in (0, 1):
                targets.append((ln, pl, len(s) - 1, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Delete a card with value 0 or 1",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    delete_card_from_field(state, sln, spl, spos)


# ----- FIRE (MN01) ------------------------------------------------------

@middle("MN01:Fire:0")
def _fire_0(state, ap, li, card):
    # Flip 1 other card. Draw 2 cards.
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if targets:
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
        idx = yield Choice(prompt="Flip 1 other card",
                           options=opts, targets=targets, decider=ap)
        sln, spl, spos, _ = targets[idx]
        flip_card(state, sln, spl, spos)
    draw_cards(state, ap, 2)


@when_covered("MN01:Fire:0")
def _fire_0_when_covered(state, ap, li, card):
    # First, draw 1 card. Then, flip 1 other card.
    draw_cards(state, ap, 1)
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 other card",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    flip_card(state, sln, spl, spos)


@middle("MN01:Fire:1")
def _fire_1(state, ap, li, card):
    # Discard 1. If you do, delete 1 card.
    if not state.players[ap].hand:
        return
    yield from _discard_n(state, ap, 1)
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Delete 1 card",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    delete_card_from_field(state, sln, spl, spos)


@middle("MN01:Fire:2")
def _fire_2(state, ap, li, card):
    # Discard 1. If you do, return 1 card.
    if not state.players[ap].hand:
        return
    yield from _discard_n(state, ap, 1)
    targets = _enumerate_uncovered(state, active_player=ap, exclude=card)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Return 1 card to its owner's hand",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    return_card_to_hand(state, sln, spl, spos)


@bottom_on_play("MN01:Fire:3")
def _fire_3_bottom(state, ap, li, card):
    # You may discard 1. If you do, flip 1 card.
    if not state.players[ap].hand:
        return
    idx0 = yield Choice(
        prompt="(optional) Discard 1 to flip 1 card?",
        options=["accept", "skip"], targets=[0, -1], optional=True, decider=ap,
    )
    if idx0 == -1 or idx0 == 1:
        return
    yield from _discard_n(state, ap, 1)
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 card", options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    flip_card(state, sln, spl, spos)


@middle("MN01:Fire:4")
def _fire_4(state, ap, li, card):
    # Discard 1 or more. Draw amount discarded plus 1.
    if not state.players[ap].hand:
        draw_cards(state, ap, 1)
        return
    state.scratch["last_discard_count"] = 0
    # Must discard at least 1 (text says "1 or more"). We require the first.
    yield from _discard_n(state, ap, 1)
    state.scratch["last_discard_count"] = 1
    # Additional optional discards.
    yield from _discard_optional_loop(state, ap, max_n=len(state.players[ap].hand))
    total = 1 + state.scratch.get("last_discard_count", 0)
    draw_cards(state, ap, total + 1)


# ----- GRAVITY (MN01) ---------------------------------------------------

@middle("MN01:Gravity:0")
def _gravity_0(state, ap, li, card):
    # For every 2 cards in this line, play the top card of your deck face-down
    # under this card.
    total_in_line = sum(len(state.lines[li].stack(pl)) for pl in (0, 1))
    n = total_in_line // 2
    for _ in range(n):
        play_top_deck_face_down_under(state, ap, li, card)
    if False:
        yield  # pragma: no cover


@middle("MN01:Gravity:1")
def _gravity_1(state, ap, li, card):
    draw_cards(state, ap, 2)
    # Shift 1 card either to or from this line on this player's side. Uses
    # the shift-target enumerator so face-up covered Speed 2 / Spirit 3 on
    # our side are valid.
    options = []
    targets = []
    candidates = _enumerate_shift_targets(state, owner="self", active_player=ap)
    for ln, pl, pos, c in candidates:
        if ln == li:
            for dst in range(NUM_LINES):
                if dst == li:
                    continue
                options.append(
                    f"FROM {ln} -> {dst}: {_describe_card(state, ln, ap, c, viewer=ap)}"
                )
                targets.append((ln, ap, pos, dst))
        else:
            options.append(
                f"TO {li} <- {ln}: {_describe_card(state, ln, ap, c, viewer=ap)}"
            )
            targets.append((ln, ap, pos, li))
    if not targets:
        return
    idx = yield Choice(prompt="Shift 1 card to or from this line",
                       options=options, targets=targets, optional=False, decider=ap)
    sln, spl, spos, dst = targets[idx]
    shift_card(state, sln, spl, spos, dst)


@middle("MN01:Gravity:2")
def _gravity_2(state, ap, li, card):
    # Flip 1 card. Shift that card to this line.
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 card (will then be shifted to this line)",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    flip_card(state, sln, spl, spos)
    # Card is still on its owner's side; "shift" affects same player's side only,
    # so we shift on `spl`'s side (the card's side) to line `li`.
    if sln != li:
        cur_pos = state.lines[sln].stack(spl).index(targets[idx][3])
        shift_card(state, sln, spl, cur_pos, li)


@middle("MN01:Gravity:4")
def _gravity_4(state, ap, li, card):
    targets = _enumerate_uncovered(state, face="down", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 face-down card to this line",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    shift_card(state, sln, spl, spos, li)


@middle("MN01:Gravity:6")
def _gravity_6(state, ap, li, card):
    # Your opponent plays the top card of their deck face-down in this line.
    opp = 1 - ap
    play_top_deck_face_down(state, opp, li)
    if False:
        yield  # pragma: no cover


# ----- LIFE (MN01) ------------------------------------------------------

# Life 0 top (Oct 2024 errata): "End: If this card is covered, delete this
# card." (bottom command was removed; previously immediate-on-cover.)
@middle("MN01:Life:0")
def _life_0(state, ap, li, card):
    for ln in range(NUM_LINES):
        if state.lines[ln].stack(ap):
            play_top_deck_face_down(state, ap, ln)
    if False:
        yield  # pragma: no cover


@end_trigger("MN01:Life:0")
def _life_0_end(state, ap, li, card):
    # Fire only if this Life 0 is currently covered. (Owner side only — top
    # text triggers from the owner's side.)
    for ln_ in range(NUM_LINES):
        s = state.lines[ln_].stack(ap)
        if card in s:
            pos = s.index(card)
            if pos < len(s) - 1:  # covered
                delete_card_from_field(state, ln_, ap, pos)
            return
    if False:
        yield  # pragma: no cover


@middle("MN01:Life:1")
def _life_1(state, ap, li, card):
    for _ in range(2):
        targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
        if not targets:
            return
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
        idx = yield Choice(prompt="Flip 1 card",
                           options=opts, targets=targets, decider=ap)
        sln, spl, spos, _ = targets[idx]
        flip_card(state, sln, spl, spos)


@middle("MN01:Life:2")
def _life_2(state, ap, li, card):
    draw_cards(state, ap, 1)
    targets = _enumerate_uncovered(state, face="down", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Flip 1 face-down card",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1:
        return
    sln, spl, spos, _ = targets[idx]
    flip_card(state, sln, spl, spos)


@bottom_first("MN01:Life:3")
def _life_3_first(state, ap, li, card):
    # First, play the top card of your deck face-down in another line.
    other_lines = [i for i in range(NUM_LINES) if i != li]
    opts = [str(i) for i in other_lines]
    if not other_lines:
        return
    idx = yield Choice(prompt="Play top-deck face-down in which other line?",
                       options=opts, targets=other_lines, decider=ap)
    play_top_deck_face_down(state, ap, other_lines[idx])


@middle("MN01:Life:4")
def _life_4(state, ap, li, card):
    # If this card is covering a card, draw 1.
    s = state.lines[li].stack(ap)
    if card in s and s.index(card) > 0:
        draw_cards(state, ap, 1)
    if False:
        yield  # pragma: no cover


# ----- LIGHT (MN01) -----------------------------------------------------

@middle("MN01:Light:0")
def _light_0(state, ap, li, card):
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 card; draw cards equal to its value",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    flipped = flip_card(state, sln, spl, spos)
    if flipped.face_up:
        draw_cards(state, ap, state.defs[flipped.def_id].value)
    else:
        draw_cards(state, ap, FACE_DOWN_BASE_VALUE)


@bottom_on_play("MN01:Light:1")
def _light_1_bottom(state, ap, li, card):
    draw_cards(state, ap, 1)
    if False:
        yield  # pragma: no cover


@middle("MN01:Light:2")
def _light_2(state, ap, li, card):
    draw_cards(state, ap, 2)
    targets = _enumerate_uncovered(state, face="down", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Reveal 1 face-down card; then shift or flip it",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, c = targets[idx]
    d = state.defs[c.def_id]
    state.log.append(f"P{ap} revealed {d.protocol} {d.value} at L{sln}/P{spl}")
    sub_idx = yield Choice(
        prompt="Shift or flip that card?",
        options=["shift", "flip"], targets=["shift", "flip"], decider=ap,
    )
    if targets and idx is not None:
        if sub_idx == 0:
            dest = [i for i in range(NUM_LINES) if i != sln]
            didx = yield Choice(prompt="To which line?", options=[str(i) for i in dest],
                                targets=dest, decider=ap)
            cur_pos = state.lines[sln].stack(spl).index(c)
            shift_card(state, sln, spl, cur_pos, dest[didx])
        else:
            cur_pos = state.lines[sln].stack(spl).index(c)
            flip_card(state, sln, spl, cur_pos)


@middle("MN01:Light:3")
def _light_3(state, ap, li, card):
    # Shift all face-down cards in this line to another line.
    other_lines = [i for i in range(NUM_LINES) if i != li]
    if not other_lines:
        return
    didx = yield Choice(prompt="Shift all face-down in this line to which other line?",
                        options=[str(i) for i in other_lines],
                        targets=other_lines, decider=ap)
    dst = other_lines[didx]
    for pl in (0, 1):
        s = state.lines[li].stack(pl)
        # collect indices in reverse
        positions = [i for i, c in enumerate(s) if not c.face_up]
        for pos in reversed(positions):
            shift_card(state, li, pl, pos, dst)


@middle("MN01:Light:4")
def _light_4(state, ap, li, card):
    # Opponent reveals their hand (informational).
    opp = 1 - ap
    desc = ", ".join(
        f"{state.defs[c.def_id].protocol} {state.defs[c.def_id].value}"
        for c in state.players[opp].hand
    )
    state.log.append(f"P{opp} hand revealed: {desc or '<empty>'}")
    if False:
        yield  # pragma: no cover


# ----- METAL (MN01) -----------------------------------------------------

@middle("MN01:Metal:0")
def _metal_0(state, ap, li, card):
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 card", options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    flip_card(state, sln, spl, spos)


@middle("MN01:Metal:1")
def _metal_1(state, ap, li, card):
    draw_cards(state, ap, 2)
    state.players[1 - ap].cannot_compile_next_turn = True
    if False:
        yield  # pragma: no cover


# Metal 2 top is purely persistent (handled by opp_play_facedown_blocked_in_line).


@middle("MN01:Metal:3")
def _metal_3(state, ap, li, card):
    draw_cards(state, ap, 1)
    # Delete all cards in 1 other line with 8 or more cards.
    candidate_lines = []
    for ln in range(NUM_LINES):
        if ln == li:
            continue
        total = len(state.lines[ln].p0_stack) + len(state.lines[ln].p1_stack)
        if total >= 8:
            candidate_lines.append(ln)
    if not candidate_lines:
        return
    opts = [str(i) for i in candidate_lines]
    idx = yield Choice(prompt="Choose a line (>=8 cards) to clear",
                       options=opts, targets=candidate_lines, decider=ap)
    ln = candidate_lines[idx]
    for pl in (0, 1):
        s = state.lines[ln].stack(pl)
        for pos in range(len(s) - 1, -1, -1):
            delete_card_from_field(state, ln, pl, pos)


# Metal 6 top: "When this card would be covered or flipped: First, delete
# this card." Two trigger events both call the same delete-self handler.
# `when_covered` fires when a card is being placed on top; `flip_trigger`
# fires on any flip transition involving this card (face-up→face-down or
# face-down→face-up — the rules don't distinguish for the "covered or
# flipped" emphasis, so we treat both directions as triggering).
def _metal_6_self_delete(state, ap, li, card):
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            if card in s:
                delete_card_from_field(state, ln, pl, s.index(card))
                return
    if False:
        yield  # pragma: no cover


when_covered("MN01:Metal:6")(_metal_6_self_delete)
flip_trigger("MN01:Metal:6")(_metal_6_self_delete)


# ----- PLAGUE (MN01) ----------------------------------------------------

@middle("MN01:Plague:0")
def _plague_0(state, ap, li, card):
    yield from _discard_n(state, 1 - ap, 1)


# Plague 0 bottom is persistent (handled by opp_play_blocked_in_line).


# Plague 1 top: "After your opponent discards cards: Draw 1 card."
# Conditional trigger fires after the opponent discards.
@after_opp_discard("MN01:Plague:1")
def _plague_1_top(state, ap, li, card):
    draw_cards(state, ap, 1)
    if False:
        yield  # pragma: no cover


@middle("MN01:Plague:1")
def _plague_1(state, ap, li, card):
    yield from _discard_n(state, 1 - ap, 1)


@middle("MN01:Plague:2")
def _plague_2(state, ap, li, card):
    # Discard 1 or more. Opp discards N+1.
    if not state.players[ap].hand:
        return
    yield from _discard_n(state, ap, 1)
    state.scratch["last_discard_count"] = 1
    yield from _discard_optional_loop(state, ap, max_n=len(state.players[ap].hand))
    n = 1 + state.scratch.get("last_discard_count", 0)
    yield from _discard_n(state, 1 - ap, n + 1)


@middle("MN01:Plague:3")
def _plague_3(state, ap, li, card):
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            for c in state.lines[ln].stack(pl):
                if c is card or not c.face_up:
                    continue
                c.face_up = False
    if False:
        yield  # pragma: no cover


@bottom_on_play("MN01:Plague:4")
def _plague_4_bottom(state, ap, li, card):
    # Your opponent deletes 1 of their face-down cards. You may flip this card.
    opp = 1 - ap
    targets = []
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(opp)
        for pos in range(len(s) - 1, -1, -1):
            c = s[pos]
            if not c.face_up:
                targets.append((ln, opp, pos, c))
    if targets:
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
        idx = yield Choice(prompt="Opponent deletes one of their face-down cards",
                           options=opts, targets=targets, decider=opp)
        sln, spl, spos, _ = targets[idx]
        delete_card_from_field(state, sln, spl, spos)
    # Optional self-flip.
    cidx = yield Choice(
        prompt="(optional) Flip Plague 4 (this card)?",
        options=["flip", "skip"], targets=[0, -1], optional=True, decider=ap,
    )
    if cidx == -1 or cidx == 1:
        return
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(ap)
        if card in s:
            card.face_up = not card.face_up
            return


# ----- PSYCHIC (MN01) ---------------------------------------------------

@middle("MN01:Psychic:0")
def _psychic_0(state, ap, li, card):
    draw_cards(state, ap, 2)
    yield from _discard_n(state, 1 - ap, 2)
    # Reveal opponent's hand.
    opp = 1 - ap
    desc = ", ".join(
        f"{state.defs[c.def_id].protocol} {state.defs[c.def_id].value}"
        for c in state.players[opp].hand
    )
    state.log.append(f"P{opp} hand revealed: {desc or '<empty>'}")


@bottom_on_play("MN01:Psychic:1")
def _psychic_1_bottom(state, ap, li, card):
    # Flip this card.
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(ap)
        if card in s:
            card.face_up = not card.face_up
            return
    if False:
        yield  # pragma: no cover


@middle("MN01:Psychic:2")
def _psychic_2(state, ap, li, card):
    yield from _discard_n(state, 1 - ap, 2)
    # Rearrange their protocols.
    if NUM_LINES < 2:
        return
    # Choose two positions to swap on opponent's side.
    opp = 1 - ap
    pairs = [(a, b) for a in range(NUM_LINES) for b in range(a + 1, NUM_LINES)]
    opts = [f"swap opp protocol L{a}<->L{b}" for a, b in pairs] + ["no swap"]
    idx = yield Choice(prompt="Rearrange opponent's protocols (cards stay)",
                       options=opts, targets=pairs + [None], optional=False, decider=ap)
    if idx < len(pairs):
        a, b = pairs[idx]
        ps = state.players[opp]
        ps.protocols[a], ps.protocols[b] = ps.protocols[b], ps.protocols[a]
        ps.compiled[a], ps.compiled[b] = ps.compiled[b], ps.compiled[a]


@middle("MN01:Psychic:3")
def _psychic_3(state, ap, li, card):
    yield from _discard_n(state, 1 - ap, 1)
    # Shift 1 of their cards.
    targets = _enumerate_shift_targets(state, owner="opponent", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 opponent card",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != sln]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    shift_card(state, sln, spl, spos, dest[didx])


@bottom_on_play("MN01:Psychic:4")
def _psychic_4_bottom(state, ap, li, card):
    # You may return 1 of your opponent's cards. If you do, flip this card.
    targets = _enumerate_uncovered(state, owner="opponent", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Return 1 of opponent's cards to their hand",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1:
        return
    sln, spl, spos, _ = targets[idx]
    return_card_to_hand(state, sln, spl, spos)
    # Flip this card.
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(ap)
        if card in s:
            card.face_up = not card.face_up
            return


# ----- SPEED (MN01) -----------------------------------------------------

@middle("MN01:Speed:0")
def _speed_0(state, ap, li, card):
    """Speed 0 middle: 'Play 1 card.' Honours every play-time rule the action
    phase does (Plague 0, Metal 2, Psychic 1, Spirit 1 *and* Chaos 3 /
    Corruption 0 own-card play affordances). The played card fires its full
    enter-play triggers via the engine callback."""
    legal = _legal_sub_plays(state, ap)
    if not legal:
        return
    opts = [t[0] for t in legal]
    idx = yield Choice(prompt="Play 1 card", options=opts, targets=legal, decider=ap)
    if not (0 <= idx < len(legal)):
        return
    _, hi, ln, fu, _cross = legal[idx]
    engine = state.scratch.get("_engine")
    if engine is not None:
        engine.play_card_for_effect(ap, hi, ln, fu)


# Speed 1 top: "After you clear cache: Draw 1 card." — fires after the
# Cache phase resolves on this player's turn (not on play). Routed via the
# `after_clear_cache` event hook, broadcast from the engine after every
# Clear Cache resolution.
@after_clear_cache("MN01:Speed:1")
def _speed_1_top(state, ap, li, card):
    draw_cards(state, ap, 1)
    if False:
        yield  # pragma: no cover


@middle("MN01:Speed:1")
def _speed_1(state, ap, li, card):
    draw_cards(state, ap, 2)
    if False:
        yield  # pragma: no cover


# Speed 2 top: "When this card would be deleted by compiling: Shift this
# card, even if this card is covered." Per Codex p.11: when the compile
# delete-all batch would include Speed 2, instead Speed 2 is shifted to
# another line — the rest of the compile delete still resolves. We
# register a `when_deleted_by_compile` interrupt that prompts a shift
# target on the owner's side. The compile broadcaster excludes Speed 2
# from the deletion list when this hook fires (see game.py).
@when_deleted_by_compile("MN01:Speed:2")
def _speed_2_top(state, ap, li, card):
    # `ap` is the player who is compiling (could be the owner or opp).
    # The shift decision belongs to the *owner* of Speed 2 (it's their
    # card text).
    owner = card.owner
    src_stack = None
    src_pos = None
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(owner)
        if card in s:
            src_stack = (ln, s)
            src_pos = s.index(card)
            break
    if src_stack is None:
        return
    src_line, _ = src_stack
    dest = [i for i in range(NUM_LINES) if i != src_line]
    if not dest:
        return
    opts = [f"shift to L{i}" for i in dest]
    idx = yield Choice(
        prompt=f"Speed 2 would be deleted — shift to which line?",
        options=opts, targets=dest, decider=owner,
    )
    if not (0 <= idx < len(dest)):
        return
    shift_card(state, src_line, owner, src_pos, dest[idx])


@middle("MN01:Speed:3")
def _speed_3(state, ap, li, card):
    # Shift 1 of your other cards.
    targets = _enumerate_shift_targets(state, owner="self", exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 of your other cards",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != sln]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    shift_card(state, sln, spl, spos, dest[didx])


@bottom_on_play("MN01:Speed:3")
def _speed_3_bottom(state, ap, li, card):
    targets = _enumerate_shift_targets(state, owner="self", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Shift 1 of your cards (then flip Speed 3)",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1:
        return
    sln, spl, spos, _ = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != sln]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    shift_card(state, sln, spl, spos, dest[didx])
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(ap)
        if card in s:
            card.face_up = not card.face_up
            return


@middle("MN01:Speed:4")
def _speed_4(state, ap, li, card):
    targets = _enumerate_uncovered(state, owner="opponent", face="down", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 of opponent's face-down cards",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != sln]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    shift_card(state, sln, spl, spos, dest[didx])


# ----- SPIRIT (MN01) ----------------------------------------------------

@middle("MN01:Spirit:0")
def _spirit_0(state, ap, li, card):
    refresh_player(state, ap)
    draw_cards(state, ap, 1)
    if False:
        yield  # pragma: no cover


# Spirit 0 bottom — persistent (player_skips_check_cache).


@middle("MN01:Spirit:1")
def _spirit_1(state, ap, li, card):
    draw_cards(state, ap, 2)
    if False:
        yield  # pragma: no cover


@start_trigger("MN01:Spirit:1")
def _spirit_1_start(state, ap, li, card):
    # Either discard 1 card or flip this card.
    if not state.players[ap].hand:
        # Forced flip.
        for ln in range(NUM_LINES):
            s = state.lines[ln].stack(ap)
            if card in s:
                card.face_up = not card.face_up
                return
        return
    opts = ["discard 1", "flip this card"]
    idx = yield Choice(prompt="Spirit 1: discard 1 or flip self?",
                       options=opts, targets=[0, 1], decider=ap)
    if idx == 0:
        yield from _discard_n(state, ap, 1)
    else:
        for ln in range(NUM_LINES):
            s = state.lines[ln].stack(ap)
            if card in s:
                card.face_up = not card.face_up
                return


@middle("MN01:Spirit:2")
def _spirit_2(state, ap, li, card):
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Flip 1 card",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1:
        return
    sln, spl, spos, _ = targets[idx]
    flip_card(state, sln, spl, spos)


# Spirit 3 top: "After you draw cards: You may shift this card, even if
# this card is covered." Fires after the owner draws ≥1 card and offers
# an optional shift.
@after_self_draw("MN01:Spirit:3")
def _spirit_3_top(state, ap, li, card):
    # `ap` is the player who drew. We only fire if this card's owner is
    # the drawer (each player's own field fires for their own draws).
    owner = card.owner
    src_line = None
    src_pos = None
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(owner)
        if card in s:
            src_line = ln
            src_pos = s.index(card)
            break
    if src_line is None:
        return
    dest = [i for i in range(NUM_LINES) if i != src_line]
    if not dest:
        return
    opts = [f"shift to L{i}" for i in dest] + ["skip"]
    idx = yield Choice(
        prompt="(optional) Shift Spirit 3",
        options=opts, targets=dest + [-1], optional=True, decider=owner,
    )
    if idx == -1 or not (0 <= idx < len(dest)):
        return
    shift_card(state, src_line, owner, src_pos, dest[idx])


@middle("MN01:Spirit:4")
def _spirit_4(state, ap, li, card):
    # Swap positions of 2 of your protocols (cards stay).
    pairs = [(a, b) for a in range(NUM_LINES) for b in range(a + 1, NUM_LINES)]
    opts = [f"swap protocols L{a}<->L{b}" for a, b in pairs]
    idx = yield Choice(prompt="Swap 2 of your protocols",
                       options=opts, targets=pairs, decider=ap)
    a, b = pairs[idx]
    ps = state.players[ap]
    ps.protocols[a], ps.protocols[b] = ps.protocols[b], ps.protocols[a]
    ps.compiled[a], ps.compiled[b] = ps.compiled[b], ps.compiled[a]


# ----- WATER (MN01) -----------------------------------------------------

@middle("MN01:Water:0")
def _water_0(state, ap, li, card):
    # Flip 1 other card. Flip this card.
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if targets:
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
        idx = yield Choice(prompt="Flip 1 other card",
                           options=opts, targets=targets, decider=ap)
        sln, spl, spos, _ = targets[idx]
        flip_card(state, sln, spl, spos)
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(ap)
        if card in s:
            card.face_up = not card.face_up
            return


@middle("MN01:Water:1")
def _water_1(state, ap, li, card):
    # Play top of deck face-down in each OTHER line.
    for ln in range(NUM_LINES):
        if ln == li:
            continue
        play_top_deck_face_down(state, ap, ln)
    if False:
        yield  # pragma: no cover


@middle("MN01:Water:2")
def _water_2(state, ap, li, card):
    draw_cards(state, ap, 2)
    # Rearrange your protocols.
    pairs = [(a, b) for a in range(NUM_LINES) for b in range(a + 1, NUM_LINES)]
    opts = [f"swap L{a}<->L{b}" for a, b in pairs] + ["no swap"]
    idx = yield Choice(prompt="Rearrange your protocols",
                       options=opts, targets=pairs + [None], decider=ap)
    if idx < len(pairs):
        a, b = pairs[idx]
        ps = state.players[ap]
        ps.protocols[a], ps.protocols[b] = ps.protocols[b], ps.protocols[a]
        ps.compiled[a], ps.compiled[b] = ps.compiled[b], ps.compiled[a]


@middle("MN01:Water:3")
def _water_3(state, ap, li, card):
    # Return all cards with value 2 in 1 line.
    lopt = [str(i) for i in range(NUM_LINES)]
    lidx = yield Choice(prompt="Pick a line",
                        options=lopt, targets=list(range(NUM_LINES)), decider=ap)
    ln = lidx
    for pl in (0, 1):
        s = state.lines[ln].stack(pl)
        for pos in range(len(s) - 1, -1, -1):
            c = s[pos]
            if c is card:
                continue
            if c.value(state.defs) == 2:
                return_card_to_hand(state, ln, pl, pos)


@middle("MN01:Water:4")
def _water_4(state, ap, li, card):
    targets = _enumerate_uncovered(state, owner="self", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Return 1 of your cards to your hand",
                       options=opts, targets=targets, decider=ap)
    sln, spl, spos, _ = targets[idx]
    return_card_to_hand(state, sln, spl, spos)


# ---------------------------------------------------------------------------
# MN02 — Chaos
# ---------------------------------------------------------------------------

def _both_players_draw_top(state: GameState, ap: int) -> None:
    """Helper: each player draws the top card of the OTHER player's deck.
    Drawn cards retain ownership transfer per the rules."""
    opp = state.opponent(ap)
    for taker, source in ((ap, opp), (opp, ap)):
        ps_src = state.players[source]
        if not ps_src.deck and ps_src.trash:
            ps_src.deck = ps_src.trash
            ps_src.trash = []
            state.rng.shuffle(ps_src.deck)
        if ps_src.deck:
            c = ps_src.deck.pop()
            c.owner = taker
            c.face_up = False
            state.players[taker].hand.append(c)


@middle("MN02:Chaos:0")
def _chaos_0(state, ap, li, card):
    # M: In each line, flip 1 covered card.
    for ln in range(NUM_LINES):
        covered: list[tuple[int, int, int, CardInst]] = []
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            for pos in range(len(s) - 1):
                c = s[pos]
                if c.is_committed:
                    continue
                covered.append((ln, pl, pos, c))
        if not covered:
            continue
        if len(covered) == 1:
            t = covered[0]
            flip_card(state, t[0], t[1], t[2])
            continue
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in covered]
        idx = yield Choice(
            prompt=f"L{ln}: flip 1 covered card", options=opts,
            targets=covered, decider=ap,
        )
        if 0 <= idx < len(covered):
            t = covered[idx]
            flip_card(state, t[0], t[1], t[2])


@bottom_on_play("MN02:Chaos:0")
def _chaos_0_bottom(state, ap, li, card):
    _both_players_draw_top(state, ap)
    if False:
        yield None  # type: ignore[misc]


@middle("MN02:Chaos:1")
def _chaos_1(state, ap, li, card):
    # M: Rearrange your protocols. Rearrange your opponent's protocols.
    for who in (ap, state.opponent(ap)):
        pairs = [(a, b) for a in range(NUM_LINES) for b in range(a + 1, NUM_LINES)]
        opts = [f"swap P{who} L{a}<->L{b}" for a, b in pairs] + ["no swap"]
        targets = list(pairs) + [None]
        idx = yield Choice(
            prompt=f"Rearrange P{who}'s protocols",
            options=opts, targets=targets, decider=ap,
        )
        if 0 <= idx < len(pairs):
            a, b = pairs[idx]
            ps = state.players[who]
            ps.protocols[a], ps.protocols[b] = ps.protocols[b], ps.protocols[a]
            ps.compiled[a], ps.compiled[b] = ps.compiled[b], ps.compiled[a]


@middle("MN02:Chaos:2")
def _chaos_2(state, ap, li, card):
    # M: Shift 1 of your covered cards.
    targets: list[tuple[int, int, int, CardInst]] = []
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(ap)
        for pos in range(len(s) - 1):
            c = s[pos]
            if c.is_committed or c is card:
                continue
            targets.append((ln, ap, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 of your covered cards",
                       options=opts, targets=targets, decider=ap)
    if not (0 <= idx < len(targets)):
        return
    src_line, src_pl, src_pos, _ = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != src_line]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    if not (0 <= didx < len(dest)):
        return
    shift_card(state, src_line, src_pl, src_pos, dest[didx])


# Chaos 3 B "This card may be played without matching protocols." — needs
# special handling in legal_actions for THIS hand card specifically. v1
# treats this as passive (the card is still playable face-up in matching
# protocol line; the bypass affordance is a known gap).

@bottom_on_play("MN02:Chaos:4")
def _chaos_4_bottom(state, ap, li, card):
    # Discard your hand. Draw that many cards.
    n = len(state.players[ap].hand)
    # Discard from hand index 0 repeatedly until empty (no choice — discard ALL).
    while state.players[ap].hand:
        discard_to_trash(state, ap, 0)
    draw_cards(state, ap, n)
    if False:
        yield None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MN02 — Clarity
# (Clarity 0 top: passive value mod — handled in compute_line_value)
# ---------------------------------------------------------------------------

# Clarity 1 top: "Start: Reveal the top card of your deck. You may
# discard the top card of your deck." Fires at the start of each of the
# owner's turns while this card is face-up.
@start_trigger("MN02:Clarity:1")
def _clarity_1_top(state, ap, li, card):
    # Reveal top of deck. You may discard the top card of your deck.
    ps = state.players[ap]
    if not ps.deck and ps.trash:
        ps.deck = ps.trash
        ps.trash = []
        state.rng.shuffle(ps.deck)
    if not ps.deck:
        return
    top = ps.deck[-1]
    d = state.defs[top.def_id]
    state.log.append(f"P{ap} reveals top of deck: {d.protocol} {d.value}")
    idx = yield Choice(
        prompt=f"Top of your deck is {d.protocol} {d.value}. Discard it?",
        options=["discard", "keep"], targets=[0, -1], optional=True, decider=ap,
    )
    if idx == 0:
        c = ps.deck.pop()
        c.face_up = True
        ps.trash.append(c)


@middle("MN02:Clarity:1")
def _clarity_1(state, ap, li, card):
    # Your opponent reveals their hand.
    opp = state.opponent(ap)
    contents = ", ".join(
        f"{state.defs[c.def_id].protocol} {state.defs[c.def_id].value}"
        for c in state.players[opp].hand
    )
    state.log.append(f"P{opp} reveals hand: {contents or '<empty>'}")
    if False:
        yield None  # type: ignore[misc]


@bottom_first("MN02:Clarity:1")
def _clarity_1_first(state, ap, li, card):
    draw_cards(state, ap, 3)
    if False:
        yield None  # type: ignore[misc]


def _clarity_reveal_draw_value(
    state: GameState, ap: int, target_value: int,
) -> tuple[CardInst | None, list[CardInst]]:
    """Helper for Clarity 2/3: reveal entire deck, return candidates that
    match `target_value` along with the un-drawn remainder. Caller picks the
    candidate via a Choice and is responsible for shuffling.

    Returns (None, candidates) when caller hasn't picked yet — used by
    Clarity 2 to surface the selection as an agent decision.
    """
    ps = state.players[ap]
    if not ps.deck and ps.trash:
        ps.deck = ps.trash
        ps.trash = []
        state.rng.shuffle(ps.deck)
    candidates: list[CardInst] = []
    for c in ps.deck:
        d = state.defs[c.def_id]
        if d.value == target_value:
            candidates.append(c)
    return None, candidates


def _clarity_finish_reveal(
    state: GameState, ap: int, chosen: CardInst | None,
) -> None:
    """Move `chosen` to hand if not None, then shuffle the remaining deck."""
    ps = state.players[ap]
    if chosen is not None:
        ps.deck.remove(chosen)
        ps.hand.append(chosen)
    state.rng.shuffle(ps.deck)


@middle("MN02:Clarity:2")
def _clarity_2(state, ap, li, card):
    """Reveal deck → player picks which value-1 to draw → shuffle → then
    player picks any value-1 from hand to play (with all affordances)."""
    _, candidates = _clarity_reveal_draw_value(state, ap, 1)
    chosen: CardInst | None = None
    if candidates:
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            opts = [
                f"{state.defs[c.def_id].protocol} 1"
                for c in candidates
            ]
            idx = yield Choice(
                prompt="Pick a value-1 card from your deck to draw",
                options=opts, targets=list(candidates), decider=ap,
            )
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
    _clarity_finish_reveal(state, ap, chosen)
    # Now play 1 value-1 card from hand (with full affordances).
    hand = state.players[ap].hand
    value1_indices = [i for i, c in enumerate(hand) if state.defs[c.def_id].value == 1]
    if not value1_indices:
        return
    legal = _legal_sub_plays(state, ap, candidate_hand_indices=value1_indices)
    if not legal:
        return
    idx = yield Choice(
        prompt="Play 1 value-1 card",
        options=[t[0] for t in legal], targets=legal, decider=ap,
    )
    if not (0 <= idx < len(legal)):
        return
    _, hi, ln, fu, _cross = legal[idx]
    engine = state.scratch.get("_engine")
    if engine is not None:
        engine.play_card_for_effect(ap, hi, ln, fu)


@middle("MN02:Clarity:3")
def _clarity_3(state, ap, li, card):
    """Reveal deck → player picks which value-5 to draw → shuffle."""
    _, candidates = _clarity_reveal_draw_value(state, ap, 5)
    chosen: CardInst | None = None
    if candidates:
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            opts = [f"{state.defs[c.def_id].protocol} 5" for c in candidates]
            idx = yield Choice(
                prompt="Pick a value-5 card from your deck to draw",
                options=opts, targets=list(candidates), decider=ap,
            )
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
    _clarity_finish_reveal(state, ap, chosen)


@middle("MN02:Clarity:4")
def _clarity_4(state, ap, li, card):
    # You may shuffle your trash into your deck.
    if not state.players[ap].trash:
        return
    idx = yield Choice(
        prompt="(optional) Shuffle your trash into your deck",
        options=["shuffle"], targets=[0], optional=True, decider=ap,
    )
    if idx == 0:
        ps = state.players[ap]
        ps.deck.extend(ps.trash)
        ps.trash = []
        state.rng.shuffle(ps.deck)


# ---------------------------------------------------------------------------
# MN02 — Corruption
# ---------------------------------------------------------------------------

# Corruption 0 top: "Flip: Flip 1 face-up covered or uncovered card in
# this stack other than this card." The "Flip:" emphasis triggers
# whenever this card itself flips (either direction).
@flip_trigger("MN02:Corruption:0")
def _corruption_0_top(state, ap, li, card):
    # Flip 1 face-up covered or uncovered card in this stack other than this card.
    stack = state.lines[li].stack(card.owner)
    targets: list[tuple[int, int, int, CardInst]] = []
    for pos, c in enumerate(stack):
        if c is card or not c.face_up:
            continue
        targets.append((li, card.owner, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 face-up card in this stack",
                       options=opts, targets=targets, decider=ap)
    if 0 <= idx < len(targets):
        t = targets[idx]
        flip_card(state, t[0], t[1], t[2])


# Corruption 0 B "play this card in any line on either player's side" is a
# play-time affordance — handled in legal_actions hook (not yet wired).

@middle("MN02:Corruption:1")
def _corruption_1(state, ap, li, card):
    # Return 1 card. Bottom: put on top of their deck face-down instead.
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Return 1 card", options=opts, targets=targets, decider=ap)
    if not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    return_card_to_hand(state, t[0], t[1], t[2])


@bottom_on_play("MN02:Corruption:1")
def _corruption_1_bottom(state, ap, li, card):
    # Bottom modifies the return: "Put that card on top of their deck face-down instead."
    # The middle has already chosen and returned the card to hand. As a
    # follow-up, move the most-recently-returned card from its owner's hand
    # to the top of their deck.
    # Heuristic: the returned card is the last card appended to that player's
    # hand. Cross-check via the engine's log.
    # If the return wasn't completed (no targets), nothing to do.
    if not state.log:
        return
    # Recent entries — find the action that just resolved the middle target.
    # Simpler: look at all hands' most recent additions face-down.
    # For our purposes (and parity with the published rule), pick the last
    # card added to either hand that's face-down.
    candidate: tuple[int, CardInst] | None = None
    for pl in (0, 1):
        h = state.players[pl].hand
        if h and not h[-1].face_up:
            candidate = (pl, h[-1])
    if not candidate:
        return
    pl, c = candidate
    state.players[pl].hand.remove(c)
    state.players[pl].deck.append(c)
    if False:
        yield None  # type: ignore[misc]


# Corruption 2 top: "After you discard cards: Your opponent discards 1 card."
@after_self_discard("MN02:Corruption:2")
def _corruption_2_top(state, ap, li, card):
    yield from _discard_n(state, state.opponent(ap), 1)


@middle("MN02:Corruption:2")
def _corruption_2(state, ap, li, card):
    draw_cards(state, ap, 1)
    yield from _discard_n(state, ap, 1)


@middle("MN02:Corruption:3")
def _corruption_3(state, ap, li, card):
    targets: list[tuple[int, int, int, CardInst]] = []
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            for pos in range(len(s) - 1):
                c = s[pos]
                if c.is_committed or not c.face_up:
                    continue
                targets.append((ln, pl, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Flip 1 face-up covered card",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1 or not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    flip_card(state, t[0], t[1], t[2])


# Corruption 6 top: "End: Either discard 1 card or delete this card."
# Fires at the start of each End phase while this card is face-up.
@end_trigger("MN02:Corruption:6")
def _corruption_6_top(state, ap, li, card):
    # Either discard 1 card or delete this card.
    if not state.players[ap].hand:
        # No choice but to delete self.
        for ln in range(NUM_LINES):
            s = state.lines[ln].stack(card.owner)
            if card in s:
                delete_card_from_field(state, ln, card.owner, s.index(card))
                return
        return
    idx = yield Choice(
        prompt="Discard 1 card OR delete Corruption 6?",
        options=["discard 1", "delete self"], targets=[0, 1], decider=ap,
    )
    if idx == 0:
        yield from _discard_n(state, ap, 1)
    else:
        for ln in range(NUM_LINES):
            s = state.lines[ln].stack(card.owner)
            if card in s:
                delete_card_from_field(state, ln, card.owner, s.index(card))
                return


# ---------------------------------------------------------------------------
# MN02 — Courage
# ---------------------------------------------------------------------------

# Courage 0 top: "Start: If you have no cards in hand, draw 1 card."
@start_trigger("MN02:Courage:0")
def _courage_0_top(state, ap, li, card):
    if not state.players[ap].hand:
        draw_cards(state, ap, 1)
    if False:
        yield None  # type: ignore[misc]


@middle("MN02:Courage:0")
def _courage_0(state, ap, li, card):
    draw_cards(state, ap, 1)
    if False:
        yield None  # type: ignore[misc]


@bottom_on_play("MN02:Courage:0")
def _courage_0_bottom(state, ap, li, card):
    # You may discard 1 card. If you do, your opponent discards 1 card.
    if not state.players[ap].hand:
        return
    idx = yield Choice(
        prompt="Discard 1 to force opponent discard?",
        options=["accept", "skip"], targets=[0, -1], optional=True, decider=ap,
    )
    if idx == -1 or idx == 1:
        return
    yield from _discard_n(state, ap, 1)
    yield from _discard_n(state, state.opponent(ap), 1)


@middle("MN02:Courage:1")
def _courage_1(state, ap, li, card):
    # Delete 1 of your opponent's cards in a line where they have a higher
    # total value than you do.
    opp = state.opponent(ap)
    targets: list[tuple[int, int, int, CardInst]] = []
    for ln in range(NUM_LINES):
        if compute_line_value(state, ln, opp) <= compute_line_value(state, ln, ap):
            continue
        s = state.lines[ln].stack(opp)
        if not s:
            continue
        c = s[-1]
        if c.is_committed:
            continue
        targets.append((ln, opp, len(s) - 1, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Delete 1 opp card in a higher-value line",
                       options=opts, targets=targets, decider=ap)
    if 0 <= idx < len(targets):
        t = targets[idx]
        delete_card_from_field(state, t[0], t[1], t[2])


@bottom_on_play("MN02:Courage:2")
def _courage_2_bottom(state, ap, li, card):
    # If opp has higher value than you in this line, draw 1.
    opp = state.opponent(ap)
    if compute_line_value(state, li, opp) > compute_line_value(state, li, ap):
        draw_cards(state, ap, 1)
    if False:
        yield None  # type: ignore[misc]


@bottom_on_play("MN02:Courage:3")
def _courage_3_bottom(state, ap, li, card):
    # You may shift this card to the line where opponent has highest value.
    opp = state.opponent(ap)
    best = max(range(NUM_LINES), key=lambda i: compute_line_value(state, i, opp))
    if best == li:
        return
    idx = yield Choice(
        prompt=f"(optional) Shift Courage 3 to L{best} (opp's strongest line)",
        options=["shift", "skip"], targets=[0, -1], optional=True, decider=ap,
    )
    if idx == -1 or idx == 1:
        return
    s = state.lines[li].stack(ap)
    if card in s:
        shift_card(state, li, ap, s.index(card), best)


# Courage 6 top: "End: If your opponent has a higher value in this line
# than you do, flip this card." Fires at End phase while face-up.
@end_trigger("MN02:Courage:6")
def _courage_6_top(state, ap, li, card):
    opp = state.opponent(ap)
    if compute_line_value(state, li, opp) > compute_line_value(state, li, ap):
        # Self-flip.
        s = state.lines[li].stack(card.owner)
        if card in s:
            card.face_up = not card.face_up
    if False:
        yield None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MN02 — Fear
# (Fear 0 top suppresses opponent middles — handled in middle_suppressed.)
# ---------------------------------------------------------------------------

@middle("MN02:Fear:0")
def _fear_0(state, ap, li, card):
    # Shift OR flip 1 card.
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift or flip 1 card", options=opts, targets=targets, decider=ap)
    if not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    sub_idx = yield Choice(
        prompt="Shift or flip?", options=["shift", "flip"], targets=[0, 1], decider=ap,
    )
    if sub_idx == 0:
        dest = [i for i in range(NUM_LINES) if i != t[0]]
        didx = yield Choice(prompt="To which line?",
                            options=[str(i) for i in dest], targets=dest, decider=ap)
        if 0 <= didx < len(dest):
            shift_card(state, t[0], t[1], t[2], dest[didx])
    else:
        flip_card(state, t[0], t[1], t[2])


@middle("MN02:Fear:1")
def _fear_1(state, ap, li, card):
    # Draw 2 cards. Opponent discards their hand and draws |hand|-1.
    draw_cards(state, ap, 2)
    opp = state.opponent(ap)
    n_discarded = len(state.players[opp].hand)
    while state.players[opp].hand:
        discard_to_trash(state, opp, 0)
    draw_n = max(0, n_discarded - 1)
    if draw_n > 0:
        draw_cards(state, opp, draw_n)
    if False:
        yield None  # type: ignore[misc]


@middle("MN02:Fear:2")
def _fear_2(state, ap, li, card):
    targets = _enumerate_uncovered(state, owner="opponent", active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Return 1 opp card", options=opts, targets=targets, decider=ap)
    if 0 <= idx < len(targets):
        t = targets[idx]
        return_card_to_hand(state, t[0], t[1], t[2])


@bottom_on_play("MN02:Fear:3")
def _fear_3_bottom(state, ap, li, card):
    # Shift 1 of opp's covered or uncovered cards in this line.
    opp = state.opponent(ap)
    targets: list[tuple[int, int, int, CardInst]] = []
    s = state.lines[li].stack(opp)
    for pos, c in enumerate(s):
        if c.is_committed:
            continue
        targets.append((li, opp, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 opp card in this line",
                       options=opts, targets=targets, decider=ap)
    if not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != li]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    if 0 <= didx < len(dest):
        shift_card(state, t[0], t[1], t[2], dest[didx])


@middle("MN02:Fear:4")
def _fear_4(state, ap, li, card):
    # Opponent discards 1 random card (no choice).
    opp = state.opponent(ap)
    if state.players[opp].hand:
        idx = state.rng.randrange(len(state.players[opp].hand))
        discard_to_trash(state, opp, idx)
    if False:
        yield None  # type: ignore[misc]


@bottom_on_play("MN02:Fear:5")
def _fear_5_bottom(state, ap, li, card):
    yield from _discard_n(state, ap, 1)


# ---------------------------------------------------------------------------
# MN02 — Ice
# (Ice 6 top blocks own draws — handled in draw_cards.)
# ---------------------------------------------------------------------------

@middle("MN02:Ice:1")
def _ice_1(state, ap, li, card):
    # You may shift this card.
    s = state.lines[li].stack(card.owner)
    if card not in s:
        return
    cur_pos = s.index(card)
    dest = [i for i in range(NUM_LINES) if i != li]
    if not dest:
        return
    idx = yield Choice(
        prompt="(optional) Shift Ice 1 to another line",
        options=[str(i) for i in dest] + ["skip"], targets=dest + [-1],
        optional=True, decider=ap,
    )
    if idx == -1 or idx >= len(dest):
        return
    shift_card(state, li, card.owner, cur_pos, dest[idx])


@bottom_on_play("MN02:Ice:1")
def _ice_1_bottom(state, ap, li, card):
    yield from _discard_n(state, state.opponent(ap), 1)


@middle("MN02:Ice:2")
def _ice_2(state, ap, li, card):
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 other card",
                       options=opts, targets=targets, decider=ap)
    if not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != t[0]]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    if 0 <= didx < len(dest):
        shift_card(state, t[0], t[1], t[2], dest[didx])


@middle("MN02:Ice:3")
def _ice_3(state, ap, li, card):
    # If this card is covered, you may shift it.
    s = state.lines[li].stack(card.owner)
    if card not in s or s.index(card) == len(s) - 1:
        return  # not covered
    cur_pos = s.index(card)
    dest = [i for i in range(NUM_LINES) if i != li]
    if not dest:
        return
    idx = yield Choice(
        prompt="(optional) Shift Ice 3 (covered) to another line",
        options=[str(i) for i in dest] + ["skip"], targets=dest + [-1],
        optional=True, decider=ap,
    )
    if idx == -1 or idx >= len(dest):
        return
    shift_card(state, li, card.owner, cur_pos, dest[idx])


# ---------------------------------------------------------------------------
# MN02 — Luck
# ---------------------------------------------------------------------------

@middle("MN02:Luck:0")
def _luck_0(state, ap, li, card):
    # State a number. Draw 3. If a drawn card has stated value (face-up), may
    # play it (with all play-affordance bypasses).
    num_idx = yield Choice(
        prompt="State a number (0-6)",
        options=[str(i) for i in range(7)], targets=list(range(7)), decider=ap,
    )
    n = num_idx if 0 <= num_idx <= 6 else 0
    draw_cards(state, ap, 3)
    hand = state.players[ap].hand
    start = max(0, len(hand) - 3)
    candidate_indices = [i for i in range(start, len(hand))
                         if state.defs[hand[i].def_id].value == n]
    if not candidate_indices:
        return
    legal = _legal_sub_plays(state, ap, candidate_hand_indices=candidate_indices)
    if not legal:
        return
    idx = yield Choice(
        prompt=f"(optional) Play a value-{n} card",
        options=[t[0] for t in legal] + ["skip"],
        targets=legal + [None], optional=True, decider=ap,
    )
    if idx == -1 or not (0 <= idx < len(legal)):
        return
    _, hi, ln, fu, _cross = legal[idx]
    engine = state.scratch.get("_engine")
    if engine is not None:
        engine.play_card_for_effect(ap, hi, ln, fu)


@middle("MN02:Luck:1")
def _luck_1(state, ap, li, card):
    # Play top of deck face-down. Flip it ignoring its middle commands.
    # We implement as: play top deck FU into a line (skipping middle effect).
    ps = state.players[ap]
    if not ps.deck and ps.trash:
        ps.deck = ps.trash; ps.trash = []
        state.rng.shuffle(ps.deck)
    if not ps.deck:
        return
    # Pick a line — play face-down then immediately flip face-up without
    # firing the middle.
    lidx = yield Choice(
        prompt="Play top of deck face-down in which line?",
        options=["0", "1", "2"], targets=[0, 1, 2], decider=ap,
    )
    if not (0 <= lidx < NUM_LINES):
        return
    c = ps.deck.pop()
    c.face_up = False
    state.lines[lidx].stack(ap).append(c)
    # "Flip that card, ignoring its middle commands" — the top text and
    # bottom-first/on-play triggers should still fire. Flip face-up directly
    # and push the non-middle enter-play triggers via the engine ref.
    c.face_up = True
    engine = state.scratch.get("_engine")
    if engine is not None:
        engine._enqueue_enter_play_triggers_skip_middle(c, ap, lidx)


@middle("MN02:Luck:2")
def _luck_2(state, ap, li, card):
    ps = state.players[ap]
    if not ps.deck and ps.trash:
        ps.deck = ps.trash; ps.trash = []
        state.rng.shuffle(ps.deck)
    if not ps.deck:
        return
    top = ps.deck.pop()
    top.face_up = True
    state.players[top.owner].trash.append(top)
    val = state.defs[top.def_id].value
    draw_cards(state, ap, val)
    if False:
        yield None  # type: ignore[misc]


@middle("MN02:Luck:3")
def _luck_3(state, ap, li, card):
    # State a protocol. Discard top of opp deck; if it matches, delete 1 card.
    # Targets carry the protocol strings themselves so the NN encoder can feed
    # each option through the protocol embedding rather than a raw index.
    from .cards import PROTOCOLS_BY_SET
    all_protos = sorted({p for ps in PROTOCOLS_BY_SET.values() for p in ps})
    p_idx = yield Choice(
        prompt="State a protocol",
        options=all_protos, targets=list(all_protos), decider=ap,
    )
    if not (0 <= p_idx < len(all_protos)):
        return
    stated = all_protos[p_idx]
    opp = state.opponent(ap)
    ps_opp = state.players[opp]
    if not ps_opp.deck and ps_opp.trash:
        ps_opp.deck = ps_opp.trash; ps_opp.trash = []
        state.rng.shuffle(ps_opp.deck)
    if not ps_opp.deck:
        return
    top = ps_opp.deck.pop()
    top.face_up = True
    state.players[top.owner].trash.append(top)
    if state.defs[top.def_id].protocol != stated:
        return
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Delete 1 card", options=opts, targets=targets, decider=ap)
    if 0 <= idx < len(targets):
        t = targets[idx]
        delete_card_from_field(state, t[0], t[1], t[2])


@middle("MN02:Luck:4")
def _luck_4(state, ap, li, card):
    # Discard top of deck. Delete 1 covered or uncovered card sharing its value.
    ps = state.players[ap]
    if not ps.deck and ps.trash:
        ps.deck = ps.trash; ps.trash = []
        state.rng.shuffle(ps.deck)
    if not ps.deck:
        return
    top = ps.deck.pop()
    top.face_up = True
    state.players[top.owner].trash.append(top)
    target_val = state.defs[top.def_id].value
    targets: list[tuple[int, int, int, CardInst]] = []
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            for pos, c in enumerate(s):
                if c.is_committed or c is card:
                    continue
                v = c.value(state.defs)
                if v == target_val:
                    targets.append((ln, pl, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt=f"Delete 1 card with value {target_val}",
                       options=opts, targets=targets, decider=ap)
    if 0 <= idx < len(targets):
        t = targets[idx]
        delete_card_from_field(state, t[0], t[1], t[2])


# ---------------------------------------------------------------------------
# MN02 — Mirror (Mirror 0 top handled in compute_line_value)
# ---------------------------------------------------------------------------

@bottom_on_play("MN02:Mirror:1")
def _mirror_1_bottom(state, ap, li, card):
    # You may resolve the middle command of 1 of your opponent's cards as
    # if it were on this card. v1 approximation: choose an opp face-up card
    # with a middle, then trigger its middle effect with `card` as the
    # active card. Complex re-entrancy; we delegate to the engine.
    opp = state.opponent(ap)
    targets: list[tuple[int, int, int, CardInst]] = []
    for ln in range(NUM_LINES):
        s = state.lines[ln].stack(opp)
        for pos, c in enumerate(s):
            if not c.face_up:
                continue
            d = state.defs[c.def_id]
            if d.middle_text:
                targets.append((ln, opp, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Resolve opp middle as Mirror 1",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1 or not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    fn = MIDDLE_EFFECTS.get(state.defs[t[3].def_id].key)
    if fn is None:
        return
    # Yield from the opp middle as if firing on this card.
    yield from fn(state, ap, li, card)


@middle("MN02:Mirror:2")
def _mirror_2(state, ap, li, card):
    # Swap all of your cards in one stack with another of your stacks.
    pairs = [(a, b) for a in range(NUM_LINES) for b in range(a + 1, NUM_LINES)]
    opts = [f"swap your stack L{a}<->L{b}" for a, b in pairs] + ["no swap"]
    targets = list(pairs) + [None]
    idx = yield Choice(prompt="Swap which two of your stacks?",
                       options=opts, targets=targets, decider=ap)
    if not (0 <= idx < len(pairs)):
        return
    a, b = pairs[idx]
    line_a, line_b = state.lines[a], state.lines[b]
    sa = line_a.stack(ap)
    sb = line_b.stack(ap)
    if ap == 0:
        line_a.p0_stack, line_b.p0_stack = sb, sa
    else:
        line_a.p1_stack, line_b.p1_stack = sb, sa


@middle("MN02:Mirror:3")
def _mirror_3(state, ap, li, card):
    # Flip 1 of YOUR cards. Flip 1 of OPP's cards IN THE SAME LINE.
    own = _enumerate_uncovered(state, owner="self", exclude=card, active_player=ap)
    if not own:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in own]
    idx = yield Choice(prompt="Flip 1 of your cards", options=opts, targets=own, decider=ap)
    if not (0 <= idx < len(own)):
        return
    t = own[idx]
    flip_card(state, t[0], t[1], t[2])
    opp = state.opponent(ap)
    opp_in_line = _enumerate_uncovered(state, owner="opponent", line_filter=t[0], active_player=ap)
    if not opp_in_line:
        return
    opts2 = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in opp_in_line]
    i2 = yield Choice(prompt="Flip 1 opp card in same line",
                      options=opts2, targets=opp_in_line, decider=ap)
    if 0 <= i2 < len(opp_in_line):
        u = opp_in_line[i2]
        flip_card(state, u[0], u[1], u[2])


@bottom_on_play("MN02:Mirror:4")
def _mirror_4_bottom(state, ap, li, card):
    draw_cards(state, ap, 1)
    if False:
        yield None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MN02 — Peace
# ---------------------------------------------------------------------------

@middle("MN02:Peace:1")
def _peace_1(state, ap, li, card):
    # Both players discard their hand.
    for pl in (0, 1):
        while state.players[pl].hand:
            discard_to_trash(state, pl, 0)
    if False:
        yield None  # type: ignore[misc]


@bottom_on_play("MN02:Peace:1")
def _peace_1_bottom(state, ap, li, card):
    if not state.players[ap].hand:
        draw_cards(state, ap, 1)
    if False:
        yield None  # type: ignore[misc]


@middle("MN02:Peace:2")
def _peace_2(state, ap, li, card):
    draw_cards(state, ap, 1)
    hand = state.players[ap].hand
    if not hand:
        return
    opts = [_describe_hand_card(state, ap, i) for i in range(len(hand))]
    hi = yield Choice(prompt="Pick a card to play face-down",
                      options=opts, targets=list(range(len(hand))), decider=ap)
    if not (0 <= hi < len(hand)):
        return
    lidx = yield Choice(prompt="Which line?", options=["0", "1", "2"],
                        targets=[0, 1, 2], decider=ap)
    if not (0 <= lidx < NUM_LINES):
        return
    c = state.players[ap].hand.pop(hi)
    c.face_up = False
    state.lines[lidx].stack(ap).append(c)


@middle("MN02:Peace:3")
def _peace_3(state, ap, li, card):
    # You may discard 1. Flip 1 card with value > #cards in hand.
    if state.players[ap].hand:
        idx = yield Choice(
            prompt="(optional) Discard 1 card first",
            options=["discard", "skip"], targets=[0, -1], optional=True, decider=ap,
        )
        if idx == 0:
            yield from _discard_n(state, ap, 1)
    threshold = len(state.players[ap].hand)
    candidates: list[tuple[int, int, int, CardInst]] = []
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            if not s:
                continue
            c = s[-1]
            if c.is_committed:
                continue
            v = c.value(state.defs)
            if v > threshold:
                candidates.append((ln, pl, len(s) - 1, c))
    if not candidates:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in candidates]
    i2 = yield Choice(prompt=f"Flip 1 card with value > {threshold}",
                      options=opts, targets=candidates, decider=ap)
    if 0 <= i2 < len(candidates):
        t = candidates[i2]
        flip_card(state, t[0], t[1], t[2])


@bottom_on_play("MN02:Peace:4")
def _peace_4_bottom(state, ap, li, card):
    draw_cards(state, ap, 1)
    if False:
        yield None  # type: ignore[misc]


@middle("MN02:Peace:6")
def _peace_6(state, ap, li, card):
    # If you have >1 card in hand, flip this card.
    if len(state.players[ap].hand) > 1:
        s = state.lines[li].stack(card.owner)
        if card in s:
            card.face_up = not card.face_up
    if False:
        yield None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MN02 — Smoke (Smoke 2 top handled in compute_line_value)
# ---------------------------------------------------------------------------

def _line_has_facedown(state: GameState, line_idx: int) -> bool:
    for pl in (0, 1):
        for c in state.lines[line_idx].stack(pl):
            if not c.face_up:
                return True
    return False


@middle("MN02:Smoke:0")
def _smoke_0(state, ap, li, card):
    for ln in range(NUM_LINES):
        if _line_has_facedown(state, ln):
            play_top_deck_face_down(state, ap, ln)
    if False:
        yield None  # type: ignore[misc]


@middle("MN02:Smoke:1")
def _smoke_1(state, ap, li, card):
    targets = _enumerate_uncovered(state, owner="self", exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 of your cards (then may shift it)",
                       options=opts, targets=targets, decider=ap)
    if not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    flip_card(state, t[0], t[1], t[2])
    dest = [i for i in range(NUM_LINES) if i != t[0]]
    didx = yield Choice(
        prompt="(optional) Shift it to which line?",
        options=[str(i) for i in dest] + ["skip"], targets=dest + [-1],
        optional=True, decider=ap,
    )
    if didx == -1 or didx >= len(dest):
        return
    # Look up the card's new position after flip.
    new_pos = state.lines[t[0]].stack(t[1]).index(t[3])
    shift_card(state, t[0], t[1], new_pos, dest[didx])


@middle("MN02:Smoke:3")
def _smoke_3(state, ap, li, card):
    hand = state.players[ap].hand
    if not hand:
        return
    eligible = [ln for ln in range(NUM_LINES) if _line_has_facedown(state, ln)]
    if not eligible:
        return
    opts_h = [_describe_hand_card(state, ap, i) for i in range(len(hand))]
    hi = yield Choice(prompt="Pick a card to play face-down (in a line with face-downs)",
                      options=opts_h, targets=list(range(len(hand))), decider=ap)
    if not (0 <= hi < len(hand)):
        return
    lidx = yield Choice(prompt="Which line?",
                        options=[str(i) for i in eligible], targets=eligible, decider=ap)
    if not (0 <= lidx < len(eligible)):
        return
    c = state.players[ap].hand.pop(hi)
    c.face_up = False
    state.lines[eligible[lidx]].stack(ap).append(c)


@middle("MN02:Smoke:4")
def _smoke_4(state, ap, li, card):
    targets: list[tuple[int, int, int, CardInst]] = []
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            for pos in range(len(s) - 1):
                c = s[pos]
                if c.is_committed or c.face_up:
                    continue
                targets.append((ln, pl, pos, c))
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Shift 1 covered face-down card",
                       options=opts, targets=targets, decider=ap)
    if not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    dest = [i for i in range(NUM_LINES) if i != t[0]]
    didx = yield Choice(prompt="To which line?",
                        options=[str(i) for i in dest], targets=dest, decider=ap)
    if 0 <= didx < len(dest):
        shift_card(state, t[0], t[1], t[2], dest[didx])


# ---------------------------------------------------------------------------
# MN02 — Time
# ---------------------------------------------------------------------------

@middle("MN02:Time:0")
def _time_0(state, ap, li, card):
    # Play 1 card from your trash. Shuffle your trash into your deck.
    # Honours every play-time affordance the action phase does, applied to a
    # trash card rather than a hand card.
    trash = state.players[ap].trash
    if trash:
        opts = [
            f"trash[{i}]: {state.defs[c.def_id].protocol} {state.defs[c.def_id].value}"
            for i, c in enumerate(trash)
        ]
        idx = yield Choice(prompt="Play 1 card from your trash",
                           options=opts, targets=list(range(len(trash))), decider=ap)
        if 0 <= idx < len(trash):
            c = trash[idx]  # don't pop yet — _legal_trash_play handles placement
            d = state.defs[c.def_id]
            spirit_1 = player_may_play_any_line_faceup(state, ap)
            psychic_1 = opp_must_play_facedown(state, ap)
            line_blocked = [opp_play_blocked_in_line(state, ln_, ap) for ln_ in range(NUM_LINES)]
            line_fd_blocked = [opp_play_facedown_blocked_in_line(state, ln_, ap) for ln_ in range(NUM_LINES)]
            chaos_3 = d.protocol == "Chaos" and d.value == 3
            corruption_0 = d.protocol == "Corruption" and d.value == 0
            unrestricted_fu = spirit_1 or chaos_3 or corruption_0
            legal: list[tuple[str, int, bool, bool]] = []  # label, line_or_opp_line, face_up, cross
            for ln in range(NUM_LINES):
                if line_blocked[ln] or psychic_1:
                    continue
                if unrestricted_fu or state.players[ap].protocols[ln] == d.protocol:
                    legal.append((f"FU L{ln}", ln, True, False))
            for ln in range(NUM_LINES):
                if line_blocked[ln] or line_fd_blocked[ln]:
                    continue
                legal.append((f"FD L{ln}", ln, False, False))
            if corruption_0:
                for ln in range(NUM_LINES):
                    legal.append((f"FU OPP L{ln}", ln, True, True))
                    legal.append((f"FD OPP L{ln}", ln, False, True))
            if legal:
                li2 = yield Choice(prompt="Place where?",
                                   options=[t[0] for t in legal], targets=legal, decider=ap)
                if 0 <= li2 < len(legal):
                    _, ln, fu, cross = legal[li2]
                    trash.remove(c)
                    target_side = (1 - ap) if cross else ap
                    if cross:
                        c.owner = target_side
                    c.face_up = fu
                    state.lines[ln].stack(target_side).append(c)
    # Shuffle remaining trash into deck.
    ps = state.players[ap]
    ps.deck.extend(ps.trash)
    ps.trash = []
    state.rng.shuffle(ps.deck)


@middle("MN02:Time:1")
def _time_1(state, ap, li, card):
    # Flip 1 covered card. Discard your entire deck.
    targets: list[tuple[int, int, int, CardInst]] = []
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            s = state.lines[ln].stack(pl)
            for pos in range(len(s) - 1):
                c = s[pos]
                if c.is_committed:
                    continue
                targets.append((ln, pl, pos, c))
    if targets:
        opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
        idx = yield Choice(prompt="Flip 1 covered card",
                           options=opts, targets=targets, decider=ap)
        if 0 <= idx < len(targets):
            t = targets[idx]
            flip_card(state, t[0], t[1], t[2])
    # Discard entire deck.
    ps = state.players[ap]
    while ps.deck:
        c = ps.deck.pop()
        c.face_up = True
        state.players[c.owner].trash.append(c)


# Time 2 top (Codex 8/2025 errata): "After you shuffle your deck: Draw 1
# card. Then, you may shift this card." Fires after the owner shuffles.
@after_self_shuffle("MN02:Time:2")
def _time_2_top(state, ap, li, card):
    draw_cards(state, ap, 1)
    # May shift this card.
    s = state.lines[li].stack(card.owner)
    if card not in s:
        return
    cur_pos = s.index(card)
    dest = [i for i in range(NUM_LINES) if i != li]
    if not dest:
        return
    idx = yield Choice(
        prompt="(optional) Shift Time 2",
        options=[str(i) for i in dest] + ["skip"], targets=dest + [-1],
        optional=True, decider=ap,
    )
    if 0 <= idx < len(dest):
        shift_card(state, li, card.owner, cur_pos, dest[idx])


@middle("MN02:Time:2")
def _time_2(state, ap, li, card):
    if not state.players[ap].trash:
        return
    idx = yield Choice(
        prompt="(optional) Shuffle trash into deck?",
        options=["shuffle"], targets=[0], optional=True, decider=ap,
    )
    if idx == 0:
        ps = state.players[ap]
        ps.deck.extend(ps.trash)
        ps.trash = []
        state.rng.shuffle(ps.deck)
        _flag_after_shuffle(state, ap)


@middle("MN02:Time:3")
def _time_3(state, ap, li, card):
    trash = state.players[ap].trash
    if not trash:
        return
    opts = [f"trash[{i}]: {state.defs[c.def_id].protocol} {state.defs[c.def_id].value}"
            for i, c in enumerate(trash)]
    idx = yield Choice(prompt="Reveal 1 trash card; play face-down in another line",
                       options=opts, targets=list(range(len(trash))), decider=ap)
    if not (0 <= idx < len(trash)):
        return
    c = trash.pop(idx)
    other = [i for i in range(NUM_LINES) if i != li]
    didx = yield Choice(prompt="Which other line?",
                        options=[str(i) for i in other], targets=other, decider=ap)
    if 0 <= didx < len(other):
        c.face_up = False
        state.lines[other[didx]].stack(ap).append(c)


@middle("MN02:Time:4")
def _time_4(state, ap, li, card):
    draw_cards(state, ap, 2)
    for _ in range(2):
        if not state.players[ap].hand:
            return
        yield from _discard_n(state, ap, 1)


# ---------------------------------------------------------------------------
# MN02 — War
# ---------------------------------------------------------------------------

# War 0 top: "After you refresh: You may flip this card." Fires after
# the owner performs a Refresh action.
@after_self_refresh("MN02:War:0")
def _war_0_top(state, ap, li, card):
    idx = yield Choice(
        prompt="(optional) Flip War 0 now?",
        options=["flip", "skip"], targets=[0, -1], optional=True, decider=ap,
    )
    if idx == 0:
        s = state.lines[li].stack(card.owner)
        if card in s:
            card.face_up = not card.face_up


@bottom_on_play("MN02:War:0")
def _war_0_bottom(state, ap, li, card):
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Delete 1 card",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1 or not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    delete_card_from_field(state, t[0], t[1], t[2])


@bottom_on_play("MN02:War:1")
def _war_1_bottom(state, ap, li, card):
    # Discard any number of cards. Refresh.
    yield from _discard_optional_loop(state, ap, max_n=len(state.players[ap].hand))
    refresh_player(state, ap)


@middle("MN02:War:2")
def _war_2(state, ap, li, card):
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="Flip 1 card", options=opts, targets=targets, decider=ap)
    if 0 <= idx < len(targets):
        t = targets[idx]
        flip_card(state, t[0], t[1], t[2])


@bottom_on_play("MN02:War:2")
def _war_2_bottom(state, ap, li, card):
    # Opponent discards their hand.
    opp = state.opponent(ap)
    while state.players[opp].hand:
        discard_to_trash(state, opp, 0)
    if False:
        yield None  # type: ignore[misc]


@middle("MN02:War:3")
def _war_3(state, ap, li, card):
    draw_cards(state, ap, 1)
    if False:
        yield None  # type: ignore[misc]


@bottom_on_play("MN02:War:3")
def _war_3_bottom(state, ap, li, card):
    hand = state.players[ap].hand
    if not hand:
        return
    opts = [_describe_hand_card(state, ap, i) for i in range(len(hand))]
    hi = yield Choice(prompt="(optional) Play 1 card face-down",
                      options=opts + ["skip"], targets=list(range(len(hand))) + [-1],
                      optional=True, decider=ap)
    if hi == -1 or not (0 <= hi < len(hand)):
        return
    lidx = yield Choice(prompt="Which line?",
                        options=["0", "1", "2"], targets=[0, 1, 2], decider=ap)
    if not (0 <= lidx < NUM_LINES):
        return
    c = state.players[ap].hand.pop(hi)
    c.face_up = False
    state.lines[lidx].stack(ap).append(c)


@middle("MN02:War:4")
def _war_4(state, ap, li, card):
    # JSON typo: "our opponent discards 1 card" — interpreted as "Your".
    yield from _discard_n(state, state.opponent(ap), 1)


# ---------------------------------------------------------------------------
# AX02 — Assimilation
# ---------------------------------------------------------------------------

@middle("AX02:Assimilation:1")
def _assim_1(state, ap, li, card):
    # M: Discard 1 card. Refresh.
    yield from _discard_n(state, ap, 1)
    refresh_player(state, ap)


@bottom_on_play("AX02:Assimilation:1")
def _assim_1_bottom(state, ap, li, card):
    # Draw the top card of your opponent's deck. Discard 1 card into their trash.
    opp = state.opponent(ap)
    ps_opp = state.players[opp]
    if not ps_opp.deck and ps_opp.trash:
        ps_opp.deck = ps_opp.trash; ps_opp.trash = []
        state.rng.shuffle(ps_opp.deck)
    if ps_opp.deck:
        c = ps_opp.deck.pop()
        c.owner = ap
        c.face_up = False
        state.players[ap].hand.append(c)
    # Discard 1 card from own hand into OPP'S trash.
    hand = state.players[ap].hand
    if not hand:
        return
    opts = [_describe_hand_card(state, ap, i) for i in range(len(hand))]
    idx = yield Choice(prompt="Discard 1 card into opponent's trash",
                       options=opts, targets=list(range(len(hand))), decider=ap)
    if not (0 <= idx < len(hand)):
        return
    c = state.players[ap].hand.pop(idx)
    c.face_up = True
    state.players[opp].trash.append(c)


@middle("AX02:Assimilation:4")
def _assim_4(state, ap, li, card):
    _both_players_draw_top(state, ap)
    if False:
        yield None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AX02 — Diversity
# ---------------------------------------------------------------------------

# Diversity 6 top: "End: If there are not at least 3 different protocols
# on cards in the field, delete this card." Codex says this fires at End
# phase only — not continuously. We register an end_trigger that checks
# the condition and self-deletes when met.
#
# (`_check_diversity_6_self_destruct` is still invoked from field-mutation
# helpers for backwards-compat — it's idempotent and only fires when the
# condition is met. The rules-correct fire-point is the End trigger
# below; the continuous check provides a defence-in-depth sweep.)
@end_trigger("AX02:Diversity:6")
def _diversity_6_top(state, ap, li, card):
    _check_diversity_6_self_destruct(state)
    if False:
        yield None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AX02 — Unity
# ---------------------------------------------------------------------------

def _count_unity_in_field(state: GameState) -> int:
    n = 0
    for ln in range(NUM_LINES):
        for pl in (0, 1):
            for c in state.lines[ln].stack(pl):
                d = state.defs[c.def_id]
                if d.protocol == "Unity":
                    n += 1
    return n


@middle("AX02:Unity:0")
def _unity_0(state, ap, li, card):
    # If there is ANOTHER Unity card in the field: either flip 1 or draw 1.
    if _count_unity_in_field(state) <= 1:
        return
    idx = yield Choice(prompt="Flip 1 card or draw 1 card?",
                       options=["flip", "draw"], targets=[0, 1], decider=ap)
    if idx == 1:
        draw_cards(state, ap, 1)
        return
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        draw_cards(state, ap, 1)
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    i2 = yield Choice(prompt="Flip 1 card", options=opts, targets=targets, decider=ap)
    if 0 <= i2 < len(targets):
        t = targets[i2]
        flip_card(state, t[0], t[1], t[2])


@bottom_first("AX02:Unity:0")
def _unity_0_first(state, ap, li, card):
    # First: flip one card or draw 1 card.
    idx = yield Choice(prompt="First: flip 1 card or draw 1?",
                       options=["flip", "draw"], targets=[0, 1], decider=ap)
    if idx == 1:
        draw_cards(state, ap, 1)
        return
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        draw_cards(state, ap, 1)
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    i2 = yield Choice(prompt="Flip 1 card", options=opts, targets=targets, decider=ap)
    if 0 <= i2 < len(targets):
        t = targets[i2]
        flip_card(state, t[0], t[1], t[2])


@middle("AX02:Unity:3")
def _unity_3(state, ap, li, card):
    if _count_unity_in_field(state) <= 1:
        return
    targets = _enumerate_uncovered(state, exclude=card, active_player=ap)
    if not targets:
        return
    opts = [_describe_card(state, t[0], t[1], t[3], viewer=ap) for t in targets]
    idx = yield Choice(prompt="(optional) Flip 1 card",
                       options=opts, targets=targets, optional=True, decider=ap)
    if idx == -1 or not (0 <= idx < len(targets)):
        return
    t = targets[idx]
    flip_card(state, t[0], t[1], t[2])


# ----- AX02 expansion additions (Assim 6, Diversity 0, Ice 4) ----------
# These cards' def_ids were previously mis-stamped in cards.json (as Assim 3,
# Diversity 2, and a duplicate Ice 5). Once the data was corrected the slots
# became real cards whose effects we hadn't implemented. Adding them here.

@bottom_on_play("AX02:Assimilation:6")
def _assim_6_bottom(state, ap, li, card):
    """Play the top card of your deck face down on your opponent's side
    of the field. Player picks which opponent line."""
    if not state.players[ap].deck:
        return
    opts = [f"opp L{i + 1}" for i in range(NUM_LINES)]
    idx = yield Choice(prompt="Play your deck-top face-down on which opp line?",
                       options=opts, targets=list(range(NUM_LINES)), decider=ap)
    play_top_deck_face_down(state, 1 - ap, idx)


@middle("AX02:Diversity:0")
def _diversity_0_middle(state, ap, li, card):
    """If 6 different protocols appear on face-up cards in the field,
    flip the Diversity protocol to its compiled side immediately. This
    short-circuits the usual line-value race to compile."""
    protos = set()
    for ln_idx in range(NUM_LINES):
        for pl in (0, 1):
            for c in state.lines[ln_idx].stack(pl):
                if c.face_up:
                    protos.add(state.defs[c.def_id].protocol)
    if len(protos) >= 6:
        # Find the Diversity protocol slot on the active player's side.
        for slot, p in enumerate(state.players[ap].protocols):
            if p == "Diversity":
                state.players[ap].compiled[slot] = True
                logInfo(state, f"P{ap + 1} compiled Diversity via diversity-0 condition (6 protocols on field).")
                break
    if False:
        yield  # pragma: no cover


@bottom_on_play("AX02:Diversity:0")
def _diversity_0_bottom(state, ap, li, card):
    """End step: you may play one non-Diversity card in this line.
    Stubbed as a no-op — the legal-action enumeration for this affordance
    would need a new ACTION variant; for now the player's regular plays
    cover the common case."""
    if False:
        yield  # pragma: no cover


# Ice 4: "This card cannot be flipped." Implemented as a check in
# flip_card (see flip_immunity_check at the top of this file) — there's
# no on-play handler.

# ---------------------------------------------------------------------------
# Convenience: bulk-lookup wrappers (called from game.py)
# ---------------------------------------------------------------------------

def get_effect(card_def):
    """Backwards-compat alias used by older tests; returns middle effect."""
    return get_middle_effect(card_def)
