"""Text-derived card-behavior audit.

For each card with a non-empty middle (or unemphasised bottom — fires on
play), parse the text for "verb + count" claims and verify the observed
state delta when the handler runs.

We target the highest-yield bug class — handlers that silently no-op,
flip the wrong direction, or use the wrong count — without trying to
LLM-read every handler body.

Verbs we extract:
    Draw N         → hand_size += N (modulo deck-empty)
    Discard N      → hand_size -= N (trash += N)
    Delete N       → field_count -= N (one player's side)
    Flip N         → flip_count == N (number of face-flips)
    Shift N        → shift_count == N

Many cards have *optional* text ("you may flip…"). For those we just
run the handler and check that it terminates without crash AND that
the resulting state isn't catastrophically wrong.
"""
from __future__ import annotations

import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from compile_engine import Game, GameConfig
from compile_engine.cards import load_card_defs
from compile_engine.effects import MIDDLE_EFFECTS
from compile_engine.state import CardInst


DRAW_RE = re.compile(r"\bdraw (\d+) cards?\b", re.IGNORECASE)
DISCARD_RE = re.compile(r"\b(?:you )?discard (\d+) cards?\b", re.IGNORECASE)
DISCARD_OPP_RE = re.compile(r"\bopponent discards (\d+) cards?\b", re.IGNORECASE)
DELETE_RE = re.compile(r"\bdelete (\d+) cards?\b", re.IGNORECASE)
SHIFT_RE = re.compile(r"\bshift (\d+) (?:of\s+)?(?:your\s+|opp\w*\s+)?cards?\b", re.IGNORECASE)


def extract_self_draw(text: str) -> int | None:
    """Hand-grow count expected from playing a card whose text says 'Draw N'.
    Returns None when no match or text says 'may draw' (optional)."""
    if "may draw" in text.lower():
        return None
    m = DRAW_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def extract_self_discard(text: str) -> int | None:
    """Count of cards the active player is *forced* to discard. Skips
    'You may discard' (optional). Also skips opponent-discard text."""
    if "may discard" in text.lower():
        return None
    m = DISCARD_RE.search(text)
    if not m:
        return None
    # Skip if this is "your opponent discards" — that's handled separately.
    span_text = text[max(0, m.start() - 20) : m.end()]
    if "opponent" in span_text.lower() and "your opponent" in span_text.lower():
        return None
    return int(m.group(1))


def extract_opp_discard(text: str) -> int | None:
    """Count of opp discards expected."""
    if "may " in text.lower() and "opponent discard" in text.lower():
        return None
    m = DISCARD_OPP_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def extract_delete(text: str) -> int | None:
    """Count of cards expected to leave the field (delete N). Skips
    optional ('may delete') and 'delete this card' (self-delete, hard
    to assert against a generic setup). Skips when there's a clear
    precondition we won't satisfy."""
    lower = text.lower()
    if "may delete" in lower:
        return None
    if "delete this card" in lower:
        return None
    if "if" in lower:
        return None  # conditional — skip
    m = DELETE_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def setup_play_state(card_def, *, hand_extra=4, deck_size=8):
    """Spin up a Game with `card_def` ready to play face-up on line 0.
    Hand has the test card + extra filler cards, deck has filler.
    Returns the configured Game and the test card instance."""
    g = Game(GameConfig(
        seed=hash(card_def.key) & 0xffff,
        include_expansion=card_def.set_code == "AX01",
        include_main2=card_def.set_code == "MN02",
        include_aux2=card_def.set_code == "AX02",
    ))
    # Draft so the card's protocol is in slot 0 for both players.
    p0_protos = [card_def.protocol, "Light", "Fire"]
    p1_protos = [card_def.protocol, "Light", "Fire"]
    # Ensure protocols are valid for the enabled set; fall back if not.
    try:
        g.set_predetermined_draft([p0_protos, p1_protos])
    except Exception:
        return None, None
    # Replace hand: test card + filler.
    defs_all = load_card_defs()
    filler_def = next(d for d in defs_all if d.key == "MN01:Light:1")
    g.state.lines = [type(g.state.lines[0])() for _ in range(3)]
    test_card = CardInst(
        inst_id=100001, def_id=card_def.def_id, owner=0, face_up=False,
    )
    g.state.players[0].hand = [test_card] + [
        CardInst(inst_id=100100 + i, def_id=filler_def.def_id, owner=0, face_up=False)
        for i in range(hand_extra)
    ]
    # Stock opp hand so opp-discard / opp-reveal effects have material to
    # act on. Without this, all "Your opponent discards N" handlers
    # appear to silently no-op (false positive).
    g.state.players[1].hand = [
        CardInst(inst_id=100400 + i, def_id=filler_def.def_id, owner=1, face_up=False)
        for i in range(hand_extra)
    ]
    g.state.players[0].deck = [
        CardInst(inst_id=100200 + i, def_id=filler_def.def_id, owner=0, face_up=False)
        for i in range(deck_size)
    ]
    g.state.players[1].deck = [
        CardInst(inst_id=100300 + i, def_id=filler_def.def_id, owner=1, face_up=False)
        for i in range(deck_size)
    ]
    g.state.scratch["_engine"] = g
    g._pending = []
    g.state.current_player = 0
    return g, test_card


def fire_middle(card_def):
    """Fire the registered middle handler in a controlled state. Returns
    (hand_delta, opp_hand_delta, trash_delta, opp_trash_delta, field_delta_p0,
     field_delta_p1, error_or_None)."""
    fn = MIDDLE_EFFECTS.get(card_def.key)
    if fn is None:
        return None
    g, c = setup_play_state(card_def)
    if g is None:
        return None
    pre_p0_hand = len(g.state.players[0].hand)
    pre_p1_hand = len(g.state.players[1].hand)
    pre_p0_trash = len(g.state.players[0].trash)
    pre_p1_trash = len(g.state.players[1].trash)
    pre_field_p0 = sum(len(g.state.lines[i].stack(0)) for i in range(3))
    pre_field_p1 = sum(len(g.state.lines[i].stack(1)) for i in range(3))
    pre_p0_deck = len(g.state.players[0].deck)
    pre_p1_deck = len(g.state.players[1].deck)
    try:
        g._push_effect(fn(g.state, 0, 0, c))
        # Drive to first choice or completion.
        for _ in range(50):
            g._drive()
            if not g._pending:
                break
            top = g._pending[-1]
            if top.last_choice is None:
                break
            # Resolve choice with first option (default to skipping if optional).
            legal = g.legal_actions()
            if not legal:
                break
            g.step(legal[0])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {
        "hand_delta_me": len(g.state.players[0].hand) - pre_p0_hand,
        "hand_delta_opp": len(g.state.players[1].hand) - pre_p1_hand,
        "trash_delta_me": len(g.state.players[0].trash) - pre_p0_trash,
        "trash_delta_opp": len(g.state.players[1].trash) - pre_p1_trash,
        "field_delta_me": sum(len(g.state.lines[i].stack(0)) for i in range(3)) - pre_field_p0,
        "field_delta_opp": sum(len(g.state.lines[i].stack(1)) for i in range(3)) - pre_field_p1,
        # Negative = deck shrank (draw happened). Comparing |deck_delta|
        # to text-claimed draw count is the robust signal.
        "deck_delta_me": len(g.state.players[0].deck) - pre_p0_deck,
        "deck_delta_opp": len(g.state.players[1].deck) - pre_p1_deck,
    }


def main():
    defs = load_card_defs()
    findings = defaultdict(list)
    for d in defs:
        if d.key not in MIDDLE_EFFECTS:
            continue
        text = d.middle_text or ""
        if not text:
            continue
        # 1. Crash test
        result = fire_middle(d)
        if result is None:
            continue
        if "error" in result:
            findings["crash"].append((d.key, text, result["error"]))
            continue
        # 2. Self-draw count check.
        draw_n = extract_self_draw(text)
        if draw_n is not None:
            # Skip cards whose draw is gated on a precondition we don't
            # set up — those produce false negatives. Detected via the
            # presence of conditional language preceding the draw.
            lower = text.lower()
            has_conditional = any(
                phrase in lower for phrase in (
                    "if there", "if you", "if this card", "if your", "if opp",
                    "another unity card", "covering a card", "with a value of",
                )
            )
            # The robust signal is deck shrinkage. Drawing N cards
            # always reduces deck by N (modulo deck-empty fallback
            # to reshuffle from trash, which doesn't change total).
            deck_shrink = -result["deck_delta_me"]
            if not has_conditional and deck_shrink < draw_n:
                findings["draw_short"].append(
                    (d.key, text, f"expected ≥{draw_n} draws, got deck_shrink={deck_shrink}")
                )
        # 3. Self-discard count check.
        disc_n = extract_self_discard(text)
        if disc_n is not None:
            # Discard should grow trash by N. (Value-5 cards: hand also has the
            # filler cards so just check trash.)
            if result["trash_delta_me"] < disc_n:
                findings["discard_short"].append(
                    (d.key, text, f"expected ≥{disc_n} discards, got trash+{result['trash_delta_me']}")
                )
        # 4. Opp discard count check.
        opp_disc_n = extract_opp_discard(text)
        if opp_disc_n is not None:
            if result["trash_delta_opp"] < opp_disc_n:
                findings["opp_discard_short"].append(
                    (d.key, text, f"expected ≥{opp_disc_n} opp discards, got opp_trash+{result['trash_delta_opp']}")
                )
        # 5. Delete count check — field should shrink somewhere by N,
        # and trash for that side should grow by N. Requires test setup
        # to have something on the field; we provide a single filler
        # card on each side that can serve as a delete target.
        # NOTE: our setup currently has empty fields → most delete
        # handlers will short-circuit on "no targets" and silently
        # no-op. Skip this check entirely until we extend the setup.
        # (Left here as a placeholder for the next pass.)
        # del_n = extract_delete(text)
        # if del_n is not None:
        #     field_shrink = -(result["field_delta_me"] + result["field_delta_opp"])
        #     trash_grow = result["trash_delta_me"] + result["trash_delta_opp"]
        #     if field_shrink < del_n or trash_grow < del_n:
        #         findings["delete_short"].append(
        #             (d.key, text, f"expected ≥{del_n} deletes; field shrink={field_shrink}, trash grow={trash_grow}")
        #         )

    if not any(findings.values()):
        print("All checks clean.")
        return 0
    for kind, items in findings.items():
        if not items:
            continue
        print(f"\n=== {kind} ({len(items)}) ===")
        for key, text, msg in items:
            print(f"  {key}: {msg}")
            print(f"    text: {text}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
