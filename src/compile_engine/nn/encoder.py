"""Game-state and action tensorization for the policy/value network.

Two responsibilities:
  - `encode_state(game, perspective)` → dict of numpy arrays representing the
    observation from `perspective`'s point of view (side 0 = me, side 1 = opp).
    Faithfully respects imperfect info: opponent hand and opponent face-down
    field cards are encoded as anonymous `HIDDEN` tokens.
  - `encode_actions(game, legal, perspective)` → (action_feats, mask) where
    `action_feats` is [MAX_ACTIONS, action_input_dim] and `mask` is a boolean
    array marking real (vs padded) actions.

The encoder is pure numpy and has no torch dependency; the model layer
converts to tensors. This keeps the rollout workers torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..actions import Action, ActionType
from ..cards import (
    AUX2_PROTOCOLS,
    BASE_PROTOCOLS,
    CARD_STATIC_FEATS_DIM,
    EXPANSION_PROTOCOLS,
    MAIN2_PROTOCOLS,
    load_card_defs,
)
from ..state import NUM_LINES, FACE_DOWN_BASE_VALUE, Phase

if TYPE_CHECKING:
    from ..game import Game
    from ..state import CardInst


# ---------------------------------------------------------------------------
# Vocabularies and shapes (mirrored in docs/nn_design.md).
# ---------------------------------------------------------------------------

NUM_CARDS = 180          # 90 (MN01+AX01) + 90 (MN02+AX02)
NUM_PROTOCOLS = 30       # 12+3+12+3 protocols across all 4 sets
MAX_STACK = 10
MAX_HAND = 12
MAX_ACTIONS = 32

PAD_TOKEN = 0
HIDDEN_TOKEN = 1
_CARD_TOKEN_OFFSET = 2  # def_id 0..89 → token 2..91

CARD_VOCAB_SIZE = NUM_CARDS + _CARD_TOKEN_OFFSET  # 92
PROTO_VOCAB_SIZE = NUM_PROTOCOLS + 1              # 16  (0 = none/unknown)

# Per-card slot raw feature dim (token id + flags + position normalised):
#   [token, face_up, committed, position_norm]
PER_CARD_FEATS = 4

PROTO_LIST = (
    list(BASE_PROTOCOLS) + list(EXPANSION_PROTOCOLS) +
    list(MAIN2_PROTOCOLS) + list(AUX2_PROTOCOLS)
)
_PROTO_TO_ID: dict[str, int] = {p: i + 1 for i, p in enumerate(PROTO_LIST)}

# Phase one-hot length.
_NUM_PHASES = len(Phase)

# ---------------------------------------------------------------------------
# Prompt-category taxonomy (A1: pending-choice context).
# ---------------------------------------------------------------------------
# When the engine yields a `Choice`, the encoder maps the Choice.prompt to a
# small categorical so the network can reason about what kind of sub-decision
# it's being asked to make ("discard 1" vs "flip 1" vs "state a protocol"
# vs "compile-control rearrange") rather than treating every CHOOSE_TARGET
# identically.

PROMPT_CAT_NONE = 0
PROMPT_CAT_DISCARD_OWN = 1
PROMPT_CAT_DISCARD_OPP = 2
PROMPT_CAT_FLIP = 3
PROMPT_CAT_DELETE = 4
PROMPT_CAT_SHIFT = 5
PROMPT_CAT_RETURN_OR_STEAL = 6
PROMPT_CAT_PICK_LINE = 7
PROMPT_CAT_PICK_PROTOCOL = 8
PROMPT_CAT_PICK_NUMBER = 9
PROMPT_CAT_PLAY_SUB_CARD = 10
PROMPT_CAT_OPTIONAL_ACCEPT = 11
PROMPT_CAT_REVEAL = 12
PROMPT_CAT_GIVE = 13
PROMPT_CAT_CONTROL_REARRANGE = 14
PROMPT_CAT_REARRANGE_PROTOS = 15
PROMPT_CAT_OTHER = 16
NUM_PROMPT_CATEGORIES = 17


def _classify_prompt(prompt: str) -> int:
    """Bucket a Choice.prompt into one of the canonical categories above.

    Keyword-matched, order-sensitive: earlier branches win on overlap.
    """
    if not prompt:
        return PROMPT_CAT_NONE
    p = prompt.lower()
    # Control rearrange (added in PR #26) — most specific prefix.
    if "control component" in p or ("rearrange" in p and "whose" in p):
        return PROMPT_CAT_CONTROL_REARRANGE
    if "give" in p:
        return PROMPT_CAT_GIVE
    if "reveal" in p:
        return PROMPT_CAT_REVEAL
    if "state a protocol" in p:
        return PROMPT_CAT_PICK_PROTOCOL
    if "state a number" in p:
        return PROMPT_CAT_PICK_NUMBER
    if "rearrange" in p or ("swap" in p and "protocol" in p) or "stack" in p:
        return PROMPT_CAT_REARRANGE_PROTOS
    if "accept" in p or ("optional" in p and "?" in p):
        return PROMPT_CAT_OPTIONAL_ACCEPT
    if "play 1 card" in p or "play a value-" in p:
        return PROMPT_CAT_PLAY_SUB_CARD
    if ("which line" in p or "to which line" in p or "choose a line" in p
            or "pick a line" in p or "select a line" in p):
        return PROMPT_CAT_PICK_LINE
    if "discard" in p and ("opp" in p or "opponent" in p):
        return PROMPT_CAT_DISCARD_OPP
    if "discard" in p:
        return PROMPT_CAT_DISCARD_OWN
    if "flip" in p:
        return PROMPT_CAT_FLIP
    if "delete" in p:
        return PROMPT_CAT_DELETE
    if "shift" in p:
        return PROMPT_CAT_SHIFT
    if "return" in p or "steal" in p:
        return PROMPT_CAT_RETURN_OR_STEAL
    return PROMPT_CAT_OTHER


def _pending_source_card_token(game: "Game", perspective: int) -> int:
    """Identify the card whose generator yielded the current Choice, return
    its encoder token (def_id + offset), or PAD_TOKEN if none.

    We introspect the suspended generator frame's local `card` variable —
    every effect handler in effects.py names its originating card that way.
    System-level generators (control_rearrange_gen, refresh_player,
    compile_finalizer_gen, uncommit_sentinel) lack a `card` local and fall
    through to PAD.
    """
    if not game._pending:
        return PAD_TOKEN
    pending = game._pending[-1]
    if pending.last_choice is None:
        return PAD_TOKEN
    gen = pending.gen
    fr = getattr(gen, "gi_frame", None)
    if fr is None:
        return PAD_TOKEN
    card = fr.f_locals.get("card")
    if card is None or not hasattr(card, "def_id"):
        return PAD_TOKEN
    # Hide opp face-down identity from the perspective player.
    if card.owner != perspective and not card.face_up:
        return HIDDEN_TOKEN
    return card.def_id + _CARD_TOKEN_OFFSET


_PENDING_DEPTH_NORM = 8.0  # ~depth at which the engine caps anyway


# ---------------------------------------------------------------------------
# Shape helpers (used by the model to size its first layer).
# ---------------------------------------------------------------------------

def card_repr_dim(d_card: int) -> int:
    """Total per-card representation dim: learned embedding + static features.

    The static features (keyword multi-hot, value one-hot, text-presence flags)
    are looked up from a model buffer at the same time as the learned card
    embedding. Together they form the card's full vector for downstream MLPs.
    """
    return d_card + CARD_STATIC_FEATS_DIM


def state_input_dim(d_card: int = 32, d_proto: int = 16) -> int:
    """Total dense input dim AFTER per-stack aggregation (see model.py)."""
    cdim = card_repr_dim(d_card)
    # Per (side, line) stack we aggregate (mean + max) of card_repr ⊕ per-slot
    # flags (face_up, committed, position_norm = 3), plus 3 meta scalars
    # (fu_count, fd_count, depth_norm).
    per_stack = 2 * (cdim + 3) + 3
    field = 3 * 2 * per_stack
    # Protocols: 2 sides * 3 lines * (d_proto + 1 compiled flag)
    protocols = 2 * 3 * (d_proto + 1)
    # Hand: aggregated (mean + max) of card representations + size_norm
    hand = 2 * cdim + 1
    # Trash: 2 sides * NUM_CARDS counts (normalised)
    trash = 2 * NUM_CARDS
    # Per-side per-line precomputed values
    line_vals = 3 * 2
    # Scalars: turn_norm, control, expansion, cannot_compile_me,
    # cannot_compile_opp, is_my_turn, opp_hand_size_norm, draft_pick_idx_norm
    # + phase one-hot
    scalars = 8 + _NUM_PHASES
    # Pending-choice context (A1): yielding-card embedding + prompt-category
    # one-hot + pending-stack depth (normalised).
    pending = cdim + NUM_PROMPT_CATEGORIES + 1
    return field + protocols + hand + trash + line_vals + scalars + pending


def action_input_dim(d_card: int = 32, d_proto: int = 16) -> int:
    """Per-action raw feature dim (input to the action MLP)."""
    n_action_types = len(ActionType)
    return (
        n_action_types        # action-type one-hot
        + 1                   # hand_index normalised
        + 4                   # src line one-hot (+ none)
        + 4                   # dst line one-hot (+ none)
        + 1                   # choice_index normalised
        + card_repr_dim(d_card)  # learned card emb + static card features
        + d_proto             # protocol embedding (raw — emb lookup)
        + 8                   # target meta one-hot (includes _TGT_PROTOCOL)
        + 1                   # stated_value_norm (for state-a-number)
        + 2                   # A5: soon-covered (present, face-up) flags
        + card_repr_dim(d_card)  # A5: soon-covered card embedding
        + ACTION_LOOKAHEAD_DIM   # A3: closed-form delta features
    )


# A3: closed-form lookahead delta features per action — see
# `_compute_action_lookahead` below.
ACTION_LOOKAHEAD_DIM = 7


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def _card_token(c: "CardInst", *, perspective: int, defs) -> int:
    """Token to use for `c` in encoder vocab. `HIDDEN` for opponent face-down."""
    if c.owner != perspective and not c.face_up:
        return HIDDEN_TOKEN
    return c.def_id + _CARD_TOKEN_OFFSET


def encode_state(game: "Game", perspective: int) -> dict[str, np.ndarray]:
    """Encode game state from `perspective`'s POV. Returns numpy arrays.

    The model loads these into tensors and does embedding lookups. Keeping
    the encoder torch-free makes rollouts cheap to multi-process.
    """
    st = game.state
    me = perspective
    opp = 1 - me
    defs = game.defs

    # field[line, side, slot] = (token_id, face_up, committed, position_norm)
    field = np.zeros((NUM_LINES, 2, MAX_STACK, PER_CARD_FEATS), dtype=np.int64)
    # We split: token_ids go to int64; flags + position can also live in same
    # array since they fit. We'll extract floats inside the model.
    # face_up_count / face_down_count / depth_norm per (line, side):
    field_meta = np.zeros((NUM_LINES, 2, 3), dtype=np.float32)

    for ln in range(NUM_LINES):
        for ps_idx, pl in ((0, me), (1, opp)):
            stack = st.lines[ln].stack(pl)
            fu_count = 0
            fd_count = 0
            for pos, c in enumerate(stack[:MAX_STACK]):
                tok = _card_token(c, perspective=me, defs=defs)
                fu = 1 if c.face_up else 0
                comm = 1 if c.is_committed else 0
                pos_norm = pos / max(1, MAX_STACK - 1)
                field[ln, ps_idx, pos] = (tok, fu, comm, int(pos_norm * 1000))
                if c.face_up:
                    fu_count += 1
                else:
                    fd_count += 1
            depth = min(len(stack), MAX_STACK)
            field_meta[ln, ps_idx] = (fu_count, fd_count, depth / MAX_STACK)

    # Protocols: [2 sides, 3 lines, (proto_id, compiled)]
    protocols = np.zeros((2, NUM_LINES, 2), dtype=np.int64)
    for ps_idx, pl in ((0, me), (1, opp)):
        ps = st.players[pl]
        for ln in range(NUM_LINES):
            if ln < len(ps.protocols):
                protocols[ps_idx, ln, 0] = _PROTO_TO_ID.get(ps.protocols[ln], 0)
                protocols[ps_idx, ln, 1] = int(ps.compiled[ln])

    # Hand: perspective's hand as card tokens.
    hand_tokens = np.zeros(MAX_HAND, dtype=np.int64)
    my_hand = st.players[me].hand
    for i, c in enumerate(my_hand[:MAX_HAND]):
        hand_tokens[i] = c.def_id + _CARD_TOKEN_OFFSET
    hand_size = float(len(my_hand)) / MAX_HAND

    # Opp hand size (count only — content is hidden):
    opp_hand_size = float(len(st.players[opp].hand)) / MAX_HAND

    # Trash counts.
    trash = np.zeros((2, NUM_CARDS), dtype=np.float32)
    for ps_idx, pl in ((0, me), (1, opp)):
        for c in st.players[pl].trash:
            if 0 <= c.def_id < NUM_CARDS:
                trash[ps_idx, c.def_id] += 1
    trash /= 18.0  # rough normalisation: each player has 18 cards total

    # Line values
    line_vals = np.zeros((NUM_LINES, 2), dtype=np.float32)
    for ln in range(NUM_LINES):
        for ps_idx, pl in ((0, me), (1, opp)):
            from ..effects import compute_line_value
            line_vals[ln, ps_idx] = compute_line_value(st, ln, pl) / 20.0

    # Game scalars
    ctrl = st.control_holder
    ctrl_flag = 0.0 if ctrl is None else (1.0 if ctrl == me else -1.0)
    # Draft pick index: where we are in the 6-pick snake. 0.0 at the first
    # pick, 1.0 once the draft is fully resolved. Visible during play so the
    # network can still distinguish 'I picked first' vs 'I picked last'
    # if it ever finds that useful for board interpretation.
    n_draft_picks = max(1, len(game._draft_schedule))
    draft_pick_idx_norm = min(game._draft_idx, n_draft_picks) / n_draft_picks
    scalars = np.array(
        [
            st.turn / float(st.config.max_turns),
            ctrl_flag,
            float(int(st.config.include_expansion)),
            float(int(st.players[me].cannot_compile_next_turn)),
            float(int(st.players[opp].cannot_compile_next_turn)),
            float(int(game.decider() == me)),
            opp_hand_size,
            float(draft_pick_idx_norm),
        ],
        dtype=np.float32,
    )
    phase_one_hot = np.zeros(_NUM_PHASES, dtype=np.float32)
    phase_one_hot[st.phase.value - 1] = 1.0

    # Pending-choice context (A1). When the engine is in mid-effect-chain
    # and asking the agent for a sub-decision, surface (a) which card
    # initiated the chain — so the agent can disambiguate "Mirror 1 asked
    # me to copy an opp middle" from "Plague 0 forced opp to discard" —
    # (b) what category of sub-decision is being asked, and (c) how deep
    # the pending stack is (signals that we're inside a longer chain like
    # Speed 0 sub-play).
    pending_card_token = _pending_source_card_token(game, me)
    pending_category = PROMPT_CAT_NONE
    pending_depth = 0
    if game._pending and game._pending[-1].last_choice is not None:
        pending_category = _classify_prompt(game._pending[-1].last_choice.prompt)
        pending_depth = len(game._pending)
    pending_category_one_hot = np.zeros(NUM_PROMPT_CATEGORIES, dtype=np.float32)
    pending_category_one_hot[pending_category] = 1.0
    pending_depth_norm = np.array(
        [min(pending_depth, _PENDING_DEPTH_NORM) / _PENDING_DEPTH_NORM],
        dtype=np.float32,
    )

    # Position normalisation: stored as int (×1000) above to share the
    # int64 array with the token id. Convert back to float in the output.
    field_flags_f = field[..., 1:].astype(np.float32)
    field_flags_f[..., 2] /= 1000.0  # position_norm back to [0, 1]

    return {
        "field_tokens": field[..., 0].astype(np.int64),         # [3, 2, MAX_STACK]
        "field_flags": field_flags_f,                            # [3, 2, MAX_STACK, 3]
        "field_meta": field_meta,                                # [3, 2, 3]
        "protocols": protocols,                                   # [2, 3, 2]
        "hand_tokens": hand_tokens,                              # [MAX_HAND]
        "hand_size": np.array([hand_size], dtype=np.float32),
        "trash": trash,                                          # [2, NUM_CARDS]
        "line_vals": line_vals,                                  # [3, 2]
        "scalars": scalars,                                      # [8]
        "phase": phase_one_hot,                                  # [_NUM_PHASES]
        # A1: pending-choice context.
        "pending_card_token": np.array(pending_card_token, dtype=np.int64),
        "pending_category": pending_category_one_hot,            # [NUM_PROMPT_CATEGORIES]
        "pending_depth_norm": pending_depth_norm,                # [1]
    }


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

# Target-meta codes for CHOOSE_TARGET (when a Choice prompted a sub-decision).
_TGT_NONE = 0
_TGT_FIELD_CARD = 1
_TGT_HAND_CARD = 2
_TGT_LINE = 3
_TGT_INT = 4
_TGT_STR = 5
_TGT_SENTINEL = 6
_TGT_PROTOCOL = 7   # for "state a protocol" / draft-style protocol picks


def _classify_target(target, prompt: str = "") -> tuple[int, int | None, int | None, int | None]:
    """For a Choice.target, return (target_meta_code, card_def_id, line_idx, proto_id).

    `prompt` is the Choice's prompt string — used to disambiguate ambiguous
    int targets (e.g. "State a number" vs "Pick a line"). Provides the
    agent with structured signal for the new MN02 choices (state-a-number,
    state-a-protocol, swap-pair).
    """
    # Field card tuple from _enumerate_uncovered / _enumerate_all etc.
    if isinstance(target, tuple) and len(target) >= 4:
        c = target[3]
        if hasattr(c, "def_id"):
            return _TGT_FIELD_CARD, c.def_id, target[0], None
    # String target: most likely a protocol name (e.g. Luck 3's "state a protocol").
    if isinstance(target, str):
        if target in _PROTO_TO_ID:
            return _TGT_PROTOCOL, None, None, _PROTO_TO_ID[target]
        return _TGT_STR, None, None, None
    # Integer target:
    if isinstance(target, int):
        if target == -1:
            return _TGT_SENTINEL, None, None, None
        # If the prompt mentions a protocol, the int is a protocol-list index.
        if "protocol" in prompt.lower():
            return _TGT_PROTOCOL, None, None, target + 1  # +1 to dodge PAD=0
        # If the prompt is about lines, mark TGT_LINE.
        if "line" in prompt.lower() and 0 <= target <= 2:
            return _TGT_LINE, None, target, None
        return _TGT_INT, None, None, None
    if target is None:
        return _TGT_NONE, None, None, None
    return _TGT_NONE, None, None, None


def _action_card_def_id(game: "Game", a: Action, perspective: int) -> int | None:
    """For actions that imply a specific card, return its def_id (or None)."""
    st = game.state
    if a.type in (ActionType.PLAY_FACE_UP, ActionType.PLAY_FACE_DOWN, ActionType.DISCARD_CARD):
        if 0 <= a.hand_index < len(st.players[perspective].hand):
            return st.players[perspective].hand[a.hand_index].def_id
    if a.type is ActionType.SHIFT_OWN_CARD:
        stack = st.lines[a.line_index].stack(perspective)
        if 0 <= a.hand_index < len(stack):
            return stack[a.hand_index].def_id
    if a.type is ActionType.CHOOSE_TARGET and game._pending and game._pending[-1].last_choice is not None:
        choice = game._pending[-1].last_choice
        targets = choice.targets
        if 0 <= a.choice_index < len(targets):
            meta, def_id, _, _ = _classify_target(targets[a.choice_index], choice.prompt)
            if def_id is not None:
                return def_id
    return None


def _action_target_meta(game: "Game", a: Action) -> tuple[int, int | None, int | None]:
    """Return (meta_code, line_idx, proto_id) for a CHOOSE_TARGET action."""
    if a.type is not ActionType.CHOOSE_TARGET:
        return _TGT_NONE, None, None
    if not game._pending or game._pending[-1].last_choice is None:
        return _TGT_NONE, None, None
    choice = game._pending[-1].last_choice
    targets = choice.targets
    if 0 <= a.choice_index < len(targets):
        meta, _, line_idx, proto_id = _classify_target(targets[a.choice_index], choice.prompt)
        return meta, line_idx, proto_id
    return _TGT_NONE, None, None


_NUM_ACTION_TYPES = len(ActionType)


# Cards with a `when_covered` handler — used by A3 lookahead to flag PLAY
# actions that would land on top of such a card. Populated lazily on first
# use so we don't import effects.py at module load time (avoids circular
# import with the engine).
_WHEN_COVERED_KEYS: set[str] | None = None


def _when_covered_keys() -> set[str]:
    global _WHEN_COVERED_KEYS
    if _WHEN_COVERED_KEYS is None:
        from ..effects import WHEN_COVERED_EFFECTS
        _WHEN_COVERED_KEYS = set(WHEN_COVERED_EFFECTS.keys())
    return _WHEN_COVERED_KEYS


def _soon_covered_for_play(game, a: Action, perspective: int) -> tuple[int, int, int]:
    """For a PLAY_FACE_UP / PLAY_FACE_DOWN action, return
    (soon_covered_token, soon_covered_face_up_flag, triggers_under_when_covered).

    If the destination stack is empty, returns (PAD, 0, 0).
    """
    if a.type not in (ActionType.PLAY_FACE_UP, ActionType.PLAY_FACE_DOWN):
        return PAD_TOKEN, 0, 0
    st = game.state
    # Corruption 0's cross-side play uses line_index in [3, 6). Resolve to
    # opp's actual line + side.
    line_idx = a.line_index
    target_side = perspective
    NUM = NUM_LINES
    # We can only tell if it's cross-side by inspecting the card being played.
    ps = st.players[perspective]
    if 0 <= a.hand_index < len(ps.hand):
        c = ps.hand[a.hand_index]
        d = st.defs[c.def_id]
        if (d.protocol == "Corruption" and d.value == 0
                and NUM <= line_idx < 2 * NUM):
            target_side = 1 - perspective
            line_idx = line_idx - NUM
    if not (0 <= line_idx < NUM):
        return PAD_TOKEN, 0, 0
    stack = st.lines[line_idx].stack(target_side)
    if not stack:
        return PAD_TOKEN, 0, 0
    top = stack[-1]
    token = _card_token(top, perspective=perspective, defs=st.defs)
    fu_flag = 1 if top.face_up else 0
    when_cov = 0
    if top.face_up:
        key = st.defs[top.def_id].key
        if key in _when_covered_keys():
            when_cov = 1
    return token, fu_flag, when_cov


def _compute_action_lookahead(
    game, a: Action, perspective: int,
) -> np.ndarray:
    """A3: closed-form predicted state deltas for `a`. No engine clone —
    purely analytic. Returns a [ACTION_LOOKAHEAD_DIM] float vector:

        [would_compile, triggers_under_when_covered,
         card_value_added_norm, dest_line_diff_post_norm,
         hand_delta_me_norm, control_resets, is_pure_target]

    Cascading effects (middles, when_covered chains, sub-plays) are NOT
    simulated — those residuals are what the value head learns. The point
    here is to give the agent a cheap "naive next-state preview" so it
    doesn't have to derive trivial deltas (compile threshold, hand
    bookkeeping, control consumption) from scratch.
    """
    from ..effects import compute_line_value
    feats = np.zeros(ACTION_LOOKAHEAD_DIM, dtype=np.float32)
    st = game.state
    me = perspective
    opp = 1 - me

    is_pure_target = a.type is ActionType.CHOOSE_TARGET
    feats[6] = 1.0 if is_pure_target else 0.0
    if is_pure_target:
        # Closed-form prediction is unreliable here — the choice's effect
        # depends entirely on which card asked. Leave deltas at zero.
        return feats

    # --- card_value_added_norm + dest line preview ---
    if a.type in (ActionType.PLAY_FACE_UP, ActionType.PLAY_FACE_DOWN):
        ps = st.players[me]
        if not (0 <= a.hand_index < len(ps.hand)):
            return feats
        c = ps.hand[a.hand_index]
        d = st.defs[c.def_id]
        face_up = a.type is ActionType.PLAY_FACE_UP
        # Resolve cross-side via Corruption 0.
        line_idx = a.line_index
        target_side = me
        if (d.protocol == "Corruption" and d.value == 0
                and NUM_LINES <= line_idx < 2 * NUM_LINES):
            target_side = opp
            line_idx -= NUM_LINES
        if not (0 <= line_idx < NUM_LINES):
            return feats
        value_added = d.value if face_up else FACE_DOWN_BASE_VALUE
        feats[2] = value_added / 12.0
        # Predicted post-line difference (us minus them) at dest line.
        cur_us = compute_line_value(st, line_idx, target_side)
        cur_them = compute_line_value(st, line_idx, 1 - target_side)
        # When target_side == me, value adds to "us"; otherwise to "them".
        if target_side == me:
            post_diff = (cur_us + value_added) - cur_them
        else:
            post_diff = cur_us - (cur_them + value_added)
        feats[3] = max(min(post_diff / 20.0, 1.0), -1.0)
        # would_compile: only valid for face-up plays on our side.
        if face_up and target_side == me:
            from ..effects import player_can_compile
            post_us = cur_us + value_added
            if post_us >= 10 and post_us > cur_them and player_can_compile(st, me):
                feats[0] = 1.0
        # hand_delta_me_norm: PLAY consumes 1 from hand.
        feats[4] = -1.0 / 5.0
        # triggers_under_when_covered: from A5 helper.
        _, _, when_cov = _soon_covered_for_play(game, a, perspective)
        feats[1] = float(when_cov)
        return feats

    if a.type is ActionType.REFRESH:
        need = max(0, 5 - len(st.players[me].hand))
        feats[4] = need / 5.0
        if st.control_holder == me:
            feats[5] = 1.0  # control resets on refresh
        return feats

    if a.type is ActionType.COMPILE_LINE:
        # The compile finalizer wipes BOTH sides of the line. The
        # post-compile dest-line diff is therefore zero.
        feats[3] = 0.0
        if st.control_holder == me:
            feats[5] = 1.0  # control resets on compile
        return feats

    if a.type is ActionType.DISCARD_CARD:
        feats[4] = -1.0 / 5.0  # we discard one from hand
        return feats

    if a.type is ActionType.SHIFT_OWN_CARD:
        # Approximate: source line value drops by the shifted card's
        # current contribution; dest line value rises by the same amount.
        # `hand_index` here is the stack pos; `choice_index` is dst line.
        if 0 <= a.line_index < NUM_LINES:
            stack = st.lines[a.line_index].stack(me)
            if 0 <= a.hand_index < len(stack):
                c = stack[a.hand_index]
                v = (st.defs[c.def_id].value if c.face_up else FACE_DOWN_BASE_VALUE)
                # Add to dst diff, subtract from src — we expose only the
                # dst delta here (simpler one-scalar summary).
                feats[2] = v / 12.0
                if 0 <= a.choice_index < NUM_LINES:
                    dst = a.choice_index
                    cur_us = compute_line_value(st, dst, me)
                    cur_them = compute_line_value(st, dst, 1 - me)
                    post_diff = (cur_us + v) - cur_them
                    feats[3] = max(min(post_diff / 20.0, 1.0), -1.0)
        return feats

    if a.type is ActionType.DRAFT_PROTOCOL:
        # Draft is pre-game; no in-game lookahead applies.
        return feats

    return feats


def encode_actions(
    game: "Game", legal: list[Action], perspective: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode legal actions for the policy head.

    Returns
    -------
    raw_feats           : [MAX_ACTIONS, raw_dim]   — float32 fixed-size features
                                                     (includes A3 lookahead deltas)
    card_ids            : [MAX_ACTIONS]            — int64 primary card token
    proto_ids           : [MAX_ACTIONS]            — int64 protocol id
    extra_card_ids      : [MAX_ACTIONS]            — int64 A5 soon-covered card token
                                                     (PAD for non-PLAY actions)
    mask                : [MAX_ACTIONS]            — bool, True for real actions
    """
    # raw_dim layout:
    #   type_one_hot(N_ATYPES) + hand_idx_norm(1) + src_line(4) + dst_line(4) +
    #   choice_idx_norm(1) + target_meta(8) + stated_value(1) +
    #   A5 soon_covered_present(1) + A5 soon_covered_face_up(1) +
    #   A3 lookahead deltas (ACTION_LOOKAHEAD_DIM).
    raw_dim = _NUM_ACTION_TYPES + 1 + 4 + 4 + 1 + 8 + 1 + 2 + ACTION_LOOKAHEAD_DIM
    raw_feats = np.zeros((MAX_ACTIONS, raw_dim), dtype=np.float32)
    card_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    proto_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    extra_card_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)

    for i, a in enumerate(legal[:MAX_ACTIONS]):
        feat = np.zeros(raw_dim, dtype=np.float32)
        # type one-hot
        feat[a.type.value - 1] = 1.0
        offset = _NUM_ACTION_TYPES
        # hand_index norm
        feat[offset] = (a.hand_index / MAX_HAND) if a.hand_index >= 0 else -1.0
        offset += 1
        # src line one-hot (+ "none" slot)
        if 0 <= a.line_index < 3:
            feat[offset + a.line_index] = 1.0
        else:
            feat[offset + 3] = 1.0
        offset += 4
        # dst line one-hot (used for SHIFT_OWN_CARD where choice_index = dst)
        if a.type is ActionType.SHIFT_OWN_CARD and 0 <= a.choice_index < 3:
            feat[offset + a.choice_index] = 1.0
        else:
            feat[offset + 3] = 1.0
        offset += 4
        # choice_index normalised (for CHOOSE_TARGET)
        feat[offset] = (a.choice_index / MAX_ACTIONS) if a.choice_index >= 0 else -1.0
        offset += 1
        # target meta one-hot (8 slots)
        meta, _line, choice_proto_id = _action_target_meta(game, a)
        feat[offset + meta] = 1.0
        offset += 8
        # stated_value_norm: state-a-number choices encode the int magnitude.
        stated_val = -1.0
        if a.type is ActionType.CHOOSE_TARGET and game._pending and game._pending[-1].last_choice is not None:
            choice = game._pending[-1].last_choice
            if "number" in choice.prompt.lower():
                t = choice.targets[a.choice_index] if 0 <= a.choice_index < len(choice.targets) else None
                if isinstance(t, int) and 0 <= t <= 6:
                    stated_val = t / 6.0
        feat[offset] = stated_val
        offset += 1

        # A5: soon-covered card flags (PLAY actions only). The card emb
        # itself is carried via `extra_card_ids` and looked up in the model.
        sc_token, sc_face_up, _sc_when_cov = _soon_covered_for_play(game, a, perspective)
        feat[offset] = 1.0 if sc_token != PAD_TOKEN else 0.0
        feat[offset + 1] = float(sc_face_up)
        offset += 2

        # A3: closed-form lookahead deltas.
        feat[offset:offset + ACTION_LOOKAHEAD_DIM] = _compute_action_lookahead(
            game, a, perspective,
        )
        offset += ACTION_LOOKAHEAD_DIM

        raw_feats[i] = feat

        # Primary card / protocol embedding lookup ids.
        def_id = _action_card_def_id(game, a, perspective)
        if def_id is not None:
            card_ids[i] = def_id + _CARD_TOKEN_OFFSET
        if a.type is ActionType.DRAFT_PROTOCOL and a.protocol:
            proto_ids[i] = _PROTO_TO_ID.get(a.protocol, 0)
        elif choice_proto_id is not None:
            # State-a-protocol choices feed the protocol embedding too.
            proto_ids[i] = choice_proto_id

        # A5: secondary "soon-covered" card embedding lookup id.
        extra_card_ids[i] = sc_token

        mask[i] = True

    return raw_feats, card_ids, proto_ids, extra_card_ids, mask
