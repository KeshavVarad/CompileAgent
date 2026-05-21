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
    # Per (side, line) stack we aggregate to (mean + max) of card representations
    # plus a few flags: 2 * cdim + 3 (face_up_count, fd_count, depth_norm)
    per_stack = 2 * cdim + 3
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
    return field + protocols + hand + trash + line_vals + scalars


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
        + 6                   # target meta one-hot
    )


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

    return {
        "field_tokens": field[..., 0].astype(np.int64),         # [3, 2, MAX_STACK]
        "field_flags": field[..., 1:].astype(np.float32) / 1.0,  # [3, 2, MAX_STACK, 3]
        "field_meta": field_meta,                                # [3, 2, 3]
        "protocols": protocols,                                   # [2, 3, 2]
        "hand_tokens": hand_tokens,                              # [MAX_HAND]
        "hand_size": np.array([hand_size], dtype=np.float32),
        "trash": trash,                                          # [2, NUM_CARDS]
        "line_vals": line_vals,                                  # [3, 2]
        "scalars": scalars,                                      # [7]
        "phase": phase_one_hot,                                  # [9]
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


def encode_actions(
    game: "Game", legal: list[Action], perspective: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode legal actions for the policy head.

    Returns
    -------
    raw_feats : [MAX_ACTIONS, raw_dim]   — float32 fixed-size features
    card_ids  : [MAX_ACTIONS]            — int64 card token (0 = PAD); model
                                            looks up embedding and concatenates
    proto_ids : [MAX_ACTIONS]            — int64 protocol id (0 = none)
    mask      : [MAX_ACTIONS]            — bool, True for real actions
    """
    n_legal = min(len(legal), MAX_ACTIONS)
    # raw_dim excludes the embedded card/proto parts (model concatenates).
    # Layout: [type_one_hot(N_ATYPES), hand_idx_norm(1), src_line_one_hot(4),
    #          dst_line_one_hot(4), choice_idx_norm(1), target_meta_one_hot(8),
    #          stated_value_norm(1)]
    # The stated_value slot is filled when the choice is a "state a number"
    # (Luck 0) — gives the agent a direct numeric feature to learn over.
    raw_dim = _NUM_ACTION_TYPES + 1 + 4 + 4 + 1 + 8 + 1
    raw_feats = np.zeros((MAX_ACTIONS, raw_dim), dtype=np.float32)
    card_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    proto_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
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
        # target meta one-hot (8 slots, includes _TGT_PROTOCOL)
        meta, _line, choice_proto_id = _action_target_meta(game, a)
        feat[offset + meta] = 1.0
        offset += 8
        # stated_value_norm: when the choice is a state-a-number, encode the
        # actual integer value (0..6) so the agent can reason over magnitude.
        stated_val = -1.0
        if a.type is ActionType.CHOOSE_TARGET and game._pending and game._pending[-1].last_choice is not None:
            choice = game._pending[-1].last_choice
            if "number" in choice.prompt.lower():
                t = choice.targets[a.choice_index] if 0 <= a.choice_index < len(choice.targets) else None
                if isinstance(t, int) and 0 <= t <= 6:
                    stated_val = t / 6.0
        feat[offset] = stated_val
        offset += 1

        raw_feats[i] = feat

        # Card / protocol embedding lookup ids
        def_id = _action_card_def_id(game, a, perspective)
        if def_id is not None:
            card_ids[i] = def_id + _CARD_TOKEN_OFFSET
        if a.type is ActionType.DRAFT_PROTOCOL and a.protocol:
            proto_ids[i] = _PROTO_TO_ID.get(a.protocol, 0)
        elif choice_proto_id is not None:
            # State-a-protocol choices feed the protocol embedding too.
            proto_ids[i] = choice_proto_id

        mask[i] = True

    return raw_feats, card_ids, proto_ids, mask
