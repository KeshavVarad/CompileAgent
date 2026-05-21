"""Mutable game state, configured for fast cloning and serialization.

Conventions
- Player indices are 0 and 1.
- Lines are indexed 0..2. Each line has a protocol on each player's side.
- Stacks grow top-of-list = uncovered (newest card).
- Face-down cards have a base value of 2 (Compile core rule).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cards import CardDef

FACE_DOWN_BASE_VALUE = 2
COMPILE_THRESHOLD = 10
HAND_SIZE_LIMIT = 5
STARTING_HAND = 5
NUM_LINES = 3
NUM_PROTOCOLS_PER_PLAYER = 3
NUM_CARDS_PER_PROTOCOL = 6


class Phase(IntEnum):
    DRAFT = auto()
    START = auto()
    CHECK_CONTROL = auto()
    CHECK_COMPILE = auto()
    ACTION = auto()
    CHECK_CACHE = auto()
    END = auto()
    RESOLVING_EFFECT = auto()
    GAME_OVER = auto()


@dataclass(slots=True)
class CardInst:
    """A specific copy of a card in the game (each player has their own deck)."""
    inst_id: int
    def_id: int
    owner: int
    face_up: bool = False
    # Committed cards are mid-placement (play or shift) and per the Compile
    # Codex (16 Dec 2024) cannot be selected as targets by effects. Flipped
    # back to False once the placement settles and its enter-play triggers
    # have fired.
    is_committed: bool = False

    def value(self, defs: "list[CardDef]") -> int:
        if not self.face_up:
            return FACE_DOWN_BASE_VALUE
        return defs[self.def_id].value

    def protocol(self, defs: "list[CardDef]") -> str:
        return defs[self.def_id].protocol


@dataclass(slots=True)
class PlayerState:
    idx: int
    deck: list[CardInst] = field(default_factory=list)        # top is end
    hand: list[CardInst] = field(default_factory=list)
    trash: list[CardInst] = field(default_factory=list)
    # protocols[line_idx] = (protocol name, compiled?)
    protocols: list[str] = field(default_factory=list)
    compiled: list[bool] = field(default_factory=list)
    # Flag set by Metal 1: opponent cannot compile on their next turn.
    cannot_compile_next_turn: bool = False

    def all_compiled(self) -> bool:
        return len(self.compiled) == NUM_LINES and all(self.compiled)

    def draw(self, n: int) -> int:
        """Draw up to n cards; reshuffle trash if needed. Returns how many drawn."""
        drawn = 0
        for _ in range(n):
            if not self.deck:
                if not self.trash:
                    return drawn
                # Reshuffle trash into deck. RNG handled by the Game layer
                # (caller will shuffle); here we just move them.
                self.deck = self.trash
                self.trash = []
                # Caller is responsible for shuffling deck after this transfer.
                # Signal via a sentinel: we just move, no shuffle here.
                # See Game.draw_cards for the actual shuffle.
            if self.deck:
                self.hand.append(self.deck.pop())
                drawn += 1
        return drawn


@dataclass(slots=True)
class Line:
    """One of three lines on the field, with a stack for each player."""
    p0_stack: list[CardInst] = field(default_factory=list)  # bottom -> top (top=uncovered)
    p1_stack: list[CardInst] = field(default_factory=list)

    def stack(self, player: int) -> list[CardInst]:
        return self.p0_stack if player == 0 else self.p1_stack

    def uncovered(self, player: int) -> CardInst | None:
        s = self.stack(player)
        return s[-1] if s else None


@dataclass(slots=True)
class GameConfig:
    """Tunable game configuration."""
    include_expansion: bool = False    # AX01 — legacy field kept for back-compat
    include_main2: bool = False        # MN02 (Chaos / Clarity / ... / War)
    include_aux2: bool = False         # AX02 (Assimilation / Diversity / Unity)
    seed: int | None = None
    # Hard turn cap to keep RL rollouts bounded.
    max_turns: int = 200
    # Defensive cap on the per-turn effect-resolution stack. Compile's rules
    # technically allow unbounded flip-chains (card A flips B which flips A);
    # rational play avoids this but random rollouts will hit it. When the
    # pending stack exceeds this depth we drop further pushes for the turn.
    max_effect_stack_depth: int = 64
    # Total cumulative effect pushes allowed in a single turn. Once exceeded,
    # the engine aborts any in-flight effect resolution and proceeds to
    # check-cache / end. Together with `max_effect_stack_depth`, this gives
    # a hard bound on per-turn work and guarantees rollouts terminate.
    max_effect_pushes_per_turn: int = 256
    # How to resolve a game that reaches `max_turns` without a compile win.
    # Compile's RAW rules have no turn cap and a rational loser can sometimes
    # stall to deny a win; for RL training we award the leader-on-progress
    # by default so the loser is incentivised to keep trying.
    #   "leader_wins": player with more compiled protocols wins (ties = draw)
    #   "draw":        always a draw (winner=None)
    turn_cap_resolution: str = "leader_wins"
    # If True, instead of letting the draft logic pick protocols, the caller
    # provides them via Game.set_predetermined_draft(...).
    deterministic_draft: bool = False
    # Probability of including the expansion when sampling. Used by env
    # constructors to mix training distributions; ignored when set_predetermined.
    expansion_sample_prob: float | None = None


@dataclass(slots=True)
class GameState:
    config: GameConfig
    defs: "list[CardDef]"
    players: tuple[PlayerState, PlayerState]
    lines: list[Line]
    current_player: int = 0
    turn: int = 0
    phase: Phase = Phase.DRAFT
    control_holder: int | None = None  # None = neutral
    rng: random.Random = field(default_factory=random.Random)
    winner: int | None = None
    log: list[str] = field(default_factory=list)
    # Per-turn flags
    compiled_this_turn: bool = False
    # Trigger queue for events raised by effects mid-resolution. Items are
    # ("face_up" | "uncover", line_idx, player, CardInst). LIFO; drained by
    # the engine before/after each effect generator step.
    triggers: list[tuple] = field(default_factory=list)
    # Per-effect scratch space (used for multi-step effects like "discard N").
    scratch: dict = field(default_factory=dict)
    # Per-turn counter of effect pushes (resets at end of turn).
    effect_pushes_this_turn: int = 0

    def opponent(self, player: int) -> int:
        return 1 - player

    def line_value(self, line_idx: int, player: int) -> int:
        """Total value of player's stack in this line, including face-down base
        plus persistent (top) modifiers from face-up uncovered+covered cards."""
        from .effects import compute_line_value
        return compute_line_value(self, line_idx, player)

    def line_for_protocol(self, player: int, protocol: str) -> int | None:
        ps = self.players[player]
        for i, p in enumerate(ps.protocols):
            if p == protocol:
                return i
        return None
