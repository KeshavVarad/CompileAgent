"""Action and Choice types.

The engine exposes a single decision interface: at any non-terminal point the
current decision-maker must pick from a list of legal `Action` (or `Choice`)
options returned by Game.legal_actions(). This unifies "macro" turn actions
(play / refresh / compile) and "micro" effect resolution prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto


class ActionType(IntEnum):
    # Draft phase
    DRAFT_PROTOCOL = auto()
    # Main action phase
    PLAY_FACE_UP = auto()
    PLAY_FACE_DOWN = auto()
    REFRESH = auto()
    # Forced compile (only legal action when conditions met)
    COMPILE_LINE = auto()
    # Discard during Clear Cache
    DISCARD_CARD = auto()
    # Shift own card affordance (Speed 2 / Spirit 3 owner — can shift even if
    # covered, as an alternative to play/refresh during the action phase).
    SHIFT_OWN_CARD = auto()
    # Choice-resolution actions (for in-effect prompts)
    CHOOSE_TARGET = auto()
    SKIP_OPTIONAL = auto()
    # End-of-game sentinel
    NOOP = auto()


@dataclass(frozen=True, slots=True)
class Action:
    """A single decision the current decider can make.

    Fields are interpreted based on `type`:
      - DRAFT_PROTOCOL: payload = protocol name (str)
      - PLAY_FACE_UP / PLAY_FACE_DOWN: hand_index (int), line_index (int)
      - REFRESH: no payload
      - SHIFT_OWN_CARD: line_index (source line), hand_index (stack position),
        choice_index (destination line)
      - COMPILE_LINE: line_index (int)
      - DISCARD_CARD: hand_index (int)
      - CHOOSE_TARGET: choice_index (int)   -- index into the current Choice options
      - SKIP_OPTIONAL: no payload (only legal if the current choice is optional)
    """
    type: ActionType
    hand_index: int = -1
    line_index: int = -1
    choice_index: int = -1
    protocol: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = [self.type.name]
        if self.hand_index >= 0:
            parts.append(f"hand={self.hand_index}")
        if self.line_index >= 0:
            parts.append(f"line={self.line_index}")
        if self.choice_index >= 0:
            parts.append(f"choice={self.choice_index}")
        if self.protocol:
            parts.append(f"protocol={self.protocol}")
        return f"Action({', '.join(parts)})"


@dataclass(slots=True)
class Choice:
    """A prompt raised by a card effect that needs a player decision.

    `options` is a list of human-readable descriptions; the agent picks one by
    submitting an Action of type CHOOSE_TARGET with `choice_index` in
    [0, len(options)). If `optional` is True, SKIP_OPTIONAL is also legal.
    The engine knows how to translate the index back into the concrete target
    via `targets` (any opaque list parallel to `options`).
    """
    prompt: str
    options: list[str]
    targets: list[object]
    optional: bool = False
    decider: int = 0  # which player decides (usually the active player)
