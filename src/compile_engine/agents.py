"""Baseline agents: random + a simple value-greedy heuristic.

An agent is any callable that, given a Game (or its state view) and the list
of legal Actions, returns one Action. We keep the surface tiny so that RL
policies can plug in as a class with `.choose(game, legal_actions)`.
"""

from __future__ import annotations

import random
from typing import Protocol

from .actions import Action, ActionType
from .effects import compute_line_value
from .game import Game


class Agent(Protocol):
    def choose(self, game: Game, legal: list[Action]) -> Action: ...


class RandomAgent:
    """Uniformly random action selection. Deterministic given a seed."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, game: Game, legal: list[Action]) -> Action:
        return self.rng.choice(legal)


class GreedyAgent:
    """Heuristic: prefer face-up plays that increase our line value most,
    then face-down plays, then refresh. Drafts in declared protocol order
    when offered. Picks random target during effects.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, game: Game, legal: list[Action]) -> Action:
        if not legal:
            raise RuntimeError("No legal actions")

        # Drafting: pick the first offered protocol.
        if legal[0].type is ActionType.DRAFT_PROTOCOL:
            return legal[0]

        # Effect-choice: pick first option deterministically (placeholder).
        if any(a.type is ActionType.CHOOSE_TARGET for a in legal):
            return next(a for a in legal if a.type is ActionType.CHOOSE_TARGET)

        # Forced compile
        if any(a.type is ActionType.COMPILE_LINE for a in legal):
            return next(a for a in legal if a.type is ActionType.COMPILE_LINE)

        # Clear cache: discard a low-value card (face-up value).
        if any(a.type is ActionType.DISCARD_CARD for a in legal):
            ap = game.state.current_player
            hand = game.state.players[ap].hand
            scored = [
                (game.defs[hand[a.hand_index].def_id].value, a)
                for a in legal if a.type is ActionType.DISCARD_CARD
            ]
            scored.sort(key=lambda x: x[0])
            return scored[0][1]

        # Action phase: rank plays.
        ap = game.state.current_player
        st = game.state
        best_score = -1
        best_action: Action | None = None
        for a in legal:
            if a.type is ActionType.PLAY_FACE_UP:
                c = st.players[ap].hand[a.hand_index]
                score = 100 + st.defs[c.def_id].value  # face-up gets value bonus
            elif a.type is ActionType.PLAY_FACE_DOWN:
                score = 30  # face-down contributes 2 (default)
            elif a.type is ActionType.REFRESH:
                # Refresh is worse than playing, better than nothing.
                score = 10 if len(st.players[ap].hand) >= 3 else 50
            else:
                score = 0
            if score > best_score:
                best_score = score
                best_action = a
        assert best_action is not None
        return best_action
