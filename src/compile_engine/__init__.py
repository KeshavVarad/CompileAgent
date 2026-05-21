"""Compile card game engine, designed for scalable self-play / RL training."""

from .cards import CardDef, BASE_PROTOCOLS, EXPANSION_PROTOCOLS, load_card_defs
from .state import GameState, PlayerState, Line, Phase, GameConfig
from .game import Game, GameOver
from .actions import Action, ActionType, Choice
from .env import CompileEnv

__all__ = [
    "CardDef",
    "BASE_PROTOCOLS",
    "EXPANSION_PROTOCOLS",
    "load_card_defs",
    "GameState",
    "PlayerState",
    "Line",
    "Phase",
    "GameConfig",
    "Game",
    "GameOver",
    "Action",
    "ActionType",
    "Choice",
    "CompileEnv",
]
