"""Gym-style RL environment + vectorized rollout.

`CompileEnv` exposes the standard reset/step interface, returning numeric
observations and per-step rewards. Two-player play is handled by alternating
the "current decider" between agents (the env is single-agent from the
perspective of one side; pass an opponent policy at construction).

The observation tensor is purposely simple — easy to swap with a richer
encoding once a baseline policy works. Action space is the variable-length
list of currently legal actions; the agent's policy should output a
distribution over indices in `legal_actions()`.

For training throughput, see `vectorized_self_play()` which streams
many independent games and yields (observation, legal_actions, decider) per
step. Pure-Python; on a modest laptop this comfortably hits a few thousand
turns/sec with the random policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .actions import Action, ActionType
from .agents import Agent, RandomAgent
from .cards import load_card_defs
from .effects import compute_line_value
from .game import Game, GameOver
from .state import (
    FACE_DOWN_BASE_VALUE,
    HAND_SIZE_LIMIT,
    NUM_LINES,
    GameConfig,
    Phase,
)


# Observation layout (per perspective player):
#   For each of 3 lines:
#     - our value (1 int)
#     - opp value (1 int)
#     - our protocol id (1 int)
#     - opp protocol id (1 int)
#     - our compiled (0/1)
#     - opp compiled (0/1)
#     - our stack size, opp stack size
#   Hand summary: counts of cards per protocol-id (configurable; cap = 15)
#   Plus: control holder (-1/0/1), turn, deck/hand/trash sizes for both.

_PROTOCOL_IDS: dict[str, int] = {}


def _proto_id(name: str) -> int:
    if name not in _PROTOCOL_IDS:
        _PROTOCOL_IDS[name] = len(_PROTOCOL_IDS) + 1  # 0 reserved for "none"
    return _PROTOCOL_IDS[name]


@dataclass(slots=True)
class StepResult:
    obs: list[float]
    legal_actions: list[Action]
    decider: int
    reward: float
    done: bool
    info: dict[str, Any]


class CompileEnv:
    """RL environment wrapping a single Game.

    Args
    ----
    config: GameConfig — base config. Use `expansion_sample_prob` to mix
        modes (expansion on/off) across resets for generalization training.
    perspective: which player (0 or 1) the env's observation is centered on.
        Rewards are also signed for this perspective. Set `perspective="active"`
        to always observe from the currently-deciding player (useful for
        self-play with shared parameters).
    opponent: optional opponent agent. If provided, the env auto-plays
        opponent turns so `step()` only requires the perspective agent's
        action. If None, the caller drives both sides manually.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        perspective: int | str = 0,
        opponent: Agent | None = None,
        card_defs=None,
    ) -> None:
        self._base_config = config or GameConfig()
        self.perspective = perspective
        self.opponent = opponent
        self._defs = card_defs if card_defs is not None else load_card_defs()
        self._game: Game | None = None

    # ----------------------------------------------------------- API

    def reset(self, *, seed: int | None = None) -> StepResult:
        cfg = GameConfig(
            include_expansion=self._base_config.include_expansion,
            seed=seed if seed is not None else self._base_config.seed,
            max_turns=self._base_config.max_turns,
            deterministic_draft=self._base_config.deterministic_draft,
            expansion_sample_prob=self._base_config.expansion_sample_prob,
        )
        self._game = Game(cfg, defs=self._defs)
        self._game.start()
        # Resolve opponent decisions until the perspective agent must act.
        self._auto_opponent()
        return self._make_step_result(reward=0.0)

    def step(self, action: Action) -> StepResult:
        assert self._game is not None, "reset() first"
        if self._game.is_over():
            raise GameOver("Game over; call reset()")

        prev_state = self._snapshot_score()
        self._game.step(action)
        # Auto-play opponent until perspective must act again (or game ends).
        self._auto_opponent()
        new_state = self._snapshot_score()
        reward = self._reward(prev_state, new_state)
        return self._make_step_result(reward=reward)

    def render_text(self) -> str:
        assert self._game is not None
        return _render_text(self._game)

    # ----------------------------------------------------------- internals

    def _perspective_player(self) -> int:
        if self.perspective == "active":
            return self._game.decider()
        return int(self.perspective)

    def _auto_opponent(self) -> None:
        if self.opponent is None:
            return
        g = self._game
        while not g.is_over():
            who = g.decider()
            persp = self._perspective_player()
            if who == persp and self.perspective != "active":
                return
            legal = g.legal_actions()
            if not legal:
                return
            if self.perspective == "active":
                # Self-play setting: caller will be alternated externally.
                return
            action = self.opponent.choose(g, legal)
            g.step(action)

    def _snapshot_score(self) -> tuple[int, int]:
        g = self._game
        return (
            sum(g.state.players[0].compiled),
            sum(g.state.players[1].compiled),
        )

    def _reward(self, prev: tuple[int, int], new: tuple[int, int]) -> float:
        persp = self._perspective_player()
        opp = 1 - persp
        delta = (new[persp] - prev[persp]) - (new[opp] - prev[opp])
        if self._game.is_over():
            w = self._game.state.winner
            if w is None:
                return float(delta)  # draw: only intermediate shaping
            return float(delta) + (1.0 if w == persp else -1.0)
        return float(delta)

    def _make_step_result(self, *, reward: float) -> StepResult:
        g = self._game
        done = g.is_over()
        legal = g.legal_actions() if not done else []
        decider = g.decider() if not done else -1
        obs = encode_observation(g, perspective=self._perspective_player())
        info = {
            "turn": g.state.turn,
            "phase": g.state.phase.name,
            "winner": g.state.winner,
            "include_expansion": g.config.include_expansion,
        }
        return StepResult(
            obs=obs, legal_actions=legal, decider=decider,
            reward=reward, done=done, info=info,
        )


# ---------------------------------------------------------------- encoding

def encode_observation(game: Game, *, perspective: int) -> list[float]:
    """Compact numeric observation (list of floats). Stable across resets."""
    st = game.state
    me = perspective
    opp = 1 - me
    obs: list[float] = []
    for ln in range(NUM_LINES):
        obs.append(float(compute_line_value(st, ln, me)))
        obs.append(float(compute_line_value(st, ln, opp)))
        my_proto = st.players[me].protocols[ln] if len(st.players[me].protocols) > ln else ""
        op_proto = st.players[opp].protocols[ln] if len(st.players[opp].protocols) > ln else ""
        obs.append(float(_proto_id(my_proto) if my_proto else 0))
        obs.append(float(_proto_id(op_proto) if op_proto else 0))
        obs.append(float(int(st.players[me].compiled[ln])) if len(st.players[me].compiled) > ln else 0.0)
        obs.append(float(int(st.players[opp].compiled[ln])) if len(st.players[opp].compiled) > ln else 0.0)
        obs.append(float(len(st.lines[ln].stack(me))))
        obs.append(float(len(st.lines[ln].stack(opp))))
    obs.append(float(len(st.players[me].hand)))
    obs.append(float(len(st.players[opp].hand)))
    obs.append(float(len(st.players[me].deck)))
    obs.append(float(len(st.players[opp].deck)))
    obs.append(float(len(st.players[me].trash)))
    obs.append(float(len(st.players[opp].trash)))
    ctrl = st.control_holder
    obs.append(float(0 if ctrl is None else (1 if ctrl == me else -1)))
    obs.append(float(st.turn))
    obs.append(float(int(st.config.include_expansion)))
    obs.append(float(int(st.current_player == me)))
    return obs


# ---------------------------------------------------------------- rollout

def play_game(
    *,
    agent0: Agent,
    agent1: Agent,
    config: GameConfig | None = None,
    card_defs=None,
) -> Game:
    """Run a full self-play game with two agents. Returns the finished Game."""
    defs = card_defs if card_defs is not None else load_card_defs()
    game = Game(config, defs=defs)
    game.start()
    agents = (agent0, agent1)
    while not game.is_over():
        who = game.decider()
        legal = game.legal_actions()
        if not legal:
            break
        action = agents[who].choose(game, legal)
        game.step(action)
    return game


def parallel_random_rollouts(
    n_games: int,
    *,
    workers: int = 0,
    include_expansion: bool | None = None,
    expansion_sample_prob: float | None = None,
    seed: int = 0,
    max_turns: int = 300,
) -> list[dict]:
    """Run N random-vs-random games across worker processes. Returns per-game
    summaries (turns, winner, expansion mode). Pure multiprocessing; no deps.

    `workers=0` picks os.cpu_count(). Set `expansion_sample_prob` to a value
    in [0,1] to randomly mix expansion-on/off across games; otherwise the
    `include_expansion` flag is used for all games.
    """
    import multiprocessing as mp

    if workers == 0:
        import os
        workers = max(1, (os.cpu_count() or 1))

    seeds = list(range(seed, seed + n_games))
    args = [
        (s, include_expansion, expansion_sample_prob, max_turns) for s in seeds
    ]
    if workers == 1:
        return [_rollout_one(a) for a in args]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(_rollout_one, args, chunksize=max(1, n_games // (workers * 8)))


def _rollout_one(args) -> dict:
    seed, include_expansion, exp_prob, max_turns = args
    from .agents import RandomAgent
    cfg = GameConfig(
        include_expansion=bool(include_expansion) if include_expansion is not None else False,
        seed=seed,
        max_turns=max_turns,
        expansion_sample_prob=exp_prob if include_expansion is None else None,
    )
    g = play_game(
        agent0=RandomAgent(seed=seed),
        agent1=RandomAgent(seed=seed + 1_000_003),
        config=cfg,
    )
    return {
        "seed": seed,
        "turns": g.state.turn,
        "winner": g.state.winner,
        "include_expansion": g.config.include_expansion,
    }


def vectorized_self_play(
    n_games: int,
    *,
    agent_factory: Callable[[int], tuple[Agent, Agent]],
    config_factory: Callable[[int], GameConfig] | None = None,
) -> Iterator[Game]:
    """Yield N finished games. Single-process, but designed to be embarrassingly
    parallel: spawn this in worker processes for true vectorization.

    `agent_factory(i)` returns (p0_agent, p1_agent) for game i.
    `config_factory(i)` returns a per-game GameConfig (e.g. to randomize
    `include_expansion` from `expansion_sample_prob`).
    """
    defs = load_card_defs()
    for i in range(n_games):
        cfg = config_factory(i) if config_factory else None
        a0, a1 = agent_factory(i)
        yield play_game(agent0=a0, agent1=a1, config=cfg, card_defs=defs)


# ---------------------------------------------------------------- pretty-print

def _render_text(game: Game) -> str:
    st = game.state
    lines = []
    lines.append(
        f"Turn {st.turn} | phase={st.phase.name} | current={st.current_player} "
        f"| control={st.control_holder} | expansion={st.config.include_expansion}"
    )
    for ln in range(NUM_LINES):
        p0_p = st.players[0].protocols[ln] if len(st.players[0].protocols) > ln else "?"
        p1_p = st.players[1].protocols[ln] if len(st.players[1].protocols) > ln else "?"
        p0_v = compile_line_value_safe(st, ln, 0)
        p1_v = compile_line_value_safe(st, ln, 1)
        p0_compiled = "*" if (len(st.players[0].compiled) > ln and st.players[0].compiled[ln]) else " "
        p1_compiled = "*" if (len(st.players[1].compiled) > ln and st.players[1].compiled[ln]) else " "
        lines.append(
            f"  Line {ln}: P0[{p0_compiled}] {p0_p:8s} ({p0_v:2d}) "
            f"vs ({p1_v:2d}) {p1_p:8s} [{p1_compiled}] P1   "
            f"stacks={len(st.lines[ln].p0_stack)}/{len(st.lines[ln].p1_stack)}"
        )
    for pl in (0, 1):
        ps = st.players[pl]
        lines.append(
            f"  P{pl}: hand={len(ps.hand)} deck={len(ps.deck)} trash={len(ps.trash)} "
            f"compiled={sum(ps.compiled)}/3"
        )
    if st.winner is not None:
        lines.append(f"  Winner: P{st.winner}")
    return "\n".join(lines)


def compile_line_value_safe(st, ln: int, pl: int) -> int:
    try:
        return compute_line_value(st, ln, pl)
    except Exception:
        return 0
