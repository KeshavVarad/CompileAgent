"""Information-Set MCTS agent for Compile.

Standard AlphaZero-style PUCT search modified for two features of the
game that vanilla AlphaZero doesn't handle:

  1. **Stochastic transitions.** Deck shuffles, opponent hidden plays,
     and random tie-breaks during effect resolution mean the next state
     after a single action isn't deterministic.

  2. **Imperfect information.** From a player's perspective the
     opponent's hand and face-down field cards are unknown.

The fix is the IS-MCTS recipe (Information-Set MCTS with
determinization):

  - At each query, sample N "determinizations" — concrete worlds
    consistent with what the perspective player can see (opponent's
    hidden cards filled in by sampling from their drafted protocols
    minus what's already visible).

  - Run a small PUCT search on each determinization independently.

  - Aggregate visit counts across the N searches and return the
    most-visited action at the root.

The simulation policy uses the network for both seats: at perspective
nodes we expand with the policy as PUCT prior; at opponent nodes we
take a single forward-pass argmax (we don't search opp's subtree —
that's an approximation that keeps cost bounded). The value head
provides leaf estimates so we never roll out to terminal.

Mid-effect decisions (CHOOSE_TARGET / SKIP_OPTIONAL inside a yielded
effect generator) skip MCTS entirely and fall back to policy argmax —
search is reserved for top-level actions where the cost is justified.

Compute budget: on MPS with our small model (~580k params,
~10ms/forward), default (`n_determinizations=4, sims_per=25`)
≈ 400 forward passes per move ≈ 1.5–2s. Tunable via MCTSConfig.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..actions import Action, ActionType
from ..cards import defs_for_protocol
from ..game import Game
from ..state import CardInst, GameState
from .encoder import encode_actions, encode_state
from .model import PolicyValueNet

# Sentinel "stop here" action for the dummy parent slot in the root.
_ROOT_ACTION = Action(type=ActionType.NOOP)


@dataclass
class MCTSConfig:
    """Hyperparameters for the search. Defaults tuned for Mac CPU/MPS."""
    n_determinizations: int = 4
    sims_per_determinization: int = 25
    # PUCT exploration constant — AlphaZero used 1.25 for chess/Go.
    c_puct: float = 1.25
    # Cap on search-tree depth in moves before we just trust the value
    # head. 16 is more than a typical Compile decision sequence, so this
    # rarely fires.
    max_depth: int = 16
    # Mid-effect choice handling: if a CHOOSE_TARGET sub-decision appears
    # during simulation, we don't expand it — we just take the policy
    # argmax to keep the simulator moving.
    simulate_choice_with_argmax: bool = True


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------


class _Node:
    """One node in the search tree.

    `game` is a deep-copied Game at this node's position; cloning lets
    each sim trajectory mutate the engine state without aliasing.
    """
    __slots__ = (
        "game",
        "parent",
        "action_from_parent",
        "children",
        "prior",
        "n_visits",
        "total_value",
        "untried_actions",
        "is_terminal",
        "_priors",
    )

    def __init__(
        self,
        game: Game,
        parent: Optional["_Node"],
        action_from_parent: Action,
        prior: float,
    ) -> None:
        self.game = game
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: dict[int, _Node] = {}  # keyed by index in `untried_actions`
        self.prior = prior
        self.n_visits = 0
        self.total_value = 0.0
        self.untried_actions: list[Action] = []
        self.is_terminal = game.is_over()
        self._priors: np.ndarray | None = None

    def q(self) -> float:
        if self.n_visits == 0:
            return 0.0
        return self.total_value / self.n_visits

    def is_expanded(self) -> bool:
        return len(self.untried_actions) > 0 or bool(self.children) or self.is_terminal


# ---------------------------------------------------------------------------
# Determinization
# ---------------------------------------------------------------------------


def _opp_unknown_cards(state: GameState, opp: int, defs) -> list[int]:
    """Card def_ids the opponent could still be holding hidden somewhere.

    Total possible opp cards = the 6 cards of each of their 3 drafted
    protocols (18). Subtract every opp card the perspective player can
    actually see: face-up on field, in trash. Whatever's left is "could
    be in opp.hand / opp.deck / opp's face-down field positions."
    """
    opp_ps = state.players[opp]
    full: list[int] = []
    for proto in opp_ps.protocols:
        full.extend(d.def_id for d in defs_for_protocol(defs, proto))
    # Remove cards we've seen (opp's face-up board cards + opp trash).
    seen: list[int] = []
    for ln in state.lines:
        for c in ln.stack(opp):
            if c.face_up:
                seen.append(c.def_id)
    for c in opp_ps.trash:
        seen.append(c.def_id)
    # Multiset subtraction: each visible card removes one occurrence
    # from the candidate pool (a card def can in principle appear once
    # per protocol-set, but in practice Compile has unique cards per
    # protocol so this collapses to set subtraction).
    pool = list(full)
    for s in seen:
        try:
            pool.remove(s)
        except ValueError:
            pass
    return pool


def _determinize(game: Game, perspective: int, rng: random.Random) -> Game:
    """Return a deep-copied Game with opponent hidden info concrete.

    Mutates the cloned engine state's `opp.hand` def_ids, `opp.deck`
    def_ids, and any opp face-down field card def_ids by sampling from
    the unknowns. The perspective player's own state stays intact.
    """
    cloned = copy.deepcopy(game)
    st = cloned.state
    opp = 1 - perspective
    pool = _opp_unknown_cards(st, opp, cloned.defs)
    rng.shuffle(pool)

    # Walk every "hidden to perspective" opp card slot and assign one
    # def_id from the pool. In draw order: hand first, face-down field
    # cards next, deck last (deck order doesn't matter for the search
    # since the engine reshuffles on cache).
    opp_ps = st.players[opp]
    cursor = 0
    def _next_def() -> int:
        nonlocal cursor
        if cursor >= len(pool):
            # Pool exhausted (shouldn't happen unless trash + face-ups
            # already cover the whole drafted set, which means hand and
            # deck are empty — but defensive fallback).
            return 0
        d = pool[cursor]
        cursor += 1
        return d

    # Hand cards (all hidden).
    for c in opp_ps.hand:
        c.def_id = _next_def()
    # Face-down field cards.
    for ln in st.lines:
        for c in ln.stack(opp):
            if not c.face_up:
                c.def_id = _next_def()
    # Deck cards — assign from the remaining pool. Engine reshuffles on
    # cache so order here is informative-only.
    for c in opp_ps.deck:
        c.def_id = _next_def()
    return cloned


# ---------------------------------------------------------------------------
# Policy / value queries against the trained model
# ---------------------------------------------------------------------------


def _policy_and_value(
    model: PolicyValueNet,
    game: Game,
    legal: list[Action],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Return (probs over `legal`, value scalar from current decider's POV).
    Probabilities are renormalised over the legal-action prefix; padded
    slots in the model output are ignored."""
    perspective = game.decider()
    state = encode_state(game, perspective)
    raw, card_ids, proto_ids, mask = encode_actions(game, legal, perspective)
    s = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in state.items()}
    ar = torch.from_numpy(raw).unsqueeze(0).to(device)
    ac = torch.from_numpy(card_ids).unsqueeze(0).to(device)
    ap = torch.from_numpy(proto_ids).unsqueeze(0).to(device)
    am = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = model(s, ar, ac, ap, am)
    logits = logits[0].detach().cpu().numpy()
    v = float(value[0].item())
    # Softmax over the legal-action prefix only — padded slots already
    # have -inf logits, so adding them in is just noise.
    n = min(len(legal), logits.shape[0])
    sub = logits[:n]
    sub = sub - sub.max()
    expd = np.exp(sub)
    probs = expd / max(1e-9, expd.sum())
    return probs, v


def _value_only(
    model: PolicyValueNet,
    game: Game,
    device: torch.device,
) -> float:
    """Cheap value-only forward pass used at expansion / leaf nodes."""
    legal = game.legal_actions()
    if not legal:
        return 0.0
    _, v = _policy_and_value(model, game, legal, device)
    return v


# ---------------------------------------------------------------------------
# IS-MCTS agent
# ---------------------------------------------------------------------------


class MCTSAgent:
    """Drop-in Agent for the eval pipeline (and training, if desired).

    Tree is rebuilt from scratch for every `.choose()` — no cross-move
    tree reuse. Cross-move reuse is a small win on top but adds tree-
    surgery code that's easy to get wrong; skipping it for v1.
    """

    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device | str = "cpu",
        cfg: MCTSConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.cfg = cfg or MCTSConfig()
        self.rng = random.Random(seed)

    def choose(self, game: Game, legal: list[Action]) -> Action:
        if not legal:
            raise RuntimeError("MCTSAgent.choose called with empty legal actions")
        if len(legal) == 1:
            return legal[0]

        # Mid-effect (CHOOSE_TARGET) → policy argmax. The simulator
        # would need to enter an in-flight effect generator otherwise.
        if _mid_effect(game):
            probs, _ = _policy_and_value(self.model, game, legal, self.device)
            return legal[int(np.argmax(probs[: len(legal)]))]

        perspective = game.decider()

        # Aggregate visit counts across determinizations. We index legal
        # actions by their position in the *original* legal list, which
        # is stable across determinizations (engine action enumeration
        # is deterministic given a state, and determinization only
        # touches hidden card identities, not legality).
        total_visits = np.zeros(len(legal), dtype=np.int64)
        for d_idx in range(self.cfg.n_determinizations):
            det_game = _determinize(game, perspective, self.rng)
            root = _Node(game=det_game, parent=None,
                         action_from_parent=_ROOT_ACTION, prior=1.0)
            root_legal = det_game.legal_actions()
            # Align determinized legal to original legal order — they
            # should match by construction; if there's any drift, fall
            # back to whichever legal list the determinization produced
            # (rare; happens only when hidden info affected legality,
            # which Compile doesn't normally do at top-level decisions).
            self._expand(root, root_legal)
            for _ in range(self.cfg.sims_per_determinization):
                self._simulate(root, perspective, depth=0)
            # Roll up visits from root children
            for idx, child in root.children.items():
                # Map by position in det_legal → original legal.
                # When the lists match exactly, idx == idx in legal.
                if 0 <= idx < len(total_visits):
                    total_visits[idx] += child.n_visits

        best_idx = int(np.argmax(total_visits))
        return legal[best_idx]

    # ------------------------------------------------------------------
    # PUCT mechanics
    # ------------------------------------------------------------------

    def _expand(self, node: _Node, legal: list[Action]) -> float:
        """Populate node's children with priors. Returns leaf value
        from the current decider's perspective."""
        if not legal:
            node.is_terminal = True
            return 0.0
        probs, value = _policy_and_value(self.model, node.game, legal, self.device)
        node.untried_actions = list(legal)
        # Each child gets lazily realised on first selection (we just
        # remember the priors here).
        node._priors = probs
        return value

    def _select_child(self, node: _Node, perspective: int) -> tuple[int, Action]:
        """PUCT selection. Returns (child_index_into_legal, action)."""
        # Sum visits across all children of `node`.
        total_n = max(1, node.n_visits)
        priors = node._priors

        best_idx = -1
        best_score = -float("inf")
        decider = node.game.decider()
        for i, action in enumerate(node.untried_actions):
            child = node.children.get(i)
            n_i = child.n_visits if child else 0
            # Q from the decider's POV. Flip sign when the decider at
            # the child is the opponent so the value is consistently
            # "good for me" from the root's perspective.
            if child is None:
                q = 0.0
            else:
                q = child.q()
                if decider != perspective:
                    q = -q
            p = float(priors[i]) if priors is not None and i < len(priors) else 1.0 / len(node.untried_actions)
            u = self.cfg.c_puct * p * math.sqrt(total_n) / (1 + n_i)
            score = q + u
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, node.untried_actions[best_idx]

    def _simulate(self, root: _Node, perspective: int, depth: int) -> float:
        """One PUCT trajectory from root → leaf. Returns the leaf value
        (relative to the perspective player, +1 = perspective wins)."""
        node = root
        path: list[tuple[_Node, int]] = []  # (parent, child_idx_in_parent)
        # Selection
        while node.is_expanded() and not node.is_terminal and depth < self.cfg.max_depth:
            if not node.untried_actions:
                break
            child_idx, action = self._select_child(node, perspective)
            path.append((node, child_idx))
            existing = node.children.get(child_idx)
            if existing is not None:
                node = existing
                depth += 1
                continue
            # Need to expand this child. Apply action to a fresh clone,
            # then drain any in-flight effect choices with policy argmax
            # so the resulting state is back at a top-level decision
            # point (deep-copyable for the next sim step).
            child_game = copy.deepcopy(node.game)
            try:
                child_game.step(action)
                self._drain_mid_effect(child_game)
            except Exception:
                # Engine refused the action; mark this branch unproductive.
                # Should be rare — if it happens, the determinization
                # produced an inconsistent state. We back off to value 0.
                value = 0.0
                # Don't add the child; just back the prior up.
                self._backprop(path, value, perspective)
                return value
            # Drive engine to next decision-or-terminal.
            prior = 0.0
            if node._priors is not None and child_idx < len(node._priors):
                prior = float(node._priors[child_idx])
            child_node = _Node(
                game=child_game,
                parent=node,
                action_from_parent=action,
                prior=prior,
            )
            node.children[child_idx] = child_node
            node = child_node
            depth += 1
            break  # leave the selection loop to expand below

        # Evaluation
        if node.is_terminal:
            # Terminal: known reward.
            w = node.game.state.winner
            if w is None:
                value = 0.0
            elif w == perspective:
                value = 1.0
            else:
                value = -1.0
        else:
            legal = node.game.legal_actions()
            if not legal:
                node.is_terminal = True
                value = 0.0
            elif _mid_effect(node.game) and self.cfg.simulate_choice_with_argmax:
                # Don't expand mid-effect; advance the simulator with
                # policy argmax and re-evaluate.
                probs, value = _policy_and_value(self.model, node.game, legal, self.device)
                # Use value of this state (the policy-argmax move is
                # already "expected" under the network's belief).
            else:
                value = self._expand(node, legal)

        # Value above is from the node's current decider's POV. Convert
        # to perspective POV (positive = good for the root's perspective
        # player).
        if node.game.decider() != perspective:
            value = -value

        self._backprop(path, value, perspective)
        return value

    def _drain_mid_effect(self, game: Game, max_steps: int = 64) -> None:
        """Advance the engine past any in-flight effect choices using
        policy argmax, so subsequent deepcopy operations see a state
        without generator objects in `_pending`. Bounded for safety."""
        steps = 0
        while not game.is_over() and _mid_effect(game) and steps < max_steps:
            legal = game.legal_actions()
            if not legal:
                break
            probs, _ = _policy_and_value(self.model, game, legal, self.device)
            n = min(len(legal), len(probs))
            if n == 0:
                break
            idx = int(np.argmax(probs[:n]))
            game.step(legal[idx])
            steps += 1

    def _backprop(self, path, value: float, perspective: int) -> None:
        """Bump n_visits + total_value on every node along the path.

        `path` is a list of (parent, child_idx) edges. Each entry contributes
        exactly one visit to the child node; the root's visit is bumped
        externally (it gets one per call). Walking each edge once and
        touching only the child avoids the double-count where an
        intermediate node would otherwise get bumped both as "parent of
        next edge" and "child of previous edge".
        """
        # Root gets one visit per simulation.
        if path:
            path[0][0].n_visits += 1
            path[0][0].total_value += value
        for parent, child_idx in path:
            child = parent.children.get(child_idx)
            if child is not None:
                child.n_visits += 1
                child.total_value += value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mid_effect(game: Game) -> bool:
    """True if the engine is mid-resolution of an effect (any pending
    generator). We use this as a "skip MCTS, fall back to policy
    argmax" sentinel because:
      (a) Choice-resolution decisions are tactical sub-steps that
          rarely benefit from search.
      (b) The engine's `_pending` list holds Python generator objects,
          and generators can't be deep-copied — so MCTS literally
          can't simulate forward from a mid-effect state.
    A non-empty `_pending` with or without `last_choice` set is enough
    to make deepcopy fail, so the check is "any pending at all"."""
    pend = getattr(game, "_pending", None)
    return bool(pend)
