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
    # Dirichlet noise at root only (AlphaZero technique). Without this,
    # a peaked policy (avg top_prob ≈ 0.95) saturates PUCT — no realistic
    # c_puct overcomes a 1000× prior ratio between top and tail actions.
    # The mix `(1-eps)*p + eps*Dir(alpha)` forces some visits onto
    # non-policy actions regardless of how confident the policy is.
    # Set eps=0 to disable (default — preserves the original behavior).
    # AZ used eps=0.25, alpha=0.3 (chess) / 0.03 (Go). Compile's branching
    # factor (~16 legal actions/move) is closer to chess than Go.
    dirichlet_eps: float = 0.0
    dirichlet_alpha: float = 0.3
    # Root top-k pruning. When > 0, after the root expands we keep only
    # the `k` actions with the highest policy priors and flatten those
    # priors to uniform 1/k. The tail (low-prior actions) gets zero
    # sims. Combined with `root_min_visits_per_action`, this implements
    # "trust the policy enough to ignore the tail, but give every top-k
    # action a fair share of sims so the value head can pick the best
    # one." Set to 0 to disable.
    root_top_k: int = 0
    # Root round-robin guarantee. When > 0, the first
    # `len(untried_actions) * root_min_visits_per_action` sims at the
    # root MUST round-robin across every untried action — PUCT only
    # takes over after each action has been visited that many times.
    # Fixes the bug where a peaked policy + non-zero Q on the first sim
    # locks PUCT onto the top action forever.
    root_min_visits_per_action: int = 0
    # Leaf batching for forward-pass throughput. When > 1, the search
    # collects up to `batch_size` sims' leaves before doing one batched
    # network forward pass — amortizing per-call MPS dispatch overhead
    # across the batch. Virtual loss is applied along each in-flight
    # path during collection so subsequent sims in the batch don't all
    # collapse onto the same trajectory. batch_size=1 reproduces the
    # original one-sim-at-a-time behavior.
    batch_size: int = 1
    # Virtual loss magnitude applied during batched collection (and
    # reverted before real-value backprop). Sign is uniform across the
    # path; in two-player play the negamax-correct sign would alternate
    # by decider, but for small batch sizes the diversification effect
    # is dominated by root-level visits where the sign is correct anyway.
    virtual_loss: float = 1.0
    # Skip search entirely when policy top_prob >= this threshold.
    # The diagnostic shows zero disagreements between MCTS and policy
    # argmax in the top_prob > 0.9 bucket across hundreds of decisions,
    # so search is pure compute waste on confident states. Set to 0.0
    # to disable; 0.9 is a safe choice given the observed structure.
    skip_search_top_prob: float = 0.0
    # Gumbel root selection (Danihelka et al. 2022, "Policy improvement
    # by planning with Gumbel"). When True, replace UCB at the root with
    # Gumbel-Top-k action selection + completed-Q-based policy target.
    # The improved policy target is GUARANTEED to be better than the
    # current policy (vanilla AZ has no such guarantee at low sim counts),
    # which fixes the "policy collapses onto safe actions" failure mode
    # observed when training under a tight sim budget.
    #
    # When True, `root_top_k`, `root_min_visits_per_action`, and
    # `dirichlet_eps` are ignored at the root — Gumbel sampling provides
    # the exploration mechanism instead.
    use_gumbel_root: bool = False
    # Top-m candidate count for Gumbel selection. Each sim at root visits
    # one of the top-m actions by (gumbel + log_prior). 0 = use all legal.
    gumbel_n_candidates: int = 16
    # Scaling constants for the sigma function in completed-Q:
    #   sigma(q) = (c_visit + max_visits) * c_scale * q
    # These match the paper's defaults. Don't touch unless you've read
    # the paper.
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0


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
        "_gumbel",          # Gumbel noise vector at root (None elsewhere)
        "_gumbel_allowed",  # set of allowed root action indices under top-m cap
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
        self._gumbel: np.ndarray | None = None
        self._gumbel_allowed: set[int] | None = None

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
    raw, card_ids, proto_ids, extra_card_ids, mask = encode_actions(
        game, legal, perspective,
    )
    s = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in state.items()}
    ar = torch.from_numpy(raw).unsqueeze(0).to(device)
    ac = torch.from_numpy(card_ids).unsqueeze(0).to(device)
    ap = torch.from_numpy(proto_ids).unsqueeze(0).to(device)
    ae = torch.from_numpy(extra_card_ids).unsqueeze(0).to(device)
    am = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = model(s, ar, ac, ap, ae, am)
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


@dataclass
class _LeafResult:
    """One sim's trajectory result, ready for (batched) evaluation + backprop.

    `mode` discriminates how the caller should finalize:
      - "terminal":   leaf is game-over; `terminal_value` is the perspective-POV
                      reward (+1/-1/0). No network call needed.
      - "expand":     leaf is unexplored; needs policy+value to seed priors and
                      back up a value estimate.
      - "mid_effect": leaf is mid-effect (CHOOSE_TARGET); we don't expand it,
                      we just need its value (the policy-argmax move from here
                      is what subsequent sims will simulate).
      - "failed":     the engine refused the action during expansion. Caller
                      backs up 0; virtual loss should NOT have been applied.
    """
    path: list[tuple["_Node", int]]
    leaf: Optional["_Node"]
    mode: str
    legal: list[Action]
    terminal_value: float


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
        # Separate numpy RNG used only for Dirichlet draws so the
        # determinization sampling (above, via `random.Random`) stays bit-
        # identical when noise is disabled.
        self.np_rng = np.random.default_rng(seed)

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

        # Skip-when-confident: if the policy is already very sure of one
        # move, search has no upside and just burns compute. Diagnostic
        # data shows zero MCTS/policy disagreements on top_prob > 0.9
        # decisions, so this is essentially free WR with a ~10% wall-clock
        # win on Greedy matchups.
        if self.cfg.skip_search_top_prob > 0.0:
            probs, _ = _policy_and_value(self.model, game, legal, self.device)
            n = min(len(legal), len(probs))
            top_prob = float(probs[:n].max()) if n > 0 else 0.0
            if top_prob >= self.cfg.skip_search_top_prob:
                return legal[int(np.argmax(probs[:n]))]

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
            if self.cfg.use_gumbel_root:
                # Gumbel replaces top-k pruning + Dirichlet noise + UCB
                # at the root.
                self._gumbel_search(root, perspective)
            else:
                self._apply_root_top_k(root)
                self._apply_root_noise(root)
                self._search(root, perspective)
            # Roll up visits from root children. With root_top_k pruning,
            # the child index points into the pruned `untried_actions`,
            # not the original `legal`; map back via action equality.
            for child in root.children.values():
                try:
                    orig_idx = legal.index(child.action_from_parent)
                except ValueError:
                    continue
                total_visits[orig_idx] += child.n_visits

        best_idx = int(np.argmax(total_visits))
        return legal[best_idx]

    def choose_with_target(
        self,
        game: Game,
        legal: list[Action],
        tau: float = 1.0,
        return_value: bool = False,
        sample_action: bool = False,
        sample_temperature: float = 1.0,
    ) -> tuple[Action, np.ndarray] | tuple[Action, np.ndarray, float]:
        """Like `choose`, but also return a soft target distribution over
        `legal` for policy distillation. Target shape: [len(legal)].

        Target formulation (AlphaZero visit-count target with prior-
        weighted Laplace smoothing):

            score(a) = visits(a) + alpha * prior(a)
            target   ∝ score(a) ** (1 / tau)

        `visits(a)` is aggregated across determinizations. The
        `alpha * prior(a)` term is a soft prior — equivalent to one
        "virtual sim drawn from the prior" — and exists for two
        reasons:
          1. With `root_top_k` pruning, actions outside the top-k get
             zero visits. Hard-zero targets would teach the policy to
             eliminate those actions entirely; that's too strong a
             claim from what is really a search-budget heuristic.
          2. It preserves the prior's *ranking* on pruned actions
             rather than collapsing them to a flat floor.

        `tau` is a sharpening exponent (1.0 = standard visits/total,
        smaller = more peaked, larger = smoother). Defaults to 1.0;
        AlphaZero uses tau→0 late in training to commit to argmax.

        Edge cases: single-legal actions and mid-effect (CHOOSE_TARGET)
        decisions return a one-hot target on the policy/MCTS pick,
        since no search runs in those branches.
        """
        if not legal:
            raise RuntimeError("MCTSAgent.choose_with_target called with empty legal")
        if len(legal) == 1:
            target = np.array([1.0], dtype=np.float32)
            if not return_value:
                return legal[0], target
            # No search for single-action states — use NN's value head as
            # the best estimate we have of the position.
            _, v = _policy_and_value(self.model, game, legal, self.device)
            return legal[0], target, v

        # Policy prior at the *real* (non-determinized) state — this is
        # what the network sees at inference time, so it's what we want
        # to base the distillation target on. Reused for mid-effect /
        # skip-when-confident fall-throughs below.
        probs, nn_value = _policy_and_value(self.model, game, legal, self.device)
        n = min(len(legal), len(probs))
        priors = np.zeros(len(legal), dtype=np.float64)
        priors[:n] = probs[:n]

        if _mid_effect(game):
            target = np.zeros(len(legal), dtype=np.float32)
            target[int(np.argmax(priors))] = 1.0
            action = legal[int(np.argmax(priors))]
            return (action, target, nn_value) if return_value else (action, target)

        # Skip-when-confident: when search wouldn't disagree anyway,
        # return policy argmax with one-hot target. Caller may also
        # filter these out *before* invoking us; this is a safety net.
        if (
            self.cfg.skip_search_top_prob > 0.0
            and float(priors.max()) >= self.cfg.skip_search_top_prob
        ):
            target = np.zeros(len(legal), dtype=np.float32)
            target[int(np.argmax(priors))] = 1.0
            action = legal[int(np.argmax(priors))]
            return (action, target, nn_value) if return_value else (action, target)

        perspective = game.decider()

        # Aggregate visits across determinizations + accumulate the
        # root's search-refined value. Each det's root.total_value /
        # root.n_visits is the search V from `perspective`'s POV. We
        # sum total_value and n_visits separately (rather than
        # averaging per-det V) so dets with more sims weight
        # proportionally.
        total_visits = np.zeros(len(legal), dtype=np.int64)
        root_total_value_sum = 0.0
        root_visits_sum = 0
        # For Gumbel mode: accumulate per-det improved-policy targets and
        # average at the end. For vanilla mode: rely on the visit-count
        # path below.
        gumbel_target_sum: np.ndarray | None = (
            np.zeros(len(legal), dtype=np.float64)
            if self.cfg.use_gumbel_root else None
        )
        n_gumbel_dets = 0
        for _ in range(self.cfg.n_determinizations):
            det_game = _determinize(game, perspective, self.rng)
            root = _Node(game=det_game, parent=None,
                         action_from_parent=_ROOT_ACTION, prior=1.0)
            root_legal = det_game.legal_actions()
            self._expand(root, root_legal)
            if self.cfg.use_gumbel_root:
                # Gumbel replaces top-k pruning + Dirichlet noise + UCB
                # at the root.
                self._gumbel_search(root, perspective)
                # Build the Gumbel-improved target for this det, indexed
                # by the engine's stable `legal` ordering. The det's own
                # root.untried_actions may align with `legal` index-for-
                # index (engine action enumeration is deterministic given
                # state); if not, fall back to action equality.
                det_target = self._gumbel_improved_target(
                    root, perspective, n_legal=len(root.untried_actions),
                )
                # Map det's positional target into original legal order.
                for i_det, action_det in enumerate(root.untried_actions):
                    try:
                        orig_idx = legal.index(action_det)
                    except ValueError:
                        continue
                    if i_det < len(det_target):
                        gumbel_target_sum[orig_idx] += det_target[i_det]
                n_gumbel_dets += 1
            else:
                self._apply_root_top_k(root)
                self._apply_root_noise(root)
                self._search(root, perspective)
            root_total_value_sum += root.total_value
            root_visits_sum += root.n_visits
            for child in root.children.values():
                try:
                    orig_idx = legal.index(child.action_from_parent)
                except ValueError:
                    continue
                total_visits[orig_idx] += child.n_visits

        # Target:
        #   Gumbel mode → mean of per-det Gumbel-improved policy targets
        #     (each det's target is guaranteed-improving by construction)
        #   Vanilla mode → visit-count target with prior-weighted Laplace
        #     smoothing (the AlphaZero recipe at higher sim budgets)
        if gumbel_target_sum is not None and n_gumbel_dets > 0:
            target = (gumbel_target_sum / n_gumbel_dets).astype(np.float32)
            s = target.sum()
            if s > 0:
                target = (target / s).astype(np.float32)
        else:
            alpha = 1.0
            score = total_visits.astype(np.float64) + alpha * priors
            if tau != 1.0 and tau > 0:
                score = np.power(np.maximum(score, 1e-12), 1.0 / tau)
            target = (score / max(1e-9, score.sum())).astype(np.float32)

        # Action selection: argmax of visits (default) OR sample from the
        # target distribution. Sampling explores mixed strategies — for an
        # imperfect-information game like Compile, the Nash equilibrium is
        # generally mixed, so deterministic argmax play is exploitable.
        # Use sample_action=True during training for diverse trajectories.
        if sample_action and target.sum() > 0:
            # Optionally sharpen / smooth via temperature on the target.
            shaped = np.asarray(target, dtype=np.float64)
            t = float(max(sample_temperature, 1e-3))
            if abs(t - 1.0) > 1e-6:
                shaped = np.power(np.maximum(shaped, 0.0), 1.0 / t)
            # Strict re-normalize for np.random.choice (it checks sum=1
            # within ~1e-9). We renormalize after clipping any negatives
            # introduced by float error, then patch the last positive
            # bin so the sum is exactly 1.0.
            shaped = np.maximum(shaped, 0.0)
            s = shaped.sum()
            if s <= 0:
                best_idx = int(np.argmax(total_visits))
            else:
                shaped /= s
                resid = 1.0 - shaped.sum()
                if abs(resid) > 0:
                    # Add the residual to the largest bin — harmless and
                    # keeps the distribution proper.
                    shaped[int(shaped.argmax())] += resid
                best_idx = int(self.np_rng.choice(len(target), p=shaped))
        else:
            best_idx = int(np.argmax(total_visits))
        if not return_value:
            return legal[best_idx], target
        v_root = (
            root_total_value_sum / root_visits_sum if root_visits_sum > 0 else nn_value
        )
        return legal[best_idx], target, float(v_root)

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

    # ------------------------------------------------------------------
    # Gumbel root search (Danihelka et al. 2022)
    # ------------------------------------------------------------------
    def _gumbel_search(self, root: _Node, perspective: int) -> np.ndarray:
        """Replace UCB at the root with Gumbel-Top-k action selection.

        At the root we sample one Gumbel(0, 1) noise per legal action
        and pick the action that maximises `g + log_prior + sigma(q)`
        at each sim step. Below the root we use standard PUCT. This
        guarantees the resulting visit distribution is a policy
        improvement (vanilla AZ does not).

        Returns the Gumbel noise draws (one per root action) so the
        caller can compute the improved-policy target.
        """
        n = len(root.untried_actions)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        # One Gumbel draw per root action (held constant across sims
        # of this determinization).
        gumbel = self.np_rng.gumbel(size=n).astype(np.float64)
        # Stash on the root so the inner selector can read it.
        root._gumbel = gumbel  # type: ignore[attr-defined]

        # Top-m candidate restriction (analogous to root_top_k for vanilla
        # AZ, but using the Gumbel-perturbed score so the tail isn't cut
        # by prior alone). 0 = no cap, use all legal.
        m = self.cfg.gumbel_n_candidates
        log_prior = _log_prior_full(root._priors, n)
        score_initial = gumbel + log_prior
        if m > 0 and m < n:
            order = np.argsort(-score_initial)[:m]
            allowed = set(int(i) for i in order)
        else:
            allowed = set(range(n))
        root._gumbel_allowed = allowed  # type: ignore[attr-defined]

        # Use the standard batched-sim loop; only the root selector
        # changes (handled in _select_child via the _gumbel attrs).
        self._search(root, perspective)
        return gumbel

    def _gumbel_root_select(self, node: _Node, perspective: int) -> tuple[int, Action]:
        """Root selection under Gumbel: pick the candidate action that
        maximises  g + log_prior + sigma(q_complete)."""
        n = len(node.untried_actions)
        log_prior = _log_prior_full(node._priors, n)
        gumbel = getattr(node, "_gumbel", np.zeros(n))
        allowed = getattr(node, "_gumbel_allowed", set(range(n)))

        # Sigma uses max visits across root children to normalise q to
        # the same scale as log-probs (paper section 4.1).
        max_visits = max(
            (c.n_visits for c in node.children.values()),
            default=0,
        )
        sigma = (
            (self.cfg.gumbel_c_visit + max_visits) * self.cfg.gumbel_c_scale
        )

        best_idx = -1
        best_score = -float("inf")
        decider = node.game.decider()
        for i, action in enumerate(node.untried_actions):
            if i not in allowed:
                continue
            child = node.children.get(i)
            if child is None or child.n_visits == 0:
                q = 0.0
            else:
                q = child.q()
                if decider != perspective:
                    q = -q
            score = gumbel[i] + log_prior[i] + sigma * q
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            # Defensive: no allowed action somehow — fall back to argmax over allowed by prior.
            best_idx = next(iter(allowed))
        return best_idx, node.untried_actions[best_idx]

    def _gumbel_improved_target(
        self, root: _Node, perspective: int, n_legal: int,
    ) -> np.ndarray:
        """Compute the Gumbel-improved policy target over the legal-action
        prefix. Formula (paper eq. 13):
            target(a) = softmax( log_prior(a) + sigma * q_complete(a) )
        where q_complete is the search-refined q for visited actions and 0
        (the value-head baseline) for unvisited. This is the AZ training
        target that's guaranteed-improving by construction."""
        log_prior = _log_prior_full(root._priors, n_legal)
        max_visits = max((c.n_visits for c in root.children.values()), default=0)
        sigma = (self.cfg.gumbel_c_visit + max_visits) * self.cfg.gumbel_c_scale
        q = np.zeros(n_legal, dtype=np.float64)
        decider = root.game.decider()
        for i in range(n_legal):
            child = root.children.get(i)
            if child is not None and child.n_visits > 0:
                qi = child.q()
                if decider != perspective:
                    qi = -qi
                q[i] = qi
        logits = log_prior + sigma * q
        logits -= logits.max()
        e = np.exp(logits)
        return (e / max(1e-9, e.sum())).astype(np.float32)

    def _apply_root_noise(self, root: _Node) -> None:
        """Mix Dirichlet noise into the root priors (AlphaZero technique).
        Off by default — only fires when cfg.dirichlet_eps > 0. Applied
        per-determinization, so each independent search gets its own
        noise draw."""
        eps = self.cfg.dirichlet_eps
        if eps <= 0 or root._priors is None:
            return
        n = len(root._priors)
        if n <= 1:
            return
        noise = self.np_rng.dirichlet([self.cfg.dirichlet_alpha] * n)
        root._priors = (1.0 - eps) * root._priors + eps * noise.astype(root._priors.dtype)

    def _apply_root_top_k(self, root: _Node) -> None:
        """Prune root's untried_actions to the top-k by policy prior and
        flatten the kept priors to uniform 1/k. No-op when cfg.root_top_k
        is 0 or already covers everything. Must be called BEFORE any
        simulation runs so children are still empty."""
        k = self.cfg.root_top_k
        if k <= 0 or root._priors is None:
            return
        n = len(root.untried_actions)
        if k >= n:
            return
        order = np.argsort(-root._priors[:n])[:k]
        order_sorted = sorted(int(i) for i in order)
        kept_actions = [root.untried_actions[i] for i in order_sorted]
        root.untried_actions = kept_actions
        root._priors = np.full(len(kept_actions), 1.0 / len(kept_actions),
                               dtype=root._priors.dtype)

    def _select_child(self, node: _Node, perspective: int) -> tuple[int, Action]:
        """PUCT selection. Returns (child_index_into_untried, action).

        Root-only round-robin guarantee: if `cfg.root_min_visits_per_action`
        is set and any root child has fewer than that many visits, pick
        it instead of running PUCT. Once every root action has cleared
        the floor, fall through to standard PUCT for the remainder of
        the sim budget.
        """
        # Gumbel root selection — replaces PUCT + round-robin + Dirichlet
        # at the root. The Gumbel noise + sigma*q formulation gives a
        # guaranteed-improving policy target even at low sim counts.
        if node.parent is None and getattr(node, "_gumbel", None) is not None:
            return self._gumbel_root_select(node, perspective)

        # Round-robin floor at the root. `node.parent is None` is the
        # root in this design (every search rebuilds its tree from a
        # fresh root).
        if (
            node.parent is None
            and self.cfg.root_min_visits_per_action > 0
            and node.untried_actions
        ):
            floor = self.cfg.root_min_visits_per_action
            for i, action in enumerate(node.untried_actions):
                child = node.children.get(i)
                n_i = child.n_visits if child else 0
                if n_i < floor:
                    return i, action

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

    def _search(self, root: _Node, perspective: int) -> None:
        """Run `cfg.sims_per_determinization` PUCT trajectories from root.

        Batches up to `cfg.batch_size` leaves' network evaluations into
        a single forward pass. Virtual loss is applied along each path
        during the batch so subsequent sims see those edges as visited
        and explore elsewhere; it's reverted before real-value backprop
        so the final tree statistics are correct.

        batch_size=1 reproduces the serial one-sim-at-a-time path.
        """
        sims_remaining = self.cfg.sims_per_determinization
        batch_size = max(1, self.cfg.batch_size)
        while sims_remaining > 0:
            n_this = min(batch_size, sims_remaining)
            pending: list[_LeafResult] = []
            # Phase 1: walk K trajectories to their leaves, applying
            # virtual loss as we go so the batch fans out.
            for _ in range(n_this):
                leaf = self._select_to_leaf(root, perspective)
                if leaf.mode == "failed":
                    # Engine refused the action; back up zero and move on.
                    # No virtual loss was applied for failed leaves.
                    self._backprop(leaf.path, 0.0, perspective)
                    sims_remaining -= 1
                    continue
                self._apply_virtual_loss(leaf.path)
                pending.append(leaf)
                sims_remaining -= 1

            # Phase 2: batched network eval for leaves needing it.
            needs_net = [l for l in pending if l.mode in ("expand", "mid_effect")]
            net_results: dict[int, tuple[np.ndarray, float]] = {}
            if needs_net:
                results = self._batched_policy_and_value(needs_net)
                for i, r in zip([id(l) for l in needs_net], results):
                    net_results[i] = r

            # Phase 3: revert virtual loss + real backprop for each leaf.
            for leaf in pending:
                self._revert_virtual_loss(leaf.path)
                if leaf.mode == "terminal":
                    value = leaf.terminal_value  # already in perspective POV
                else:
                    priors, raw_value = net_results[id(leaf)]
                    if leaf.mode == "expand":
                        leaf.leaf.untried_actions = list(leaf.legal)
                        leaf.leaf._priors = priors
                    # Network value is from the leaf's current decider's
                    # POV; convert to perspective POV.
                    if leaf.leaf.game.decider() != perspective:
                        value = -raw_value
                    else:
                        value = raw_value
                self._backprop(leaf.path, value, perspective)

    def _select_to_leaf(self, root: _Node, perspective: int) -> "_LeafResult":
        """Walk from root via PUCT until reaching a leaf needing
        evaluation. Returns a _LeafResult describing what kind of
        evaluation the leaf needs, so the caller can either batch it
        with siblings or finalize immediately."""
        node = root
        path: list[tuple[_Node, int]] = []
        depth = 0
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
            # Expand a new child: apply action on a clone, drain any
            # in-flight effect choices to a top-level decision point.
            child_game = copy.deepcopy(node.game)
            try:
                child_game.step(action)
                self._drain_mid_effect(child_game)
            except Exception:
                # Determinization produced an inconsistent state — engine
                # refused the action. Caller backs up zero with no virtual
                # loss applied.
                return _LeafResult(path=path, leaf=None, mode="failed",
                                    legal=[], terminal_value=0.0)
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
            break

        if node.is_terminal:
            w = node.game.state.winner
            if w is None:
                tv = 0.0
            elif w == perspective:
                tv = 1.0
            else:
                tv = -1.0
            return _LeafResult(path=path, leaf=node, mode="terminal",
                                legal=[], terminal_value=tv)

        legal = node.game.legal_actions()
        if not legal:
            node.is_terminal = True
            return _LeafResult(path=path, leaf=node, mode="terminal",
                                legal=[], terminal_value=0.0)
        if _mid_effect(node.game) and self.cfg.simulate_choice_with_argmax:
            return _LeafResult(path=path, leaf=node, mode="mid_effect",
                                legal=legal, terminal_value=0.0)
        return _LeafResult(path=path, leaf=node, mode="expand",
                            legal=legal, terminal_value=0.0)

    def _batched_policy_and_value(
        self, leaves: list["_LeafResult"],
    ) -> list[tuple[np.ndarray, float]]:
        """Stack K leaves' state+action encodings into one forward pass.
        Returns list of (priors_over_leaf_legal, value) per leaf.

        Each leaf's `legal` length can differ, but `encode_actions` pads
        all of them to MAX_ACTIONS=32 so we can stack along dim 0 directly.
        Priors are returned renormalized over the leaf's legal prefix.
        """
        # Encode all leaves serially (cheap numpy work). The expensive
        # part is the single batched torch forward below.
        state_dicts = []
        raws, cards, protos, extras, masks, n_legals = [], [], [], [], [], []
        for l in leaves:
            persp = l.leaf.game.decider()
            s = encode_state(l.leaf.game, persp)
            raw, card_ids, proto_ids, extra_card_ids, mask = encode_actions(
                l.leaf.game, l.legal, persp,
            )
            state_dicts.append(s)
            raws.append(raw)
            cards.append(card_ids)
            protos.append(proto_ids)
            extras.append(extra_card_ids)
            masks.append(mask)
            n_legals.append(len(l.legal))
        # Stack into [B, ...] tensors.
        keys = list(state_dicts[0].keys())
        batched_state = {
            k: torch.from_numpy(np.stack([sd[k] for sd in state_dicts])).to(self.device)
            for k in keys
        }
        ar = torch.from_numpy(np.stack(raws)).to(self.device)
        ac = torch.from_numpy(np.stack(cards)).to(self.device)
        ap = torch.from_numpy(np.stack(protos)).to(self.device)
        ae = torch.from_numpy(np.stack(extras)).to(self.device)
        am = torch.from_numpy(np.stack(masks)).to(self.device)
        with torch.no_grad():
            logits, values = self.model(batched_state, ar, ac, ap, ae, am)
        logits_np = logits.detach().cpu().numpy()  # [B, MAX_ACTIONS]
        values_np = values.detach().cpu().numpy()  # [B]
        out: list[tuple[np.ndarray, float]] = []
        for i, n_legal in enumerate(n_legals):
            n = min(n_legal, logits_np.shape[1])
            sub = logits_np[i, :n]
            sub = sub - sub.max()
            expd = np.exp(sub)
            probs = expd / max(1e-9, expd.sum())
            out.append((probs, float(values_np[i])))
        return out

    def _apply_virtual_loss(self, path: list[tuple[_Node, int]]) -> None:
        """Temporarily make this path look less attractive to subsequent
        sims in the same batch. Sign is uniform across the path — the
        negamax-correct version alternates by decider, but at small batch
        sizes the root-level diversification (where the sign is correct
        anyway) dominates. Reverted by `_revert_virtual_loss` before real
        backprop."""
        loss = self.cfg.virtual_loss
        if not path:
            return
        path[0][0].n_visits += 1
        path[0][0].total_value -= loss
        for parent, child_idx in path:
            child = parent.children.get(child_idx)
            if child is not None:
                child.n_visits += 1
                child.total_value -= loss

    def _revert_virtual_loss(self, path: list[tuple[_Node, int]]) -> None:
        loss = self.cfg.virtual_loss
        if not path:
            return
        path[0][0].n_visits -= 1
        path[0][0].total_value += loss
        for parent, child_idx in path:
            child = parent.children.get(child_idx)
            if child is not None:
                child.n_visits -= 1
                child.total_value += loss

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


def _log_prior_full(priors: np.ndarray | None, n: int) -> np.ndarray:
    """Build a length-`n` log-prior vector that handles the priors-shorter-
    than-untried-actions case. The policy head only outputs MAX_ACTIONS
    logits (32), but the engine can occasionally return more legal actions
    than that; for those tail actions we use a uniform 1/n fallback prior,
    matching how `_select_child` handles the same edge case for PUCT."""
    out = np.full(n, math.log(1.0 / max(1, n)), dtype=np.float64)
    if priors is None or len(priors) == 0:
        return out
    n_p = min(n, len(priors))
    out[:n_p] = np.log(np.maximum(priors[:n_p].astype(np.float64), 1e-12))
    return out


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
