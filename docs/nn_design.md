# Compile NN Agent — Design

Status: design doc preceding implementation in [src/compile_engine/nn/](../src/compile_engine/nn/).
Last revised: 2026-05-20.

## 1. Goals

1. **Laptop-trainable.** All hyperparameters chosen so a single training run completes in a few hours on a modern laptop (M-series CPU / MPS or modest x86 CPU). Single-machine, single-process inference; rollouts may parallelise via `multiprocessing`.
2. **Interpretable evaluation.** The value head outputs a scalar in `[-1, +1]` representing the active player's expected outcome. `(V + 1) / 2` reads as win probability for the side to move; the same number tracked across a game gives the eval bar; per-move drops give the blunder list.
3. **Plays both seats.** A single network parameterises the policy for both players via perspective-relative observation encoding. Self-play through an opponent pool of past snapshots.
4. **Faithful to imperfect information.** The observation never leaks information that a human at the table couldn't see: opponent hand contents and opponent face-down identities are encoded as anonymous tokens. The agent learns to play under partial information just like a person does.

Non-goals for v1: MCTS, distributed training, exotic architectures (transformers, GNNs), or beating a strong human. v1 targets "comfortably beats `GreedyAgent`" as the first concrete milestone.

## 2. Architecture overview

```
                   ┌──────────────────────────────────┐
                   │  game state (perspective-relative)│
                   │  + list of legal Actions          │
                   └────────────────┬─────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                                           ▼
      ┌──────────────┐                          ┌────────────────────┐
      │ state encoder│                          │  per-action encoder │
      │  → s ∈ R^256 │                          │  → a_i ∈ R^256      │
      └──────┬───────┘                          └─────────┬──────────┘
             │                                            │
             ├──────────────► value head: V = tanh(W·s)   │
             │                                            │
             └──────────┐                                 │
                        ▼                                 │
              policy logits ℓ_i = ⟨s, a_i⟩  ◄─────────────┘
                        │
                        ▼
              masked softmax over legal actions
```

Trunk and heads are small MLPs. The "dot-product pointer policy" (used in e.g. Pointer Networks and several AlphaStar-era agents) is the right fit because the legal action set is variable per state.

## 3. Observation encoding

The observation is built from `Game.state` viewed from the **perspective player** (the agent whose turn it is — `Game.decider()`). Side 0 in the encoded tensors is always "me", side 1 is always "opponent". This makes the network seat-invariant.

### 3.1 Card tokens

A learned embedding `nn.Embedding(92, d_card=32)` over a vocabulary of:

- `0`: `PAD` — empty slot in a stack or hand.
- `1`: `HIDDEN` — face-down card on the opponent's side of the field, or any card in the opponent's hand. The agent cannot see its identity.
- `2..91`: the 90 specific card def_ids (offset by 2). Used for all face-up field cards, all cards in trash (trash is public face-up), and all cards in our own hand and on our own face-down field positions.

The Compile rules explicitly grant a player visibility into their own face-down cards after placing them — so on the perspective side we use the real def_id for face-down cards too. On the opponent side, face-down cards are `HIDDEN`.

A protocol embedding `nn.Embedding(16, d_proto=16)` indexes 1..15 (the 15 protocols) plus 0 for unknown / not-drafted.

### 3.2 Field tensor

`field: [3 lines, 2 sides, MAX_STACK=10, per_card_feats]` with `per_card_feats = (token_id, face_up, committed, position_index)`.

Stacks shorter than `MAX_STACK` are padded with `(0, 0, 0, 0)`. Side 0 is me, side 1 is opponent (after perspective swap). The position index encodes whether the card is on top (uncovered) or below — important for committed semantics and for predicting which card a future "Flip 1 card" effect can target.

### 3.3 Protocol tensor

`protocols: [2 sides, 3 lines, 2 feats]` where the two features per (side, line) are `(protocol_id, compiled_flag)`.

### 3.4 Hand tensor

`hand: [MAX_HAND=12]` of card token IDs, **for the perspective player only**. Padded with `0`.

`opp_hand_size` is a scalar.

### 3.5 Trash counts

`trash: [2 sides, 90 def_ids]` — integer count of each card def in each player's trash. Trash is public face-up information.

### 3.6 Per-side computed scalars

`line_values: [3 lines, 2 sides]` — current computed line value (post-modifiers) from `effects.compute_line_value`. Pre-computing this is a cheap form of hand-crafted feature engineering that saves the network from re-deriving Apathy/Metal/Darkness modifiers.

### 3.7 Game scalars

A small dense vector of:

- `turn / max_turns` (normalised)
- `control_flag` ∈ {-1, 0, +1} (opp / neutral / me)
- `cannot_compile_me`, `cannot_compile_opp` (each 0/1)
- `include_expansion` (0/1)
- `phase` one-hot over the 9 `Phase` values
- `is_my_turn` (0/1) — usually 1 at the agent's decision point, but can be 0 when a card effect prompts a player whose turn it isn't (rare)

### 3.8 Total state input dim

- field: `3 × 2 × 10 × 4 = 240` raw values, expanded via card-embedding lookup to `3 × 2 × 10 × (32 + 3) = 2100` then aggregated per (side, line) by a small Set-aggregator (mean + max).
- protocols: `2 × 3 × (16 + 1) = 102`
- hand: `12 × 32 = 384`
- trash: `2 × 90 = 180`
- line values: `3 × 2 = 6`
- game scalars: ~15
- opp hand size: 1

After per-stack aggregation, the concatenated dense vector is roughly **~1,000–1,200 features**, fed into the trunk MLP.

## 4. Action encoding

Each legal `Action` is encoded as a fixed-size feature vector. For a list of N legal actions we build `[N, action_input_dim]`, then map each row through a small MLP to a `[N, hidden]` matrix that is dot-producted against the state hidden.

Per-action features (concat):

| Field | Size | Source |
|---|---:|---|
| action_type one-hot | 10 | `ActionType` enum |
| hand_index normalised | 1 | `Action.hand_index / 12` or `-1` |
| src_line one-hot (+none) | 4 | `Action.line_index` |
| dst_line one-hot (+none) | 4 | `Action.choice_index` for SHIFT_OWN_CARD; else `none` |
| choice_index normalised | 1 | `Action.choice_index / 32` or `-1` |
| card_def embedding | 32 | the card token implied by the action (e.g., the hand card being played, the target card in a CHOOSE_TARGET, the protocol's card pool) — `PAD` if N/A |
| protocol embedding | 16 | for DRAFT_PROTOCOL |
| target meta one-hot | 6 | for CHOOSE_TARGET: target type — `field_card / hand_card / line_idx / int / str / sentinel / none` |

Total per-action dim ≈ 74. The action MLP projects this to the hidden dim (256).

The trickiest case is `CHOOSE_TARGET` where the target's structure varies by the currently-pending `Choice`. The `target_meta` one-hot lets the network tell apart "flip target X" from "discard card index Y" from "shift to line Z" even though they share the same `ActionType`. When the target is a field card, we additionally fill the `card_def` slot with that card's embedding.

The action list is padded to `MAX_ACTIONS=32` with a boolean mask. The mask is `True` for real actions, `False` for padding; masked entries get `-1e9` logit before softmax.

## 5. Network

```python
class PolicyValueNet(nn.Module):
    card_emb:  Embedding(92, 32, padding_idx=0)
    proto_emb: Embedding(16, 16, padding_idx=0)

    state_trunk: 2-layer MLP (state_dim → 512 → 256), LayerNorm, ReLU
    action_mlp:  2-layer MLP (action_dim → 128 → 256), LayerNorm, ReLU
    value_head:  Linear(256 → 64) → ReLU → Linear(64 → 1) → Tanh

    forward(state_feats, action_feats, action_mask) →
        logits:  state @ action.T masked by action_mask
        value:   scalar in [-1, +1]
```

Total parameter count: ~250k. This is intentionally small — comfortably trainable on CPU.

Initialisation: PyTorch default (Kaiming for linear, normal for embeddings) is fine. Final layer of the policy is the dot product (no learned weights), so no special init needed there. Value head's final linear is initialised with `gain=0.01` to start the value bar near zero, which keeps PPO updates well-behaved at the beginning of training.

## 6. PPO training

Standard PPO recipe (Schulman et al. 2017), single-machine:

- **Rollout collector**: plays games using the *current* model as agent and an *opponent* sampled from an opponent pool. Records `(state_tensors, action_features, action_mask, action_idx, log_prob_old, value_estimate, reward, done)` at every decision the agent makes (one per call to `Game.legal_actions` where `Game.decider() == agent_side`).
- **Reward**: the env's existing shaped reward (per-step `Δ compiled_protocols`) plus the terminal `+1 / −1`. Optional `−1e-3` per turn to discourage stalling once we observe stalling behaviour empirically.
- **Returns / advantage**: Generalised Advantage Estimation (GAE) with `γ=0.99`, `λ=0.95`. Returns are bootstrapped from the value head at the final step (or `0` if terminal).
- **Loss**:
  ```
  L = -E[ min( ratio * A, clip(ratio, 1-ε, 1+ε) * A ) ]    # clipped surrogate
      + c_v * E[ (V_θ(s) - R)^2 ]                          # value MSE
      - c_e * E[ entropy(π_θ(·|s)) ]                       # entropy bonus
  ```
  with `ε=0.2`, `c_v=0.5`, `c_e=0.01`. Standard everything.
- **Optimisation**: Adam, lr=3e-4, no schedule for v1.
- **Batch update**: collect ~16 games (~2k transitions) per iteration, run 4 epochs of minibatch SGD (batch=256), gradient clip 0.5.

Training loop pseudocode:

```python
opponent_pool = [RandomAgent(), GreedyAgent()]
for iter in range(num_iters):
    transitions = []
    for _ in range(games_per_iter):
        opp = sample_opponent(opponent_pool)
        transitions += play_one_episode(net, opp, agent_seat=random_seat())
    compute_gae(transitions, gamma, lam, last_value=net.value(...))
    for epoch in range(ppo_epochs):
        for mb in minibatches(transitions, batch_size=256):
            optimise_ppo_step(net, mb, optim, clip=0.2, c_v=0.5, c_e=0.01)
    if iter % snapshot_every == 0:
        opponent_pool.append(snapshot(net))
        log_eval(net, vs=[RandomAgent(), GreedyAgent(), prev_snapshots])
```

### 6.1 Two-player handling

For each rollout game we randomly assign which seat the **current network** plays; the other seat is played by the sampled opponent (from the pool). We only record transitions from the current-network seat. This avoids the off-policy headache of using current-network experience from BOTH sides while a non-stationary partner is acting.

Once the network is comfortably beating random/greedy, the pool will contain past snapshots — true self-play emerges as snapshots accumulate.

### 6.2 Decision-point semantics

Compile's "turn" is many decisions: macro turn actions (play/refresh) and micro effect target picks. We treat each call where `Game.decider()` returns the agent's seat as one transition. The reward at a transition is the env's per-step reward delivered when the agent's `step` returns (so target picks inside one card's effect typically receive 0 immediate reward, with the eventual win/loss credit assigned via GAE bootstrap).

This is the right granularity: it gives the agent control over every choice the engine actually presents to a human player, including which target to flip and whether to take the optional Speed 2 shift action.

## 7. Interpretability

Three output artefacts, all derived from forward passes of the trained net:

1. **Eval bar.** At any state, `(V(s) + 1) / 2` is the predicted win probability for the side to move. Plot this across the turns of a played game → the eval bar.
2. **Move analysis.** For each agent decision in a recorded game, compute `Δ = V(s_after) − V(s_before)` from the *agent's* perspective. Negative `Δ` ranks the move as a mistake; the magnitude approximates "lost win probability".
3. **Alternate-line recommendation.** At each decision, take `argmax` (or top-k) of the policy logits over legal actions; if the human picked a non-top-k action, surface the alternative with its eval delta. This is the chess-engine "you should have played X" output.

Calibration: the value head is trained against `±1` terminal outcomes (plus the shaping). Empirically PPO value heads are reasonably well-calibrated on simple games; if needed, add a post-hoc Platt-scaling step before exposing the eval bar to humans.

## 8. Evaluation

Three numbers, run every snapshot:

- **WR_random** — 400-game match (alternating seats) vs `RandomAgent`. Should saturate near 100% within a small number of iterations.
- **WR_greedy** — 400-game match vs `GreedyAgent` (current heuristic). The primary learning signal during early training. Target: > 70%.
- **Elo vs pool** — round-robin 200-game tournaments against the last 5 snapshots. The Elo curve is the long-run learning signal.

Plus a curated **regression suite** of hand-built positions where we know the right call:

- "P0 about to compile L1; P1 holds Hate 2 in hand" → P1 should be valued near +/-0 (Hate 2 wipes P0's L1).
- "Both players have 2 compiled protocols, P0's turn, lines tied" → V ≈ 0.
- "P0 has 3 compiled" → V = +1 (game already won).

These positions stress-test the value head's understanding more thoroughly than win-rate vs baseline can.

## 9. File layout

```
src/compile_engine/nn/
  __init__.py
  encoder.py       # game state + actions → tensors
  model.py         # PolicyValueNet
  agent.py         # NNAgent(Agent) — wraps the model in our existing Agent protocol
  buffer.py        # rollout buffer + GAE
  train.py         # PPO training loop
scripts/
  train_nn.py      # entry point: python scripts/train_nn.py --iters N --device mps
  eval_nn.py       # entry point: python scripts/eval_nn.py --ckpt path --opponent greedy
  review_game.py   # replay a game with eval bar + blunder list
tests/
  test_nn.py       # encoder shapes, model forward, PPO loss is finite, 1-iter trains
```

`torch` is opt-in: `pip install -e .[nn]` installs torch + numpy.

## 10. Default hyperparameters

| Param | Default | Notes |
|---|---:|---|
| `MAX_STACK` | 10 | per-side per-line stack cap |
| `MAX_HAND` | 12 | perspective player's hand cap |
| `MAX_ACTIONS` | 32 | legal action cap (well above observed maxes) |
| `d_card` | 32 | card embedding dim |
| `d_proto` | 16 | protocol embedding dim |
| `hidden` | 256 | trunk hidden dim |
| `games_per_iter` | 32 | rollout games per PPO update (raised from 16 after observing high update variance) |
| `ppo_epochs` | 4 | passes over rollout per update |
| `batch_size` | 256 | minibatch size in PPO update |
| `lr` | 1e-4 | Adam (lowered from 3e-4 after observing KL-spike runs) |
| `γ` | 0.99 | discount |
| `λ` | 0.95 | GAE |
| `clip ε` | 0.2 | PPO clip |
| `target_kl` | 0.03 | break PPO epoch loop when K3 approx-KL exceeds this; `None` disables |
| `c_v` | 0.5 | value loss weight |
| `c_e` | 0.01 | entropy bonus |
| `grad clip` | 0.5 | global L2 norm |
| `snapshot_every` | 10 iters | how often to checkpoint + eval (pool addition is gated, see below) |
| `pool_threshold_wr_random` | 0.7 | minimum WR-vs-random to add a snapshot to the opponent pool — checkpoints are always saved regardless |
| `device` | auto | mps if available, else cpu |

### KL early-stop and approx-KL estimator

PPO update uses the **K3 estimator** of approximate KL,

```
kl ≈ E_x~D[(r(x) - 1) - log r(x)]      where r(x) = π_new(a|x) / π_old(a|x)
```

(Schulman, *Approximating KL Divergence*, 2020). This estimator is non-negative and lower variance than the naive `E[log π_old - log π_new]` used in early CleanRL/SB3 versions. When the per-minibatch K3 KL exceeds `target_kl`, we break out of the remaining epochs for this iteration — the standard PPO trust-region safety valve. The log line includes `stop@ep=N` to make it visible when this triggers.

### Opponent-pool snapshot gating

Snapshots are saved to `runs/<name>/snapshot_<iter>.pt` every `snapshot_every` iterations, **regardless of agent quality** (so we always have a recoverable history). They are added to the live opponent pool only once `WR_random ≥ pool_threshold_wr_random`. Without this gate, early in training the pool fills with weak imitations of the still-bad current policy and the rollout opponent distribution becomes dominated by symmetric self-play — informationally near-empty and slow.

## 11. Open questions and v2 ideas

- **Set/transformer encoder over the stacks.** v1 uses mean+max aggregation per stack; a small transformer over `(token, face_up, position)` triples would respect order and could attend across lines. Worth trying once v1 is working.
- **MCTS on top.** Once the policy/value net is competent, drop in a small MCTS (e.g. 100 sims/move) using the policy as prior and the value as leaf eval. Standard AlphaZero-lite. Multiplies playing strength considerably with no new training.
- **Recurrent state.** Compile decisions have short-range dependencies (within a turn), so a stateless agent is probably fine — but if effect-chain target picks need cross-decision memory, add a small GRU over the per-step encoder output.
- **Stronger opponent modelling.** v1's `HIDDEN` token for opponent hand contents discards a lot of information that's actually inferrable (we know which protocols they drafted, hence what their deck contains). A future encoder could include a "probabilistic hand" tensor — distribution over their possible holdings.
- **Self-play vs both seats.** v1 records only from the agent's seat to avoid mid-game policy drift. Once stable, double sample efficiency by recording from both seats with the current net.

## 12. Milestone definition

v1 is "done" when:

1. Smoke test passes: encoder → model → PPO step runs without error on a couple of games.
2. WR vs `RandomAgent` exceeds 95% within ~30 minutes of training on a laptop.
3. WR vs `GreedyAgent` exceeds 60% within ~2 hours of training.
4. `scripts/review_game.py` can replay any game and produce a sensible-looking eval graph.

Past that, v2 work (MCTS, self-play from both seats, transformer encoder) is justified.
