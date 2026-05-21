# MCTS Distillation — Pipeline + Findings

Status: working pipeline shipped; first iteration regressed vs Greedy. Sweeps and recipe adjustments are open work.
Last revised: 2026-05-21.

## 1. Why this exists

PPO produced [Sparkv1](model-card-sparkv1.md) and then a higher-entropy "loose-policy"
iter 500 sibling, both topping out around 76% vs Greedy. MCTS at inference time can lift
the agent's win rate substantially (see §3), but inference search is impractical for the
browser (10–150 ms budget). The standard way to bank that search advantage into a
shippable policy is **expert iteration (ExIt)**: run MCTS at training time, distill the
search-improved decisions back into the network, ship the policy alone.

This document describes the pipeline built for that loop and what one round of it
produced.

## 2. What was built

### 2.1 MCTS efficiency upgrades — `src/compile_engine/nn/mcts.py`

The IS-MCTS implementation gained four levers, each off by default to preserve the
pre-change behaviour at `batch_size=1, skip_search_top_prob=0`:

- **Batched leaf evaluation** (`MCTSConfig.batch_size`). Collects up to `batch_size`
  sims' leaves before dispatching one batched forward pass on MPS. Virtual loss
  applied along each in-flight path so subsequent sims fan out instead of
  collapsing onto a single trajectory. Measured **2.2× wall-clock speedup** at
  `batch=8` on the iter 410 model (21.3 → 9.6 s/game). Real ceiling is higher; the
  drag is from `_drain_mid_effect` still being serial.
- **Skip-when-confident** (`MCTSConfig.skip_search_top_prob`). Bypasses search
  entirely when the policy already exceeds the threshold, with the diagnostic
  showing zero MCTS/policy disagreements on `top_prob > 0.9` decisions. Saves
  ~10% of decisions cleanly.
- **Root top-k pruning** (`root_top_k`). Keeps only the top-k actions by prior at
  the root and flattens their priors to uniform `1/k`. Stops PUCT from locking
  onto the policy argmax when the loose-policy distribution still concentrates
  most mass on a small set.
- **Root round-robin floor** (`root_min_visits_per_action`). Guarantees every
  surviving root action gets at least N visits before PUCT takes over. Together
  with `root_top_k` this turns the search budget into "fair-share across top-k
  candidates → PUCT for the remainder," which was the regime where MCTS started
  meaningfully outperforming the policy.

### 2.2 Distillation target — `MCTSAgent.choose_with_target`

Returns `(action, soft_target_dist)` over `legal`. The action is `argmax(visits)`
as in vanilla `choose`. The soft target is the **visit-count distribution with
prior-weighted Laplace smoothing**:

```
target(a) ∝ visits(a) + α · prior(a)
```

`α = 1` ≈ "one virtual sim drawn from the prior" — keeps a small floor on actions
that got pruned by `root_top_k` rather than collapsing them to hard zero. The
`tau` argument is a sharpening exponent (`target ∝ score(a)^(1/tau)`), defaulting
to 1.0; pushing it toward 0 commits to argmax-by-visits.

An earlier soft-target formulation `softmax(log(prior) + τ·Q)` (see §3.2) was
tried first and abandoned — it produced 70× less KL signal than the visit-count
target at the same MCTS budget.

### 2.3 Distillation pipeline — `scripts/distill/`

Three components:

- `generate_labels.py` — runs self-play with the source policy stochastic for
  both seats; at every top-level non-mid-effect decision where the policy is
  *not* already confident (skip threshold), runs MCTS-with-target and records
  `(state, action_features, target_dist, mask)`. Writes a single `.pt` file.
- `train.py` — loads labels and the source checkpoint, freezes `value_head`,
  fine-tunes the rest with cross-entropy against the soft target. Tracks
  `KL(target ‖ model)` pre- and post-training as the health signal. Output
  checkpoint matches the format `scripts/eval/_lib.py` expects, so the standard
  eval pipeline runs on it unmodified.
- `run.sh` — orchestrates gen → train → eval. Outputs land under
  `<run-dir>/distill/<timestamp>/`.

### 2.4 Diagnostic — `scripts/eval/mcts_diagnostic.py`

Pre-existing; added `--batch-size` and `--skip-top-prob` flags. Now reports the
new instrumentation in its settings line and validates the levers end-to-end.

## 3. Empirical findings

### 3.1 MCTS at full budget beats the policy by +32 pp (iter 410 vs Greedy)

`runs/latest/eval/snapshot_00410/mcts/vs_greedy_v2_50g.{json,log}`. 50-game
diagnostic with `dets=8, sims=50, batch=8, top_k=5, min_visits=3, skip≥0.9`:

| Agent | WR vs Greedy | sec/game |
|---|---|---|
| MCTS (400 sims/move) | **86% (43/50)** | 29.0 |
| Policy argmax | 54% (27/50) | 0.24 |
| Δ matched-seed | **+32 pp** | — |

Sign test on matched-seed disagreements: 20/24 favored MCTS, **p ≈ 7.7 × 10⁻⁴**.
The 86% absolute is also above the original 50-game policy baseline of 76–78%,
confirming the gain isn't seed cherry-picking.

The two levers that did the most work were `root_top_k=5` and
`root_min_visits_per_action=3`. With them off, agreement with policy was 86%
and the matched-seed delta was a noisier +25 pp at n=20. With them on, agreement
collapsed to ~48% (PUCT now exploring rank-3/4/5 actions), and the WR gap
widened to the +32 pp above.

### 3.2 First ExIt iteration regressed vs Greedy

Two runs against `snapshot_00500`, both at the same MCTS budget. **First run**
used the soft-target formulation `softmax(log(prior) + τ·Q)` with `τ=1`:

```
Pre-train  KL = 0.0179
Post-train KL = 0.0111
```

Almost no signal — `log(prior)` dominated; the `τ·Q` term shifted relative
rankings (which is enough to change the argmax MCTS returns) but barely moved
the distribution shape. Eval delta versus the source: −6 pp vs Greedy, +6 pp
vs Random (within 50-game noise band ≈ ±10 pp).

**Second run** switched to visit-count targets with α=1 Laplace smoothing:

```
Pre-train KL = 1.1238
  epoch 1: 0.7550
  epoch 2: 0.4550
  epoch 3: 0.3777
  epoch 4: 0.3406
  epoch 5: 0.3196
Post-train KL = 0.2959   (Δ −0.83)
```

70× more KL signal, training closed ~75% of the gap monotonically. But the
distilled checkpoint regressed:

| Matchup | iter 500 baseline | iter 500 distilled | Δ |
|---|---|---|---|
| vs Greedy | 76% (38/50) | **64% (32/50)** | **−12 pp** |
| vs Random | 92% (46/50) | 94% (47/50) | +2 pp |

The −12 pp is borderline significant at n=50 (95% CI ≈ ±13 pp) but the direction
is clear.

### 3.3 Probable causes of the regression

Three suspects, ranked by likelihood:

1. **Overshoot in training.** KL diminished sharply after epoch 2. Five epochs
   at lr=1e-4 likely drifted the trunk past the labels' actual confidence. The
   cheap recovery is an epoch sweep from the saved labels (no need to redo the
   ~55-minute label gen).
2. **Simulator/opponent mismatch.** MCTS labeling uses the model as the
   opponent (self-play), but eval is vs Greedy. The distillation pushed the
   policy toward "moves that beat the model" — which is not the same as "moves
   that beat Greedy." This is the same gap that makes MCTS at *inference* less
   useful than MCTS as a *training signal* (see "When scaling sims actually
   helps" in conversation 2026-05-21).
3. **Frozen value head + drifted trunk.** `value_head` was held fixed while
   the trunk it consumes shifted (necessarily, to absorb the policy change).
   Value-head outputs are now miscalibrated — but this only matters if MCTS
   runs at inference on the distilled model. For pure-policy inference (the
   shippable artifact) this is a non-issue.

## 4. Open work

Roughly ordered by effort/cost:

1. **Epoch sweep on saved labels.** Re-train `runs/latest/distill/20260521-131722/labels.pt`
   at epochs ∈ {1, 2, 3} and re-eval each. ~5 min per variant. Will tell us
   whether suspect #1 (overshoot) is the dominant failure mode.
2. **Mix Greedy/Random into self-play during label gen.** Address suspect #2
   by exposing the labeler to opponent distributions the model will actually
   face. Could be a `--opp-mix greedy:0.3,random:0.1,self:0.6` knob on
   `generate_labels.py`.
3. **Joint policy+value training.** Unfreeze `value_head`; add a value-bootstrap
   loss using the search-improved value at the root (visit-weighted mean Q).
   Addresses suspect #3 and aligns with standard AlphaZero loss form.
4. **Tree reuse across moves in MCTS.** Free 2× on sim budget at inference time.
   Doesn't affect the training loop but improves all diagnostics.
5. **Gumbel root selection** (Danihelka et al. 2022). Provable policy
   improvement at 16–50 sims per move instead of 400. ~5× sim-budget reduction
   for label gen. Moderate code change at the root-level selection only.

## 5. File map

| Path | Purpose |
|---|---|
| `src/compile_engine/nn/mcts.py` | IS-MCTS agent. `MCTSConfig`, `MCTSAgent.choose`, `choose_with_target`. |
| `scripts/eval/mcts_diagnostic.py` | A/B vs policy argmax + rank/agreement breakdown. |
| `scripts/distill/generate_labels.py` | Self-play + MCTS labeling → `.pt` file. |
| `scripts/distill/train.py` | Cross-entropy distill against soft targets; saves a standard snapshot. |
| `scripts/distill/run.sh` | End-to-end orchestrator (gen → train → eval). |
| `runs/<run>/distill/<ts>/` | Per-pipeline outputs (gitignored): `labels.pt`, `*_distilled.pt`, `eval/`, logs. |
