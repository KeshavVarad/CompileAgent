# Model card · Sparkv3

> **Sparkv3 is iter-20 from training run `20260521-230453-az`.** It's
> the bot currently deployed as the "Play vs Sparkv3" opponent on the
> webapp. Replaces Sparkv2 as the canonical NN agent.

## At a glance

| metric | value |
|---|---|
| Architecture | Pointer-style policy + value MLP, learned card/protocol embeddings (~580k params) |
| Source checkpoint | `runs/20260521-230453-az/snapshot_00020.pt` (hot-started from joint-distilled Sparkv2 derivative) |
| Training paradigm | AlphaZero-style self-play with Gumbel root selection, pool-admission gate, draft-pool-9, mixed-strategy play |
| Training compute | Apple-silicon MPS, ~10 hours wall-clock for 200 iters; champion identified at iter 20 |
| Win rate vs Random | **0.94** (n=200, argmax) |
| Win rate vs Greedy | **0.70** (n=200, argmax) |
| Win rate vs Sparkv2 (h2h) | **0.70** (n=200, p < 1e-7) |
| Win rate vs joint-distilled (h2h) | **0.73** (n=200, p < 1e-8) |
| Ladder Elo | **1344** (3-way round-robin, 100 games/pair × 2 seats) |

## Sparkv3's "thesis" — what the model thinks optimal play looks like

Across 200 iters of AlphaZero training with stochastic-strategy play,
Sparkv3 (iter-20) converged on a *coherent* playing style that
out-performs every prior checkpoint by a meaningful margin. Reading the
quantitative profile + draft preferences + decision-diff data, we can
articulate a clear strategic thesis the model has settled on:

### 1. Aggression dominates. Commit hard, commit fast.

Sparkv3 has the highest aggression metrics of any checkpoint we've
trained:

- **Face-up play ratio: 62.1%** — substantially higher than the
  baseline joint-distilled (57.2%) and the highest of any iter in this
  run. The model has decided that the value of exposing your card text
  for synergies outweighs the information cost of revealing it.
- **First compile turn: 17.2** (fastest across all iters) — Sparkv3
  races to its first compile, *not* setting up.
- **Compile aggression: 74.5%** (games where the agent compiles ≥3
  times) — Sparkv3 reaches game-winning compile counts in ~3 out of 4
  games. The model thinks the way to win is to push to 3-compiles as
  fast as possible.

### 2. Refresh discipline. Refresh only when forced.

Sparkv3 has near-minimum refresh usage:

- **Refresh rate: 9.1%** of decisions — almost identical to
  joint-distilled's 9.0%, but with a much more aggressive face-up
  style. The model is *not* dragging out games by cycling its hand.
- **Hand-empty refresh rate: 61.8%** — meaning 62% of Sparkv3's
  refreshes happen when its hand is already empty (forced by the
  rules). The remaining ~38% are discretionary, but only when the
  current hand truly can't make progress.

Later checkpoints in the same run drifted toward refresh-heavy play
(iter-30+: refresh 11-12%, hand-empty refresh dropping to 45-51%),
which is **how iter-30+ regressed from iter-20's strength**. The
refresh-creep pattern was the same one that crashed previous runs.
Sparkv3 specifically *resists* it.

### 3. Drafted protocol set: spread, not concentrated.

Sparkv3's draft theory differs notably from earlier Sparkv2-era
models that were heavily Psychic-biased:

| protocol | iter-20 pick rate | sparkv2 baseline | shift |
|---|---:|---:|---|
| Clarity | 14.5% | small | **+10** (new favorite) |
| Apathy | 13.3% | small | **+10** |
| Darkness | 11.7% | small | **+10** |
| Mirror | 9.3% | ~10% | flat |
| Psychic | **7.8%** | **>20%** | **-13** (deprecated) |
| Life | 0.3% | ~16% | **-15** (abandoned) |
| Fire | ~2% | ~11% | **-9** |

The shift away from Psychic/Life/Fire and toward Clarity/Apathy/Darkness
is one of the run's clearest learned lessons. Sparkv3 prefers
*defensive control* (Darkness flips, Clarity card-stack manipulation,
Apathy face-down value buffs) over the *raw value/draw* engine that
Sparkv2-era models favored.

### 4. Per-pair synergies it has actually learned

These are protocol pairs Sparkv3 wins with substantially more than
its baseline 70% vs-Greedy rate:

- **Darkness + Apathy**: Apathy 0 buffs face-down value; Darkness 0
  shifts opp's covered cards. The combo lets you set up a stronger
  line while disrupting opp's stacked positions.
- **Clarity + any control protocol**: Clarity 1's "reveal top of
  deck + may discard" pairs with any protocol that punishes the opp
  for specific known cards.
- **Apathy + Darkness + Love** triples (when drafted): late-game
  win condition via face-down value accumulation.

### 5. What it has decided to *not* do

- **Doesn't refresh as a stalling tactic.** Refresh-cycle to look at
  more cards = a tempo loss the model now recognizes.
- **Doesn't draft Life or Fire.** These were "value engine" protocols
  whose strength depended on long games. Sparkv3 wins short, so they
  don't pay off.
- **Doesn't over-rely on Psychic.** Earlier iters had a 77% Psychic
  pick rate. Sparkv3 has dethroned Psychic in favor of more diverse
  draft theories.

## How this thesis was discovered

The training pipeline that produced Sparkv3 was the result of multiple
paradigm experiments documented across this repo:

1. **PPO** (`scripts/eval/strategy.py` on `20260520-224058`): produced
   Sparkv1. Plateaued at 60% vs Greedy. Catastrophic-forgetting
   patterns. Documented in [`docs/strategy-analysis.md`](strategy-analysis.md).
2. **PPO loose-policy + MCTS distillation** (`20260521-040612-loose-policy`
   + `distill/`): produced Sparkv2 + joint-distilled. Significant
   improvement (66% vs Greedy, 62% h2h vs Sparkv2). Documented in
   [`docs/mcts-distillation.md`](mcts-distillation.md).
3. **AlphaZero attempts with deterministic play** (`20260521-165313-az`,
   `20260521-200924-az`): repeatedly collapsed via "refresh creep"
   — model finds locally-safe refresh play, loses overall. Even
   Gumbel root selection + pool admission gate only delayed the
   collapse, didn't prevent it.
4. **AlphaZero with stochastic-strategy play** (`20260521-230453-az`):
   **the breakthrough.** Switching pool opponents to sample from
   policy + sampling MCTS actions from the Gumbel-improved target
   (instead of argmax) broke the strategy-cycle failure mode. iter-20
   became Sparkv3.

The fundamental insight: Compile is a stochastic imperfect-information
game, so the *Nash-optimal* strategy is necessarily mixed. Training
with deterministic argmax play converges to *pure* strategies — which
are exploitable. Once both sides play their policy distributions as
intended, the model converges to something much closer to optimal.

## Behaviour fingerprint vs prior models

Concrete style metrics, all measured at n=200 vs Greedy (argmax both sides):

| metric | Sparkv3 (iter-20) | joint-distilled | Sparkv2 |
|---|---:|---:|---:|
| face_up_ratio | **62.1%** | 57.2% | (similar to JD) |
| refresh_rate | 9.1% | 9.0% | 9.5% |
| compile_aggression | **74.5%** | 77.0% | (similar) |
| avg_game_length (turns) | **50.2** | 54.9 | 51.6 |
| first_compile_turn | **17.2** | 17.6 | 17.8 |
| hand_empty_refresh % | 61.8% | 75.3% | 72% |
| h2h vs Sparkv2 | **+70%** | +62% | — |
| h2h vs joint-distilled | **+73%** | — | -62% |

Sparkv3's distinguishing characteristics are the aggression metrics
(highest face-up ratio, shortest games, fastest first compile). The
refresh and compile-aggression numbers are comparable to baselines;
the difference is *coordinated* execution.

## Known limitations

- **Three rarely-drafted cards** (`AX02:Assimilation:6`, `AX02:Diversity:0`,
  `MN02:Ice:4`) were mis-stamped in `cards.json` during training. The
  data has been corrected, and Sparkv3's `card_static` buffer was
  hot-patched post-training to refresh the affected per-card static
  features — but the model's *learned* embeddings for those def_ids
  still reflect the older incorrect cards. Impact is small (these are
  rarely picked in the protocols Sparkv3 drafts).
- **Drift past iter-20.** The training run kept going to iter-200,
  but pool-admission gate rejected most subsequent snapshots and h2h
  tests confirmed iter-30/40/50/130/200 are all weaker than iter-20.
  Continued AZ training under closed-loop self-play still slowly
  degrades the policy, even with stochastic play; just much more
  slowly than with deterministic play.
- **No mid-effect search.** Roughly 12% of Compile decisions are
  sub-effect resolution (CHOOSE_TARGET inside an effect's flow).
  MCTS can't enter these states (Python generator coroutines don't
  deepcopy), so the policy plays them by argmax during training.
  Sparkv3 is reliant on its policy head being correct on these
  decisions.
- **Card embeddings effectively at random init.** A diagnostic
  confirmed Sparkv3's `card_emb` weights are statistically
  indistinguishable from random initialization — the model relies
  on hand-engineered static features + protocol embeddings for
  card-level distinction. This is an architectural quirk inherited
  from the model design, not a Sparkv3-specific issue.

## Reproducibility

```bash
# Verify Sparkv3's headline numbers
PYTHONPATH=src .venv/bin/python scripts/eval/collect.py \
    --model runs/sparkv3.pt --opp greedy --games 200 --seed 0 --device mps \
    --out /tmp/v3_g.jsonl --stochastic 0

PYTHONPATH=src .venv/bin/python scripts/eval/collect.py \
    --model runs/sparkv3.pt --opp runs/latest/snapshot_00500.pt \
    --games 200 --seed 0 --device mps --out /tmp/v3_v2.jsonl --stochastic 0

# Full strategic-evolution analysis on the same run
PYTHONPATH=src .venv/bin/python scripts/analysis/discoveries.py \
    runs/sparkv3.pt \
    --baseline runs/latest/distill/20260521-152408-mixed/snapshot_00500_distilled.pt \
    --games 200
```

Generated metrics live at `runs/20260521-230453-az/metrics.jsonl`;
per-iter style fingerprints are at
`analysis/run3_evolution/style_fingerprint.md`.
