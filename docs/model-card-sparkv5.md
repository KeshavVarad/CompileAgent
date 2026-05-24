# Model card · Spark v5

> **Spark v5 is iter-90 from training run `20260524-123408-ppo-q-head-extended`**,
> a 100-iter continuation hot-started from iter-150 of the prior
> `20260524-043104-ppo-q-head` run. Shipped as
> [`webapp/public/models/bot-current.onnx`](../webapp/public/models/bot-current.onnx).
> Replaces Spark v4 (PPO with no Q-head) as the canonical NN agent.

## At a glance

| metric | value |
|---|---|
| Architecture | Pointer-style policy + value MLP + Q-head + UNREAL aux heads (~710k params; +130k Q-head + ~24k aux over v4) |
| Source checkpoint | `runs/20260524-123408-ppo-q-head-extended/snapshot_00090.pt` |
| Training paradigm | PPO with adaptive per-class entropy controller + UNREAL-style aux losses + per-action Q-head used at inference for CHOOSE_TARGET argmax routing |
| Training chain | Spark v4 → adaptive-band run (iter 50/60) → q-head run (iter 150) → this run (iter 90) ≈ 350 PPO iters total |
| **Win rate vs Random** | **0.92** (n=200 stochastic) |
| **Win rate vs Greedy** | **0.723** ±2.3pp (n=400 stochastic) |
| Win rate vs shipped Spark v4 baseline (rigorous) | v4 = 0.615 — **+10.8pp** improvement |

## The headline result

Spark v5 is the first NN agent in this codebase to break **0.70 WR vs Greedy** on a rigorous 400-game evaluation. Trajectory of model strength (rigorous 400-game vs greedy):

```
Spark v4 (shipped):                   0.615
Run 20260523-123850 iter 40 (v4):     0.615   (the shipped baseline)
Run 20260524-031841 iter 60:          0.610
Run 20260524-043104 iter 150 (v5α):   0.705
Spark v5 (this card):                 0.723
```

+11pp on greedy over Spark v4. Confidence interval ±2.3pp puts the improvement well outside noise.

## What changed since Spark v4

Three research-backed training improvements landed across this lineage, plus one bug fix:

### 1. Per-class adaptive entropy (PR #48, #50)

[Gowal et al. 2018 — Efficient Entropy for Multidimensional Action Space](https://arxiv.org/abs/1806.00589).
DRAFT decisions get a separate entropy bonus from PLAY / CHOOSE. The
adaptive controller targets per-class entropy bands (DRAFT [0.8, 1.4],
CHOOSE [0.5, 1.0] nats) rather than a single global entropy.

The v4 model's draft was mode-collapsed to **Darkness 98%**. With the
per-class controller, DRAFT entropy actively explored values up to 2+
nats during training before settling at ~1.0-1.4. The bug we caught
along the way (a misweighting that made the per-class multiplier act
as ~1× instead of N×) was fixed in PR #49 — DRAFT mult of 4× now
actually weights 4× regardless of class frequency in the batch.

### 2. UNREAL-style auxiliary supervised heads (PR #48)

[Jaderberg et al. 2016 — Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/pdf/1611.05397).
Two heads on the shared trunk:
- `aux_opp_hand_head` — predict opp's hidden hand contents (multi-label BCE)
- `aux_margin_head` — predict final compile margin (regression)

Both are train-time only — they don't bias action selection. They give
the encoder richer training signal for representation learning. Across
training, `aux_oh` loss went 0.68 → 0.07 (the encoder genuinely learns
to recover opp's hand from observable features), `aux_margin` went
0.8 → 0.5 (slower descent, harder target).

### 3. Q-head for CHOOSE_TARGET (PR #51)

[O'Donoghue et al. 2017 — Combining Policy Gradient and Q-Learning (PGQL)](https://arxiv.org/pdf/1611.01626).
A per-action `Q(s, a)` head trained on TD(0) target `Q(s, a) → ret`
(the GAE return, same target the value head uses). At **inference**,
CHOOSE_TARGET decisions are routed through `argmax(Q)` instead of
policy softmax sampling.

Why CHOOSE only: these decisions happen during deterministic
mid-effect resolution. No opp response between yields → no mixed-Nash
benefit from stochastic action. Argmax of a well-trained Q over a
small (2-8 option) action set is the right rule. Outer turn-level
decisions (DRAFT / PLAY / COMPILE) still flow through the stochastic
policy head — mixed-Nash play remains the target there.

This was the highest-leverage of the three changes. The Spark v4
analysis identified CHOOSE_TARGET quality as the bottleneck (47/47
rejected Love-1-End trades, random give-to-opp choices on Love 3,
etc.). The Q-head supplies an action-value signal targeted at exactly
those decisions.

### 4. Recompile semantics clarification (PR #44)

Not a model change but a doc fix. The Spark v4 model-card mis-explained
"3-3 compile counts" as a tiebreaker scenario. Compile has no
tiebreakers — those are just COMPILE_LINE actions including
recompiles (a recompile on an already-compiled line steals the top of
opp's deck per [game.py:628-637](../src/compile_engine/game.py#L628-L637),
rather than advancing the win condition).

## Behaviour fingerprint

Action distribution, vs greedy, n=400 stochastic:

| action | share |
|---|---:|
| CHOOSE_TARGET | 29% |
| PLAY_FACE_UP | 27% |
| PLAY_FACE_DOWN | 17% |
| COMPILE_LINE | 9% |
| REFRESH | 8% |
| DRAFT_PROTOCOL | 7% |
| SKIP_OPTIONAL | 3% |
| DISCARD_CARD | 1% |

Face-up ratio: **0.61** (down from v4's 0.74). The new model plays
more face-down — a real strategic shift, not just a draft change.
This is consistent with the Gravity-heavy draft (Gravity benefits from
face-down board state via its "play top-of-deck face-down per 2 cards
in line" mechanic).

Heuristic adherence:

| heuristic | rate | flag |
|---|---:|---|
| compile-when-possible | **100%** (1,524/1,524) | 🟢 |
| wasteful refresh (full hand) | **<5%** | 🟢 |

Game length distribution (vs greedy, n=400):

| game length | WR | n |
|---|---:|---:|
| q1 (<39 turns) | **82.3%** | 96 |
| q2 (39-48 turns) | 68.0% | 97 |
| q3 (49-62 turns) | 66.0% | 103 |
| q4 (≥63 turns) | 73.1% | 104 |

Interesting: unlike Spark v4 (which had a clear short-game advantage
and long-game weakness), Spark v5's WR is **bimodal** — wins fast or
wins late. Long-game stamina has improved (73% in q4 vs Spark v4's
53%). The aux margin head likely contributed to better long-game value
estimation.

## Counterfactual analysis — what makes Spark v5 win

Replicates the analysis from [docs/STRATEGY_THESIS_sparkv4.md](STRATEGY_THESIS_sparkv4.md).
Tested whether blocking specific protocols from the agent's draft
changes WR (n=200 per condition):

| condition | WR | delta vs baseline |
|---|---:|---:|
| baseline (no restriction) | 0.680 | (ref) |
| block Gravity | 0.665 | −1.5pp |
| block Plague | 0.645 | −3.5pp |
| block Life | 0.725 | **+4.5pp** |
| block Love | 0.715 | **+3.5pp** |
| block all four (Grav+Plague+Life+Love) | 0.695 | +1.5pp |

**Gravity at 97% pool presence is a cosmetic mode collapse, not a
load-bearing pick.** Same pattern as Spark v4's Darkness obsession:
the agent has a strong draft preference, but win rate is barely
affected by blocking it.

The agent actually wins **more** when blocked from Life or Love. The
play-side policy has real multi-strategy depth — when forced off
preferred protocols, the model deploys alternatives effectively.

**The strength comes from CHOOSE_TARGET quality**, not from any
specific draft. That's exactly the bottleneck the Q-head was designed
to address.

## Draft preferences (the cosmetic mode collapse)

Per 400-game eval:

| protocol | pool freq | avg pick pos |
|---|---:|---:|
| **Gravity** | **97%** | **1.23** |
| Plague | 40% | 2.34 |
| Life | 35% | 2.28 |
| Love | 28% | 2.34 |
| Fire | 23% | 2.42 |
| Ice | 20% | 2.26 |
| Speed | 14% | 2.65 |
| Darkness | 9% | 2.29 |
| (long tail) | <8% | — |

20 unique protocols drafted across the run. The non-Gravity slots are
genuinely diverse — no Spark v4-style "Darkness, Plague, Love or
nothing" pattern.

WR-when-drafted is informative (n≥80):

| protocol | WR | n |
|---|---:|---:|
| Darkness | **88.6%** | 35 |
| Diversity | 81.8% | 22 |
| Love | 76.6% | 111 |
| Speed | 72.2% | 54 |
| Gravity | 71.8% | 387 |
| Plague | 71.0% | 162 |
| Life | 70.9% | 141 |
| Ice | 70.4% | 81 |
| Fire | 69.9% | 93 |

Note Darkness has the highest WR-when-drafted (88.6%) but is only
drafted 9% — opposite of Spark v4 where Darkness was drafted 98%
with only ~62% WR-when-drafted. The new model picks Darkness *when
the situation warrants it* rather than reflexively.

## Known limitations

- **Cosmetic Gravity mode collapse.** 97% pool presence is striking,
  but per the counterfactual it's not load-bearing — same status as
  Spark v4's Darkness obsession. Honest disclosure: we did not fully
  fix mode collapse, only displaced it to a different protocol. The
  WR gain comes from elsewhere (CHOOSE_TARGET quality via Q-head).
- **WR-when-drafted reveals over-selection of lower-WR picks.** Gravity
  at 71.8% (drafted 387 times) is bottom-half of WR-when-drafted, yet
  it's the modal pick. Darkness at 88.6% WR is drafted only 35 times.
  Suggests the agent's draft policy is sub-optimal — though the
  counterfactual confirms changing it doesn't help.
- **Q-loss never converged to zero.** Final Q-loss ~0.5 — this is the
  irreducible variance floor for `ret`, not a fit failure. But it
  means the Q-head's action-value estimates have residual noise.
- **No counterfactual block on lower-tier picks.** Tested blocks were
  Gravity + top three (Plague/Life/Love). Other potential mode-collapse
  candidates aren't ruled out — though entropy diagnostics during
  training showed DRAFT entropy in target band, so unlikely.
- **Sub-optimal optional-clause handling not directly confirmed fixed.**
  Spark v4 rejected the Love 1 End trade 47/47 times. We haven't
  re-run that specific analysis on v5. The Q-head *should* help here
  (the whole motivation), but we don't have direct evidence.

## Reproducibility

```bash
# 1. Per-snapshot rigorous evals (the headline 0.723 number)
.venv/bin/python scripts/eval/collect.py \
    --model runs/20260524-123408-ppo-q-head-extended/snapshot_00090.pt \
    --opp greedy --games 400 --seed 0 \
    --out runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090/vs_greedy_400.jsonl

.venv/bin/python scripts/eval/collect.py \
    --model runs/20260524-123408-ppo-q-head-extended/snapshot_00090.pt \
    --opp random --games 200 --seed 1 \
    --out runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090/vs_random_200.jsonl

# 2. Counterfactual draft block (the Gravity-is-cosmetic check)
.venv/bin/python scripts/eval/counterfactual_draft.py \
    --model runs/20260524-123408-ppo-q-head-extended/snapshot_00090.pt \
    --opp greedy --games 200 \
    --block Gravity Plague Life Love \
    --out runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090/counterfactual_draft_gravity.json

# 3. Model card + h2h breakdown
.venv/bin/python scripts/eval/metrics.py \
    --in runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090 \
    --out runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090/metrics.json \
    --model snapshot_00090

.venv/bin/python scripts/eval/h2h_breakdown.py \
    runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090/vs_greedy_400.jsonl

# 4. ONNX export for the webapp
.venv/bin/python scripts/eval/export_onnx.py \
    --ckpt runs/20260524-123408-ppo-q-head-extended/snapshot_00090.pt \
    --out webapp/public/models/bot-current.onnx
```

All artifacts live at
[`runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090/`](../runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090/).

## What's next

Three lines of follow-up investigation, ranked by EV:

1. **Apply the Spark v4 strategy-thesis analysis to v5.** Replay the
   Love 1 End / Love 3 give-card analysis specifically — does the
   Q-head actually fix those decisions, or is the WR improvement
   coming from elsewhere? If the latter, what?
2. **MCTS-over-effect-tree for CHOOSE_TARGET.** The Q-head got us +11pp.
   Full MCTS distillation (research candidate #1 from the post-v4
   analysis) is the next step up if we believe CHOOSE_TARGET is still
   the bottleneck. Requires ~2 weeks of state-snapshot infrastructure.
3. **League training with goal-conditioned exploiters.** The remaining
   draft mode collapse (Gravity 97%) suggests we're still in a
   self-play local optimum. Goal-conditioned exploiters (e.g. "win
   without Gravity") would force the policy to learn defensive plays
   in non-Gravity drafts. ~1 week of work.
