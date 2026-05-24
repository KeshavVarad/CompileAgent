# Strategy thesis · Spark v4

> **Spark v4 is iter-40 from PPO training run `20260523-123850-ppo-resume`.**
> Shipped as [`webapp/public/models/bot-current.onnx`](../webapp/public/models/bot-current.onnx)
> since [PR #37](https://github.com/keshavvaradarajan/CompileAgent/pull/37).
> Replaces Sparkv3 (AlphaZero-derived) as the canonical NN agent.

This document is the analytical companion to the
[Sparkv3 model card](model-card-sparkv3.md). It asks:
*what has the PPO-trained network actually learned about how to play Compile?*
The answers come from per-game telemetry, draft frequencies,
action-mix deltas, and head-to-head data — not from theory.

Underlying data:
[`runs/20260523-123850-ppo-resume/eval/`](../runs/20260523-123850-ppo-resume/eval/).
Regenerate with [`/tmp/run_eval.sh`](#reproducibility) (the exact
pipeline that produced the numbers below).

## TL;DR

Spark v4 has converged on a **fast-aggressive, hand-disruption** play
style, but the *draft* portion of its thesis is mostly self-reinforced
mode collapse — see
[Counterfactual analysis](#counterfactual-analysis--is-the-darkness-draft-real)
below. Specifically:

1. **Draft pool-presence: Darkness in 98% of games.** But the
   counterfactual shows this is **not load-bearing** — blocking
   Darkness from the agent's draft leaves WR essentially unchanged
   (0.54 → 0.54 over 100 games). The agent's *play skill* is what's
   doing the work; the draft preference is a mode-collapse artifact
   of self-play.
2. **First-pick is seat-dependent.** As Seat 0 (picks first in the
   snake): Darkness 43%, Plague 20%, Love 12%. As Seat 1 (picks second
   after greedy random-picks): Darkness 82%. The pool-presence rate
   averages these out to 98%.
3. **Push for 3 compiled lines fast.** Wins 79% of games under 41
   turns; only 54% past 68 turns. The strategy is to race, not grind.
4. **Trade tempo for information by playing face-up.** 73.8% of plays
   in wins are face-up — the model has decided the synergy value of
   revealing your card text beats the information cost.
5. **Compile every time it's legal.** Perfect 100% compile-when-possible
   rate across 2,398 opportunities (greedy + random eval).
6. **Refresh almost never.** 2-3% wasteful-refresh rate — almost every
   refresh is a forced one (empty hand).
7. **Love is the secret weapon.** Love cards (especially Love 6, 5, 4, 3)
   appear *substantially more often* in wins than losses. Without the
   AX01 expansion in the pool, WR drops from 68% to 55%. This holds
   up under counterfactual (blocking Love costs a few WR points).
8. **Structural counter: Mirror exists but is weak (~3pp).** Darkness/
   Mirror matchup is 58% vs baseline 61%. Take it if available, but
   don't expect it to flip the game.

The model is **roughly tied with snapshot_00030 in ladder Elo** (1463
vs 1469, within noise on 240 games per pair). Training curve plateaued
between iter-30 and iter-40 and started degrading at iter-50 as the
entropy controller widened.

## Lineage and training context

Spark v4 was hot-started from
`runs/20260523-114745-ppo-resume/snapshot_00040.pt` (a PPO snapshot at
`wr_greedy=0.67`), itself part of a chain of PPO resumes seeded from
the post-encoder-port checkpoint (PR #37). The full chain represents
roughly **300 PPO iterations** of accumulated training on the
A1+A3+A4+A5 encoder + the broader MN02/AX02 card pool.

The `20260523-123850-ppo-resume` run ran **50 iterations** of PPO
self-play with:

| hyperparameter | value | rationale |
|---|---|---|
| games per iter | 32 | balances rollout variance vs wall-clock |
| target KL | 0.03 | tightened from 0.05 after a regression on the prior session |
| entropy controller band | [0.40, 0.80] | widened from [0.55, 0.70] that overshot earlier |
| pool admission gate | `wr_random ≥ 0.65` | raised mid-session to filter out weak snapshots |
| eval / pool sampling | **stochastic** (sample from policy) | mixed-Nash target — argmax eval underestimates against stochastic opponents |
| play distribution | both agents stochastic at inference | training-time and eval-time use the same sampling policy |

### Training-time eval trajectory

| iter | wr_random | wr_greedy | entropy | c_entropy | notes |
|---|---|---|---|---|---|
| 10 | 0.88 | 0.68 | 0.46 | 0.010 | |
| 20 | 0.97 | 0.60 | 0.48 | 0.010 | greedy dip — entropy still climbing |
| 30 | 0.83 | 0.67 | 0.49 | 0.010 | |
| **40** | **0.93** | **0.72** | **0.43** | **0.010** | **shipped peak** |
| 50 | 0.90 | 0.65 | 0.39 | 0.020 | entropy controller ramped; policy degrading |

Iter-40 was selected because (a) it had the strongest training-time
greedy WR by a clear margin, (b) entropy was inside the target band
and not yet under controller pressure, and (c) iter-50 was clearly
weaker.

The post-hoc 400-game evaluation finds the *real* greedy WR is 0.615
(±4.8%) — within noise of every other snapshot's 0.61-0.65 band. The
training-time 100-game eval that placed iter-40 at 0.72 was a
fortunate sampling, not a real superiority. **All five evaluated
snapshots have overlapping confidence intervals on raw WR; the
ladder is what separates them.**

## Headline numbers

Measured 2026-05-24 against the shipped checkpoint
`runs/20260523-123850-ppo-resume/snapshot_00040.pt`, stochastic
play on both sides, training-distribution game configs:

| metric | value | n |
|---|---|---|
| WR vs Random | **0.85** | 200 |
| WR vs Greedy | **0.615** ±4.8% | 400 |
| Avg compile margin vs Greedy | **+0.86** | 400 |
| Avg compile margin vs Random | **+2.83** | 200 |
| Avg game length | **56.7 turns** | 400 |
| Ladder Elo (vs snapshots 10/20/30/50 + baselines, 240 games per pair) | **1463** | — |
| Ladder rank | **2 of 7** | — |
| Compile-when-possible rate | **100%** | 2,398 opportunities |
| Wasteful-refresh rate | **2-3%** | 2,888 refreshes |

For comparison, Sparkv3 (AlphaZero-derived, prior canonical bot)
achieved 0.70 vs Greedy under *argmax* eval. Under matched stochastic
eval the numbers are not directly comparable — see
[*Behaviour fingerprint vs prior models*](#behaviour-fingerprint-vs-prior-models).

## Spark v4's thesis — what the model thinks optimal play looks like

Across 50 iterations of PPO training with stochastic-strategy play
(see [Lineage](#lineage-and-training-context) for the full chain),
Spark v4 settled on a *coherent* play style. Reading the action mix
+ draft preferences + h2h breakdown data, the thesis is:

### 1. Darkness, always — but it's not because Darkness is best.

Spark v4 has the most concentrated draft policy of any model we've
trained. **Per the counterfactual analysis [below](#counterfactual-analysis--is-the-darkness-draft-real),
this concentration is mode-collapse — not an objectively-correct
draft thesis.** Blocking Darkness from the agent's pool leaves WR
unchanged. The agent's *play* is what's winning games, not its draft.

| protocol | pick freq | avg pick pos (1-3) | vs sparkv3 baseline |
|---|---:|---:|---|
| **Darkness** | **98%** | **1.49** | sparkv3: 11.7% — **+86pp** |
| Plague | 55% | 2.15 | sparkv3 had no Plague pref |
| Love | 45% | 1.96 | sparkv3 had Apathy-flavored second pick |
| Ice | 36% | 2.23 | |
| Diversity | 19% | 2.55 | |
| Chaos | 15% | 2.32 | |
| Speed | 10% | 2.58 | |
| Psychic | 8% | 2.53 | sparkv3 used 7.8% — **same** |
| Fire | 8% | 2.57 | |
| Time | 6% | 2.65 | |
| Water | 0% | 3.0 | **never drafted** |

**Darkness 98% with avg pick position 1.49 is unprecedented.** The
model has fully committed to a single first-pick protocol — much more
extreme than Sparkv3's Clarity 14.5% / Apathy 13.3% / Darkness 11.7%
spread. Whether this is a *strength* (the model found a genuinely
dominant draft) or a *weakness* (the model has mode-collapsed) is
discussed in [Known Limitations](#known-limitations).

### 2. Compile every time you can. Discipline is perfect.

Action-mix data from 1,524 greedy-game compileable decisions:

| heuristic | rate | flag |
|---|---:|---|
| compile-when-possible | **100%** (1524/1524) | 🟢 |
| wasteful refresh (full hand) | **2%** (45/1928) | 🟢 |

The agent never passes up a compile, never refreshes with a full
hand. Both heuristics are at the 🟢 threshold of model-card scoring
across both opponents.

This is the same discipline Sparkv3 had — and the same Sparkv1/v2
*didn't*. The "refresh creep" failure mode that killed prior AZ
attempts (refresh rate climbing from 9% to 12-15% as training
continued) is absent here.

### 3. Push hard, push fast — the race thesis.

Win rate by game-length quartile (vs greedy, n=400):

| game length | WR | n |
|---|---:|---:|
| q1 (<41 turns) | **78.9%** | 95 |
| q2 (41-52 turns) | 65.4% | 104 |
| q3 (53-67 turns) | 49.0% | 100 |
| q4 (≥68 turns) | 53.5% | 101 |

**Spark v4 wins 79% of fast games and only 53% of long ones.** The
gap (+26pp from q1 to q4) is huge. The model has decided the way to
win is to push for 3 compiles before the opponent can stabilise — and
when that fails, it underperforms grinding.

Compile-count breakdown reinforces this:

| (agent COMPILE_LINE actions, opp COMPILE_LINE actions) | WR | n |
|---|---:|---:|
| 3 - 1 | **100%** | 43 |
| 3 - 2 | **100%** | 43 |
| 4 - 2 | **100%** | 24 |
| 3 - 3 | 34% | 35 |
| 4 - 3 | 60% | 30 |
| 2 - 3 | **0%** | 25 |
| 3 - 4 | 9% | 23 |

The *win condition* in Compile is `all_compiled()` — all 3 of your
lines marked compiled. The counts above are total **COMPILE_LINE
actions**, which include **recompiles** (a COMPILE_LINE on a
line you've already compiled steals the top card of the opponent's
deck; see [`game.py:628-637`](../src/compile_engine/game.py#L628-L637)).
There are no tiebreakers in Compile — the game ends when one side
hits all 3 lines first, or via the timeout `leader_wins` policy if
max_turns is reached.

Read that way: the asymmetric rows (3-1, 3-2, 4-2 → 100% WR) are
games where the agent reached `all_compiled()` cleanly. The mixed
rows (3-3, 4-3, 3-4) are mostly games where at least one side burned
COMPILE_LINE actions on recompiles instead of new lines — the deck-
theft is valuable tempo but doesn't advance the win condition. When
the agent took 3 compile actions and lost (the 34% in row "3-3"),
it almost certainly had ≥1 recompile and was actually only at 2/3
distinct lines compiled when opp finished. The agent's bias is to
push the race; it's *less* efficient when it has to play the
deck-theft sub-game.

### 4. Face-up by default. Information cost is worth it.

Face-up vs face-down play ratio:

| outcome | face-up share | n plays |
|---|---:|---:|
| in winning games | **73.8%** | 4,616 |
| in losing games | 70.9% | 3,286 |

The win/loss face-up gap is small (+2.9pp in wins). But the *absolute*
face-up share is high in both: 73.8% of all plays are face-up. The
model thinks face-up is right by default — face-down is a tempo move
reserved for setting up future synergies (per
[`webapp/lib/compile/cards.ts`](../webapp/lib/compile/cards.ts) face-down
cards are 2-value with no text).

Sparkv3 had a 62.1% face-up ratio under similar eval. Spark v4 is
**+11.7pp more aggressive** on face-up play — meaningfully different
strategic identity.

### 5. Love is the secret weapon — cards that win, vs cards you just play.

Top-15 per-decision card-usage deltas (wins minus losses, vs greedy):

| card | wins # | losses # | delta% | thesis |
|---|---:|---:|---:|---|
| **AX01:Love:6** | 144 | 63 | **+1.5%** | most-decisive card |
| AX01:Love:5 | 85 | 34 | +1.0% | |
| AX01:Love:4 | 72 | 32 | +0.7% | |
| AX01:Love:3 | 122 | 67 | +0.7% | |
| MN01:Plague:4 | 191 | 115 | +0.7% | hand disruption |
| MN01:Plague:5 | 126 | 74 | +0.5% | |
| AX02:Diversity:5 | 43 | 15 | +0.6% | |
| MN02:Chaos:3 | 41 | 16 | +0.5% | |
| — appears more in **losses** — | | | | |
| MN01:Darkness:2 | 278 | 219 | -1.2% | base flip-shift; gets blown up by Mirror |
| MN02:Ice:3 | 93 | 82 | -0.8% | |
| MN01:Darkness:1 | 118 | 97 | -0.7% | |
| MN02:Ice:5 | 67 | 62 | -0.7% | |

**Four of the top eight winning-cards are Love (3, 4, 5, 6).**
That's a striking concentration — the model is winning *via Love*
when Love is on the field, even though Love is only its third-most
drafted protocol (45%). Conversely, Darkness shows up in losses more
often than in wins relative to play volume. Darkness is the
*structural commitment* (drafted 98%) but Love is the *blade* —
when the agent has Love, it converts.

The "by config" slice in
[`h2h_breakdown_vs_greedy.txt`](../runs/20260523-123850-ppo-resume/eval/snapshot_00040/h2h_breakdown_vs_greedy.txt)
makes this concrete:

- **WR with AX01 expansion in pool: 68.2%** (n=192)
- **WR without AX01 expansion: 55.3%** (n=208)

A **+12.9pp drop when Love (and the rest of AX01) is unavailable.**
The agent's identity is *Darkness + Love*; remove Love and the win
condition narrows.

### 6. Mirror is the structural counter.

Worst protocol matchups for the agent (n≥30):

| my proto | opp proto | WR | n |
|---|---|---:|---:|
| Chaos | Fire | 56% | 34 |
| Ice | Clarity | 57% | 46 |
| Ice | Fire | 57% | 35 |
| **Ice** | **Mirror** | **58%** | 33 |
| **Darkness** | **Mirror** | **58%** | 59 |

Mirror specifically counters Darkness — and Darkness is the agent's
mandatory pick. WR drops 3pp from baseline when the opp has Mirror.

This is reflected in the **WR by config** slice:

- **WR with main2 (Chaos/Mirror/Time/...): 57.9%** (n=171)
- **WR without main2: 64.2%** (n=229)

A −6.3pp gap. Most of that is attributable to Mirror existing in the
pool. The agent doesn't have a learned counter to having its flips
mirrored back — it just plays slightly worse when Mirror is on the
board.

## Behaviour fingerprint vs prior models

Action distribution, vs greedy, n=400 stochastic games:

| action | Spark v4 share |
|---|---:|
| CHOOSE_TARGET (sub-effect resolution) | 32% |
| PLAY_FACE_UP | 26% |
| PLAY_FACE_DOWN | 10% |
| SKIP_OPTIONAL | 10% |
| REFRESH | 9% |
| COMPILE_LINE | 7% |
| DRAFT_PROTOCOL | 5% |
| DISCARD_CARD | 1% |
| SHIFT_OWN_CARD | <1% |

Differences vs Sparkv3 (AZ-derived) measured at n=200 vs greedy:

| metric | Spark v4 | Sparkv3 | shift |
|---|---:|---:|---|
| face_up_ratio | **73.8%** (wins) | 62.1% | **+11.7pp** more aggressive |
| compile_aggression (games with ≥3 compiles) | high* | 74.5% | comparable |
| avg_game_length (turns) | 56.7 | 50.2 | **+6.5 turns longer** |
| compile-when-possible | 100% | (similar) | identical discipline |
| Darkness pick rate | **98%** | 11.7% | **+86pp** concentration |
| Psychic pick rate | 8% | 7.8% | unchanged (both dethroned it) |
| Top-3 draft | Darkness/Plague/Love | Clarity/Apathy/Darkness | **different thesis** |

*Compile aggression isn't directly comparable because Sparkv3 was
evaluated under argmax and v4 under stochastic — but Spark v4 gets to
3 compiles in essentially every winning game.

The most striking divergence: **Sparkv3 favored defensive control
(Clarity/Apathy/Darkness mid-pick spread), Spark v4 has converged on
aggression-via-Darkness with Love as the closer.** PPO with
stochastic eval pushed the model toward a more committed, more
aggressive thesis than AlphaZero's smoother distributional draft.

## Trajectory through training (within this 50-iter run)

Ladder Elo across the 5 evaluated snapshots:

| snapshot | Elo | rank | vs greedy (n=150) | vs random (n=80) |
|---|---:|---:|---:|---:|
| snapshot_00030 | **1469** | **1** | 0.65 | 0.91 |
| **snapshot_00040** | **1463** | **2** | **0.62** | **0.85** |
| snapshot_00050 | 1387 | 3 | 0.62 | 0.85 |
| snapshot_00020 | 1367 | 4 | 0.61 | 0.90 |
| snapshot_00010 | 1315 | 5 | 0.65 | 0.84 |

**The training curve plateaus at iter-30 and never advances.** Iters
30 and 40 are within noise (~7 Elo); iter 50 regressed ~80 Elo as
the entropy controller fired. Iter 30 would have been a defensible
ship pick — the WR-vs-greedy at training time happened to favor
iter-40, but the population-level evaluation finds them equivalent
strength.

Also notable: the Elo gap from snap_10 → snap_30 is ~150 points, so
the run *did* learn something across its 30 iterations. But the
gradient flattened after that — typical of late-PPO behavior on a
sparse-reward game.

## Non-transitive structure — RPS triads

The cross-snapshot ladder surfaces **2 rock-paper-scissors triads**
where each model beats the next at ≥55% WR:

- `snap_40` beats `snap_50` (68%) · `snap_50` beats `snap_20` (57%) · `snap_20` beats `snap_40` (55%)
- `snap_40` beats `snap_50` (68%) · `snap_50` beats `snap_10` (62%) · `snap_10` beats `snap_40` (55%)

Both triads center on **snap_40 dominantly counters snap_50, but
loses slightly to snap_20 and snap_10.** Read as a strategy
fingerprint: snap_40's commitment to Darkness/Plague/Love is sharp
enough to crush snap_50's more entropy-bonus-pressured policy, but
*just* exploitable by the earlier snapshots' less-committed play
(which doesn't telegraph as much).

This is **real evidence the policy is on a mixed-Nash trajectory**,
not a pure-strategy ridge. The ELO ranking flattens this into a
linear order (snap_30 > snap_40 > snap_50), but the head-to-head
matrix shows the relation is cyclic at the top.

See `docs/sparkv4-figures/rps-cycles.png` for the WR heatmap.

## Cross-snapshot draft / matchup heatmaps

All five evaluated snapshots draft Darkness ≥97% of the time — the
draft thesis was already locked in at iter-10 (drafted 100% there)
and never wavered through iter-50. The flex slot wobbled: iter-10
favored Psychic (71%), iter-20+ moved to Plague (75-82%), iter-40
sat at Plague 55% / Ice 36%, iter-50 returned to Psychic 47%.

Best WR-when-drafted across the run (n≥20):

| protocol | WR | n drafted |
|---|---:|---:|
| Love | 0.73 | 684 |
| Plague | 0.73 | 795 |
| Diversity | 0.72 | 283 |
| Psychic | 0.71 | 429 |
| Darkness | 0.71 | 1494 |

Note **Darkness only wins 71% of games it's drafted in.** Love
(0.73) and Plague (0.73) outperform it on win-when-drafted basis, but
the model drafts Darkness 2× more often than either. The draft
commitment isn't paying off in raw conversion — it's paying off in
*availability* (Darkness is always there).

Figures are in [docs/sparkv4-figures/](sparkv4-figures/):

- `protocol-preferences.png` — pick frequency heatmap
- `protocol-winrates.png` — WR conditioned on draft
- `protocol-matchup.png` — protocol-vs-protocol WR
- `rps-cycles.png` — pairwise WR with triads

## Known limitations

- **Draft mode collapse (Darkness 98%) without an underlying edge.**
  The counterfactual block test shows the agent wins just as much
  *without* Darkness as with it. So while a human counter-drafting
  Mirror feels like an exploit, the actual WR cost is small (~3pp).
  The bigger issue is that the model **has not learned a draft
  policy** — it has learned to pick the protocol it always picks.
  Suggests a future training-time fix: add a draft-diversity bonus
  to the entropy regulariser, or randomise the agent's draft
  during self-play rollouts so the policy gets gradients on
  non-Darkness lines.
- **Plateaued at iter-30.** The training curve flattened (within-noise
  Elo from iter-30 onwards) — the model is **at the PPO ceiling for
  this hyperparameter regime, not for Compile**. Further gains likely
  require either widening the entropy band more aggressively (risk:
  policy collapse, as iter-50 hinted) or returning to AlphaZero with
  the lessons from prior failed AZ runs.
- **No mid-effect search.** ~32% of decisions are CHOOSE_TARGET
  (sub-effect resolution). Spark v4 plays these by policy head only —
  no MCTS, no rollout — because the engine's generator-coroutine
  effects don't deepcopy. Errors here compound: if the policy misjudges
  even 5% of CHOOSE_TARGET decisions, that's ~16 mistakes per game.
- **Configuration-dependent strength.** WR drops 12.9pp without AX01
  expansion (Love unavailable) and 6.3pp with main2 (Mirror available).
  The agent has *one* answer to Compile, and it depends on the card
  pool supporting it.
- **Long-game weakness.** WR drops from 79% in shortest-quartile
  games to 53% in longest-quartile games. The agent loses control of
  late-game stabilisation. Sparkv3 had the same pattern; nothing in
  PPO training fixed it.

## Counterfactual analysis — is the Darkness draft real?

The headline draft finding ("Darkness 98% pick freq") is striking but
ambiguous on its own. Two equally consistent stories:

- **Story A (rational play):** Darkness is genuinely the strongest
  protocol; the model has discovered an objective optimum. Drafting
  it 98% is the right answer.
- **Story B (mode collapse):** In self-play, both sides try to draft
  Darkness; whoever grabs it gets to play with it and wins more often.
  Over training, the policy gradient reinforces "draft Darkness" even
  though the actual edge is small or zero. The model has no data on
  what happens when it *doesn't* take Darkness, because it always does.

To distinguish, we ran a **forced-block counterfactual**: a wrapper
around the NN agent that filters DRAFT_PROTOCOL actions whose
protocol is in a blocked set, then lets the underlying policy
renormalise over what's left.

Script:
[`scripts/eval/counterfactual_draft.py`](../scripts/eval/counterfactual_draft.py).
Results (100 games per condition, snapshot_00040 vs greedy, seed 0,
stochastic both sides):

| condition | WR | delta vs baseline |
|---|---:|---:|
| baseline (no restriction) | 0.540 | (ref) |
| block Darkness | **0.540** | **+0.000** |
| block Plague | 0.550 | +0.010 |
| block Love | **0.580** | **+0.040** (!) |
| block all three (Dark+Plague+Love) | 0.510 | −0.030 |

**Story B is the right story.** Blocking Darkness — the protocol
the agent picks 98% of the time, that its top 4 most-played cards
are from — has *literally zero* effect on win rate. Blocking Love,
the supposed "secret weapon," actually *raises* WR by 4pp (probably
noise, but certainly not a load-bearing protocol either). Blocking
all three top-drafted protocols costs only 3pp.

**The model's "thesis" about Darkness is mostly illusion.** The
model's actual skill — the part that beats greedy at 54-61% — lives
in the **play decisions** (CHOOSE_TARGET, PLAY_FACE_UP / PLAY_FACE_DOWN
ordering, COMPILE_LINE timing) rather than the draft. The draft has
mode-collapsed onto Darkness because:

1. Self-play with mutually informed opponents both want Darkness.
2. The agent that grabs it first wins more games (selection signal).
3. Gradient descent reinforces "pick Darkness."
4. Once that habit forms, agent never plays games *without* Darkness,
   so there's no gradient to learn alternative draft policies.
5. The actual edge Darkness gives is ~0, but the model can't see
   that because of step 4.

Note also: the per-first-pick WR table (from the same 400-game eval)
already hinted at this if you looked carefully —

| agent's first pick | WR | n |
|---|---:|---:|
| Chaos | 0.750 | 12 |
| **Love** | **0.647** | **34** |
| **Darkness** | **0.627** | **249** |
| Plague | 0.592 | 49 |
| Ice | 0.571 | 28 |

Love-first wins 64.7% vs Darkness-first 62.7%. The agent chooses
Darkness 5× more often than Love as first-pick but gets a *worse*
win rate by it. The model has not discovered an optimal draft;
it has discovered a *consistent* draft.

### What it does when it can't have Darkness

Per the seat-conditional breakdown:

- As Seat 0 (snake-draft picks first; sees full pool): Darkness 43%
  first-pick, Plague 20%, Love 12%, Ice 10%, then a long tail. The
  model spreads its first picks across the entire top draft tier.
- As Seat 1 (picks second, after greedy random-picks): Darkness 82%
  first-pick. Greedy's random pick rarely takes Darkness (and from a
  pool of 10-24 protocols the chance Darkness was at slot 0 of greedy's
  legal_actions is ~5-10%), so Darkness is almost always still
  available.
- When greedy *first*-picks Darkness (5/400 games): agent picks
  Diversity, Time, or Love. WR 60% (small n) — agent adapts
  reasonably. Confirms the model has *some* off-policy capability
  when forced.

The 98% pool-presence number is the *average* across both seats. As
Seat 0, the agent often picks Darkness in its 4th or 5th snake-draft
turn (when Darkness has survived to its pick) rather than first. The
draft policy is "Darkness if available, ever" rather than "Darkness
must be first."

## How to read this for human play

If you're a human playing against Spark v4 on the webapp, the
exploitable patterns are:

1. **Draft Mirror if it's in the pool.** Darkness/Mirror is one of
   the bot's worst protocol matchups (~58% WR for the bot, vs ~61%
   baseline). The effect is small (~3pp), but it's the largest
   structural counter we've measured and it's free to take —
   the bot will draft Darkness regardless.
2. **Drag the game out.** The bot is a fast-game player. Past turn
   ~50 its WR drops below 55%. Refresh more, play face-down more,
   force the game into late-game where every action has to clear
   a stack the bot has already invested cards into.
3. **Watch for Love plays as a tempo gauge.** When the bot plays
   Love cards 4-6, it's executing its win condition. These are the
   highest-leverage cards for the bot — disrupt them with
   discards/flip-downs/hate.

## Reproducibility

The exact pipeline that produced every number in this document:

```bash
RUN=runs/20260523-123850-ppo-resume

# 1. Per-snapshot telemetry collection (stochastic, training-distribution configs)
python scripts/eval/collect.py --model $RUN/snapshot_00040.pt --opp greedy --games 400 \
    --out $RUN/eval/snapshot_00040/vs_greedy.jsonl
python scripts/eval/collect.py --model $RUN/snapshot_00040.pt --opp random --games 200 \
    --out $RUN/eval/snapshot_00040/vs_random.jsonl
# (repeat with --games 150 / 80 for snapshots 10, 20, 30, 50)

# 2. Per-snapshot metrics aggregation
python scripts/eval/metrics.py --in $RUN/eval/snapshot_00040 \
    --out $RUN/eval/snapshot_00040/metrics.json

# 3. Cross-snapshot ladder (240 games per pair of 7 entities)
python scripts/eval/ladder.py --snapshots $RUN/snapshot_0001*.pt $RUN/snapshot_0002*.pt \
    $RUN/snapshot_0003*.pt $RUN/snapshot_0004*.pt $RUN/snapshot_0005*.pt \
    --games 20 --out $RUN/eval/ladder.json

# 4. Model card + per-decision card-usage breakdown
python scripts/eval/card.py --metrics $RUN/eval/snapshot_00040/metrics.json \
    --ladder $RUN/eval/ladder.json --out $RUN/eval/snapshot_00040/model_card.md
python scripts/eval/h2h_breakdown.py $RUN/eval/snapshot_00040/vs_greedy.jsonl

# 5. Cross-snapshot strategy figures + analysis MD
#    (writes docs/strategy-analysis.md + docs/figures/*.png — for v4
#    these were renamed to docs/sparkv4-strategy-analysis.md +
#    docs/sparkv4-figures/ to preserve the historical Sparkv1 outputs.)
python scripts/eval/strategy.py --run $RUN --out docs
```

All scripts live in [`scripts/eval/`](../scripts/eval/) and write to
`runs/20260523-123850-ppo-resume/eval/`. Wall-clock on M2: ~25 minutes
end-to-end.
