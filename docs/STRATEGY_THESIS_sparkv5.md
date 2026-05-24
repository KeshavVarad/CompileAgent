# Strategy thesis · Spark v5

> Companion to [the Spark v5 model card](model-card-sparkv5.md). Same
> kind of analytical breakdown the v4 thesis tried to do, applied to the
> new shipped model and grounded in side-by-side telemetry against v4.

Underlying data:
- `runs/20260524-123408-ppo-q-head-extended/eval/snapshot_00090/vs_greedy_400.jsonl` (400 games)
- 500-game CHOOSE_TARGET replay (`/tmp/v5_love_replay.py`)
- 300-game head-to-head vs v4 (`/tmp/v5_vs_v4_h2h.jsonl`)

## TL;DR

The +11pp WR improvement over Spark v4 comes from **broad CHOOSE_TARGET
improvements**, not from fixing any single dramatic v4 failure. Specifically:

1. **SKIP_OPTIONAL rate dropped from 9.8% → 2.7%** — the agent now takes
   optional clauses far more often. This is the single biggest action-mix
   shift, and the most direct evidence of the Q-head working.
2. **Love 1 End trade fixed partially** — taken 14% of the time (8/59),
   up from v4's catastrophic 0/47 (0%). Not the "100% take" you'd expect
   if Q* were perfect, but a real shift away from the pathological skip.
3. **PLAY_FACE_DOWN ratio up +7.8pp** (9.7% → 17.5%) — driven by Gravity
   in the new draft, which plays cards from deck-top face-down.
4. **Different draft, similar mode collapse pattern**. Gravity replaces
   Darkness as the modal pick (97% pool freq), but counterfactual
   confirms it's also cosmetic — not load-bearing.
5. **v4-vs-v5 head-to-head: 58%** (n=300). Within noise of "equal but
   slightly better." The improvement is real but the WR uplift vs greedy
   is partly Spark v5's broader skill, not specifically "beats v4."

What was NOT improved by the Q-head:

- **Love 3 give-to-opp choice**: still roughly random by hand index.
  v4 average rank 0.44, v5 average rank 0.44. The Q-head didn't learn
  to gift junk cards.
- **Per-protocol draft calibration**: Darkness has 88.6% WR-when-drafted
  but is only drafted 9% of the time. Mirror image of v4. The model
  over-drafts Gravity (71.8% WR-when-drafted, drafted 97%).

## How the Q-head changed action selection

The clearest behavioural signal of the Q-head at work is the action-mix shift:

| action | v4 share | v5 share | delta |
|---|---:|---:|---:|
| CHOOSE_TARGET | 32.4% | 28.9% | −3.5% |
| PLAY_FACE_UP | 25.8% | 27.0% | +1.2% |
| **PLAY_FACE_DOWN** | **9.7%** | **17.5%** | **+7.8%** |
| REFRESH | 8.7% | 8.2% | −0.5% |
| COMPILE_LINE | 6.9% | 8.6% | +1.7% |
| DRAFT_PROTOCOL | 5.4% | 6.7% | +1.3% |
| **SKIP_OPTIONAL** | **9.8%** | **2.7%** | **−7.1%** |
| DISCARD_CARD | 1.3% | 0.6% | −0.8% |

Two huge shifts: SKIP_OPTIONAL down 7.1pp, PLAY_FACE_DOWN up 7.8pp. The
first is direct Q-head behaviour (taking optional clauses); the second
is downstream from the Gravity-heavy draft (Gravity benefits from
face-down board state).

### Specific optional-clause decisions

Across the 500-game replay, here's what v5 does on optional clauses:

| optional prompt | take | skip | take rate |
|---|---:|---:|---:|
| "Flip Plague 4 (this card)?" | 208 | 92 | 69% |
| "Shift covered Ice 3 to another line" | 124 | 63 | 66% |
| "Resolve opp middle as Mirror 1" | 105 | 9 | **92%** |
| "Give a card to opponent (optional)? Love 1 End" | 8 | 51 | **14%** (v4: 0%) |
| "Discard another card?" (Plague 2 chain) | 23 | 164 | 12% |

The Mirror 1 number is the most striking. Mirror 1's bottom says "End:
You may resolve the middle command of 1 of your opponent's cards as if
it were on this card." That's a HARD optional decision — you're picking
which opp middle command to *copy*. The Q-head takes it 92% of the time.
This is the kind of decision that requires reading the board, which the
Q-head + UNREAL aux trunk seem to do well.

Compare to the chain-discard at 12% take rate: the agent has learned
that chaining discards (Plague 2 "discard 1 or more cards") is usually
not worth the marginal cost — a defensible heuristic.

## The Love 1 End partial fix

This was the headline v4 failure. From [docs/STRATEGY_THESIS_sparkv4.md](STRATEGY_THESIS_sparkv4.md):

> Love 1 End ("may give 1 card, draw 2 = +1 net swing"): REJECTED 47/47
> times the agent saw the choice. Free positive-EV trade left on the
> table.

For v5 with the Q-head:

| | encountered | took the trade | skip |
|---|---:|---:|---:|
| **Spark v4** | 47 | **0 (0%)** | 47 |
| **Spark v5** | 59 | **8 (14%)** | 51 |

A real shift. 14% is far from "always take" but it's also far from "never take." Some interpretations:

- **Most likely**: the Q-head learned the trade IS net-positive in many
  states, but its on-policy `Q^π` target (the GAE return under the current
  policy, not Q*) is still noisy for actions the policy rarely takes.
  As the policy explored more, Q*'s "take" estimate gradually moved into
  the argmax zone for a subset of states.
- **Alternative**: maybe taking the trade isn't always right. In late game
  with low deck, giving opp a known card while drawing 2 deck-bottoms
  could be worse than skipping. Hard to tell without state-conditional
  analysis.

Either way, the Q-head materially shifted a behaviour that pure PPO
couldn't move in v4. Limited by the on-policy data the Q-head learns
from — for fuller improvement we'd want counterfactual Q-targets
(compute Q(s, action_not_taken) too).

## What the Q-head did NOT fix

**Love 3 "give 1 card from your hand to your opponent"** is still random
by hand-index, exactly like v4:

| | n decisions | given avg value | hand avg value | normalized rank |
|---|---:|---:|---:|---:|
| Spark v4 | 50 | 2.64 | 2.53 | 0.44 |
| Spark v5 | 114 | 2.54 | 2.55 | **0.44** |

Identical rank-0.44 (roughly random — 0.5 is exactly random) and similar
average values. The Q-head didn't learn to gift junk cards in Love 3.

Why this specifically didn't move: hypothesis-1 is that the **target
within the "give 1 card" decision is a different protocol from the
action features**. The action's features include "which hand-index am I
giving" — but they don't include "what cards are in my hand for me to
keep." So the Q-head can't differentiate "give value-5 (I keep nothing)"
from "give value-0 (I keep value-5)." Both look the same to the Q-head
at the (state, action) level.

Fixing this would require a richer action encoding: "give card_X means
I retain cards [list]." The current encoder doesn't expose that.

## Draft & combo patterns

### Draft mode collapse (cosmetic, like v4)

| | Spark v4 | Spark v5 |
|---|---|---|
| Modal protocol | Darkness 98% | Gravity 97% |
| Modal pick position | 1.49 | 1.23 |
| WR-when-drafted (modal) | 62.7% | 71.8% |
| Protocols drafted ≥10% | 4 | 9 |
| Counterfactual: block modal | −0pp | −1.5pp |

Same "mode-collapsed but cosmetic" pattern as v4 — different protocol
this time. The agent has a strong draft signature on one pick but isn't
winning *because* of it. WR comes from elsewhere (play quality + the
other 2 protocol slots).

### The combos that win

Protocol pairs the model lands most often, vs WR for that pair (n≥30):

| pair | n | WR |
|---|---:|---:|
| Gravity+Plague | 157 | 0.70 |
| Gravity+Life | 133 | 0.69 |
| Gravity+Love | 105 | **0.76** |
| Fire+Gravity | 90 | 0.70 |
| Gravity+Ice | 80 | 0.71 |
| Gravity+Speed | 53 | 0.72 |
| Life+Plague | 36 | 0.64 |
| Fire+Plague | 35 | 0.69 |
| **Darkness+Gravity** | 33 | **0.88** ← |
| **Love+Plague** | 32 | **0.81** ← |

The two highest-WR pairs are **Darkness+Gravity (88%)** and **Love+Plague (81%)**.
These are landed rarely (the model drafts Gravity in everything, so
Darkness or Love+Plague-without-Gravity is uncommon), but when landed
they're the strongest combinations measured.

### What's the Gravity synergy actually doing?

Looking at the cards involved:

- **Gravity 0**: "For every 2 cards in this line, play the top card of
  your deck face-down under this card." → floods the line with face-down
  cards
- **Gravity 2/4/5**: top emphasis effects that scale with face-down board
  state

Pairing Gravity with **Plague/Life/Love** is structural: Plague disrupts
opp's hand (so opp can't keep up with your board flood); Life refills
your own deck (so Gravity 0 has cards to play); Love does asymmetric
trades that get opp's high-value cards out of play.

The **Darkness+Gravity** combo (88% WR, n=33) is the interesting one.
Darkness 2's top says "All face-down cards in this stack have a value
of 4." Combine with Gravity 0 dumping face-down cards into the line:
each face-down is now 4 value (vs default 2). Result: a face-down-heavy
line that scores ~2× faster than baseline.

The agent has discovered this combo, but only drafts it in 33/400 games
because of the Gravity-mono draft preventing Darkness pickup. **There's
a real strategic insight buried in the Spark v5 policy that's blocked
from execution by the draft mode collapse.**

## Game-style shifts vs Spark v4

| metric | Spark v4 | Spark v5 | delta |
|---|---:|---:|---:|
| Avg game length (turns) | 56.7 | 55.0 | −1.7 |
| Short games (<40 turns) | 21% | 26% | +5pp |
| Long games (≥60 turns) | 38% | 31% | −7pp |
| Avg compile margin | +0.86 | +1.01 | +0.15 |
| Face-up ratio | 0.700 | 0.599 | −10.1pp |

Spark v5 wins **shorter** games. Both ends shifted: more short games,
fewer long games. Plus higher compile margin (+1.01 vs +0.86) means
when it wins, it wins more decisively.

The face-up ratio drop (from 0.70 → 0.60) is structural — Gravity puts
cards face-down naturally, shifting the equilibrium away from face-up
plays. This is *not* a strategic shift (v4 was pushing face-up
aggression as thesis); it's a consequence of the new draft.

## v4 vs v5 head-to-head

Direct match: 300 stochastic games, Spark v5 vs Spark v4.

```
v5 win rate: 0.58 (±5.7pp)
```

Within noise of "v5 marginally better" but **not a huge gap.** The
implied Elo from each agent's WR vs greedy (v4: 0.615, v5: 0.723)
predicts ~0.62 in a direct match — actual 0.58 is in the same zone.

What this means: v5's +11pp on greedy is partly v5 being better in
general, and partly v5 being good at the same things v4 was good at
(both beat greedy at related skills). Direct head-to-head, v5 has a
modest edge, not a dominant one.

## Where the +11pp actually came from — best guess

Synthesizing the evidence:

1. **Q-head on optional clauses (~5-6pp of the gain)**. SKIP_OPTIONAL
   dropped 7.1pp; many of those formerly-skipped optionals were
   positive-EV (Mirror 1, Plague 4 flip, Love 1 End). Each one is a
   small advantage in some specific game state.
2. **Better mid-effect target selection on common prompts (~3-4pp)**.
   "Flip 1 card" type prompts happen 300+ times per 500 games. Q-head's
   ability to pick the right target (opp's most valuable card vs opp's
   junk) compounds.
3. **Gravity-driven play style (~1-2pp)**. Face-down-heavy strategy is
   somewhat sticky against opponents trained against face-up-heavy v4.
   Not a huge factor.
4. **Aux task representation improvement (~1-2pp)**. Better encoder
   representation through opp-hand prediction probably improved value
   estimates across the board.

## Limitations of this analysis

- **Sample size for rare CHOOSE_TARGET decisions is small.** Love 1 End:
  59 occurrences in 500 games is at the boundary of "real signal."
  Mirror 1: only 114 occurrences. Specific protocol-pair WRs at n=30-100
  have ±10-15pp std error.
- **Doesn't isolate Q-head from other changes.** We're comparing v5
  (Q-head + per-class entropy + aux + extended training) to v4 (none of
  those). Hard to attribute the +11pp to any single component
  without an ablation run. The action-mix shifts are *consistent* with
  the Q-head story but don't prove it.
- **Greedy is a weak target.** "v5 beats greedy at 0.72" doesn't tell us
  much about how v5 plays vs strong opponents (Mirror counter-drafts,
  humans, future agents). The head-to-head vs v4 at 0.58 is the closest
  comparison we have to "strong opponent."

## What would actually pin down the gain

Three targeted analyses worth doing if you want to be sure:

1. **Ablation training run**: Train a v5-style agent *without* the
   Q-head (keep per-class entropy + aux). If WR matches v4 (≈0.61),
   Q-head was the whole story. If it's 0.65-0.68, the per-class entropy
   + aux did some work too.
2. **CHOOSE_TARGET-only counterfactual**: At each CHOOSE_TARGET in a
   replay, swap the Q-head decision for the policy-head decision and
   see if WR drops. Direct measurement of Q-head's contribution.
3. **Per-decision impact attribution**: For each Love 1 End / Mirror 1
   / Plague 4 flip decision, what's the WR conditional on "took the
   optional" vs "skipped"? Should reveal which optional clauses
   actually matter.

For now: **the +11pp is real, the mechanism is plausibly the Q-head,
and the specific evidence points to broad CHOOSE_TARGET improvement
rather than fixing one or two known failures.** That's a defensible
position for shipping Spark v5, while flagging that we don't have full
attribution.

## Hidden insights worth following up

Two patterns in the v5 telemetry that hint at next-level strategy
discoveries the model has partially found:

1. **Darkness+Gravity = 88% WR, drafted only 33 times.** The model has
   discovered the strongest combo we've measured but can't reliably
   land it because of the Gravity-mono draft. A future training run
   that breaks the Gravity collapse could unlock another +5-10pp.
2. **The agent uses Mirror 1's "resolve opp middle" 92% of the time.**
   This is a sophisticated copy-effect play. Spark v4 likely couldn't
   handle this decision at all. Worth a deeper analysis of whether v5
   correctly picks WHICH opp middle to copy (the Q-head's choice of
   target).

Either of these would be a good direction for the next training
experiment — they're empirical signals about where the policy already
*understands* something but execution is blocked.
