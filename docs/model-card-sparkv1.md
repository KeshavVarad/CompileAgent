# Model card · Sparkv1

> **Sparkv1 is PPO snapshot 00120 from training run `20260520-224058`.**
> It's the bot currently deployed on the webapp as the "Play vs Sparkv1"
> opponent.

## At a glance

| metric | value |
|---|---|
| Architecture | Pointer-style policy + value MLP, learned card/protocol embeddings |
| Parameters | ~580k (≈2.4 MB ONNX) |
| Training run | `runs/20260520-224058/` |
| Training iter | 120 of 500 (still climbing on the original schedule when checkpointed) |
| Training compute | Apple-silicon MPS, ~38 min wall-clock to iter 120 |
| Win rate vs Random | 0.80 (50 deterministic games) |
| Win rate vs Greedy | 0.60 (50 deterministic games) |
| Compile margin vs Greedy | -0.08 (essentially even on compiles) |
| Average game length vs Greedy | 79 turns |

## Why this checkpoint?

PPO bots usually trade off two things: **how well they exploit their
training pool** (other NN snapshots in self-play), and **how broadly
they generalise** to the heuristic baselines. As the run progressed we
watched both axes:

- **Iter 90–120** sat at the local optimum against the Greedy baseline
  (60% WR), with average compile margins near zero — meaning Sparkv1
  wasn't blowing Greedy out, but it was beating it more often than not.
- **Iter 150 onward** kept gaining Elo against later self-play snapshots
  but lost ground against Greedy specifically (dropping to 36–48% WR vs
  Greedy by iter 200–230). This is consistent with the policy
  overfitting to the opponent pool: it became hard to beat for itself
  and other recent snapshots, but Greedy's simple "max immediate line
  value" exposed a hole.
- **Iter 230 currently leads the ladder Elo at ~1518**, but only 36% vs
  Greedy. We deliberately pick the checkpoint that's *robust* across
  opponents instead of the one that crushes its peers — humans
  evaluating the bot will play more like Greedy (heuristic-ish) than
  like a self-play partner.

The full Elo + WR trajectories are plotted in the
[README](../README.md#training-progress); the eval pipeline that produced
them lives in [scripts/eval/](../../scripts/eval/).

## Behaviour highlights

These come from the bot's 50-game runs vs Random and Greedy (see
[snapshot_00120/model_card.md](../runs/20260520-224058/eval/snapshot_00120/model_card.md)
for the raw table; **note: the per-run eval directory is gitignored**, but
the model card is small enough to commit if you'd like a checked-in copy).

- **Drafts Plague consistently** (50% of games) and Darkness (50%), with
  Psychic in the third slot ~28% of the time. Plague's commands draft
  well with Sparkv1's commitment-heavy style.
- **Top cards played**: Plague 0, Plague 4, Darkness 4/5 — high-value
  field cards that win lines on their own.
- **Face-down play ratio is low (~9%)** — Sparkv1 commits face-up rather
  than bluffs. This is a structural weakness; a more balanced player
  would use more face-downs both defensively (information hiding) and
  offensively (on-flip effects).
- **Compile-when-possible adherence is 100%** — when a compile is the
  only legal move, it takes it. Wasteful refreshes (refreshing while
  holding ≥5 cards) are 0% — fixed early in training.
- **Loss modes vs Greedy**: 60% of the 20 losses were "blowouts" (margin
  ≥2 compiles). Some of these are positions where Greedy's brute
  line-value approach exposed Sparkv1's lack of defensive line shifts.

## Known weaknesses

These are worth knowing if you're playing Sparkv1 and looking for
exploits:

1. **It doesn't bluff.** Almost never plays face-down for denial or
   surprise effects.
2. **Stuck around 60% WR vs Greedy.** That ceiling, despite further
   training, looks like a structural feature of the current reward
   shaping (Δ compiled protocols) rather than something more training
   alone would fix. See `docs/figures/eval-sweep.png` for the curve.
3. **Late-game starvation.** When refresh + small hand combine,
   Sparkv1 sometimes ends up in compile-poor positions. Forcing many
   turns of effect resolution against it can exhaust its hand.
4. **Protocol-collapsed drafts.** It loves Plague + Darkness; if you can
   draft to deny those, it picks weaker third options.

## Swapping the bot

The webapp reads its current bot from
[`webapp/lib/bot-config.ts`](../webapp/lib/bot-config.ts). To replace
Sparkv1 with a new snapshot:

```bash
# 1. Export the new checkpoint to ONNX (self-contained file).
python scripts/eval/export_onnx.py \
  --ckpt runs/<run>/snapshot_NNNNN.pt \
  --out  webapp/public/models/bot-current.onnx

# 2. Edit CURRENT_BOT in webapp/lib/bot-config.ts:
#    - id: short identifier baked into game rows
#    - displayLabel: human-visible name
#    - trainingIter, evalSummary: filled in from your eval results

# 3. Redeploy.
cd webapp && vercel deploy --prod
```

That's it — the rest (UI, API validation, the game-creation flow) reads
from `CURRENT_BOT` so a single edit + redeploy ships the new bot.

## Reproducing the eval

```bash
RUN=runs/20260520-224058
PY=.venv/bin/python

# Per-snapshot behaviour vs Random + Greedy
$PY scripts/eval/collect.py --model $RUN/snapshot_00120.pt \
  --opp random --games 50 --device cpu \
  --out $RUN/eval/snapshot_00120/vs_random.jsonl
$PY scripts/eval/collect.py --model $RUN/snapshot_00120.pt \
  --opp greedy --games 50 --device cpu \
  --out $RUN/eval/snapshot_00120/vs_greedy.jsonl
$PY scripts/eval/metrics.py --in $RUN/eval/snapshot_00120/ \
  --out $RUN/eval/snapshot_00120/metrics.json --model snapshot_00120

# Cross-snapshot ladder (12 games per ordered pair, 9 nodes + 2 anchors)
$PY scripts/eval/ladder.py \
  --snapshots $RUN/snapshot_00010.pt $RUN/snapshot_00030.pt \
              $RUN/snapshot_00060.pt $RUN/snapshot_00090.pt \
              $RUN/snapshot_00120.pt $RUN/snapshot_00150.pt \
              $RUN/snapshot_00170.pt $RUN/snapshot_00200.pt \
              $RUN/snapshot_00230.pt \
  --games 12 --device cpu --out $RUN/eval/ladder.json

# Plots
$PY scripts/eval/plot.py --run $RUN --out docs/figures/
```
