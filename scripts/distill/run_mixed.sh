#!/usr/bin/env bash
# Distillation with opponent mixing. Generates labels from a controlled
# mix of opponents (self/greedy/random) so the label state-distribution
# matches the eval distribution, then runs a 3-epoch distillation +
# standard eval.
#
# Usage:
#   scripts/distill/run_mixed.sh runs/latest/snapshot_00500.pt
#   GAMES=100 MIX_SELF=0.4 MIX_GREEDY=0.4 MIX_RANDOM=0.2 \
#       scripts/distill/run_mixed.sh runs/latest/snapshot_00500.pt

set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT="${1:?usage: $0 <source_ckpt.pt>}"

GAMES="${GAMES:-80}"
DETS="${DETS:-8}"
SIMS="${SIMS:-50}"
BATCH="${BATCH:-8}"
TOP_K="${TOP_K:-5}"
MIN_VISITS="${MIN_VISITS:-3}"
SKIP_TOP_PROB="${SKIP_TOP_PROB:-0.9}"
TAU="${TAU:-1.0}"
MIX_SELF="${MIX_SELF:-0.5}"
MIX_GREEDY="${MIX_GREEDY:-0.3}"
MIX_RANDOM="${MIX_RANDOM:-0.2}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH_TRAIN="${BATCH_TRAIN:-64}"
VALUE_COEF="${VALUE_COEF:-0.0}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-0}"

CKPT_BASENAME="$(basename "$CKPT" .pt)"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$(dirname "$CKPT")/distill/${TS}-mixed"
mkdir -p "$OUT_DIR"

LABELS="$OUT_DIR/labels.pt"
DISTILLED="$OUT_DIR/${CKPT_BASENAME}_distilled.pt"
EVAL_DIR="$OUT_DIR/eval"

echo "==> Mixed-opponent distillation"
echo "    source ckpt:  $CKPT"
echo "    output dir:   $OUT_DIR"
echo "    games:        $GAMES (mix self=$MIX_SELF greedy=$MIX_GREEDY random=$MIX_RANDOM)"
echo "    distill:      $EPOCHS epochs at lr=$LR"
echo ""

echo "==> Step 1: generate MCTS labels"
.venv/bin/python scripts/distill/generate_labels.py \
    --ckpt "$CKPT" \
    --out "$LABELS" \
    --games "$GAMES" \
    --dets "$DETS" --sims "$SIMS" --batch-size "$BATCH" \
    --root-top-k "$TOP_K" --root-min-visits "$MIN_VISITS" \
    --skip-top-prob "$SKIP_TOP_PROB" --tau "$TAU" \
    --mix-self "$MIX_SELF" --mix-greedy "$MIX_GREEDY" --mix-random "$MIX_RANDOM" \
    --device "$DEVICE" --seed "$SEED" \
    2>&1 | tee "$OUT_DIR/generate.log"

echo ""
echo "==> Step 2: fine-tune policy ($EPOCHS epochs)"
.venv/bin/python scripts/distill/train.py \
    --ckpt "$CKPT" \
    --labels "$LABELS" \
    --out "$DISTILLED" \
    --epochs "$EPOCHS" --batch-size "$BATCH_TRAIN" --lr "$LR" \
    --value-coef "$VALUE_COEF" \
    --device "$DEVICE" --seed "$SEED" \
    2>&1 | tee "$OUT_DIR/train.log"

echo ""
echo "==> Step 3: eval distilled checkpoint vs Random + Greedy"
mkdir -p "$EVAL_DIR"
for opp in random greedy; do
    PYTHONPATH=src .venv/bin/python scripts/eval/collect.py \
        --model "$DISTILLED" --opp "$opp" \
        --games 50 --seed "$SEED" --device "$DEVICE" \
        --out "$EVAL_DIR/vs_$opp.jsonl" \
        2>&1 | tee -a "$OUT_DIR/eval.log"
done
PYTHONPATH=src .venv/bin/python scripts/eval/metrics.py \
    --in "$EVAL_DIR" --model "${CKPT_BASENAME}_distilled_mixed" \
    --out "$EVAL_DIR/metrics.json" \
    2>&1 | tee -a "$OUT_DIR/eval.log"

echo ""
echo "==> Done. Distilled: $DISTILLED"
echo "    Metrics:        $EVAL_DIR/metrics.json"
