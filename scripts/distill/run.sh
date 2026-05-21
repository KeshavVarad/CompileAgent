#!/usr/bin/env bash
# Orchestrates the full distillation pipeline:
#   1. MCTS-label states from policy self-play with skip-when-confident.
#   2. Fine-tune the policy head against soft targets (value head frozen).
#   3. Run the standard eval pipeline on the distilled checkpoint.
#
# Defaults target the "cheap-wins" recipe: offline labeling, skip ≥ 0.9
# threshold, Gumbel-AlphaZero-style soft target (log prior + tau * Q).
#
# Usage:
#   scripts/distill/run.sh runs/latest/snapshot_00500.pt
#   GAMES=100 EPOCHS=5 scripts/distill/run.sh runs/latest/snapshot_00500.pt

set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT="${1:?usage: $0 <source_ckpt.pt>}"
GAMES="${GAMES:-50}"
DETS="${DETS:-8}"
SIMS="${SIMS:-50}"
BATCH="${BATCH:-8}"
TOP_K="${TOP_K:-5}"
MIN_VISITS="${MIN_VISITS:-3}"
SKIP_TOP_PROB="${SKIP_TOP_PROB:-0.9}"
TAU="${TAU:-1.0}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
BATCH_TRAIN="${BATCH_TRAIN:-64}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-0}"

# Derive output paths from source checkpoint name.
CKPT_BASENAME="$(basename "$CKPT" .pt)"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$(dirname "$CKPT")/distill/${TS}"
mkdir -p "$OUT_DIR"

LABELS="$OUT_DIR/labels.pt"
DISTILLED="$OUT_DIR/${CKPT_BASENAME}_distilled.pt"
EVAL_DIR="$OUT_DIR/eval"

echo "==> Distillation pipeline"
echo "    source ckpt:   $CKPT"
echo "    output dir:    $OUT_DIR"
echo "    labels:        $LABELS"
echo "    distilled:     $DISTILLED"
echo ""

echo "==> Step 1: generate MCTS labels ($GAMES self-play games)"
.venv/bin/python scripts/distill/generate_labels.py \
    --ckpt "$CKPT" \
    --out "$LABELS" \
    --games "$GAMES" \
    --dets "$DETS" --sims "$SIMS" --batch-size "$BATCH" \
    --root-top-k "$TOP_K" --root-min-visits "$MIN_VISITS" \
    --skip-top-prob "$SKIP_TOP_PROB" --tau "$TAU" \
    --device "$DEVICE" --seed "$SEED" \
    2>&1 | tee "$OUT_DIR/generate.log"

echo ""
echo "==> Step 2: fine-tune policy ($EPOCHS epochs, lr=$LR)"
.venv/bin/python scripts/distill/train.py \
    --ckpt "$CKPT" \
    --labels "$LABELS" \
    --out "$DISTILLED" \
    --epochs "$EPOCHS" --batch-size "$BATCH_TRAIN" --lr "$LR" \
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
    --in "$EVAL_DIR" --model "${CKPT_BASENAME}_distilled" \
    --out "$EVAL_DIR/metrics.json" \
    2>&1 | tee -a "$OUT_DIR/eval.log"

echo ""
echo "==> Done. Distilled checkpoint: $DISTILLED"
echo "    Compare to source eval at: $(dirname "$CKPT")/eval/${CKPT_BASENAME}/metrics.json"
echo "    Distilled eval at:         $EVAL_DIR/metrics.json"
