#!/usr/bin/env bash
# Distillation epoch sweep. Reuses an existing labels.pt and produces
# one distilled checkpoint per epoch count, with eval against Random +
# Greedy after each. Use to diagnose whether the regression at 5 epochs
# is an overshoot (lower epoch counts win) or a deeper problem (none
# beat the source).
#
# Usage:
#   scripts/distill/epoch_sweep.sh <source_ckpt.pt> <labels.pt> [epochs...]
#   scripts/distill/epoch_sweep.sh runs/latest/snapshot_00500.pt \
#       runs/latest/distill/20260521-131722/labels.pt 1 2 3

set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT="${1:?usage: $0 <source_ckpt.pt> <labels.pt> [epochs...]}"
LABELS="${2:?usage: $0 <source_ckpt.pt> <labels.pt> [epochs...]}"
shift 2
if [[ $# -eq 0 ]]; then
    EPOCHS_LIST=(1 2 3)
else
    EPOCHS_LIST=("$@")
fi

GAMES="${GAMES:-50}"
LR="${LR:-1e-4}"
BATCH_TRAIN="${BATCH_TRAIN:-64}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-0}"

CKPT_BASENAME="$(basename "$CKPT" .pt)"
LABELS_DIR="$(dirname "$LABELS")"
TS="$(date +%Y%m%d-%H%M%S)"
SWEEP_DIR="${LABELS_DIR}/epoch_sweep_${TS}"
mkdir -p "$SWEEP_DIR"

SUMMARY="$SWEEP_DIR/summary.tsv"
printf "epochs\tvs_greedy_wr\tvs_greedy_record\tvs_random_wr\tvs_random_record\n" > "$SUMMARY"

echo "==> Epoch sweep on $LABELS"
echo "    source ckpt: $CKPT"
echo "    sweep dir:   $SWEEP_DIR"
echo "    epochs:      ${EPOCHS_LIST[*]}"
echo ""

for E in "${EPOCHS_LIST[@]}"; do
    RUN_DIR="$SWEEP_DIR/epochs_${E}"
    mkdir -p "$RUN_DIR"
    DISTILLED="$RUN_DIR/${CKPT_BASENAME}_distilled.pt"
    EVAL_DIR="$RUN_DIR/eval"
    mkdir -p "$EVAL_DIR"

    echo "==> Epochs=$E: train"
    .venv/bin/python scripts/distill/train.py \
        --ckpt "$CKPT" \
        --labels "$LABELS" \
        --out "$DISTILLED" \
        --epochs "$E" --batch-size "$BATCH_TRAIN" --lr "$LR" \
        --device "$DEVICE" --seed "$SEED" \
        2>&1 | tee "$RUN_DIR/train.log"

    echo "==> Epochs=$E: eval"
    for opp in random greedy; do
        PYTHONPATH=src .venv/bin/python scripts/eval/collect.py \
            --model "$DISTILLED" --opp "$opp" \
            --games "$GAMES" --seed "$SEED" --device "$DEVICE" \
            --out "$EVAL_DIR/vs_$opp.jsonl" \
            2>&1 | tee -a "$RUN_DIR/eval.log"
    done
    PYTHONPATH=src .venv/bin/python scripts/eval/metrics.py \
        --in "$EVAL_DIR" --model "${CKPT_BASENAME}_distilled_e${E}" \
        --out "$EVAL_DIR/metrics.json" \
        2>&1 | tee -a "$RUN_DIR/eval.log"

    # Extract win rates for the summary row.
    G_WR=$(.venv/bin/python -c "import json; m=json.load(open('$EVAL_DIR/metrics.json'))['matchups']['greedy']; print(f\"{m['win_rate']:.2f}\t{m['wins']}-{m['losses']}-{m['draws']}\")")
    R_WR=$(.venv/bin/python -c "import json; m=json.load(open('$EVAL_DIR/metrics.json'))['matchups']['random']; print(f\"{m['win_rate']:.2f}\t{m['wins']}-{m['losses']}-{m['draws']}\")")
    printf "%s\t%s\t%s\n" "$E" "$G_WR" "$R_WR" >> "$SUMMARY"
    echo ""
done

echo "==> Sweep complete."
echo ""
column -t -s $'\t' "$SUMMARY"
echo ""
echo "    full results: $SWEEP_DIR"
