#!/usr/bin/env bash
# AlphaZero-style training run. Hot-starts from a PPO snapshot and runs
# the search-as-data-engine loop in src/compile_engine/nn/train_alphazero.py.
#
# Usage:
#   scripts/az/train.sh runs/latest/snapshot_00500.pt
#   ITERS=200 GAMES=16 DETS=4 SIMS=32 SKIP=0.85 \
#       scripts/az/train.sh runs/latest/snapshot_00500.pt

set -euo pipefail
cd "$(dirname "$0")/../.."

INIT_CKPT="${1:?usage: $0 <init_ckpt.pt>}"

ITERS="${ITERS:-200}"
GAMES="${GAMES:-16}"
BUF="${BUF:-4}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-64}"
LR="${LR:-5e-5}"
DETS="${DETS:-4}"
SIMS="${SIMS:-32}"
MCTS_BATCH="${MCTS_BATCH:-8}"
SKIP="${SKIP:-0.85}"
DRAFT_POOL="${DRAFT_POOL:-9}"
GUMBEL="${GUMBEL:-1}"
GUMBEL_N="${GUMBEL_N:-8}"
POOL_WR="${POOL_WR:-0.55}"
POOL_GAMES="${POOL_GAMES:-30}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-10}"
EVAL_GAMES="${EVAL_GAMES:-60}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-mps}"

TS="$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="${SAVE_DIR:-runs/${TS}-az}"
mkdir -p "$SAVE_DIR"

echo "==> AlphaZero training"
echo "    init ckpt:      $INIT_CKPT"
echo "    save dir:       $SAVE_DIR"
echo "    iters:          $ITERS"
echo "    games_per_iter: $GAMES (buffer=$BUF iters)"
echo "    sgd:            $EPOCHS epochs, batch=$BATCH, lr=$LR"
echo "    mcts:           dets=$DETS sims=$SIMS batch=$MCTS_BATCH skip=$SKIP"
echo "    gumbel root:    $GUMBEL (top-m=$GUMBEL_N)"
echo "    pool gate:      admit only if WR vs best >= $POOL_WR (over $POOL_GAMES games)"
echo "    draft pool:     $DRAFT_POOL (per-game subset of 30 protocols)"
echo "    device:         $DEVICE"
echo ""

PYTHONPATH=src .venv/bin/python -u -m compile_engine.nn.train_alphazero \
    --init-ckpt "$INIT_CKPT" \
    --save-dir "$SAVE_DIR" \
    --iters "$ITERS" \
    --games-per-iter "$GAMES" \
    --buffer-iters "$BUF" \
    --sgd-epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --mcts-dets "$DETS" \
    --mcts-sims "$SIMS" \
    --mcts-batch "$MCTS_BATCH" \
    --skip-top-prob "$SKIP" \
    --use-gumbel-root "$GUMBEL" \
    --gumbel-n-candidates "$GUMBEL_N" \
    --pool-admission-wr "$POOL_WR" \
    --pool-admission-games "$POOL_GAMES" \
    --draft-pool-size "$DRAFT_POOL" \
    --snapshot-every "$SNAPSHOT_EVERY" \
    --eval-games "$EVAL_GAMES" \
    --seed "$SEED" \
    --device "$DEVICE" \
    2>&1 | tee "$SAVE_DIR/train.log"

echo ""
echo "==> Done. Run dir: $SAVE_DIR"
