#!/usr/bin/env bash
# Train the Compile NN agent with loosened policy regularization.
#
# Why a separate script: the previous Sparkv1 run collapsed entropy from
# 0.83 → 0.23 (top_prob ≈ 0.97) and the adaptive entropy mechanism never
# recovered because PPO was bailing out of the epoch loop after 1 epoch
# every iter (target_kl=0.03 hit on a single minibatch). The MCTS
# diagnostic confirmed the symptom: 99% of search decisions match
# policy argmax. This recipe loosens the trust region so the entropy
# bonus has room to land.
#
# Knobs (relative to scripts/train.sh defaults):
#   target_kl       0.03 → 0.10   most important — let updates land
#   entropy_floor    0.4 → 0.8    aim for top_prob ≈ 0.85, not 0.97
#   entropy_ceiling 0.55 → 1.0    proportional
#   c_entropy_start 0.01 → 0.05   don't waste early iters climbing
#   max_pool_size      6 → 16     diversity from PFSP pool
#   seed              0 → 1       fresh sample, comparable to old run 0
#
# Usage:
#   ./scripts/train_loose_policy.sh
#   ./scripts/train_loose_policy.sh --iters 300         # short run
#   ITERS=1000 ./scripts/train_loose_policy.sh          # via env
#   RUN_NAME=loose-v2 ./scripts/train_loose_policy.sh
#
# Wrapped in `caffeinate -i` so the Mac doesn't idle-sleep during the
# multi-hour run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Pick the Python interpreter: prefer the project venv if it exists.
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON=".venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="python"
    else
        PYTHON="python3"
    fi
fi

default_device() {
    "$PYTHON" - <<'PY'
import torch
if torch.backends.mps.is_available():
    print("mps")
elif torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
PY
}

TS="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-${TS}-loose-policy}"
RUN_DIR="${RUN_DIR:-runs/$RUN_NAME}"
mkdir -p "$RUN_DIR"

ITERS="${ITERS:-500}"
DEVICE="${DEVICE:-$(default_device)}"
SEED="${SEED:-1}"

# Loosened regularization knobs (the whole point of this script).
TARGET_KL="${TARGET_KL:-0.10}"
ENTROPY_FLOOR="${ENTROPY_FLOOR:-0.8}"
ENTROPY_CEILING="${ENTROPY_CEILING:-1.0}"
C_ENTROPY_START="${C_ENTROPY_START:-0.05}"
MAX_POOL_SIZE="${MAX_POOL_SIZE:-16}"

LOG_FILE="$RUN_DIR/train.log"
ARGS_FILE="$RUN_DIR/args.txt"

{
    echo "recipe:           loose-policy"
    echo "repo:             $REPO"
    echo "ts:               $TS"
    echo "device:           $DEVICE"
    echo "iters:            $ITERS"
    echo "seed:             $SEED"
    echo "target_kl:        $TARGET_KL"
    echo "entropy_floor:    $ENTROPY_FLOOR"
    echo "entropy_ceiling:  $ENTROPY_CEILING"
    echo "c_entropy_start:  $C_ENTROPY_START"
    echo "max_pool_size:    $MAX_POOL_SIZE"
    echo "extra:            $*"
    echo "python:           $($PYTHON -c 'import sys; print(sys.executable)')"
    echo "torch:            $($PYTHON -c 'import torch; print(torch.__version__)')"
    echo "git_head:         $(git rev-parse --short HEAD 2>/dev/null || echo 'not-a-git-repo')"
    echo "git_branch:       $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
} | tee "$ARGS_FILE"

echo
echo "Logging to $LOG_FILE"
echo "Checkpoints will be written under $RUN_DIR/"
echo "Tail entropy/KL live:"
echo "  tail -f $RUN_DIR/metrics.jsonl | jq -c '{iter, ent: .entropy, kl: .approx_kl, stop: .stopped_at_epoch}'"
echo

# `caffeinate -i` blocks idle sleep for the duration of the python
# process and releases automatically on exit. `-u` on python keeps
# stdout unbuffered so `tee` streams the log live.
caffeinate -i "$PYTHON" -u scripts/train_nn.py \
    --iters "$ITERS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --target-kl "$TARGET_KL" \
    --entropy-floor "$ENTROPY_FLOOR" \
    --entropy-ceiling "$ENTROPY_CEILING" \
    --c-entropy-start "$C_ENTROPY_START" \
    --max-pool-size "$MAX_POOL_SIZE" \
    --save-dir "$RUN_DIR" \
    "$@" 2>&1 | tee "$LOG_FILE"

ln -sfn "$(basename "$RUN_DIR")" runs/latest

echo
echo "Done. Latest checkpoints:"
ls -1t "$RUN_DIR"/snapshot_*.pt 2>/dev/null | head -3 || echo "  (no snapshots written — was --snapshot-every larger than --iters?)"
