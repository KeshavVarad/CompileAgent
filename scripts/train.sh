#!/usr/bin/env bash
# Train the Compile NN agent. Each run is saved under runs/<timestamp>/ with
# checkpoints, the resolved CLI args, and a tee'd log file.
#
# Usage:
#   ./scripts/train.sh                       # defaults
#   ./scripts/train.sh --iters 500 --device mps
#   RUN_NAME=long ./scripts/train.sh --iters 1000
#   ./scripts/train.sh --resume runs/20260520-200000  # not yet implemented; see notes
#
# Defaults are tuned for an Apple-silicon laptop. Override any of them by
# passing the corresponding train_nn.py flag through.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Pick the Python interpreter: prefer the project venv if it exists, else
# fall back to whatever `python` is on PATH. Settable via $PYTHON if needed.
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON=".venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="python"
    else
        PYTHON="python3"
    fi
fi

# Auto-pick device: mps on Apple silicon, cuda if available, else cpu.
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

# Resolve the run directory: runs/<RUN_NAME or timestamp>/.
TS="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-$TS}"
RUN_DIR="${RUN_DIR:-runs/$RUN_NAME}"
mkdir -p "$RUN_DIR"

# Defaults — override on the command line, e.g. `./scripts/train.sh --iters 1000`.
ITERS="${ITERS:-500}"
GAMES_PER_ITER="${GAMES_PER_ITER:-32}"
LR="${LR:-3e-4}"
SEED="${SEED:-0}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-10}"
EVAL_GAMES="${EVAL_GAMES:-60}"
EXPANSION_PROB="${EXPANSION_PROB:-0.5}"
MAIN2_PROB="${MAIN2_PROB:-0.4}"
AUX2_PROB="${AUX2_PROB:-0.4}"
MAX_POOL_SIZE="${MAX_POOL_SIZE:-6}"
DEVICE="${DEVICE:-$(default_device)}"

LOG_FILE="$RUN_DIR/train.log"
ARGS_FILE="$RUN_DIR/args.txt"

# Record exactly what we're invoking, so the run is reproducible later.
{
    echo "repo:      $REPO"
    echo "ts:        $TS"
    echo "device:    $DEVICE"
    echo "iters:     $ITERS"
    echo "g/iter:    $GAMES_PER_ITER"
    echo "lr:        $LR"
    echo "seed:      $SEED"
    echo "snap_eve:  $SNAPSHOT_EVERY"
    echo "eval_g:    $EVAL_GAMES"
    echo "exp_prob:  $EXPANSION_PROB"
    echo "main2_pb:  $MAIN2_PROB"
    echo "aux2_pb:   $AUX2_PROB"
    echo "pool_max:  $MAX_POOL_SIZE"
    echo "extra:     $*"
    echo "python:    $($PYTHON -c 'import sys; print(sys.executable)')"
    echo "torch:     $($PYTHON -c 'import torch; print(torch.__version__)')"
    echo "git_head:  $(git rev-parse --short HEAD 2>/dev/null || echo 'not-a-git-repo')"
} | tee "$ARGS_FILE"

echo
echo "Logging to $LOG_FILE"
echo "Checkpoints will be written under $RUN_DIR/"
echo

# Run training. `-u` for unbuffered stdout so the log streams live.
"$PYTHON" -u scripts/train_nn.py \
    --iters "$ITERS" \
    --games-per-iter "$GAMES_PER_ITER" \
    --lr "$LR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --snapshot-every "$SNAPSHOT_EVERY" \
    --eval-games "$EVAL_GAMES" \
    --expansion-prob "$EXPANSION_PROB" \
    --main2-prob "$MAIN2_PROB" \
    --aux2-prob "$AUX2_PROB" \
    --max-pool-size "$MAX_POOL_SIZE" \
    --save-dir "$RUN_DIR" \
    "$@" 2>&1 | tee "$LOG_FILE"

# Convenience symlink to the latest run.
ln -sfn "$(basename "$RUN_DIR")" runs/latest

echo
echo "Done. Latest checkpoint:"
ls -1t "$RUN_DIR"/snapshot_*.pt 2>/dev/null | head -3 || echo "  (no snapshots written — was --snapshot-every larger than --iters?)"
