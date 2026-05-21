#!/usr/bin/env bash
# Run the full eval pipeline for a training run: collect telemetry vs the
# baseline opponents, compute metrics, run the all-pairs Elo ladder, and
# render a Markdown + JSON model card per snapshot.
#
# Lays everything down under <run-dir>/eval/:
#     ladder.json
#     snapshot_NNNNN/vs_random.jsonl
#     snapshot_NNNNN/vs_greedy.jsonl
#     snapshot_NNNNN/metrics.json
#     snapshot_NNNNN/model_card.md
#     snapshot_NNNNN/model_card.json
#
# Usage:
#   ./scripts/eval_run.sh                                # runs/latest, every 50 iters
#   ./scripts/eval_run.sh runs/20260521-040612-loose-policy
#   STRIDE=20 ./scripts/eval_run.sh                      # eval more snapshots
#   GAMES=100 LADDER_GAMES=50 ./scripts/eval_run.sh      # tighter Elo
#   SNAPSHOTS="00100 00250 00410" ./scripts/eval_run.sh  # explicit list

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_DIR="${1:-runs/latest}"
RUN_DIR="$(readlink -f "$RUN_DIR" 2>/dev/null || python -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "$RUN_DIR")"
if [[ ! -d "$RUN_DIR" ]]; then
    echo "run dir not found: $RUN_DIR" >&2
    exit 1
fi

# Pick the Python interpreter.
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python"
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

DEVICE="${DEVICE:-$(default_device)}"
GAMES="${GAMES:-50}"            # games per matchup per snapshot
LADDER_GAMES="${LADDER_GAMES:-30}"  # games per ordered pair in ladder
STRIDE="${STRIDE:-50}"          # which snapshots to eval (every Nth iter)
SEED="${SEED:-0}"

EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$EVAL_DIR"

# Resolve snapshot list. Explicit SNAPSHOTS env wins; else iterate by stride.
if [[ -n "${SNAPSHOTS:-}" ]]; then
    SNAP_TAGS=( $SNAPSHOTS )
else
    SNAP_TAGS=()
    for path in $(ls "$RUN_DIR"/snapshot_*.pt 2>/dev/null | sort); do
        tag="$(basename "$path" .pt)"     # snapshot_00050
        n="${tag#snapshot_}"              # 00050
        # decimal-safe modulo (strip leading zeros)
        if (( 10#$n % STRIDE == 0 )); then
            SNAP_TAGS+=("$n")
        fi
    done
    # Always include the very last snapshot even if not a stride boundary.
    last_path="$(ls "$RUN_DIR"/snapshot_*.pt 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "$last_path" ]]; then
        last_tag="$(basename "$last_path" .pt)"
        last_n="${last_tag#snapshot_}"
        if [[ ${#SNAP_TAGS[@]} -eq 0 || "${SNAP_TAGS[-1]}" != "$last_n" ]]; then
            SNAP_TAGS+=("$last_n")
        fi
    fi
fi

if [[ ${#SNAP_TAGS[@]} -eq 0 ]]; then
    echo "no snapshots matched stride=$STRIDE under $RUN_DIR" >&2
    exit 1
fi

echo "Run dir:    $RUN_DIR"
echo "Eval dir:   $EVAL_DIR"
echo "Device:     $DEVICE"
echo "Games/MU:   $GAMES        (per matchup, per snapshot)"
echo "Ladder/pr:  $LADDER_GAMES (per ordered pair)"
echo "Snapshots:  ${SNAP_TAGS[*]}"
echo

# --- Step 1: per-snapshot telemetry + metrics -------------------------------
for n in "${SNAP_TAGS[@]}"; do
    tag="snapshot_$n"
    pt="$RUN_DIR/$tag.pt"
    if [[ ! -f "$pt" ]]; then
        echo "  [skip] $tag — checkpoint not found"
        continue
    fi
    out_dir="$EVAL_DIR/$tag"
    mkdir -p "$out_dir"

    echo "=== $tag ==="
    for opp in random greedy; do
        out="$out_dir/vs_$opp.jsonl"
        if [[ -f "$out" ]] && [[ "$(wc -l < "$out")" -ge "$GAMES" ]]; then
            echo "  [cache] $out ($(wc -l < "$out") games)"
            continue
        fi
        echo "  collect vs $opp …"
        PYTHONPATH=src "$PYTHON" -u scripts/eval/collect.py \
            --model "$pt" --opp "$opp" \
            --games "$GAMES" --seed "$SEED" --device "$DEVICE" \
            --out "$out"
    done
    echo "  metrics …"
    PYTHONPATH=src "$PYTHON" -u scripts/eval/metrics.py \
        --in "$out_dir" --model "$tag" \
        --out "$out_dir/metrics.json"
done

# --- Step 2: all-pairs Elo ladder over the same snapshots -------------------
echo
echo "=== ladder ==="
LADDER_OUT="$EVAL_DIR/ladder.json"
SNAP_PATHS=()
for n in "${SNAP_TAGS[@]}"; do
    p="$RUN_DIR/snapshot_$n.pt"
    if [[ -f "$p" ]]; then SNAP_PATHS+=("$p"); fi
done
PYTHONPATH=src "$PYTHON" -u scripts/eval/ladder.py \
    --snapshots "${SNAP_PATHS[@]}" \
    --games "$LADDER_GAMES" --seed "$SEED" --device "$DEVICE" \
    --out "$LADDER_OUT"

# --- Step 3: model cards (need both metrics and ladder) ---------------------
echo
echo "=== model cards ==="
for n in "${SNAP_TAGS[@]}"; do
    tag="snapshot_$n"
    metrics="$EVAL_DIR/$tag/metrics.json"
    [[ -f "$metrics" ]] || { echo "  [skip] $tag — no metrics.json"; continue; }
    PYTHONPATH=src "$PYTHON" -u scripts/eval/card.py \
        --metrics "$metrics" --ladder "$LADDER_OUT" \
        --out "$EVAL_DIR/$tag/model_card.md"
    echo "  wrote $EVAL_DIR/$tag/model_card.md"
done

echo
echo "Done. Inspect:"
echo "  cat $EVAL_DIR/snapshot_${SNAP_TAGS[-1]}/model_card.md"
echo "  cat $EVAL_DIR/ladder.json | jq '.ratings'"
