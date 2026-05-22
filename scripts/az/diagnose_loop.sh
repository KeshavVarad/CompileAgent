#!/usr/bin/env bash
# Background watcher that runs scripts/az/diagnose.py on every snapshot
# in a run dir, picking up new ones as they land. The diagnose script
# skips work if its report.md already exists, so this is safe to leave
# running across multiple training iterations.
#
# Usage:
#   scripts/az/diagnose_loop.sh runs/<az-run-dir> <source_ckpt>
#   # Optional env: POLL_S=60 GAMES=100
#
# Tip: run in the background and tee to a log:
#   scripts/az/diagnose_loop.sh runs/.../-az ... > /tmp/diag.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_DIR="${1:?usage: $0 <run-dir> <source-ckpt>}"
SOURCE="${2:?usage: $0 <run-dir> <source-ckpt>}"
POLL_S="${POLL_S:-60}"
GAMES="${GAMES:-100}"
DEVICE="${DEVICE:-mps}"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "no such run dir: $RUN_DIR" >&2
    exit 2
fi
if [[ ! -f "$SOURCE" ]]; then
    echo "no such source ckpt: $SOURCE" >&2
    exit 2
fi

echo "Watching $RUN_DIR for new snapshots (poll every ${POLL_S}s)"
echo "Source for comparison: $SOURCE"
echo ""

while true; do
    for ckpt in "$RUN_DIR"/snapshot_*.pt; do
        # Bash gives us the literal pattern if there are no matches.
        [[ -f "$ckpt" ]] || continue
        stem=$(basename "$ckpt" .pt)
        report="$RUN_DIR/diagnostic_${stem}/report.md"
        if [[ -f "$report" ]]; then
            continue
        fi
        echo "[$(date +%H:%M:%S)] diagnosing $stem"
        .venv/bin/python scripts/az/diagnose.py \
            "$ckpt" --source "$SOURCE" \
            --games "$GAMES" --device "$DEVICE" \
            2>&1 | sed 's/^/  /'
    done
    sleep "$POLL_S"
done
