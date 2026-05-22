# Strategic-discoveries analysis suite

A set of scripts that analyse a trained Compile checkpoint and produce
shareable markdown + JSON + PNG artifacts describing what the model has
learned. Intended for publishing findings (blog posts, GitHub releases,
research notes) rather than for in-loop training diagnostics.

All scripts:

- Take a checkpoint path (`.pt` snapshot saved by `train.py` or
  `train_alphazero.py`).
- Run on top of [`scripts/eval/collect.py`](../eval/collect.py) telemetry,
  caching the per-game JSONL under `analysis/<ckpt_stem>/` so repeated
  runs don't redo work.
- Produce a `*.md` (human-facing), a `*.json` sidecar (machine-facing),
  and (where applicable) a PNG chart.

## The scripts

| script | purpose |
|---|---|
| [`protocol_meta.py`](protocol_meta.py) | Per-protocol pick rate + WR + 2-protocol pair synergies + heatmap |
| [`style_fingerprint.py`](style_fingerprint.py) | Playstyle metrics (face-up ratio, refresh rate, compile aggression, …) + radar chart. Accepts multiple checkpoints for side-by-side comparison |
| [`decision_diff.py`](decision_diff.py) | Where does model A pick different actions than model B on the same states? Reports disagreement frequency by decision class + outcome correlation |
| [`ladder.py`](ladder.py) | Round-robin among N checkpoints → Elo ranking + head-to-head table + Elo-curve chart |
| [`discoveries.py`](discoveries.py) | Master orchestrator: runs all of the above for one checkpoint and assembles a top-level index report |

## Quick start

```bash
# Single-checkpoint report (no comparison)
.venv/bin/python scripts/analysis/discoveries.py \
    runs/latest/snapshot_00500.pt

# With a baseline checkpoint for diff
.venv/bin/python scripts/analysis/discoveries.py \
    runs/latest/snapshot_00500.pt \
    --baseline runs/latest/snapshot_00100.pt

# Full report with progression ladder
.venv/bin/python scripts/analysis/discoveries.py \
    runs/latest/snapshot_00500.pt \
    --baseline runs/latest/snapshot_00100.pt \
    --ladder runs/latest/snapshot_00200.pt runs/latest/snapshot_00300.pt runs/latest/snapshot_00400.pt
```

Output lands at `analysis/<ckpt_stem>/discoveries.md` and links to the
sub-reports.

## Compute footprint

For each checkpoint:

- `protocol_meta.py`: 200 vs Greedy + 200 vs Random → ~60s on MPS
- `style_fingerprint.py`: reuses the protocol_meta data → ~0s
- `decision_diff.py`: 100 fresh games with double inference → ~3 min on MPS
- `ladder.py`: round-robin across N checkpoints → N × (N-1) / 2 × games × 2s

For typical use (single checkpoint + baseline + 3-snapshot ladder), the
full `discoveries.py` run takes about 10 minutes on a laptop.

## Layout of outputs

```
analysis/<ckpt_stem>/
├── discoveries.md                  # master index
├── protocol_meta.md                # section 1
├── protocol_meta.json
├── pair_heatmap.png
├── style_fingerprint.md            # section 2
├── style_fingerprint.json
├── style_radar.png
├── vs_greedy.jsonl                 # cached eval games (200)
├── vs_random.jsonl                 # cached eval games (200)
├── decision_diff_vs_<baseline>/    # only if --baseline given
│   ├── decision_diff.md
│   └── decision_diff.json
└── ladder/                         # only if --ladder given
    ├── ladder.md
    ├── ladder.json
    └── elo_curve.png
```

## Notes for publishing

The markdown reports are designed to be standalone — each section
explains its own table headers and metric definitions, so you can link
to any single report without context. Images are referenced by relative
path so a `analysis/<stem>/` directory works as a self-contained bundle
to upload anywhere.

JSON sidecars carry the raw numbers behind every table. If you want to
re-render the same data with different formatting, that's where to pull
from.

When publishing, consider including:
- The exact CLI command used (the master report writes this automatically)
- The training config (or a pointer to the run dir's metrics.jsonl)
- A timestamp (also in the master report)
- The Compile rulebook reference (the analyses assume base rules + any
  enabled expansions)
