"""Live monitor for AlphaZero training metrics.

Tails the metrics.jsonl that train_alphazero.py writes, prints a
formatted table that updates as new iters land, and shows a progress bar
+ ETA based on the average per-iter wall clock so far.

Usage:
    python scripts/az/watch.py runs/<timestamp>-az/metrics.jsonl
    python scripts/az/watch.py runs/<timestamp>-az            # also accepts a run dir
    python scripts/az/watch.py                                # auto-pick newest -az run
    python scripts/az/watch.py --iters 500                    # override total-iters guess

The total-iters guess defaults to whatever `--iters` is passed; if
omitted, the script tries to sniff it from the run's `train.log` and
falls back to 500. The progress bar/ETA only depend on this guess —
the data itself comes from metrics.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# ANSI colors — kept minimal, gated behind a TTY check so piped output stays clean.
def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")


C_RESET = "\033[0m" if _supports_color() else ""
C_DIM = "\033[2m" if _supports_color() else ""
C_BOLD = "\033[1m" if _supports_color() else ""
C_GREEN = "\033[32m" if _supports_color() else ""
C_YELLOW = "\033[33m" if _supports_color() else ""
C_CYAN = "\033[36m" if _supports_color() else ""
C_RED = "\033[31m" if _supports_color() else ""


def find_metrics_path(arg: str | None) -> Path:
    """Resolve a CLI arg (file, dir, or None) to a metrics.jsonl path."""
    if arg:
        p = Path(arg)
        if p.is_file():
            return p
        if p.is_dir():
            cand = p / "metrics.jsonl"
            if cand.is_file():
                return cand
        raise SystemExit(f"no metrics.jsonl at {arg}")
    # Auto-pick the newest AZ run dir. Matches both `*-az` (original
    # naming) and `*-az-fresh` / `*-az-*` variants. We don't gate on
    # metrics.jsonl existing yet — a fresh run hasn't written the
    # first record, but we still want to point at it (it'll appear
    # after iter 1).
    candidates = list(Path("runs").glob("*-az")) + list(Path("runs").glob("*-az-*"))
    if not candidates:
        raise SystemExit("no *-az* run dir found in runs/; pass a path explicitly")
    runs = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] / "metrics.jsonl"


def fmt_dt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


def progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "-"
    filled = int(width * done / total)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def row_str(rec: dict) -> str:
    it = rec["iter"]
    dt = rec["dt"]
    pol = rec["pol_loss"]
    val = rec["val_loss"]
    kl = rec["kl_to_target"]
    wr = rec["rollout_wr"]
    searched = rec.get("searched", 0)
    skipped = rec.get("skipped", 0)
    wr_r = rec.get("wr_random")
    wr_g = rec.get("wr_greedy")

    eval_cell = ""
    if wr_r is not None or wr_g is not None:
        g = f"{wr_g:.2f}" if wr_g is not None else "  - "
        r = f"{wr_r:.2f}" if wr_r is not None else "  - "
        # Color eval rows so they jump out as the real signal.
        eval_cell = f"{C_GREEN}g={g} r={r}{C_RESET}"

    # Highlight kl row if it dropped substantially.
    return (
        f"{C_BOLD}{it:5d}{C_RESET} | "
        f"{fmt_dt(dt):>5} | "
        f"{pol:5.3f}  {val:5.3f}  {C_CYAN}{kl:5.3f}{C_RESET} | "
        f"{wr:5.2f}  | "
        f"{searched:4d}/{skipped:<3d} | "
        f"{eval_cell}"
    )


def render(records: list[dict], total_iters: int, run_label: str) -> None:
    # Clear screen, redraw the whole thing. Simple and robust against
    # column-width drift.
    print("\033[2J\033[H", end="")  # ANSI clear screen + home cursor

    n_done = len(records)
    if records:
        avg_dt = sum(r["dt"] for r in records) / n_done
        remaining_iters = max(0, total_iters - n_done)
        eta_s = remaining_iters * avg_dt
        eta_str = fmt_dt(eta_s)
        finish_at = (datetime.now() + timedelta(seconds=eta_s)).strftime("%H:%M")
        pct = n_done / total_iters * 100
    else:
        avg_dt = 0
        eta_str = "?"
        finish_at = "?"
        pct = 0

    print(f"{C_BOLD}═══ AZ training: {run_label} ═══{C_RESET}")
    print(f"  progress:   {progress_bar(n_done, total_iters)} {n_done}/{total_iters} ({pct:.1f}%)")
    print(f"  avg dt:     {fmt_dt(avg_dt) if avg_dt else '?'}/iter")
    print(f"  eta:        {eta_str}  (finish ~{finish_at})")
    print()
    print(f"  {C_DIM}iter | dt    | pol    val    kl→tgt | rollout | search/skip | eval{C_RESET}")
    print(f"  {C_DIM}─────┼───────┼──────────────────────┼─────────┼─────────────┼──────────────────{C_RESET}")
    # Show the last N rows so the table fits on most terminals.
    max_rows = max(8, _terminal_height() - 10)
    visible = records[-max_rows:]
    if len(records) > len(visible):
        print(f"  {C_DIM}... {len(records) - len(visible)} earlier iters elided ...{C_RESET}")
    for rec in visible:
        print("  " + row_str(rec))

    if records and any(r.get("wr_greedy") is not None for r in records):
        # Show the latest eval values prominently at the bottom.
        last_eval = next(
            (r for r in reversed(records) if r.get("wr_greedy") is not None), None,
        )
        if last_eval is not None:
            it = last_eval["iter"]
            wr_g = last_eval["wr_greedy"]
            wr_r = last_eval["wr_random"]
            print()
            print(
                f"  {C_BOLD}Last eval{C_RESET} (iter {it}): "
                f"vs greedy = {C_GREEN}{wr_g:.2f}{C_RESET}, "
                f"vs random = {C_GREEN}{wr_r:.2f}{C_RESET}"
            )

    print()
    print(f"  {C_DIM}Ctrl-C to exit. Polling every 5s.{C_RESET}")


def _terminal_height() -> int:
    try:
        return os.get_terminal_size().lines
    except OSError:
        return 30


def read_all(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Partial write race — skip this iter for now, we'll get it next poll.
                continue
    return out


def _sniff_total_iters(run_dir: Path, fallback: int = 500) -> int:
    """Scan the run's train.log for the `--iters N` invocation arg.
    Falls back to `fallback` if not found."""
    log = run_dir / "train.log"
    if not log.exists():
        return fallback
    try:
        with log.open() as f:
            head = f.read(4096)
        # Look for `--iters 500` style; greedy scan, take the first hit.
        import re
        m = re.search(r"--iters\s+(\d+)", head)
        if m:
            return int(m.group(1))
    except OSError:
        pass
    return fallback


def main() -> int:
    # Parse args: optional positional path + optional --iters N.
    cli_iters: int | None = None
    pos_arg: str | None = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--iters" and i + 1 < len(args):
            cli_iters = int(args[i + 1])
            i += 2
        elif a.startswith("--iters="):
            cli_iters = int(a.split("=", 1)[1])
            i += 1
        else:
            pos_arg = a
            i += 1
    metrics_path = find_metrics_path(pos_arg)
    run_label = str(metrics_path.parent.name)

    # Total-iters for the progress bar / ETA: --iters > train.log sniff > 500.
    total_iters = cli_iters if cli_iters is not None else _sniff_total_iters(metrics_path.parent)

    # Render once on startup so the user sees the header + "0/N" even
    # if metrics.jsonl doesn't exist yet (e.g. a fresh run before iter 1).
    render(read_all(metrics_path), total_iters, run_label)
    last_mtime: float | None = None
    try:
        while True:
            try:
                mtime = metrics_path.stat().st_mtime
            except FileNotFoundError:
                mtime = None
            if mtime != last_mtime:
                last_mtime = mtime
                records = read_all(metrics_path)
                render(records, total_iters, run_label)
                if records and records[-1]["iter"] >= total_iters:
                    print(f"\n{C_GREEN}{C_BOLD}Training complete.{C_RESET}")
                    return 0
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nbye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
