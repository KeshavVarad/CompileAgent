"""Train the Compile NN agent via PPO.

Usage:
    python scripts/train_nn.py --iters 200 --device auto --save-dir runs/v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from compile_engine.nn.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--games-per-iter", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--save-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--expansion-prob", type=float, default=0.5,
                    help="probability of including AX01 (Apathy/Hate/Love) per game")
    ap.add_argument("--main2-prob", type=float, default=0.4,
                    help="probability of including MN02 (Chaos/Clarity/.../War) per game")
    ap.add_argument("--aux2-prob", type=float, default=0.4,
                    help="probability of including AX02 (Assimilation/Diversity/Unity) per game")
    ap.add_argument("--target-kl", type=float, default=0.03,
                    help="break PPO epoch loop once approx-KL exceeds this; <=0 disables")
    ap.add_argument("--pool-threshold", type=float, default=0.0,
                    help="only add snapshots to opponent pool once wr_random ≥ this")
    ap.add_argument("--max-pool-size", type=int, default=16,
                    help="cap total opponent pool size; the most-beaten non-anchor "
                         "snapshot is evicted on overflow")
    ap.add_argument("--entropy-floor", type=float, default=0.4,
                    help="adaptive c_entropy target floor for policy entropy. "
                         "set to <0 to disable adaptive entropy and use fixed c_entropy.")
    ap.add_argument("--entropy-ceiling", type=float, default=0.55,
                    help="upper end of the entropy comfort band (when measured "
                         "entropy is above this, c_entropy is eased down)")
    ap.add_argument("--c-entropy-start", type=float, default=0.01,
                    help="initial value for c_entropy before adaptive scaling kicks in")
    ap.add_argument("--pfsp-p", type=float, default=2.0,
                    help="PFSP exponent; sampling weight = max(min_weight, "
                         "(1-WR)^p). p=0 is uniform, higher focuses harder on "
                         "weak spots.")
    ap.add_argument("--pfsp-min-weight", type=float, default=0.04,
                    help="floor on PFSP weights so even fully-beaten opponents "
                         "still get sampled occasionally")
    ap.add_argument("--pfsp-window", type=int, default=80,
                    help="rolling window length for per-opponent win-rate")
    ap.add_argument("--init-ckpt", type=str, default=None,
                    help="Optional warm-start: path to a snapshot_*.pt file. "
                         "PPO and AZ trainers share the snapshot format, so "
                         "you can hand an AZ checkpoint to PPO here.")
    # Per-action-type entropy multipliers — DRAFT defaults to 4x to counter
    # the mode collapse documented in docs/STRATEGY_THESIS_sparkv4.md.
    # These are STARTING values; the per-class adaptive controller (flags
    # below) adjusts them each iter to keep per-class entropy in a target
    # band, preventing the runaway exploration that destroyed iter 60+
    # of runs/20260524-023022-ppo-entropy-aux.
    ap.add_argument("--draft-entropy-mult", type=float, default=4.0,
                    help="starting multiplier on c_entropy for DRAFT decisions. "
                         "set high (3-8) to break Darkness-98%% mode collapse; "
                         "the adaptive controller eases it back when DRAFT "
                         "entropy crosses --draft-entropy-ceiling.")
    ap.add_argument("--choose-entropy-mult", type=float, default=2.0,
                    help="starting multiplier on c_entropy for CHOOSE_TARGET. "
                         "boosts exploration of optional clauses (Love 1 End "
                         "was rejected 47/47 times under mult=1).")
    # Per-class adaptive controller target bands.
    ap.add_argument("--draft-entropy-floor", type=float, default=0.8,
                    help="lower bound of DRAFT entropy target band (nats). "
                         "below this, the DRAFT multiplier is bumped up by "
                         "the adaptive controller. set <0 to disable.")
    ap.add_argument("--draft-entropy-ceiling", type=float, default=1.4,
                    help="upper bound of DRAFT entropy target band (nats). "
                         "above this, the DRAFT multiplier is eased down. "
                         "Was missing in run 20260524-023022 — DRAFT ran "
                         "from 1.25 to 2.46 nats, destroying play structure.")
    ap.add_argument("--choose-entropy-floor", type=float, default=0.5,
                    help="lower bound of CHOOSE entropy target band (nats). "
                         "set <0 to disable adaptive control on CHOOSE.")
    ap.add_argument("--choose-entropy-ceiling", type=float, default=1.0,
                    help="upper bound of CHOOSE entropy target band (nats).")
    # UNREAL-style aux loss coefficients.
    ap.add_argument("--c-aux-opp-hand", type=float, default=0.05,
                    help="coefficient on auxiliary 'predict opp's hand "
                         "contents' BCE loss. 0 disables the aux head.")
    ap.add_argument("--c-aux-margin", type=float, default=0.05,
                    help="coefficient on auxiliary 'predict final compile "
                         "margin' regression loss. 0 disables.")
    args = ap.parse_args()

    cfg = TrainConfig(
        iters=args.iters,
        games_per_iter=args.games_per_iter,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
        eval_games=args.eval_games,
        expansion_prob=args.expansion_prob,
        main2_prob=args.main2_prob,
        aux2_prob=args.aux2_prob,
        target_kl=args.target_kl if args.target_kl > 0 else None,
        pool_threshold_wr_random=args.pool_threshold,
        max_pool_size=args.max_pool_size,
        c_entropy=args.c_entropy_start,
        entropy_floor=args.entropy_floor if args.entropy_floor >= 0 else None,
        entropy_ceiling=args.entropy_ceiling,
        pfsp_p=args.pfsp_p,
        pfsp_min_weight=args.pfsp_min_weight,
        pfsp_window=args.pfsp_window,
        init_ckpt=args.init_ckpt,
        # Order matches ACTION_CLASS_NAMES: DRAFT, PLAY, CHOOSE, COMPILE,
        # DISCARD, SHIFT. Only DRAFT and CHOOSE are exposed as CLI knobs;
        # the others stay at 1.0 (standard entropy bonus).
        per_class_entropy_mult=(
            args.draft_entropy_mult, 1.0, args.choose_entropy_mult,
            1.0, 1.0, 1.0,
        ),
        per_class_entropy_floor=(
            args.draft_entropy_floor if args.draft_entropy_floor >= 0 else None,
            None,
            args.choose_entropy_floor if args.choose_entropy_floor >= 0 else None,
            None, None, None,
        ),
        per_class_entropy_ceiling=(
            args.draft_entropy_ceiling,
            None,
            args.choose_entropy_ceiling,
            None, None, None,
        ),
        c_aux_opp_hand=args.c_aux_opp_hand,
        c_aux_margin=args.c_aux_margin,
    )
    train(cfg)


if __name__ == "__main__":
    main()
