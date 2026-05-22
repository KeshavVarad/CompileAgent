"""Fine-tune the policy via cross-entropy against MCTS soft targets.

This is the "distill" half of expert iteration (ExIt). The value head
is frozen — it was trained on PPO self-play and is already calibrated;
we only want to push search-improved policy decisions back into the
policy weights.

Loss:
    For each labeled state, the model produces logits over MAX_ACTIONS
    (masked at padded slots). Softmax over the legal prefix gives a
    policy distribution. Loss is cross-entropy of that policy against
    the saved soft target:
        L = - sum_a target(a) * log_softmax(logits)(a)   for a in legal

    target is already 0 on padded slots, so we just zero-out the
    contribution from padded slots to avoid 0 * -inf NaN.

The output checkpoint matches the format `scripts/eval/_lib.py` and
`scripts/eval/collect.py` expect, so the standard eval pipeline runs on
it unmodified.

Usage:
    python scripts/distill/train.py \\
        --ckpt runs/latest/snapshot_00500.pt \\
        --labels runs/latest/distill/labels.pt \\
        --out runs/latest/distill/snapshot_00500_distilled.pt \\
        --epochs 3 --lr 1e-4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from compile_engine.nn.model import PolicyValueNet  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "eval"))
from _lib import load_model_from_ckpt, resolve_device  # noqa: E402


def _to_device(arr_dict: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in arr_dict.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="source checkpoint to fine-tune")
    p.add_argument("--labels", required=True, help="labels .pt produced by generate_labels.py")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-coef", type=float, default=0.0,
                   help="coefficient on the value-MSE loss. 0 = policy-only "
                        "(frozen value head); >0 = joint policy+value (head "
                        "unfrozen, MCTS root V used as the value target).")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    model: PolicyValueNet = load_model_from_ckpt(args.ckpt, device)
    model.train()

    # Value head: frozen for policy-only distillation (value_coef=0),
    # trainable for joint mode. Joint training keeps the value head
    # aligned with the drifting trunk; pure-policy mode preserves
    # PPO-trained calibration of the value head but risks the trunk
    # drifting past it (which matters for future ExIt rounds where
    # the value head feeds MCTS leaf evaluations).
    joint = args.value_coef > 0
    for p_ in model.value_head.parameters():
        p_.requires_grad = joint
    print(f"Mode: {'joint policy+value' if joint else 'policy-only (value head frozen)'}"
          f"  value_coef={args.value_coef}")

    payload = torch.load(args.labels, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    print(f"Loaded {meta.get('n_labeled', '?')} labels from {args.labels}")
    print(f"  source ckpt:  {meta.get('ckpt', '?')}")
    print(f"  tau:          {meta.get('tau', '?')}")
    print(f"  skip prob:    {meta.get('skip_top_prob', '?')}")

    # Move tensors to device once (dataset is small).
    state_tensors = _to_device(payload["state"], device)
    action_raw = torch.from_numpy(payload["action_raw"]).to(device)
    action_card = torch.from_numpy(payload["action_card_ids"]).to(device)
    action_proto = torch.from_numpy(payload["action_proto_ids"]).to(device)
    action_mask = torch.from_numpy(payload["action_mask"]).to(device).bool()
    target = torch.from_numpy(payload["target"]).to(device)
    n = target.shape[0]
    print(f"Dataset: {n} samples on {device}")

    value_target: torch.Tensor | None = None
    if joint:
        if "value_target" not in payload:
            raise SystemExit(
                "--value-coef > 0 but labels file has no `value_target` key. "
                "Re-generate labels with the updated generate_labels.py."
            )
        value_target = torch.from_numpy(payload["value_target"]).to(device).float()
        print(f"  value target: mean={value_target.mean().item():+.3f}  "
              f"std={value_target.std().item():.3f}  "
              f"range=[{value_target.min().item():+.2f}, {value_target.max().item():+.2f}]")

    opt = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Stats trackers
    initial_kl = _eval_kl(model, state_tensors, action_raw, action_card, action_proto,
                          action_mask, target, args.batch_size)
    print(f"Pre-train KL(target || model) = {initial_kl:.4f}")

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        running_loss = 0.0
        running_pol = 0.0
        running_val = 0.0
        running_kl = 0.0
        n_batches = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            b_state = {k: v[idx] for k, v in state_tensors.items()}
            logits, value_pred = model(
                b_state,
                action_raw[idx],
                action_card[idx],
                action_proto[idx],
                action_mask[idx],
            )
            # The model already masks padded slots to -1e9 internally,
            # so log_softmax produces -inf there. Zero out those terms
            # in the per-slot CE to avoid 0 * -inf = NaN.
            log_probs = F.log_softmax(logits, dim=-1)
            t = target[idx]
            ce_per_slot = torch.where(
                action_mask[idx],
                t * log_probs,
                torch.zeros_like(t),
            )
            pol_loss = -ce_per_slot.sum(dim=-1).mean()
            if joint and value_target is not None:
                val_loss = F.mse_loss(value_pred.squeeze(-1), value_target[idx])
            else:
                val_loss = torch.zeros((), device=device)
            loss = pol_loss + args.value_coef * val_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in model.parameters() if p_.requires_grad],
                max_norm=1.0,
            )
            opt.step()
            running_loss += float(loss.item())
            running_pol += float(pol_loss.item())
            running_val += float(val_loss.item())
            # KL(target || model) for monitoring — same as CE up to
            # the constant H(target), but more interpretable.
            with torch.no_grad():
                kl = (t * (torch.log(t + 1e-12) - log_probs)).where(
                    action_mask[idx], torch.zeros_like(t)
                ).sum(dim=-1).mean()
            running_kl += float(kl.item())
            n_batches += 1

        if joint:
            print(
                f"  epoch {epoch + 1}/{args.epochs}  "
                f"loss={running_loss / max(1, n_batches):.4f}  "
                f"pol={running_pol / max(1, n_batches):.4f}  "
                f"val={running_val / max(1, n_batches):.4f}  "
                f"KL={running_kl / max(1, n_batches):.4f}  "
                f"elapsed={time.perf_counter() - t0:.1f}s"
            )
        else:
            print(
                f"  epoch {epoch + 1}/{args.epochs}  "
                f"loss={running_loss / max(1, n_batches):.4f}  "
                f"KL={running_kl / max(1, n_batches):.4f}  "
                f"elapsed={time.perf_counter() - t0:.1f}s"
            )

    final_kl = _eval_kl(model, state_tensors, action_raw, action_card, action_proto,
                        action_mask, target, args.batch_size)
    print(f"Post-train KL(target || model) = {final_kl:.4f}  (Δ {final_kl - initial_kl:+.4f})")

    # Save with the same payload schema the eval pipeline uses.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "iter": -1,  # not a PPO iteration
            "distilled_from": args.ckpt,
            "labels": args.labels,
            "epochs": args.epochs,
            "lr": args.lr,
            "n_labels": int(n),
            "initial_kl": float(initial_kl),
            "final_kl": float(final_kl),
        },
        out,
    )
    print(f"\nWrote distilled checkpoint to {out}")
    return 0


@torch.no_grad()
def _eval_kl(
    model: PolicyValueNet,
    state: dict[str, torch.Tensor],
    raw: torch.Tensor,
    card: torch.Tensor,
    proto: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> float:
    """Mean KL(target || model_policy) over the whole dataset."""
    model.eval()
    n = target.shape[0]
    total = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        b_state = {k: v[start:end] for k, v in state.items()}
        logits, _ = model(b_state, raw[start:end], card[start:end],
                          proto[start:end], mask[start:end])
        log_probs = F.log_softmax(logits, dim=-1)
        t = target[start:end]
        m = mask[start:end]
        kl = (t * (torch.log(t + 1e-12) - log_probs)).where(m, torch.zeros_like(t)).sum(dim=-1).mean()
        total += float(kl.item())
        n_batches += 1
    model.train()
    return total / max(1, n_batches)


if __name__ == "__main__":
    raise SystemExit(main())
