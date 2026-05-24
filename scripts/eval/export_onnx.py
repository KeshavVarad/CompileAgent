"""Export a PolicyValueNet checkpoint to ONNX.

The model takes a dict input (state_batch) plus four action tensors; ONNX
prefers flat positional inputs, so we wrap the model in a small adapter
module that accepts positional tensors and reconstructs the dict.

We then verify the export by running both PyTorch and onnxruntime on a
random input and checking that the outputs match within float tolerance.

Usage:
    python scripts/eval/export_onnx.py \
        --ckpt runs/20260520-224058/snapshot_00120.pt \
        --out runs/20260520-224058/snapshot_00120.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from compile_engine.nn.encoder import MAX_ACTIONS, MAX_HAND, MAX_STACK, NUM_CARDS  # noqa: E402
from compile_engine.nn.model import PolicyValueNet  # noqa: E402


# Static dims the model expects. These are pinned at export time; the TS
# encoder must produce inputs with matching shapes.
N_LINES = 3
N_PHASES = 9
SCALARS_DIM = 8
# Action raw-features dim. Bumped when the encoder added action-lookahead
# closed-form features + a pending-card-context channel. Source of truth
# is encoder.encode_actions: this constant just needs to match the second
# axis of `raw_feats` it returns. Inspect with:
#   raw, *_ = encode_actions(game, legal, perspective); raw.shape[1]
RAW_ACTION_DIM = 38


class FlatPolicyValueNet(nn.Module):
    """Adapter exposing PolicyValueNet.forward with positional inputs only.

    All inputs are batch-first; we export with batch size = 1 (and a
    dynamic axis for the batch dim) so callers can pass any batch later.
    """

    def __init__(self, inner: PolicyValueNet) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        field_tokens: torch.Tensor,       # [B, 3, 2, MAX_STACK] int64
        field_flags: torch.Tensor,        # [B, 3, 2, MAX_STACK, 3] float32
        field_meta: torch.Tensor,         # [B, 3, 2, 3]         float32
        protocols: torch.Tensor,          # [B, 2, 3, 2]         int64
        hand_tokens: torch.Tensor,        # [B, MAX_HAND]        int64
        hand_size: torch.Tensor,          # [B, 1]               float32
        trash: torch.Tensor,              # [B, 2, NUM_CARDS]    float32
        line_vals: torch.Tensor,          # [B, 3, 2]            float32
        scalars: torch.Tensor,            # [B, 8]               float32
        phase: torch.Tensor,              # [B, 9]               float32
        pending_card_token: torch.Tensor, # [B]                  int64
        pending_category: torch.Tensor,   # [B, 17]              float32
        pending_depth_norm: torch.Tensor, # [B, 1]               float32
        action_raw: torch.Tensor,             # [B, MAX_ACTIONS, RAW_ACTION_DIM] float32
        action_card_ids: torch.Tensor,        # [B, MAX_ACTIONS]     int64
        action_proto_ids: torch.Tensor,       # [B, MAX_ACTIONS]     int64
        action_extra_card_ids: torch.Tensor,  # [B, MAX_ACTIONS]     int64
        action_mask: torch.Tensor,            # [B, MAX_ACTIONS]     bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = {
            "field_tokens": field_tokens,
            "field_flags": field_flags,
            "field_meta": field_meta,
            "protocols": protocols,
            "hand_tokens": hand_tokens,
            "hand_size": hand_size,
            "trash": trash,
            "line_vals": line_vals,
            "scalars": scalars,
            "phase": phase,
            "pending_card_token": pending_card_token,
            "pending_category": pending_category,
            "pending_depth_norm": pending_depth_norm,
        }
        logits, value = self.inner(
            state, action_raw, action_card_ids, action_proto_ids,
            action_extra_card_ids, action_mask,
        )
        return logits, value


def dummy_inputs(batch: int = 1) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(0)
    return {
        "field_tokens": torch.zeros(batch, N_LINES, 2, MAX_STACK, dtype=torch.int64),
        "field_flags": torch.zeros(batch, N_LINES, 2, MAX_STACK, 3, dtype=torch.float32),
        "field_meta": torch.zeros(batch, N_LINES, 2, 3, dtype=torch.float32),
        "protocols": torch.zeros(batch, 2, N_LINES, 2, dtype=torch.int64),
        "hand_tokens": torch.zeros(batch, MAX_HAND, dtype=torch.int64),
        "hand_size": torch.zeros(batch, 1, dtype=torch.float32),
        "trash": torch.zeros(batch, 2, NUM_CARDS, dtype=torch.float32),
        "line_vals": torch.zeros(batch, N_LINES, 2, dtype=torch.float32),
        "scalars": torch.zeros(batch, SCALARS_DIM, dtype=torch.float32),
        "phase": torch.zeros(batch, N_PHASES, dtype=torch.float32),
        "pending_card_token": torch.zeros(batch, dtype=torch.int64),
        "pending_category": torch.zeros(batch, 17, dtype=torch.float32),
        "pending_depth_norm": torch.zeros(batch, 1, dtype=torch.float32),
        "action_raw": torch.tensor(
            rng.standard_normal((batch, MAX_ACTIONS, RAW_ACTION_DIM)).astype(np.float32),
        ),
        "action_card_ids": torch.zeros(batch, MAX_ACTIONS, dtype=torch.int64),
        "action_proto_ids": torch.zeros(batch, MAX_ACTIONS, dtype=torch.int64),
        "action_extra_card_ids": torch.zeros(batch, MAX_ACTIONS, dtype=torch.int64),
        # Mark the first 5 actions as legal so masked softmax has something
        # to work with during the smoke test.
        "action_mask": torch.tensor(
            [[True] * 5 + [False] * (MAX_ACTIONS - 5)] * batch
        ),
    }


def verify(onnx_path: Path, torch_model: nn.Module) -> None:
    import onnxruntime as ort

    inputs = dummy_inputs(batch=1)
    with torch.no_grad():
        torch_logits, torch_value = torch_model(*[inputs[k] for k in [
            "field_tokens", "field_flags", "field_meta", "protocols",
            "hand_tokens", "hand_size", "trash", "line_vals", "scalars", "phase",
            "pending_card_token", "pending_category", "pending_depth_norm",
            "action_raw", "action_card_ids", "action_proto_ids",
            "action_extra_card_ids", "action_mask",
        ]])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.numpy() for k, v in inputs.items()}
    onnx_logits, onnx_value = sess.run(["logits", "value"], ort_inputs)

    logit_diff = float(np.max(np.abs(torch_logits.numpy() - onnx_logits)))
    value_diff = float(np.max(np.abs(torch_value.numpy() - onnx_value)))
    print(f"  max |Δlogits| = {logit_diff:.2e}")
    print(f"  max |Δvalue|  = {value_diff:.2e}")
    if logit_diff > 1e-3 or value_diff > 1e-3:
        raise SystemExit("ONNX outputs diverge from PyTorch; aborting")
    print("  ✓ ONNX output matches PyTorch within tolerance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    print(f"loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    inner = PolicyValueNet()
    inner.load_state_dict(state["model"], strict=False)
    inner.eval()
    wrapper = FlatPolicyValueNet(inner).eval()

    inputs = dummy_inputs(batch=1)
    input_names = [
        "field_tokens", "field_flags", "field_meta", "protocols",
        "hand_tokens", "hand_size", "trash", "line_vals", "scalars", "phase",
        "pending_card_token", "pending_category", "pending_depth_norm",
        "action_raw", "action_card_ids", "action_proto_ids",
        "action_extra_card_ids", "action_mask",
    ]
    # Dynamic axis 0 (batch). MAX_ACTIONS / MAX_HAND / MAX_STACK are fixed.
    dynamic_axes = {n: {0: "batch"} for n in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    dynamic_axes["value"] = {0: "batch"}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting ONNX (opset {args.opset}) to {out_path}")
    torch.onnx.export(
        wrapper,
        tuple(inputs[k] for k in input_names),
        out_path,
        input_names=input_names,
        output_names=["logits", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    # The new torch.onnx exporter sometimes shards weights into an adjacent
    # `.onnx.data` file. We're a tiny model (~80kB), so collapse everything
    # into a single self-contained .onnx file to keep the deploy bundle
    # clean and the webapp's session loader simple.
    import onnx
    loaded = onnx.load(str(out_path), load_external_data=True)
    onnx.save(loaded, str(out_path), save_as_external_data=False)
    sidecar = out_path.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  wrote {size_mb:.1f} MB (single-file)")
    print("verifying with onnxruntime …")
    verify(out_path, wrapper)


if __name__ == "__main__":
    main()
