"""Policy/value network for the Compile NN agent.

Architecture (see docs/nn_design.md):
  - Card embedding (vocab 92) and protocol embedding (vocab 16).
  - State encoder: aggregates per-stack card embeddings (mean + max),
    concatenates with protocols, hand, trash counts, line values, scalars.
  - Trunk MLP: → 256-d hidden.
  - Per-action encoder: small MLP mapping raw action features + card/proto
    embedding lookups → 256-d.
  - Policy: logits_i = <state_hidden, action_emb_i>, masked softmax.
  - Value: tanh(W · state_hidden) ∈ [-1, +1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cards import (
    CARD_STATIC_FEATS_DIM,
    load_card_defs,
    static_features_for_def,
)
from .encoder import (
    CARD_VOCAB_SIZE,
    MAX_ACTIONS,
    MAX_HAND,
    MAX_STACK,
    PROTO_VOCAB_SIZE,
    action_input_dim,
    card_repr_dim,
    state_input_dim,
)


class _MLP(nn.Module):
    """Small fully-connected stack with LayerNorm + ReLU."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _aggregate_stack(card_embs: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Per-stack aggregation: mean + max of card embeddings, plus meta scalars.

    Args
    ----
    card_embs : [B, 3, 2, MAX_STACK, d_card]   — embedding of each slot token
                                                  (PAD lookups are zeros)
    meta      : [B, 3, 2, 3]                   — (fu_count, fd_count, depth_norm)

    Returns
    -------
    [B, 3 * 2 * (2*d_card + 3)] — flattened per-stack features.
    """
    # Build a mask of "real" slots from the non-zero token positions implied
    # by non-zero embeddings. Easier: pass meta-derived depth.
    mean_emb = card_embs.mean(dim=-2)                     # [B, 3, 2, d_card]
    max_emb = card_embs.amax(dim=-2)                      # [B, 3, 2, d_card]
    out = torch.cat([mean_emb, max_emb, meta], dim=-1)    # [B, 3, 2, 2*d_card + 3]
    B = out.shape[0]
    return out.reshape(B, -1)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        d_card: int = 32,
        d_proto: int = 16,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.d_card = d_card
        self.d_proto = d_proto
        self.hidden = hidden
        self.card_repr_dim = card_repr_dim(d_card)

        self.card_emb = nn.Embedding(CARD_VOCAB_SIZE, d_card, padding_idx=0)
        self.proto_emb = nn.Embedding(PROTO_VOCAB_SIZE, d_proto, padding_idx=0)

        # Static card features (keyword multi-hot + value one-hot + text-presence
        # flags) loaded from the card data once and registered as a fixed
        # (non-learned) buffer. Rows 0 (PAD) and 1 (HIDDEN) are all zeros.
        defs = load_card_defs()
        static = torch.zeros(CARD_VOCAB_SIZE, CARD_STATIC_FEATS_DIM, dtype=torch.float32)
        for d in defs:
            static[d.def_id + 2] = torch.tensor(
                static_features_for_def(d), dtype=torch.float32,
            )
        self.register_buffer("card_static", static)

        # State trunk
        in_dim = state_input_dim(d_card, d_proto)
        self.state_in_dim = in_dim
        self.state_trunk = _MLP(in_dim, hidden, hidden, n_layers=3)
        self.state_ln = nn.LayerNorm(hidden)

        # Action encoder
        from ..actions import ActionType
        n_atypes = len(ActionType)
        # Mirrors `encoder.encode_actions` layout: type_one_hot + hand_idx +
        # src_line(4) + dst_line(4) + choice_idx + target_meta(8) + stated_val.
        self.action_raw_dim = n_atypes + 1 + 4 + 4 + 1 + 8 + 1
        self.action_mlp = _MLP(
            self.action_raw_dim + self.card_repr_dim + d_proto, hidden, hidden, n_layers=2,
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        # Init the last value linear small so the bar starts near 0.
        nn.init.normal_(self.value_head[-2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_head[-2].bias)

    def lookup_card(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Card token → (learned embedding ‖ static features). Static features
        for PAD (0) and HIDDEN (1) are zeros, so they contribute nothing."""
        learned = self.card_emb(token_ids)            # [..., d_card]
        static = self.card_static[token_ids]          # [..., CARD_STATIC_FEATS_DIM]
        return torch.cat([learned, static], dim=-1)   # [..., card_repr_dim]

    # ------------------------------------------------------------------- state

    def encode_state_dense(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the dense state vector from a batched observation dict.

        Expected keys (each tensor is batch-first):
          field_tokens : [B, 3, 2, MAX_STACK]    int64
          field_flags  : [B, 3, 2, MAX_STACK, 3] float
          field_meta   : [B, 3, 2, 3]            float
          protocols    : [B, 2, 3, 2]            int64 (id, compiled)
          hand_tokens  : [B, MAX_HAND]           int64
          hand_size    : [B, 1]                  float
          trash        : [B, 2, NUM_CARDS]       float
          line_vals    : [B, 3, 2]               float
          scalars      : [B, 7]                  float
          phase        : [B, 9]                  float
        """
        field_tokens = batch["field_tokens"]
        card_embs = self.lookup_card(field_tokens)  # [B, 3, 2, MAX_STACK, card_repr_dim]
        field_meta = batch["field_meta"]
        field_feat = _aggregate_stack(card_embs, field_meta)  # [B, *]

        # Protocols
        proto_ids = batch["protocols"][..., 0]    # [B, 2, 3]
        compiled = batch["protocols"][..., 1].float().unsqueeze(-1)  # [B, 2, 3, 1]
        proto_e = self.proto_emb(proto_ids)       # [B, 2, 3, d_proto]
        proto_feat = torch.cat([proto_e, compiled], dim=-1)   # [B, 2, 3, d_proto+1]
        B = proto_feat.shape[0]
        proto_feat = proto_feat.reshape(B, -1)

        # Hand
        hand_e = self.lookup_card(batch["hand_tokens"])       # [B, MAX_HAND, card_repr_dim]
        hand_mean = hand_e.mean(dim=-2)
        hand_max = hand_e.amax(dim=-2)
        hand_feat = torch.cat([hand_mean, hand_max, batch["hand_size"]], dim=-1)

        # Trash (already normalised)
        trash_feat = batch["trash"].reshape(B, -1)

        # Line values
        lv = batch["line_vals"].reshape(B, -1)

        # Scalars + phase
        sc = batch["scalars"]
        ph = batch["phase"]

        dense = torch.cat(
            [field_feat, proto_feat, hand_feat, trash_feat, lv, sc, ph],
            dim=-1,
        )
        return dense

    # ----------------------------------------------------------------- actions

    def encode_actions_dense(
        self,
        raw: torch.Tensor,        # [B, MAX_ACTIONS, raw_dim]
        card_ids: torch.Tensor,   # [B, MAX_ACTIONS]
        proto_ids: torch.Tensor,  # [B, MAX_ACTIONS]
    ) -> torch.Tensor:
        card_e = self.lookup_card(card_ids)  # [B, MAX_ACTIONS, card_repr_dim]
        proto_e = self.proto_emb(proto_ids)  # [B, MAX_ACTIONS, d_proto]
        feats = torch.cat([raw, card_e, proto_e], dim=-1)
        return self.action_mlp(feats)        # [B, MAX_ACTIONS, hidden]

    # -------------------------------------------------------------- forward(s)

    def forward(
        self,
        state_batch: dict[str, torch.Tensor],
        action_raw: torch.Tensor,
        action_card_ids: torch.Tensor,
        action_proto_ids: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value).

        logits: [B, MAX_ACTIONS] — masked (padded entries set to -1e9).
        value:  [B]              — scalar in [-1, +1].
        """
        dense = self.encode_state_dense(state_batch)
        h = self.state_ln(self.state_trunk(dense))         # [B, hidden]

        a_h = self.encode_actions_dense(action_raw, action_card_ids, action_proto_ids)
        # Dot product over the hidden dim.
        logits = torch.einsum("bh,bah->ba", h, a_h)         # [B, MAX_ACTIONS]
        logits = logits.masked_fill(~action_mask, -1e9)

        value = self.value_head(h).squeeze(-1)              # [B]
        return logits, value
