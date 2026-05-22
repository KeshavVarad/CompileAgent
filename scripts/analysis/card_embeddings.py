"""Card-embedding investigation: what has the model learned about cards?

Pulls the learned card embedding matrix out of a checkpoint, runs PCA
and t-SNE to project it into 2D, and writes a markdown report with:

  * 2D scatter plots colored by protocol, value, set, and keywords
  * k-nearest-neighbour table per card (cosine distance) — surfaces
    "the model treats these as similar" pairs
  * cross-protocol neighbour analysis — cards close in embedding space
    that come from *different* protocols (these are the functional-role
    overlaps that hint at synergy)
  * (optional, --compare-to) embedding-drift analysis between two
    checkpoints — shows which cards' representations moved most as the
    model trained

Caveat: card embeddings encode "what is similar from the network's POV,"
not "what cards combo together." Use this to find functional groupings
and to spot training-induced shifts; pair-co-occurrence analysis (see
protocol_meta.py) is what actually measures synergy.

Usage:
    python scripts/analysis/card_embeddings.py runs/.../snapshot.pt
    python scripts/analysis/card_embeddings.py runs/.../snapshot.pt \\
        --compare-to runs/.../baseline.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from compile_engine.cards import (  # noqa: E402
    AUX2_PROTOCOLS,
    BASE_PROTOCOLS,
    EXPANSION_PROTOCOLS,
    MAIN2_PROTOCOLS,
    load_card_defs,
)
from compile_engine.nn.encoder import _CARD_TOKEN_OFFSET  # noqa: E402
from compile_engine.nn.model import PolicyValueNet  # noqa: E402

from _lib import analysis_dir, write_json, write_md  # noqa: E402


# Map protocol → set code for coloring.
_PROTO_SET: dict[str, str] = {}
for p in BASE_PROTOCOLS:
    _PROTO_SET[p] = "MN01"
for p in EXPANSION_PROTOCOLS:
    _PROTO_SET[p] = "AX01"
for p in MAIN2_PROTOCOLS:
    _PROTO_SET[p] = "MN02"
for p in AUX2_PROTOCOLS:
    _PROTO_SET[p] = "AX02"


def load_card_embeddings(ckpt_path: Path, device: str = "cpu") -> tuple[np.ndarray, list[dict]]:
    """Load `card_emb.weight` from a checkpoint, plus per-row metadata.

    Returns (embeddings, metadata) where embeddings has shape [N, d_card]
    (N = number of real cards, excluding PAD/HIDDEN), and metadata[i]
    describes card i with {def_id, key, protocol, value, set_code,
    has_top, has_middle, has_bottom, keywords}."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = state["model"]
    if "card_emb.weight" not in model_state:
        raise SystemExit("ckpt has no card_emb.weight — wrong model architecture?")
    table = model_state["card_emb.weight"].cpu().numpy()
    # Rows 0=PAD, 1=HIDDEN; real cards start at _CARD_TOKEN_OFFSET (=2).
    real = table[_CARD_TOKEN_OFFSET:]

    defs = load_card_defs()
    by_def: dict[int, object] = {d.def_id: d for d in defs}
    meta: list[dict] = []
    for i in range(real.shape[0]):
        def_id = i  # rows are indexed by def_id (offset already removed)
        d = by_def.get(def_id)
        if d is None:
            # Padded slot — embedding exists but no real card. Skip.
            meta.append({
                "def_id": def_id, "key": f"<unused def_{def_id}>",
                "protocol": "?", "value": -1, "set_code": "?",
                "has_top": False, "has_middle": False, "has_bottom": False,
                "keywords": [],
            })
            continue
        meta.append({
            "def_id": def_id,
            "key": d.key,
            "protocol": d.protocol,
            "value": d.value,
            "set_code": d.set_code,
            "has_top": bool(d.top_text),
            "has_middle": bool(d.middle_text),
            "has_bottom": bool(d.bottom_text),
            "keywords": sorted(d.keywords),
        })

    # Drop unused slots so the analysis works on real cards only.
    real_idx = [i for i, m in enumerate(meta) if m["protocol"] != "?"]
    real = real[real_idx]
    meta = [meta[i] for i in real_idx]
    return real, meta


def cosine_knn(emb: np.ndarray, k: int = 5) -> np.ndarray:
    """Cosine-similarity top-k for each row (excluding self). Returns
    indices shape [N, k]."""
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    # argsort gives ascending; we want descending → flip.
    order = np.argsort(-sim, axis=1)
    return order[:, :k]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def scatter(
    points: np.ndarray, meta: list[dict], color_by: str, out_path: Path,
    title: str = "", annotate_top: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    if color_by == "protocol":
        protos = sorted({m["protocol"] for m in meta})
        cmap = plt.cm.tab20.colors
        proto_to_color = {p: cmap[i % len(cmap)] for i, p in enumerate(protos)}
        for p in protos:
            idx = [i for i, m in enumerate(meta) if m["protocol"] == p]
            ax.scatter(points[idx, 0], points[idx, 1],
                       c=[proto_to_color[p]], label=p, s=40, alpha=0.85)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=7, ncol=2, frameon=False)
    elif color_by == "value":
        values = np.array([m["value"] for m in meta])
        sc = ax.scatter(points[:, 0], points[:, 1], c=values, cmap="viridis",
                        s=40, alpha=0.85, vmin=0, vmax=6)
        plt.colorbar(sc, ax=ax, label="card value")
    elif color_by == "set":
        sets = sorted({m["set_code"] for m in meta})
        cmap = plt.cm.Set1.colors
        set_to_color = {s: cmap[i % len(cmap)] for i, s in enumerate(sets)}
        for s in sets:
            idx = [i for i, m in enumerate(meta) if m["set_code"] == s]
            ax.scatter(points[idx, 0], points[idx, 1], c=[set_to_color[s]],
                       label=s, s=40, alpha=0.85)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    if annotate_top > 0:
        # Annotate outliers (largest distance from centroid).
        centroid = points.mean(axis=0)
        d2 = ((points - centroid) ** 2).sum(axis=1)
        outliers = np.argsort(-d2)[:annotate_top]
        for i in outliers:
            ax.annotate(
                f"{meta[i]['protocol']} {meta[i]['value']}",
                (points[i, 0], points[i, 1]),
                fontsize=6, color="#222",
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_title(title or f"colored by {color_by}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    ckpt: Path,
    meta: list[dict],
    nn_idx: np.ndarray,
    cross_proto_pairs: list[tuple[int, int, float]],
    drift_top: list[tuple[int, float]] | None,
    compare_to: Path | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Card embedding investigation — `{ckpt.stem}`\n")
    lines.append(f"**Checkpoint:** `{ckpt}`")
    if compare_to:
        lines.append(f"**Compared against:** `{compare_to}`")
    lines.append("")
    lines.append(
        f"Analyzes the learned 32-d card embedding table "
        f"({len(meta)} real cards). Cards close in this space are the "
        f"ones the model treats similarly — useful for spotting "
        f"functional groupings and re-learning during training."
    )
    lines.append("")

    lines.append("## Projections\n")
    lines.append("- ![PCA by protocol](pca_protocol.png)")
    lines.append("- ![PCA by value](pca_value.png)")
    lines.append("- ![PCA by set](pca_set.png)")
    lines.append("- ![t-SNE by protocol](tsne_protocol.png)")
    lines.append("- ![t-SNE by value](tsne_value.png)")
    lines.append("")

    lines.append("## What to look for in the plots\n")
    lines.append(
        "- **Tight same-protocol clusters** = the model has learned a "
        "protocol-specific niche (good — distinguishable strategy)."
    )
    lines.append(
        "- **Cross-protocol neighbours** = different protocols' cards "
        "land near each other → the model treats them as filling the "
        "same role. These are the candidates for cross-protocol synergy."
    )
    lines.append(
        "- **Outliers** = cards far from any cluster → unique-effect "
        "cards the model has learned to handle distinctly."
    )
    lines.append("")

    lines.append("## Top cross-protocol neighbour pairs\n")
    lines.append("These pairs have high cosine similarity but come from different protocols. "
                 "The model treats them as functionally similar — hypothesis: they fill "
                 "interchangeable roles in a deck.\n")
    lines.append("| card A | card B | cosine sim |")
    lines.append("|---|---|---:|")
    for i, j, sim in cross_proto_pairs[:25]:
        a = meta[i]; b = meta[j]
        lines.append(
            f"| {a['protocol']} {a['value']} (`{a['key']}`) "
            f"| {b['protocol']} {b['value']} (`{b['key']}`) "
            f"| {sim:.3f} |"
        )
    lines.append("")

    lines.append("## k-NN per card (top-5 closest)\n")
    lines.append("For each card, the five other cards closest in embedding space. "
                 "Same-protocol neighbours are expected (same protocol token shared "
                 "through training); cross-protocol neighbours are the interesting signal.\n")
    lines.append("| card | top-5 nearest |")
    lines.append("|---|---|")
    # Sort cards by protocol for readability.
    rows = sorted(
        range(len(meta)), key=lambda i: (meta[i]["protocol"], meta[i]["value"]),
    )
    for i in rows:
        m = meta[i]
        neighbours = []
        for j in nn_idx[i]:
            n = meta[j]
            marker = "" if n["protocol"] == m["protocol"] else " ⚡"
            neighbours.append(f"{n['protocol']} {n['value']}{marker}")
        lines.append(f"| **{m['protocol']} {m['value']}** | " + ", ".join(neighbours) + " |")
    lines.append("")
    lines.append("(⚡ = cross-protocol neighbour)")
    lines.append("")

    if drift_top:
        lines.append("## Embedding drift vs baseline\n")
        lines.append(
            "Cards whose embedding moved most between the baseline and this "
            "checkpoint. Cosine distance, larger = the model has changed its "
            "understanding of this card the most during training.\n"
        )
        lines.append("| card | cosine distance from baseline |")
        lines.append("|---|---:|")
        for i, dist in drift_top[:25]:
            m = meta[i]
            lines.append(f"| {m['protocol']} {m['value']} (`{m['key']}`) | {dist:.3f} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--compare-to", default=None,
                    help="baseline checkpoint for drift analysis (optional)")
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--tsne-perplexity", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out_dir = analysis_dir(ckpt) / "card_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading {ckpt}")
    emb, meta = load_card_embeddings(ckpt)
    print(f"  {emb.shape[0]} real cards, d={emb.shape[1]}")

    # PCA + t-SNE.
    print("[2/4] computing PCA + t-SNE projections")
    pca = PCA(n_components=2, random_state=args.seed).fit_transform(emb)
    tsne = TSNE(
        n_components=2, perplexity=args.tsne_perplexity,
        init="pca", random_state=args.seed, max_iter=1500,
    ).fit_transform(emb)

    # Scatter plots.
    print("[3/4] writing plots")
    scatter(pca, meta, "protocol", out_dir / "pca_protocol.png",
            title="PCA — by protocol")
    scatter(pca, meta, "value", out_dir / "pca_value.png",
            title="PCA — by card value", annotate_top=10)
    scatter(pca, meta, "set", out_dir / "pca_set.png",
            title="PCA — by card set")
    scatter(tsne, meta, "protocol", out_dir / "tsne_protocol.png",
            title=f"t-SNE — by protocol (perplexity={args.tsne_perplexity})")
    scatter(tsne, meta, "value", out_dir / "tsne_value.png",
            title="t-SNE — by card value", annotate_top=10)

    # k-NN + cross-protocol top pairs.
    nn_idx = cosine_knn(emb, k=args.knn_k)
    # Build the cross-protocol "interesting" list.
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    cross_pairs: list[tuple[int, int, float]] = []
    for i in range(len(meta)):
        for j in range(i + 1, len(meta)):
            if meta[i]["protocol"] != meta[j]["protocol"]:
                cross_pairs.append((i, j, float(sim[i, j])))
    cross_pairs.sort(key=lambda t: -t[2])

    # Drift analysis.
    drift_top: list[tuple[int, float]] | None = None
    if args.compare_to:
        print(f"[+] loading baseline for drift: {args.compare_to}")
        base_emb, base_meta = load_card_embeddings(Path(args.compare_to))
        # Align by def_id — both arrays should have the same set if the
        # model architectures match, but be defensive.
        base_by_def = {m["def_id"]: i for i, m in enumerate(base_meta)}
        dists = []
        for i, m in enumerate(meta):
            j = base_by_def.get(m["def_id"])
            if j is None:
                continue
            a = emb[i] / (np.linalg.norm(emb[i]) + 1e-12)
            b = base_emb[j] / (np.linalg.norm(base_emb[j]) + 1e-12)
            dists.append((i, float(1.0 - a @ b)))
        dists.sort(key=lambda t: -t[1])
        drift_top = dists

    print("[4/4] writing report")
    compare_path = Path(args.compare_to) if args.compare_to else None
    report = render_report(ckpt, meta, nn_idx, cross_pairs, drift_top, compare_path)
    write_md(out_dir / "card_embeddings.md", report)

    write_json(out_dir / "card_embeddings.json", {
        "ckpt": str(ckpt),
        "compare_to": str(compare_path) if compare_path else None,
        "n_cards": len(meta),
        "d_emb": int(emb.shape[1]),
        "knn_idx": nn_idx.tolist(),
        "meta": meta,
        "pca_2d": pca.tolist(),
        "tsne_2d": tsne.tolist(),
        "cross_proto_top_50": [
            {"a": meta[i]["key"], "b": meta[j]["key"], "sim": s}
            for i, j, s in cross_pairs[:50]
        ],
        "drift_top": (
            [{"key": meta[i]["key"], "distance": d} for i, d in (drift_top or [])[:50]]
            if drift_top else None
        ),
    })
    print(f"\nWrote {out_dir / 'card_embeddings.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
