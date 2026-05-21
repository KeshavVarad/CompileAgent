"""Card data: definitions loaded from data/cards.json plus protocol metadata.

Each protocol owns 6 cards. The base game (set code 'MN01') has 12 protocols;
the Aux 1 expansion (set code 'AX01') adds 3 more (Apathy, Hate, Love).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable

BASE_PROTOCOLS: tuple[str, ...] = (
    "Darkness", "Death", "Fire", "Gravity", "Life", "Light",
    "Metal", "Plague", "Psychic", "Speed", "Spirit", "Water",
)
EXPANSION_PROTOCOLS: tuple[str, ...] = ("Apathy", "Hate", "Love")
MAIN2_PROTOCOLS: tuple[str, ...] = (
    "Chaos", "Clarity", "Corruption", "Courage", "Fear", "Ice",
    "Luck", "Mirror", "Peace", "Smoke", "Time", "War",
)
AUX2_PROTOCOLS: tuple[str, ...] = ("Assimilation", "Diversity", "Unity")

BASE_SET = "MN01"
EXPANSION_SET = "AX01"
MAIN2_SET = "MN02"
AUX2_SET = "AX02"

# All sets currently shipped, in chronological order. Used by GameConfig and
# tests as the canonical enumeration.
ALL_SETS: tuple[str, ...] = (BASE_SET, EXPANSION_SET, MAIN2_SET, AUX2_SET)
PROTOCOLS_BY_SET: dict[str, tuple[str, ...]] = {
    BASE_SET: BASE_PROTOCOLS,
    EXPANSION_SET: EXPANSION_PROTOCOLS,
    MAIN2_SET: MAIN2_PROTOCOLS,
    AUX2_SET: AUX2_PROTOCOLS,
}


@dataclass(frozen=True, slots=True)
class CardDef:
    """Static definition of a single card (one of the 6 per protocol)."""
    def_id: int
    set_code: str
    protocol: str
    value: int
    top_text: str
    middle_text: str
    bottom_text: str
    keywords: frozenset[str]

    @property
    def is_expansion(self) -> bool:
        return self.set_code == EXPANSION_SET

    @property
    def key(self) -> str:
        return f"{self.set_code}:{self.protocol}:{self.value}"


def _data_path() -> Path:
    # data/cards.json sits next to the package root for ease of inspection.
    here = Path(__file__).resolve()
    return here.parents[2] / "data" / "cards.json"


def load_card_defs(path: Path | None = None, *, apply_errata: bool = True) -> list[CardDef]:
    """Load all card definitions from JSON. Stable ordering by (set, protocol, value).

    The community card database we vendor in `data/cards.json` reflects the
    launch printing. Pass `apply_errata=False` to opt out of the official
    Codex errata (last revised 16 Dec 2024) — useful for replaying old logs.
    """
    path = path or _data_path()
    raw = json.loads(path.read_text())
    sorted_raw = sorted(raw, key=lambda c: (c["set"], c["protocol"], c["value"]))
    defs: list[CardDef] = []
    for i, c in enumerate(sorted_raw):
        defs.append(
            CardDef(
                def_id=i,
                set_code=c["set"],
                protocol=c["protocol"],
                value=int(c["value"]),
                top_text=c["top"]["text"] or "",
                middle_text=c["middle"]["text"] or "",
                bottom_text=c["bottom"]["text"] or "",
                keywords=frozenset((c.get("keywords") or {}).keys()),
            )
        )
    if apply_errata:
        defs = _apply_official_errata(defs)
    return defs


# Errata sourced from the Compile Codex, last revision 16DEC2024.
# Cards whose text differs from the launch printing.
ERRATA: dict[str, dict[str, str]] = {
    "MN01:Death:1": {
        "top": "Start: You may draw 1 card. If you do, delete 1 other card. "
               "Then, delete this card.",
    },
    "MN01:Fire:0": {
        "bottom": "When this card would be covered: First, draw 1 card. "
                  "Then, flip 1 other card.",
    },
    "MN01:Life:0": {
        "top": "End: If this card is covered, delete this card.",
        "bottom": "",
    },
}


def _apply_official_errata(defs: list[CardDef]) -> list[CardDef]:
    out: list[CardDef] = []
    for d in defs:
        e = ERRATA.get(d.key)
        if not e:
            out.append(d)
            continue
        out.append(
            CardDef(
                def_id=d.def_id,
                set_code=d.set_code,
                protocol=d.protocol,
                value=d.value,
                top_text=e.get("top", d.top_text),
                middle_text=e.get("middle", d.middle_text),
                bottom_text=e.get("bottom", d.bottom_text),
                keywords=d.keywords,
            )
        )
    return out


def defs_for_protocol(defs: Iterable[CardDef], protocol: str) -> list[CardDef]:
    return [d for d in defs if d.protocol == protocol]


# Keyword vocabulary derived from the community card database. Each card's
# `keywords` field is a subset of this set. Used by the NN encoder to give
# the model an explicit multi-hot signal of "what verbs this card uses"
# alongside the learned card embedding.
KEYWORD_VOCAB: tuple[str, ...] = (
    "cache", "compile", "covered", "delete", "discard", "draw", "flip",
    "give", "line", "play", "rearrange", "refresh", "return", "reveal",
    "shift", "stack", "swap", "take",
)
NUM_KEYWORDS = len(KEYWORD_VOCAB)
VALUE_RANGE = 7  # values are 0..6 inclusive
TEXT_PRESENCE_DIM = 3  # has_top, has_middle, has_bottom flags
CARD_STATIC_FEATS_DIM = NUM_KEYWORDS + VALUE_RANGE + TEXT_PRESENCE_DIM  # 28


def static_features_for_def(d: CardDef) -> list[float]:
    """Return the static (non-learned) feature vector for one CardDef.

    Layout (NUM_KEYWORDS + VALUE_RANGE + TEXT_PRESENCE_DIM = 28):
      [0:NUM_KEYWORDS]                     — keyword multi-hot
      [NUM_KEYWORDS:NUM_KEYWORDS+VALUE_RANGE] — value one-hot
      [NUM_KEYWORDS+VALUE_RANGE:]          — (has_top, has_middle, has_bottom)
    """
    feats = [0.0] * CARD_STATIC_FEATS_DIM
    for i, kw in enumerate(KEYWORD_VOCAB):
        if kw in d.keywords:
            feats[i] = 1.0
    if 0 <= d.value < VALUE_RANGE:
        feats[NUM_KEYWORDS + d.value] = 1.0
    base = NUM_KEYWORDS + VALUE_RANGE
    feats[base + 0] = 1.0 if d.top_text else 0.0
    feats[base + 1] = 1.0 if d.middle_text else 0.0
    feats[base + 2] = 1.0 if d.bottom_text else 0.0
    return feats


def available_protocols(
    defs: Iterable[CardDef],
    include_expansion: bool = False,
    *,
    enabled_sets: tuple[str, ...] | None = None,
) -> list[str]:
    """Return the protocol names eligible for the draft pool.

    Two calling styles:
      - Legacy: `available_protocols(defs, include_expansion=True)` → MN01 + AX01.
      - Set-explicit: `enabled_sets=("MN01", "AX01", "MN02", "AX02")`.
    `enabled_sets` takes precedence when provided.
    """
    if enabled_sets is None:
        enabled_sets = (BASE_SET, EXPANSION_SET) if include_expansion else (BASE_SET,)
    seen: set[str] = set()
    for d in defs:
        if d.set_code not in enabled_sets:
            continue
        seen.add(d.protocol)
    return sorted(seen)
