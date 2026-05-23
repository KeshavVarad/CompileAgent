/**
 * Card data loader. Reads data/cards.json at build time and exposes typed
 * card definitions to the rest of the engine.
 */

import rawCards from "@/data/cards.json";

export const BASE_PROTOCOLS = [
  "Darkness", "Death", "Fire", "Gravity", "Life", "Light",
  "Metal", "Plague", "Psychic", "Speed", "Spirit", "Water",
] as const;

export const EXPANSION_PROTOCOLS = ["Apathy", "Hate", "Love"] as const;

export const MAIN2_PROTOCOLS = [
  "Chaos", "Clarity", "Corruption", "Courage", "Fear", "Ice",
  "Luck", "Mirror", "Peace", "Smoke", "Time", "War",
] as const;

export const AUX2_PROTOCOLS = ["Assimilation", "Diversity", "Unity"] as const;

export type Protocol =
  | (typeof BASE_PROTOCOLS)[number]
  | (typeof EXPANSION_PROTOCOLS)[number]
  | (typeof MAIN2_PROTOCOLS)[number]
  | (typeof AUX2_PROTOCOLS)[number];

export const BASE_SET = "MN01";
export const EXPANSION_SET = "AX01";
export const MAIN2_SET = "MN02";
export const AUX2_SET = "AX02";
export const ALL_SETS = [BASE_SET, EXPANSION_SET, MAIN2_SET, AUX2_SET] as const;
export type SetCode = (typeof ALL_SETS)[number];

// Errata to apply on top of the upstream JSON at load time. Mirrors the
// Python `ERRATA` dict in src/compile_engine/cards.py. Each tier override
// is a full {emphasis, text} pair so the UI can still surface the trigger
// label (Start: / End: / When covered:) separately from prose.
type TierOverride = { emphasis: string; text: string };
const ERRATA: Record<string, { top?: TierOverride; middle?: TierOverride; bottom?: TierOverride }> = {
  "MN01:Death:1": {
    top: { emphasis: "Start:", text: "You may draw 1 card. If you do, delete 1 other card. Then, delete this card." },
  },
  "MN01:Fire:0": {
    bottom: { emphasis: "When this card would be covered:", text: "First, draw 1 card. Then, flip 1 other card." },
  },
  "MN01:Life:0": {
    top: { emphasis: "End:", text: "If this card is covered, delete this card." },
    bottom: { emphasis: "", text: "" },
  },
};

export type CardDef = {
  defId: number;          // 0..89 — stable index across the game
  setCode: string;
  protocol: Protocol;
  value: number;          // 0..6
  topEmphasis: string;
  topText: string;
  middleEmphasis: string;
  middleText: string;
  bottomEmphasis: string;
  bottomText: string;
  keywords: string[];
  errata: string | null;
  key: string;            // "<set>:<protocol>:<value>"
};

type RawTier = { emphasis?: string; text?: string };
type RawCard = {
  set: string;
  protocol: string;
  value: number;
  top: RawTier;
  middle: RawTier;
  bottom: RawTier;
  keywords?: Record<string, boolean>;
  errata?: string;
};

const sorted = [...(rawCards as unknown as RawCard[])].sort((a, b) =>
  a.set !== b.set
    ? a.set.localeCompare(b.set)
    : a.protocol !== b.protocol
      ? a.protocol.localeCompare(b.protocol)
      : a.value - b.value,
);

export const CARD_DEFS: ReadonlyArray<CardDef> = sorted.map((c, i) => {
  const key = `${c.set}:${c.protocol}:${c.value}`;
  const patch = ERRATA[key];
  return {
    defId: i,
    setCode: c.set,
    protocol: c.protocol as Protocol,
    value: c.value,
    topEmphasis: patch?.top?.emphasis ?? c.top.emphasis ?? "",
    topText: patch?.top?.text ?? c.top.text ?? "",
    middleEmphasis: patch?.middle?.emphasis ?? c.middle.emphasis ?? "",
    middleText: patch?.middle?.text ?? c.middle.text ?? "",
    bottomEmphasis: patch?.bottom?.emphasis ?? c.bottom.emphasis ?? "",
    bottomText: patch?.bottom?.text ?? c.bottom.text ?? "",
    keywords: Object.keys(c.keywords || {}),
    errata: patch ? "applied Dec 2024 Codex errata" : (c.errata ?? null),
    key,
  };
});

export const CARDS_BY_KEY: ReadonlyMap<string, CardDef> = new Map(
  CARD_DEFS.map((d) => [d.key, d]),
);

export function defsForProtocol(p: Protocol): CardDef[] {
  return CARD_DEFS.filter((d) => d.protocol === p);
}

/** Placeholder def for cards whose identity is hidden from the recorder
 *  (used in record mode). All text empty, value 0, protocol stub. */
export const PLACEHOLDER_CARD_DEF: CardDef = {
  defId: -1,
  setCode: "PLACEHOLDER",
  protocol: "Speed" as Protocol,
  value: 0,
  topEmphasis: "",
  topText: "",
  middleEmphasis: "",
  middleText: "",
  bottomEmphasis: "",
  bottomText: "",
  keywords: [],
  errata: null,
  key: "PLACEHOLDER",
};

/** Safe accessor — returns a placeholder def for unknown identities so
 *  helpers/effects can still introspect the card without crashing. Use this
 *  on field cards where the recorder may have logged a placeholder. */
export function safeCardDef(defId: number): CardDef {
  if (defId < 0 || defId >= CARD_DEFS.length) return PLACEHOLDER_CARD_DEF;
  return CARD_DEFS[defId];
}

export function availableProtocols(includeExpansion: boolean): Protocol[] {
  return includeExpansion
    ? [...BASE_PROTOCOLS, ...EXPANSION_PROTOCOLS]
    : [...BASE_PROTOCOLS];
}

export function protocolsForSets(enabledSets: ReadonlySet<SetCode>): Protocol[] {
  const out: Protocol[] = [];
  if (enabledSets.has(BASE_SET)) out.push(...BASE_PROTOCOLS);
  if (enabledSets.has(EXPANSION_SET)) out.push(...EXPANSION_PROTOCOLS);
  if (enabledSets.has(MAIN2_SET)) out.push(...MAIN2_PROTOCOLS);
  if (enabledSets.has(AUX2_SET)) out.push(...AUX2_PROTOCOLS);
  return out;
}
