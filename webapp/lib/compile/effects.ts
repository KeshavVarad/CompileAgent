/**
 * Per-card effect registry + generators. Each effect is a JS generator that
 * may `yield` a Choice when it needs a player decision. The engine then
 * resumes via `gen.next(choiceIdx)`. This is the TS analogue of the Python
 * `effects.py` design — same control flow, same Choice payload shape.
 *
 * For brevity, effects that simply contribute card value (no active text) are
 * not registered; the engine treats unregistered cards as value-only.
 */

import { CARD_DEFS } from "./cards";
import {
  checkDiversity6SelfDestruct,
  deleteCardFromField,
  describeCard,
  describeHandCard,
  discardToTrash,
  drawCards,
  enumerateAll,
  enumerateShiftTargets,
  enumerateUncovered,
  type FieldTarget,
  flipCard,
  lineStack,
  logInfo,
  middleSuppressed,
  oppMustPlayFacedown,
  oppPlayBlockedInLine,
  oppPlayFacedownBlockedInLine,
  playerMayPlayAnyLineFaceup,
  playTopDeckFaceDown,
  refreshPlayer,
  returnCardToHand,
  shiftCard,
} from "./helpers";
import type { CardInst, Choice, GameState, PlayerIndex } from "./types";
import { FACE_DOWN_BASE_VALUE, NUM_LINES } from "./types";

// An effect generator yields Choices and is sent number indices in return.
export type EffectGen = Generator<Choice, void, number>;

export type EffectFn = (state: GameState, ap: PlayerIndex, line: number, card: CardInst) => EffectGen;

export const MIDDLE_EFFECTS: Record<string, EffectFn> = {};
export const BOTTOM_FIRST_EFFECTS: Record<string, EffectFn> = {};
export const BOTTOM_ON_PLAY_EFFECTS: Record<string, EffectFn> = {};
export const START_EFFECTS: Record<string, EffectFn> = {};
export const END_EFFECTS: Record<string, EffectFn> = {};
// Reserved for unconditional one-shot tops (e.g. Speed 0 top "Play another
// card") — TOPs with a "Start:" / "End:" / "Flip:" / "After you...:" /
// "When this card would be ...:" emphasis must register on the matching
// event-specific registry below instead.
export const TOP_TRIGGER_EFFECTS: Record<string, EffectFn> = {};
export const WHEN_COVERED_EFFECTS: Record<string, EffectFn> = {};
// Event-conditional registries — mirror the Python engine.
export const AFTER_CLEAR_CACHE_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_SELF_DISCARD_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_OPP_DISCARD_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_SELF_DELETE_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_SELF_DRAW_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_SELF_SHUFFLE_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_SELF_REFRESH_EFFECTS: Record<string, EffectFn> = {};
export const FLIP_TRIGGER_EFFECTS: Record<string, EffectFn> = {};
export const WHEN_DELETED_BY_COMPILE_EFFECTS: Record<string, EffectFn> = {};

export function hasWhenCoveredEffect(defId: number): boolean {
  return CARD_DEFS[defId].key in WHEN_COVERED_EFFECTS;
}

export function hasWhenDeletedByCompileEffect(defId: number): boolean {
  return CARD_DEFS[defId].key in WHEN_DELETED_BY_COMPILE_EFFECTS;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function* chooseFieldTarget(
  prompt: string,
  targets: FieldTarget[],
  state: GameState,
  decider: PlayerIndex,
  optional = false,
): EffectGen {
  if (targets.length === 0) return;
  const options = targets.map((t) => describeCard(state, t));
  const idx: number = yield {
    prompt, options, targets, optional, decider,
  };
  if (idx === -1 || idx < 0 || idx >= targets.length) return;
  // Caller acts on `targets[idx]`. But because TS generators can't return a
  // value via yield easily, the calling effect re-derives the target by
  // matching on the returned index. We provide a stash on state.scratch.
  state.scratch["_last_target_idx"] = idx;
}

function* discardN(state: GameState, player: PlayerIndex, n: number): EffectGen {
  for (let i = 0; i < n; i++) {
    if (state.players[player].hand.length === 0) {
      const remaining = n - i;
      logInfo(state, `P${player + 1} skipped ${remaining} of ${n} forced discard(s) — hand empty.`);
      return;
    }
    const options = state.players[player].hand.map((_, idx) =>
      describeHandCard(state, player, idx),
    );
    const targets = state.players[player].hand.map((_, idx) => idx);
    const idx: number = yield {
      prompt: "Discard 1 card from hand",
      options,
      targets,
      optional: false,
      decider: player,
    };
    if (idx < 0 || idx >= state.players[player].hand.length) return;
    discardToTrash(state, player, idx);
  }
}

function* discardOptionalLoop(state: GameState, player: PlayerIndex, maxN: number): EffectGen {
  let discarded = 0;
  while (discarded < maxN && state.players[player].hand.length > 0) {
    const options = state.players[player].hand.map((_, idx) =>
      describeHandCard(state, player, idx),
    );
    const targets = state.players[player].hand.map((_, idx) => idx);
    const idx: number = yield {
      prompt: `Discard another card? (${discarded} so far)`,
      options,
      targets,
      optional: true,
      decider: player,
    };
    if (idx === -1) break;
    discardToTrash(state, player, idx);
    discarded++;
  }
  state.scratch["last_discard_count"] = discarded;
}

function register(reg: Record<string, EffectFn>, key: string, fn: EffectFn): void {
  reg[key] = fn;
}

// ---------------------------------------------------------------------------
// Value-5 "You discard 1 card." across all 15 protocols.
// ---------------------------------------------------------------------------

const v5Discard: EffectFn = function* (state, ap) {
  yield* discardN(state, ap, 1);
};

for (const [proto, set] of [
  // MN01
  ["Darkness", "MN01"], ["Death", "MN01"], ["Fire", "MN01"], ["Gravity", "MN01"],
  ["Life", "MN01"], ["Light", "MN01"], ["Metal", "MN01"], ["Plague", "MN01"],
  ["Psychic", "MN01"], ["Speed", "MN01"], ["Spirit", "MN01"], ["Water", "MN01"],
  // AX01
  ["Apathy", "AX01"], ["Hate", "AX01"], ["Love", "AX01"],
  // MN02
  ["Chaos", "MN02"], ["Clarity", "MN02"], ["Corruption", "MN02"], ["Courage", "MN02"],
  ["Ice", "MN02"], ["Luck", "MN02"], ["Mirror", "MN02"], ["Peace", "MN02"],
  ["Smoke", "MN02"], ["Time", "MN02"], ["War", "MN02"],
  // AX02
  ["Assimilation", "AX02"], ["Diversity", "AX02"], ["Unity", "AX02"],
] as const) {
  MIDDLE_EFFECTS[`${set}:${proto}:5`] = v5Discard;
}

// ---------------------------------------------------------------------------
// APATHY (AX01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "AX01:Apathy:1", function* (state, ap, li, card) {
  // Flip all other face-up cards in this line.
  for (const pl of [0, 1] as PlayerIndex[]) {
    for (const c of lineStack(state.lines[li], pl)) {
      if (c === card || !c.faceUp) continue;
      c.faceUp = false;
    }
  }
  if (false) yield {} as Choice;
});

register(BOTTOM_FIRST_EFFECTS, "AX01:Apathy:2", function* (state, ap, li, card) {
  // First, flip this card. (Self-flip.)
  card.faceUp = !card.faceUp;
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "AX01:Apathy:3", function* (state, ap) {
  // Flip 1 of your opponent's face-up cards.
  const targets = enumerateUncovered(state, { owner: "opponent", face: "up", activePlayer: ap });
  yield* chooseFieldTarget("Flip 1 of your opponent's face-up cards", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "AX01:Apathy:4", function* (state, ap) {
  // (optional) Flip 1 of your face-up covered cards.
  const targets: FieldTarget[] = [];
  for (let li = 0; li < NUM_LINES; li++) {
    const stack = lineStack(state.lines[li], ap);
    for (let pos = 0; pos < stack.length - 1; pos++) {
      const c = stack[pos];
      if (c.faceUp && !c.isCommitted) targets.push({ line: li, player: ap, pos, card: c });
    }
  }
  yield* chooseFieldTarget("(optional) Flip 1 of your face-up covered cards", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

// ---------------------------------------------------------------------------
// HATE (AX01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "AX01:Hate:0", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  yield* chooseFieldTarget("Delete 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "AX01:Hate:1", function* (state, ap, li, card) {
  yield* discardN(state, ap, 3);
  for (let k = 0; k < 2; k++) {
    const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
    yield* chooseFieldTarget("Delete 1 card", targets, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i == null || !targets[i]) return;
    deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
  }
});

register(MIDDLE_EFFECTS, "AX01:Hate:2", function* (state, ap) {
  // Delete your highest value uncovered card. Delete opponent's highest value uncovered card.
  for (const who of [ap, (1 - ap) as PlayerIndex]) {
    const targets = enumerateUncovered(state, {
      owner: who === ap ? "self" : "opponent",
      activePlayer: ap,
    });
    if (targets.length === 0) continue;
    const scored = targets
      .map((t) => ({ val: t.card.faceUp ? CARD_DEFS[t.card.defId].value : FACE_DOWN_BASE_VALUE, t }))
      .sort((a, b) => b.val - a.val);
    const topVal = scored[0].val;
    const tied = scored.filter((s) => s.val === topVal).map((s) => s.t);
    if (tied.length === 1) {
      deleteCardFromField(state, tied[0].line, tied[0].player, tied[0].pos);
    } else {
      yield* chooseFieldTarget("Break tie: delete which?", tied, state, ap);
      const i = state.scratch["_last_target_idx"] as number | undefined;
      if (i == null || !tied[i]) continue;
      deleteCardFromField(state, tied[i].line, tied[i].player, tied[i].pos);
    }
  }
});

// Hate 3 top: "After you delete cards: Draw 1 card."
register(AFTER_SELF_DELETE_EFFECTS, "AX01:Hate:3", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(BOTTOM_FIRST_EFFECTS, "AX01:Hate:4", function* (state, ap, li) {
  // First, delete the lowest-value covered card in this line.
  const targets: FieldTarget[] = [];
  for (const pl of [0, 1] as PlayerIndex[]) {
    const stack = lineStack(state.lines[li], pl);
    for (let pos = 0; pos < stack.length - 1; pos++) {
      targets.push({ line: li, player: pl, pos, card: stack[pos] });
    }
  }
  if (targets.length === 0) return;
  const scored = targets
    .map((t) => ({ v: t.card.faceUp ? CARD_DEFS[t.card.defId].value : FACE_DOWN_BASE_VALUE, t }))
    .sort((a, b) => a.v - b.v);
  const lo = scored[0].v;
  const tied = scored.filter((s) => s.v === lo).map((s) => s.t);
  if (tied.length === 1) {
    deleteCardFromField(state, tied[0].line, tied[0].player, tied[0].pos);
  } else {
    yield* chooseFieldTarget("Break tie", tied, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i == null || !tied[i]) return;
    deleteCardFromField(state, tied[i].line, tied[i].player, tied[i].pos);
  }
});

// ---------------------------------------------------------------------------
// LOVE (AX01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "AX01:Love:1", function* (state, ap) {
  // Draw the top card of your opponent's deck. (Ownership transfers.)
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const ps = state.players[opp];
  if (ps.deck.length > 0) {
    const c = ps.deck.pop()!;
    c.owner = ap;
    c.faceUp = false;
    state.players[ap].hand.push(c);
  }
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "AX01:Love:2", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  drawCards(state, opp, 1);
  refreshPlayer(state, ap);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "AX01:Love:6", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  drawCards(state, opp, 2);
  if (false) yield {} as Choice;
});

// ---------------------------------------------------------------------------
// DARKNESS (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Darkness:0", function* (state, ap) {
  drawCards(state, ap, 3);
  // Shift 1 of opponent's covered cards.
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const targets: FieldTarget[] = [];
  for (let li = 0; li < NUM_LINES; li++) {
    const stack = lineStack(state.lines[li], opp);
    for (let pos = 0; pos < stack.length - 1; pos++) {
      targets.push({ line: li, player: opp, pos, card: stack[pos] });
    }
  }
  yield* chooseFieldTarget("Shift 1 of opponent's covered cards", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const lineOpts = [0, 1, 2].map((n) => String(n));
  const lineTargets = [0, 1, 2];
  const dstIdx: number = yield {
    prompt: "To which line?",
    options: lineOpts,
    targets: lineTargets,
    optional: false,
    decider: ap,
  };
  if (dstIdx < 0 || dstIdx > 2) return;
  shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dstIdx);
});

register(MIDDLE_EFFECTS, "MN01:Darkness:1", function* (state, ap) {
  // "Flip 1 of your opponent's cards. You may shift that card."
  const targets = enumerateUncovered(state, { owner: "opponent", activePlayer: ap });
  yield* chooseFieldTarget("Flip 1 of opponent's cards", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const tgt = targets[i];
  flipCard(state, tgt.line, tgt.player, tgt.pos);
  // The "that card" reference survives mid-effect even if the flip moved
  // / deleted the card (Codex p.3 "Selecting and Targeting"). Re-locate
  // the flipped card by identity; if it's no longer on the field (e.g.
  // a flip-triggered self-delete on Metal 6), skip the shift.
  let srcLine = -1;
  let srcPos = -1;
  for (let ln = 0; ln < NUM_LINES; ln++) {
    const stack = lineStack(state.lines[ln], tgt.player);
    const idx = stack.indexOf(tgt.card);
    if (idx >= 0) { srcLine = ln; srcPos = idx; break; }
  }
  if (srcLine < 0) return;
  const dest = [0, 1, 2].filter((l) => l !== srcLine);
  if (dest.length === 0) return;
  const dIdx: number = yield {
    prompt: "(optional) Shift the flipped card to another line",
    options: dest.map((l) => `shift to L${l + 1}`).concat(["skip"]),
    targets: (dest as number[]).concat([-1]),
    optional: true,
    decider: ap,
  };
  if (dIdx === -1 || dest[dIdx] == null) return;
  shiftCard(state, srcLine, tgt.player, srcPos, dest[dIdx]);
});

register(MIDDLE_EFFECTS, "MN01:Darkness:2", function* (state, ap, li) {
  // (optional) Flip 1 covered card in this line.
  const targets: FieldTarget[] = [];
  for (const pl of [0, 1] as PlayerIndex[]) {
    const stack = lineStack(state.lines[li], pl);
    for (let pos = 0; pos < stack.length - 1; pos++) {
      targets.push({ line: li, player: pl, pos, card: stack[pos] });
    }
  }
  yield* chooseFieldTarget("(optional) Flip 1 covered card in this line", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN01:Darkness:3", function* (state, ap, li) {
  // Play 1 card face-down in another line.
  const hand = state.players[ap].hand;
  if (hand.length === 0) return;
  const handOpts = hand.map((_, idx) => describeHandCard(state, ap, idx));
  const handTargets = hand.map((_, idx) => idx);
  const hi: number = yield {
    prompt: "Pick a card to play face-down in another line",
    options: handOpts, targets: handTargets, optional: false, decider: ap,
  };
  if (hi < 0 || hi >= hand.length) return;
  const otherLines = [0, 1, 2].filter((l) => l !== li);
  const lidx: number = yield {
    prompt: "Which other line?",
    options: otherLines.map(String), targets: otherLines, optional: false, decider: ap,
  };
  const ln = otherLines[lidx];
  if (ln == null) return;
  const c = state.players[ap].hand.splice(hi, 1)[0];
  c.faceUp = false;
  lineStack(state.lines[ln], ap).push(c);
});

register(MIDDLE_EFFECTS, "MN01:Darkness:4", function* (state, ap) {
  const targets = enumerateUncovered(state, { face: "down", activePlayer: ap });
  yield* chooseFieldTarget("Shift 1 face-down card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== targets[i].line);
  const didx: number = yield {
    prompt: "To which line?",
    options: dest.map(String), targets: dest, optional: false, decider: ap,
  };
  const dl = dest[didx];
  if (dl == null) return;
  shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dl);
});

// ---------------------------------------------------------------------------
// DEATH (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Death:0", function* (state, ap, li) {
  for (let ln = 0; ln < NUM_LINES; ln++) {
    if (ln === li) continue;
    const targets = enumerateUncovered(state, { lineFilter: ln, activePlayer: ap });
    if (targets.length === 0) continue;
    if (targets.length === 1) {
      const t = targets[0];
      deleteCardFromField(state, t.line, t.player, t.pos);
      continue;
    }
    yield* chooseFieldTarget(`Delete one card from line ${ln}`, targets, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i == null || !targets[i]) continue;
    deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
  }
});

register(START_EFFECTS, "MN01:Death:1", function* (state, ap, li, card) {
  // Errata: "Start: You may draw 1 card. If you do, delete 1 other card. Then, delete this card."
  const deckHas = state.players[ap].deck.length > 0 || state.players[ap].trash.length > 0;
  if (!deckHas) return;
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  const idx0: number = yield {
    prompt: "(optional) Draw 1 + delete 1 other card + delete this card",
    options: ["accept", "skip"], targets: [0, -1], optional: true, decider: ap,
  };
  if (idx0 === -1 || idx0 === 1) return;
  drawCards(state, ap, 1);
  const t2 = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (t2.length > 0) {
    yield* chooseFieldTarget("Delete 1 other card", t2, state, ap);
    const i2 = state.scratch["_last_target_idx"] as number | undefined;
    if (i2 != null && t2[i2]) deleteCardFromField(state, t2[i2].line, t2[i2].player, t2[i2].pos);
  }
  // Delete this card from wherever it is.
  for (let ln = 0; ln < NUM_LINES; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = lineStack(state.lines[ln], pl);
      const idx = stack.indexOf(card);
      if (idx >= 0) {
        deleteCardFromField(state, ln, pl, idx);
        return;
      }
    }
  }
});

register(MIDDLE_EFFECTS, "MN01:Death:2", function* (state, ap, li, card) {
  const lidx: number = yield {
    prompt: "Choose a line", options: ["0", "1", "2"], targets: [0, 1, 2], optional: false, decider: ap,
  };
  if (lidx < 0 || lidx > 2) return;
  for (const pl of [0, 1] as PlayerIndex[]) {
    const stack = lineStack(state.lines[lidx], pl);
    for (let pos = stack.length - 1; pos >= 0; pos--) {
      const c = stack[pos];
      if (c === card) continue;
      const v = c.faceUp ? CARD_DEFS[c.defId].value : FACE_DOWN_BASE_VALUE;
      if (v === 1 || v === 2) deleteCardFromField(state, lidx, pl, pos);
    }
  }
});

register(MIDDLE_EFFECTS, "MN01:Death:3", function* (state, ap) {
  const targets = enumerateUncovered(state, { face: "down", activePlayer: ap });
  yield* chooseFieldTarget("Delete 1 face-down card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN01:Death:4", function* (state, ap, li, card) {
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < NUM_LINES; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = lineStack(state.lines[ln], pl);
      if (stack.length === 0) continue;
      const c = stack[stack.length - 1];
      if (c === card) continue;
      const v = c.faceUp ? CARD_DEFS[c.defId].value : FACE_DOWN_BASE_VALUE;
      if (v === 0 || v === 1) targets.push({ line: ln, player: pl, pos: stack.length - 1, card: c });
    }
  }
  yield* chooseFieldTarget("Delete a card with value 0 or 1", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

// ---------------------------------------------------------------------------
// FIRE (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Fire:0", function* (state, ap, li, card) {
  // Flip 1 other card. Draw 2 cards.
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length > 0) {
    yield* chooseFieldTarget("Flip 1 other card", targets, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
  }
  drawCards(state, ap, 2);
});

register(WHEN_COVERED_EFFECTS, "MN01:Fire:0", function* (state, ap, li, card) {
  // Errata: When this card would be covered: First, draw 1 card. Then, flip 1 other card.
  drawCards(state, ap, 1);
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 other card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN01:Fire:1", function* (state, ap, li, card) {
  if (state.players[ap].hand.length === 0) return;
  yield* discardN(state, ap, 1);
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Delete 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN01:Fire:2", function* (state, ap, li, card) {
  if (state.players[ap].hand.length === 0) return;
  yield* discardN(state, ap, 1);
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Return 1 card to its owner's hand", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  returnCardToHand(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN01:Fire:4", function* (state, ap) {
  // Discard 1 or more. Draw amount discarded plus 1.
  if (state.players[ap].hand.length === 0) {
    drawCards(state, ap, 1);
    return;
  }
  yield* discardN(state, ap, 1);
  yield* discardOptionalLoop(state, ap, state.players[ap].hand.length);
  const n = 1 + ((state.scratch["last_discard_count"] as number) || 0);
  drawCards(state, ap, n + 1);
});

// ---------------------------------------------------------------------------
// GRAVITY (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Gravity:0", function* (state, ap, li, card) {
  const total = state.lines[li].p0Stack.length + state.lines[li].p1Stack.length;
  const n = Math.floor(total / 2);
  for (let k = 0; k < n; k++) playTopDeckFaceDown(state, ap, li);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Gravity:1", function* (state, ap, li) {
  drawCards(state, ap, 2);
  const candidates = enumerateShiftTargets(state, { owner: "self", activePlayer: ap });
  if (candidates.length === 0) return;
  type Opt = { label: string; t: FieldTarget; dst: number };
  const opts: Opt[] = [];
  for (const t of candidates) {
    if (t.line === li) {
      for (const dst of [0, 1, 2]) if (dst !== li) opts.push({ label: `FROM ${t.line} -> ${dst}: ${describeCard(state, t)}`, t, dst });
    } else {
      opts.push({ label: `TO ${li} <- ${t.line}: ${describeCard(state, t)}`, t, dst: li });
    }
  }
  if (opts.length === 0) return;
  const idx: number = yield {
    prompt: "Shift 1 card to or from this line",
    options: opts.map((o) => o.label),
    targets: opts,
    optional: false, decider: ap,
  };
  const opt = opts[idx];
  if (!opt) return;
  shiftCard(state, opt.t.line, opt.t.player, opt.t.pos, opt.dst);
});

register(MIDDLE_EFFECTS, "MN01:Gravity:2", function* (state, ap, li, card) {
  // Flip 1 card. Shift that card to this line.
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 card (then it shifts to this line)", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const t = targets[i];
  flipCard(state, t.line, t.player, t.pos);
  if (t.line !== li) {
    const newPos = lineStack(state.lines[t.line], t.player).indexOf(t.card);
    if (newPos >= 0) shiftCard(state, t.line, t.player, newPos, li);
  }
});

register(MIDDLE_EFFECTS, "MN01:Gravity:4", function* (state, ap, li) {
  const targets = enumerateUncovered(state, { face: "down", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift 1 face-down card to this line", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, li);
});

register(MIDDLE_EFFECTS, "MN01:Gravity:6", function* (state, ap, li) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  playTopDeckFaceDown(state, opp, li);
  if (false) yield {} as Choice;
});

// ---------------------------------------------------------------------------
// LIFE (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Life:0", function* (state, ap) {
  for (let ln = 0; ln < NUM_LINES; ln++) {
    if (lineStack(state.lines[ln], ap).length > 0) playTopDeckFaceDown(state, ap, ln);
  }
  if (false) yield {} as Choice;
});

register(END_EFFECTS, "MN01:Life:0", function* (state, ap, li, card) {
  // Errata: End: If this card is covered, delete this card.
  for (let ln = 0; ln < NUM_LINES; ln++) {
    const stack = lineStack(state.lines[ln], ap);
    const idx = stack.indexOf(card);
    if (idx >= 0) {
      if (idx < stack.length - 1) deleteCardFromField(state, ln, ap, idx);
      return;
    }
  }
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Life:1", function* (state, ap, li, card) {
  for (let k = 0; k < 2; k++) {
    const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
    if (targets.length === 0) return;
    yield* chooseFieldTarget("Flip 1 card", targets, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i == null || !targets[i]) return;
    flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
  }
});

register(MIDDLE_EFFECTS, "MN01:Life:2", function* (state, ap) {
  drawCards(state, ap, 1);
  const targets = enumerateUncovered(state, { face: "down", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Flip 1 face-down card", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(BOTTOM_FIRST_EFFECTS, "MN01:Life:3", function* (state, ap, li) {
  const otherLines = [0, 1, 2].filter((l) => l !== li);
  if (otherLines.length === 0) return;
  const idx: number = yield {
    prompt: "Play top-deck face-down in which other line?",
    options: otherLines.map(String), targets: otherLines, optional: false, decider: ap,
  };
  const ln = otherLines[idx];
  if (ln == null) return;
  playTopDeckFaceDown(state, ap, ln);
});

register(MIDDLE_EFFECTS, "MN01:Life:4", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  const idx = stack.indexOf(card);
  if (idx > 0) drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

// ---------------------------------------------------------------------------
// LIGHT (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Light:0", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 card; draw cards equal to its value", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const flipped = flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
  const v = flipped.faceUp ? CARD_DEFS[flipped.defId].value : FACE_DOWN_BASE_VALUE;
  drawCards(state, ap, v);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN01:Light:1", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Light:2", function* (state, ap) {
  drawCards(state, ap, 2);
  // Reveal 1 face-down. Then shift or flip it.
  const targets = enumerateUncovered(state, { face: "down", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Reveal 1 face-down; then shift or flip", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const t = targets[i];
  const subIdx: number = yield {
    prompt: "Shift or flip that card?",
    options: ["shift", "flip"], targets: [0, 1], optional: false, decider: ap,
  };
  if (subIdx === 0) {
    const dest = [0, 1, 2].filter((l) => l !== t.line);
    const didx: number = yield {
      prompt: "To which line?", options: dest.map(String), targets: dest, optional: false, decider: ap,
    };
    if (dest[didx] == null) return;
    const curPos = lineStack(state.lines[t.line], t.player).indexOf(t.card);
    if (curPos >= 0) shiftCard(state, t.line, t.player, curPos, dest[didx]);
  } else if (subIdx === 1) {
    const curPos = lineStack(state.lines[t.line], t.player).indexOf(t.card);
    if (curPos >= 0) flipCard(state, t.line, t.player, curPos);
  }
});

register(MIDDLE_EFFECTS, "MN01:Light:3", function* (state, ap, li) {
  // Shift all face-down cards in this line to another line.
  const otherLines = [0, 1, 2].filter((l) => l !== li);
  if (otherLines.length === 0) return;
  const didx: number = yield {
    prompt: "Shift all face-down in this line to which other line?",
    options: otherLines.map(String), targets: otherLines, optional: false, decider: ap,
  };
  const dst = otherLines[didx];
  if (dst == null) return;
  for (const pl of [0, 1] as PlayerIndex[]) {
    const stack = lineStack(state.lines[li], pl);
    const positions: number[] = [];
    stack.forEach((c, i) => { if (!c.faceUp) positions.push(i); });
    for (let i = positions.length - 1; i >= 0; i--) {
      shiftCard(state, li, pl, positions[i], dst);
    }
  }
});

// ---------------------------------------------------------------------------
// METAL (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Metal:0", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN01:Metal:1", function* (state, ap) {
  drawCards(state, ap, 2);
  state.players[ap === 0 ? 1 : 0].cannotCompileNextTurn = true;
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Metal:3", function* (state, ap, li) {
  drawCards(state, ap, 1);
  const candidates: number[] = [];
  for (let ln = 0; ln < NUM_LINES; ln++) {
    if (ln === li) continue;
    const total = state.lines[ln].p0Stack.length + state.lines[ln].p1Stack.length;
    if (total >= 8) candidates.push(ln);
  }
  if (candidates.length === 0) return;
  const idx: number = yield {
    prompt: "Choose a line (>=8 cards) to clear",
    options: candidates.map(String), targets: candidates, optional: false, decider: ap,
  };
  const ln = candidates[idx];
  if (ln == null) return;
  for (const pl of [0, 1] as PlayerIndex[]) {
    const stack = lineStack(state.lines[ln], pl);
    for (let pos = stack.length - 1; pos >= 0; pos--) {
      deleteCardFromField(state, ln, pl, pos);
    }
  }
});

// Metal 6 top: "When this card would be covered or flipped: First, delete
// this card." Two trigger events share the same handler.
function* metal6SelfDelete(state: GameState, _ap: PlayerIndex, _li: number, card: CardInst): EffectGen {
  for (let ln = 0; ln < NUM_LINES; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = lineStack(state.lines[ln], pl);
      const idx = stack.indexOf(card);
      if (idx >= 0) { deleteCardFromField(state, ln, pl, idx); return; }
    }
  }
  if (false) yield {} as Choice;
}
register(WHEN_COVERED_EFFECTS, "MN01:Metal:6", metal6SelfDelete);
register(FLIP_TRIGGER_EFFECTS, "MN01:Metal:6", metal6SelfDelete);

// ---------------------------------------------------------------------------
// PLAGUE (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Plague:0", function* (state, ap) {
  yield* discardN(state, ap === 0 ? 1 : 0, 1);
});

// Plague 1 top: "After your opponent discards cards: Draw 1 card."
register(AFTER_OPP_DISCARD_EFFECTS, "MN01:Plague:1", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Plague:1", function* (state, ap) {
  yield* discardN(state, ap === 0 ? 1 : 0, 1);
});

register(MIDDLE_EFFECTS, "MN01:Plague:2", function* (state, ap) {
  if (state.players[ap].hand.length === 0) return;
  yield* discardN(state, ap, 1);
  yield* discardOptionalLoop(state, ap, state.players[ap].hand.length);
  const n = 1 + ((state.scratch["last_discard_count"] as number) || 0);
  yield* discardN(state, ap === 0 ? 1 : 0, n + 1);
});

register(MIDDLE_EFFECTS, "MN01:Plague:3", function* (state, ap, li, card) {
  for (let ln = 0; ln < NUM_LINES; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = lineStack(state.lines[ln], pl);
      // "each" semantics: uncovered face-up only
      if (stack.length === 0) continue;
      const top = stack[stack.length - 1];
      if (top === card || !top.faceUp) continue;
      top.faceUp = false;
    }
  }
  if (false) yield {} as Choice;
});

// ---------------------------------------------------------------------------
// PSYCHIC (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Psychic:0", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  drawCards(state, ap, 2);
  yield* discardN(state, opp, 2);
});

register(MIDDLE_EFFECTS, "MN01:Psychic:2", function* (state, ap) {
  yield* discardN(state, ap === 0 ? 1 : 0, 2);
  // Rearrange opp protocols — pick a swap (or none).
  const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
  const options = pairs.map(([a, b]) => `swap opp protocol L${a}<->L${b}`).concat(["no swap"]);
  const targets = pairs.map((p) => p).concat([null as unknown as [number, number]]);
  const idx: number = yield { prompt: "Rearrange opponent's protocols", options, targets, optional: false, decider: ap };
  if (idx < pairs.length) {
    const [a, b] = pairs[idx];
    const opp: PlayerIndex = ap === 0 ? 1 : 0;
    const ps = state.players[opp];
    [ps.protocols[a], ps.protocols[b]] = [ps.protocols[b], ps.protocols[a]];
    [ps.compiled[a], ps.compiled[b]] = [ps.compiled[b], ps.compiled[a]];
  }
});

register(MIDDLE_EFFECTS, "MN01:Psychic:3", function* (state, ap) {
  yield* discardN(state, ap === 0 ? 1 : 0, 1);
  const targets = enumerateShiftTargets(state, { owner: "opponent", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift 1 opponent card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== targets[i].line);
  const didx: number = yield {
    prompt: "To which line?", options: dest.map(String), targets: dest, optional: false, decider: ap,
  };
  if (dest[didx] == null) return;
  shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dest[didx]);
});

// ---------------------------------------------------------------------------
// SPEED (MN01)
// ---------------------------------------------------------------------------

type SubPlay = { label: string; hi: number; ln: number; fu: boolean; cross: boolean };

function legalSubPlays(
  state: GameState,
  ap: PlayerIndex,
  candidateHandIndices?: number[],
): SubPlay[] {
  const out: SubPlay[] = [];
  const spirit1 = playerMayPlayAnyLineFaceup(state, ap);
  const psychic1 = oppMustPlayFacedown(state, ap);
  const lineBlocked = [0, 1, 2].map((ln) => oppPlayBlockedInLine(state, ln, ap));
  const lineFdBlocked = [0, 1, 2].map((ln) => oppPlayFacedownBlockedInLine(state, ln, ap));
  const hand = state.players[ap].hand;
  const indices = candidateHandIndices ?? hand.map((_, i) => i);
  for (const hi of indices) {
    if (hi < 0 || hi >= hand.length) continue;
    const c = hand[hi];
    const d = CARD_DEFS[c.defId];
    const chaos3 = d.protocol === "Chaos" && d.value === 3;
    const corruption0 = d.protocol === "Corruption" && d.value === 0;
    const unrestrictedFu = spirit1 || chaos3 || corruption0;
    for (let ln = 0; ln < NUM_LINES; ln++) {
      if (lineBlocked[ln] || psychic1) continue;
      if (unrestrictedFu || state.players[ap].protocols[ln] === d.protocol) {
        out.push({ label: `FU ${d.protocol}${d.value} L${ln}`, hi, ln, fu: true, cross: false });
      }
    }
    for (let ln = 0; ln < NUM_LINES; ln++) {
      if (lineBlocked[ln] || lineFdBlocked[ln]) continue;
      out.push({ label: `FD hand[${hi}] L${ln}`, hi, ln, fu: false, cross: false });
    }
    if (corruption0) {
      for (let ln = 0; ln < NUM_LINES; ln++) {
        out.push({ label: `FU OPP L${ln}: ${d.protocol}${d.value}`, hi, ln: NUM_LINES + ln, fu: true, cross: true });
        out.push({ label: `FD OPP L${ln}`, hi, ln: NUM_LINES + ln, fu: false, cross: true });
      }
    }
  }
  return out;
}

register(MIDDLE_EFFECTS, "MN01:Speed:0", function* (state, ap) {
  // "Play 1 card." Honours every play-time rule the action phase does, including
  // Chaos 3 / Corruption 0 / Spirit 1 bypasses.
  const legal = legalSubPlays(state, ap);
  if (legal.length === 0) return;
  const idx: number = yield {
    prompt: "Play 1 card",
    options: legal.map((x) => x.label),
    targets: legal, optional: false, decider: ap,
  };
  const pick = legal[idx];
  if (!pick) return;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const engine = state.scratch["_engine"] as any;
  if (engine?.playCardForEffect) engine.playCardForEffect(ap, pick.hi, pick.ln, pick.fu);
});

// Speed 1 top: "After you clear cache: Draw 1 card."
register(AFTER_CLEAR_CACHE_EFFECTS, "MN01:Speed:1", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Speed:1", function* (state, ap) {
  drawCards(state, ap, 2);
  if (false) yield {} as Choice;
});

// Speed 2 top: "When this card would be deleted by compiling: Shift this
// card, even if this card is covered." Compile-time interrupt: the
// engine's compile path queries this registry before bulk-deleting the
// line, gives this handler a chance to shift Speed 2 out, then proceeds.
register(WHEN_DELETED_BY_COMPILE_EFFECTS, "MN01:Speed:2", function* (state, _ap, _li, card) {
  const owner = card.owner;
  let srcLine = -1;
  let srcPos = -1;
  for (let ln = 0; ln < NUM_LINES; ln++) {
    const s = lineStack(state.lines[ln], owner);
    const idx = s.indexOf(card);
    if (idx >= 0) { srcLine = ln; srcPos = idx; break; }
  }
  if (srcLine < 0) return;
  const dest = [0, 1, 2].filter((l) => l !== srcLine);
  if (dest.length === 0) return;
  const didx: number = yield {
    prompt: "Speed 2 would be deleted — shift to which line?",
    options: dest.map((l) => `shift to L${l}`),
    targets: dest,
    optional: false,
    decider: owner,
  };
  if (dest[didx] == null) return;
  shiftCard(state, srcLine, owner, srcPos, dest[didx]);
});

// Spirit 3 top: "After you draw cards: You may shift this card, even if
// this card is covered."
register(AFTER_SELF_DRAW_EFFECTS, "MN01:Spirit:3", function* (state, _ap, _li, card) {
  const owner = card.owner;
  let srcLine = -1;
  let srcPos = -1;
  for (let ln = 0; ln < NUM_LINES; ln++) {
    const s = lineStack(state.lines[ln], owner);
    const idx = s.indexOf(card);
    if (idx >= 0) { srcLine = ln; srcPos = idx; break; }
  }
  if (srcLine < 0) return;
  const dest = [0, 1, 2].filter((l) => l !== srcLine);
  if (dest.length === 0) return;
  const didx: number = yield {
    prompt: "(optional) Shift Spirit 3",
    options: dest.map((l) => `shift to L${l}`).concat(["skip"]),
    targets: (dest as number[]).concat([-1]),
    optional: true,
    decider: owner,
  };
  if (didx === -1 || dest[didx] == null) return;
  shiftCard(state, srcLine, owner, srcPos, dest[didx]);
});

register(MIDDLE_EFFECTS, "MN01:Speed:3", function* (state, ap, li, card) {
  const targets = enumerateShiftTargets(state, { owner: "self", exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift 1 of your other cards", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== targets[i].line);
  const didx: number = yield {
    prompt: "To which line?", options: dest.map(String), targets: dest, optional: false, decider: ap,
  };
  if (dest[didx] == null) return;
  shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dest[didx]);
});

register(MIDDLE_EFFECTS, "MN01:Speed:4", function* (state, ap) {
  const targets = enumerateUncovered(state, { owner: "opponent", face: "down", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift 1 opp face-down card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== targets[i].line);
  const didx: number = yield {
    prompt: "To which line?", options: dest.map(String), targets: dest, optional: false, decider: ap,
  };
  if (dest[didx] == null) return;
  shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dest[didx]);
});

// ---------------------------------------------------------------------------
// SPIRIT (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Spirit:0", function* (state, ap) {
  refreshPlayer(state, ap);
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Spirit:1", function* (state, ap) {
  drawCards(state, ap, 2);
  if (false) yield {} as Choice;
});

register(START_EFFECTS, "MN01:Spirit:1", function* (state, ap, li, card) {
  // Either discard 1 card or flip this card.
  if (state.players[ap].hand.length === 0) {
    card.faceUp = !card.faceUp; return;
  }
  const idx: number = yield {
    prompt: "Spirit 1: discard 1 or flip self?",
    options: ["discard 1", "flip self"], targets: [0, 1], optional: false, decider: ap,
  };
  if (idx === 0) yield* discardN(state, ap, 1);
  else card.faceUp = !card.faceUp;
});

register(MIDDLE_EFFECTS, "MN01:Spirit:2", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Flip 1 card", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN01:Spirit:4", function* (state, ap) {
  const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
  const options = pairs.map(([a, b]) => `swap protocols L${a}<->L${b}`);
  const idx: number = yield { prompt: "Swap 2 of your protocols", options, targets: pairs, optional: false, decider: ap };
  if (pairs[idx] == null) return;
  const [a, b] = pairs[idx];
  const ps = state.players[ap];
  [ps.protocols[a], ps.protocols[b]] = [ps.protocols[b], ps.protocols[a]];
  [ps.compiled[a], ps.compiled[b]] = [ps.compiled[b], ps.compiled[a]];
});

// ---------------------------------------------------------------------------
// WATER (MN01)
// ---------------------------------------------------------------------------

register(MIDDLE_EFFECTS, "MN01:Water:0", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length > 0) {
    yield* chooseFieldTarget("Flip 1 other card", targets, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
  }
  // Flip this card.
  card.faceUp = !card.faceUp;
});

register(MIDDLE_EFFECTS, "MN01:Water:1", function* (state, ap, li) {
  for (let ln = 0; ln < NUM_LINES; ln++) {
    if (ln === li) continue;
    playTopDeckFaceDown(state, ap, ln);
  }
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Water:2", function* (state, ap) {
  drawCards(state, ap, 2);
  const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
  const options = pairs.map(([a, b]) => `swap L${a}<->L${b}`).concat(["no swap"]);
  const targets = (pairs as ([number, number] | null)[]).concat([null]);
  const idx: number = yield { prompt: "Rearrange your protocols", options, targets, optional: false, decider: ap };
  if (idx < pairs.length) {
    const [a, b] = pairs[idx];
    const ps = state.players[ap];
    [ps.protocols[a], ps.protocols[b]] = [ps.protocols[b], ps.protocols[a]];
    [ps.compiled[a], ps.compiled[b]] = [ps.compiled[b], ps.compiled[a]];
  }
});

register(MIDDLE_EFFECTS, "MN01:Water:3", function* (state, ap, li, card) {
  const lidx: number = yield {
    prompt: "Pick a line", options: ["0", "1", "2"], targets: [0, 1, 2], optional: false, decider: ap,
  };
  if (lidx < 0 || lidx > 2) return;
  for (const pl of [0, 1] as PlayerIndex[]) {
    const stack = lineStack(state.lines[lidx], pl);
    for (let pos = stack.length - 1; pos >= 0; pos--) {
      const c = stack[pos];
      if (c === card) continue;
      const v = c.faceUp ? CARD_DEFS[c.defId].value : FACE_DOWN_BASE_VALUE;
      if (v === 2) returnCardToHand(state, lidx, pl, pos);
    }
  }
});

register(MIDDLE_EFFECTS, "MN01:Water:4", function* (state, ap) {
  const targets = enumerateUncovered(state, { owner: "self", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Return 1 of your cards", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  returnCardToHand(state, targets[i].line, targets[i].player, targets[i].pos);
});

// ===========================================================================
//                     MN02 + AX02 — full effect coverage
// ===========================================================================

function bothPlayersDrawTop(state: GameState, ap: PlayerIndex): void {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  for (const [taker, source] of [[ap, opp], [opp, ap]] as [PlayerIndex, PlayerIndex][]) {
    const psSrc = state.players[source];
    if (psSrc.deck.length === 0 && psSrc.trash.length > 0) {
      psSrc.deck = psSrc.trash; psSrc.trash = [];
      // Shuffle via state.rngState is approximate; use rng helper.
      // (helpers.drawCards handles this normally; we replicate inline here.)
      const rng = { state: state.rngState };
      const { rngShuffle } = require("./rng");
      rngShuffle(rng, psSrc.deck);
      state.rngState = rng.state;
    }
    if (psSrc.deck.length > 0) {
      const c = psSrc.deck.pop()!;
      c.owner = taker;
      c.faceUp = false;
      state.players[taker].hand.push(c);
    }
  }
}

// ----- MN02: Chaos ---------------------------------------------------------

register(MIDDLE_EFFECTS, "MN02:Chaos:0", function* (state, ap) {
  // In each line, flip 1 covered card.
  for (let ln = 0; ln < 3; ln++) {
    const covered: FieldTarget[] = [];
    for (const pl of [0, 1] as PlayerIndex[]) {
      const s = lineStack(state.lines[ln], pl);
      for (let pos = 0; pos < s.length - 1; pos++) {
        const c = s[pos];
        if (!c.isCommitted) covered.push({ line: ln, player: pl, pos, card: c });
      }
    }
    if (covered.length === 0) continue;
    if (covered.length === 1) {
      const t = covered[0];
      flipCard(state, t.line, t.player, t.pos);
      continue;
    }
    yield* chooseFieldTarget(`L${ln}: flip 1 covered card`, covered, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i != null && covered[i]) flipCard(state, covered[i].line, covered[i].player, covered[i].pos);
  }
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Chaos:0", function* (state, ap) {
  bothPlayersDrawTop(state, ap);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:Chaos:1", function* (state, ap) {
  for (const who of [ap, (ap === 0 ? 1 : 0) as PlayerIndex] as PlayerIndex[]) {
    const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
    const opts = pairs.map(([a, b]) => `swap P${who} L${a}<->L${b}`).concat(["no swap"]);
    const idx: number = yield {
      prompt: `Rearrange P${who}'s protocols`,
      options: opts, targets: [...pairs, null], optional: false, decider: ap,
    };
    if (idx < pairs.length) {
      const [a, b] = pairs[idx];
      const ps = state.players[who];
      [ps.protocols[a], ps.protocols[b]] = [ps.protocols[b], ps.protocols[a]];
      [ps.compiled[a], ps.compiled[b]] = [ps.compiled[b], ps.compiled[a]];
    }
  }
});

register(MIDDLE_EFFECTS, "MN02:Chaos:2", function* (state, ap, li, card) {
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    const s = lineStack(state.lines[ln], ap);
    for (let pos = 0; pos < s.length - 1; pos++) {
      const c = s[pos];
      if (!c.isCommitted && c !== card) targets.push({ line: ln, player: ap, pos, card: c });
    }
  }
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift 1 of your covered cards", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== targets[i].line);
  const didx: number = yield {
    prompt: "To which line?", options: dest.map(String), targets: dest,
    optional: false, decider: ap,
  };
  if (dest[didx] != null) shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dest[didx]);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Chaos:4", function* (state, ap) {
  const n = state.players[ap].hand.length;
  while (state.players[ap].hand.length > 0) discardToTrash(state, ap, 0);
  drawCards(state, ap, n);
  if (false) yield {} as Choice;
});

// ----- MN02: Clarity -------------------------------------------------------

// Clarity 1 top: "Start: Reveal the top card of your deck. You may
// discard the top card of your deck."
register(START_EFFECTS, "MN02:Clarity:1", function* (state, ap) {
  const ps = state.players[ap];
  if (ps.deck.length === 0 && ps.trash.length > 0) {
    ps.deck = ps.trash; ps.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, ps.deck);
    state.rngState = rng.state;
  }
  if (ps.deck.length === 0) return;
  const top = ps.deck[ps.deck.length - 1];
  const d = CARD_DEFS[top.defId];
  (state.scratch["_reveals"] ??= [] as string[]); (state.scratch["_reveals"] as string[]).push(`P${ap} reveals top: ${d.protocol} ${d.value}`);
  const idx: number = yield {
    prompt: `Top is ${d.protocol} ${d.value}. Discard it?`,
    options: ["discard", "keep"], targets: [0, -1], optional: true, decider: ap,
  };
  if (idx === 0) {
    const c = ps.deck.pop()!;
    c.faceUp = true;
    ps.trash.push(c);
  }
});

register(MIDDLE_EFFECTS, "MN02:Clarity:1", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const hand = state.players[opp].hand
    .map((c) => `${CARD_DEFS[c.defId].protocol} ${CARD_DEFS[c.defId].value}`)
    .join(", ");
  (state.scratch["_reveals"] ??= [] as string[]); (state.scratch["_reveals"] as string[]).push(`P${opp} reveals hand: ${hand || "<empty>"}`);
  if (false) yield {} as Choice;
});

register(BOTTOM_FIRST_EFFECTS, "MN02:Clarity:1", function* (state, ap) {
  drawCards(state, ap, 3);
  if (false) yield {} as Choice;
});

function clarityRevealCandidates(
  state: GameState, ap: PlayerIndex, targetValue: number,
): CardInst[] {
  const ps = state.players[ap];
  if (ps.deck.length === 0 && ps.trash.length > 0) {
    ps.deck = ps.trash; ps.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, ps.deck);
    state.rngState = rng.state;
  }
  return ps.deck.filter((c) => CARD_DEFS[c.defId].value === targetValue);
}

function clarityFinishReveal(state: GameState, ap: PlayerIndex, chosen: CardInst | null): void {
  const ps = state.players[ap];
  if (chosen) {
    const idx = ps.deck.indexOf(chosen);
    if (idx >= 0) ps.deck.splice(idx, 1);
    ps.hand.push(chosen);
  }
  const rng = { state: state.rngState };
  const { rngShuffle } = require("./rng");
  rngShuffle(rng, ps.deck);
  state.rngState = rng.state;
}

register(MIDDLE_EFFECTS, "MN02:Clarity:2", function* (state, ap) {
  // Reveal deck, player picks which value-1 to draw, shuffle, then play 1
  // value-1 from hand using full sub-play affordances.
  const cands = clarityRevealCandidates(state, ap, 1);
  let chosen: CardInst | null = null;
  if (cands.length === 1) chosen = cands[0];
  else if (cands.length > 1) {
    const idx: number = yield {
      prompt: "Pick a value-1 card from your deck to draw",
      options: cands.map((c) => `${CARD_DEFS[c.defId].protocol} 1`),
      targets: cands, optional: false, decider: ap,
    };
    if (idx >= 0 && idx < cands.length) chosen = cands[idx];
  }
  clarityFinishReveal(state, ap, chosen);
  // Now play 1 value-1 card from hand with full play-time affordances.
  const hand = state.players[ap].hand;
  const value1Indices = hand
    .map((c, i) => (CARD_DEFS[c.defId].value === 1 ? i : -1))
    .filter((i) => i >= 0);
  if (value1Indices.length === 0) return;
  const legal = legalSubPlays(state, ap, value1Indices);
  if (legal.length === 0) return;
  const idx2: number = yield {
    prompt: "Play 1 value-1 card",
    options: legal.map((x) => x.label),
    targets: legal, optional: false, decider: ap,
  };
  const pick = legal[idx2];
  if (!pick) return;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const engine = state.scratch["_engine"] as any;
  if (engine?.playCardForEffect) engine.playCardForEffect(ap, pick.hi, pick.ln, pick.fu);
});

register(MIDDLE_EFFECTS, "MN02:Clarity:3", function* (state, ap) {
  const cands = clarityRevealCandidates(state, ap, 5);
  let chosen: CardInst | null = null;
  if (cands.length === 1) chosen = cands[0];
  else if (cands.length > 1) {
    const idx: number = yield {
      prompt: "Pick a value-5 card from your deck to draw",
      options: cands.map((c) => `${CARD_DEFS[c.defId].protocol} 5`),
      targets: cands, optional: false, decider: ap,
    };
    if (idx >= 0 && idx < cands.length) chosen = cands[idx];
  }
  clarityFinishReveal(state, ap, chosen);
});

register(MIDDLE_EFFECTS, "MN02:Clarity:4", function* (state, ap) {
  if (state.players[ap].trash.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Shuffle trash into deck",
    options: ["shuffle"], targets: [0], optional: true, decider: ap,
  };
  if (idx === 0) {
    const ps = state.players[ap];
    ps.deck.push(...ps.trash);
    ps.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, ps.deck);
    state.rngState = rng.state;
  }
});

// ----- MN02: Corruption ----------------------------------------------------

// Corruption 0 top: "Flip: Flip 1 face-up covered or uncovered card in
// this stack other than this card."
register(FLIP_TRIGGER_EFFECTS, "MN02:Corruption:0", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], card.owner);
  const targets: FieldTarget[] = [];
  stack.forEach((c, pos) => {
    if (c !== card && c.faceUp) targets.push({ line: li, player: card.owner, pos, card: c });
  });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 face-up card in this stack", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN02:Corruption:1", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Return 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  returnCardToHand(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Corruption:1", function* (state) {
  // After return, push most-recent face-down hand addition onto that player's deck.
  for (const pl of [0, 1] as PlayerIndex[]) {
    const h = state.players[pl].hand;
    if (h.length > 0 && !h[h.length - 1].faceUp) {
      const c = h.pop()!;
      state.players[pl].deck.push(c);
      break;
    }
  }
  if (false) yield {} as Choice;
});

// Corruption 2 top: "After you discard cards: Your opponent discards 1 card."
register(AFTER_SELF_DISCARD_EFFECTS, "MN02:Corruption:2", function* (state, ap) {
  yield* discardN(state, (ap === 0 ? 1 : 0) as PlayerIndex, 1);
});

register(MIDDLE_EFFECTS, "MN02:Corruption:2", function* (state, ap) {
  drawCards(state, ap, 1);
  yield* discardN(state, ap, 1);
});

register(MIDDLE_EFFECTS, "MN02:Corruption:3", function* (state, ap) {
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const s = lineStack(state.lines[ln], pl);
      for (let pos = 0; pos < s.length - 1; pos++) {
        const c = s[pos];
        if (c.isCommitted || !c.faceUp) continue;
        targets.push({ line: ln, player: pl, pos, card: c });
      }
    }
  }
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Flip 1 face-up covered", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

// Corruption 6 top: "End: Either discard 1 card or delete this card."
register(END_EFFECTS, "MN02:Corruption:6", function* (state, ap, li, card) {
  if (state.players[ap].hand.length === 0) {
    // Forced self-delete.
    for (let ln = 0; ln < 3; ln++) {
      const s = lineStack(state.lines[ln], card.owner);
      const idx = s.indexOf(card);
      if (idx >= 0) { deleteCardFromField(state, ln, card.owner, idx); return; }
    }
    return;
  }
  const idx: number = yield {
    prompt: "Discard 1 card OR delete Corruption 6?",
    options: ["discard 1", "delete self"], targets: [0, 1],
    optional: false, decider: ap,
  };
  if (idx === 0) {
    yield* discardN(state, ap, 1);
  } else {
    for (let ln = 0; ln < 3; ln++) {
      const s = lineStack(state.lines[ln], card.owner);
      const idx2 = s.indexOf(card);
      if (idx2 >= 0) { deleteCardFromField(state, ln, card.owner, idx2); return; }
    }
  }
});

// ----- MN02: Courage -------------------------------------------------------

// Courage 0 top: "Start: If you have no cards in hand, draw 1 card."
register(START_EFFECTS, "MN02:Courage:0", function* (state, ap) {
  if (state.players[ap].hand.length === 0) drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:Courage:0", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Courage:0", function* (state, ap) {
  if (state.players[ap].hand.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Discard 1 → opp discards 1?",
    options: ["accept", "skip"], targets: [0, -1], optional: true, decider: ap,
  };
  if (idx === -1 || idx === 1) return;
  yield* discardN(state, ap, 1);
  yield* discardN(state, (ap === 0 ? 1 : 0) as PlayerIndex, 1);
});

register(MIDDLE_EFFECTS, "MN02:Courage:1", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const { computeLineValue } = require("./helpers");
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    if (computeLineValue(state, ln, opp) <= computeLineValue(state, ln, ap)) continue;
    const s = lineStack(state.lines[ln], opp);
    if (s.length === 0) continue;
    const c = s[s.length - 1];
    if (c.isCommitted) continue;
    targets.push({ line: ln, player: opp, pos: s.length - 1, card: c });
  }
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Delete 1 opp card in losing line", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Courage:2", function* (state, ap, li) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const { computeLineValue } = require("./helpers");
  if (computeLineValue(state, li, opp) > computeLineValue(state, li, ap)) drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Courage:3", function* (state, ap, li, card) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const { computeLineValue } = require("./helpers");
  let best = 0;
  let bestV = -1;
  for (let i = 0; i < 3; i++) {
    const v = computeLineValue(state, i, opp);
    if (v > bestV) { bestV = v; best = i; }
  }
  if (best === li) return;
  const idx: number = yield {
    prompt: `(optional) Shift Courage 3 to L${best} (opp's strongest)`,
    options: ["shift", "skip"], targets: [0, -1], optional: true, decider: ap,
  };
  if (idx === -1 || idx === 1) return;
  const s = lineStack(state.lines[li], ap);
  const pos = s.indexOf(card);
  if (pos >= 0) shiftCard(state, li, ap, pos, best);
});

// Courage 6 top: "End: If your opponent has a higher value in this line
// than you do, flip this card."
register(END_EFFECTS, "MN02:Courage:6", function* (state, ap, li, card) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const { computeLineValue } = require("./helpers");
  if (computeLineValue(state, li, opp) > computeLineValue(state, li, ap)) {
    const s = lineStack(state.lines[li], card.owner);
    if (s.indexOf(card) >= 0) card.faceUp = !card.faceUp;
  }
  if (false) yield {} as Choice;
});

// ----- MN02: Fear ----------------------------------------------------------

register(MIDDLE_EFFECTS, "MN02:Fear:0", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift or flip 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const t = targets[i];
  const sub: number = yield {
    prompt: "Shift or flip?",
    options: ["shift", "flip"], targets: [0, 1], optional: false, decider: ap,
  };
  if (sub === 0) {
    const dest = [0, 1, 2].filter((l) => l !== t.line);
    const didx: number = yield {
      prompt: "To which line?", options: dest.map(String), targets: dest,
      optional: false, decider: ap,
    };
    if (dest[didx] != null) shiftCard(state, t.line, t.player, t.pos, dest[didx]);
  } else {
    flipCard(state, t.line, t.player, t.pos);
  }
});

register(MIDDLE_EFFECTS, "MN02:Fear:1", function* (state, ap) {
  drawCards(state, ap, 2);
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const n = state.players[opp].hand.length;
  while (state.players[opp].hand.length > 0) discardToTrash(state, opp, 0);
  if (n > 1) drawCards(state, opp, n - 1);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:Fear:2", function* (state, ap) {
  const targets = enumerateUncovered(state, { owner: "opponent", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Return 1 opp card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) returnCardToHand(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Fear:3", function* (state, ap, li) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const s = lineStack(state.lines[li], opp);
  const targets: FieldTarget[] = [];
  s.forEach((c, pos) => {
    if (!c.isCommitted) targets.push({ line: li, player: opp, pos, card: c });
  });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift 1 opp card in this line", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== li);
  const didx: number = yield {
    prompt: "To which line?", options: dest.map(String), targets: dest,
    optional: false, decider: ap,
  };
  if (dest[didx] != null) shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dest[didx]);
});

register(MIDDLE_EFFECTS, "MN02:Fear:4", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  if (state.players[opp].hand.length === 0) return;
  const { rngInt } = require("./rng");
  const rng = { state: state.rngState };
  const idx = rngInt(rng, state.players[opp].hand.length);
  state.rngState = rng.state;
  discardToTrash(state, opp, idx);
  if (false) yield {} as Choice;
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Fear:5", function* (state, ap) {
  yield* discardN(state, ap, 1);
});

// ----- MN02: Ice -----------------------------------------------------------

register(MIDDLE_EFFECTS, "MN02:Ice:1", function* (state, ap, li, card) {
  const s = lineStack(state.lines[li], card.owner);
  if (s.indexOf(card) < 0) return;
  const dest = [0, 1, 2].filter((l) => l !== li);
  if (dest.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Shift Ice 1",
    options: dest.map(String).concat(["skip"]),
    targets: [...dest, -1], optional: true, decider: ap,
  };
  if (idx === -1 || idx >= dest.length) return;
  shiftCard(state, li, card.owner, s.indexOf(card), dest[idx]);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Ice:1", function* (state, ap) {
  yield* discardN(state, (ap === 0 ? 1 : 0) as PlayerIndex, 1);
});

register(MIDDLE_EFFECTS, "MN02:Ice:2", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift 1 other card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== targets[i].line);
  const didx: number = yield {
    prompt: "To which line?", options: dest.map(String), targets: dest,
    optional: false, decider: ap,
  };
  if (dest[didx] != null) shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dest[didx]);
});

register(MIDDLE_EFFECTS, "MN02:Ice:3", function* (state, ap, li, card) {
  const s = lineStack(state.lines[li], card.owner);
  const pos = s.indexOf(card);
  if (pos < 0 || pos === s.length - 1) return; // not covered
  const dest = [0, 1, 2].filter((l) => l !== li);
  if (dest.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Shift Ice 3 (covered)",
    options: dest.map(String).concat(["skip"]),
    targets: [...dest, -1], optional: true, decider: ap,
  };
  if (idx === -1 || idx >= dest.length) return;
  shiftCard(state, li, card.owner, pos, dest[idx]);
});

// ----- MN02: Luck ----------------------------------------------------------

register(MIDDLE_EFFECTS, "MN02:Luck:0", function* (state, ap) {
  const numIdx: number = yield {
    prompt: "State a number (0-6)",
    options: ["0", "1", "2", "3", "4", "5", "6"], targets: [0, 1, 2, 3, 4, 5, 6],
    optional: false, decider: ap,
  };
  const n = numIdx;
  drawCards(state, ap, 3);
  const hand = state.players[ap].hand;
  const start = Math.max(0, hand.length - 3);
  const candidateIndices: number[] = [];
  for (let i = start; i < hand.length; i++) {
    if (CARD_DEFS[hand[i].defId].value === n) candidateIndices.push(i);
  }
  if (candidateIndices.length === 0) return;
  const legal = legalSubPlays(state, ap, candidateIndices);
  if (legal.length === 0) return;
  const idx: number = yield {
    prompt: `(optional) Play a value-${n} card`,
    options: legal.map((x) => x.label).concat(["skip"]),
    targets: [...legal, null], optional: true, decider: ap,
  };
  if (idx === -1 || idx >= legal.length) return;
  const pick = legal[idx];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const engine = state.scratch["_engine"] as any;
  if (engine?.playCardForEffect) engine.playCardForEffect(ap, pick.hi, pick.ln, pick.fu);
});

register(MIDDLE_EFFECTS, "MN02:Luck:1", function* (state, ap) {
  const ps = state.players[ap];
  if (ps.deck.length === 0 && ps.trash.length === 0) return;
  if (ps.deck.length === 0) {
    ps.deck = ps.trash; ps.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, ps.deck);
    state.rngState = rng.state;
  }
  const lidx: number = yield {
    prompt: "Play top of deck face-down in which line?",
    options: ["0", "1", "2"], targets: [0, 1, 2], optional: false, decider: ap,
  };
  if (lidx < 0 || lidx >= 3) return;
  const c = ps.deck.pop()!;
  c.faceUp = false;
  lineStack(state.lines[lidx], ap).push(c);
  // "Flip that card, ignoring its middle commands" — set face-up directly
  // (skipping the face_up trigger that fires the middle), then fire top and
  // bottom enter-play triggers via the engine helper.
  c.faceUp = true;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const engine = state.scratch["_engine"] as any;
  if (engine?.enqueueEnterPlayTriggersSkipMiddle) {
    engine.enqueueEnterPlayTriggersSkipMiddle(c, ap, lidx);
  }
});

register(MIDDLE_EFFECTS, "MN02:Luck:2", function* (state, ap) {
  const ps = state.players[ap];
  if (ps.deck.length === 0 && ps.trash.length > 0) {
    ps.deck = ps.trash; ps.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, ps.deck);
    state.rngState = rng.state;
  }
  if (ps.deck.length === 0) return;
  const top = ps.deck.pop()!;
  top.faceUp = true;
  state.players[top.owner].trash.push(top);
  drawCards(state, ap, CARD_DEFS[top.defId].value);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:Luck:3", function* (state, ap, li, card) {
  const allProtos = Array.from(new Set(CARD_DEFS.map((d) => d.protocol))).sort();
  const pi: number = yield {
    prompt: "State a protocol", options: allProtos,
    targets: allProtos.map((_, i) => i), optional: false, decider: ap,
  };
  if (pi < 0 || pi >= allProtos.length) return;
  const stated = allProtos[pi];
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const psOpp = state.players[opp];
  if (psOpp.deck.length === 0 && psOpp.trash.length > 0) {
    psOpp.deck = psOpp.trash; psOpp.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, psOpp.deck);
    state.rngState = rng.state;
  }
  if (psOpp.deck.length === 0) return;
  const top = psOpp.deck.pop()!;
  top.faceUp = true;
  state.players[top.owner].trash.push(top);
  if (CARD_DEFS[top.defId].protocol !== stated) return;
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Delete 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "MN02:Luck:4", function* (state, ap, li, card) {
  const ps = state.players[ap];
  if (ps.deck.length === 0 && ps.trash.length > 0) {
    ps.deck = ps.trash; ps.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, ps.deck);
    state.rngState = rng.state;
  }
  if (ps.deck.length === 0) return;
  const top = ps.deck.pop()!;
  top.faceUp = true;
  state.players[top.owner].trash.push(top);
  const tv = CARD_DEFS[top.defId].value;
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const s = lineStack(state.lines[ln], pl);
      s.forEach((c, pos) => {
        if (c.isCommitted || c === card) return;
        const v = c.faceUp ? CARD_DEFS[c.defId].value : 2;
        if (v === tv) targets.push({ line: ln, player: pl, pos, card: c });
      });
    }
  }
  if (targets.length === 0) return;
  yield* chooseFieldTarget(`Delete 1 card with value ${tv}`, targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

// ----- MN02: Mirror --------------------------------------------------------

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Mirror:1", function* (state, ap, li, card) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    const s = lineStack(state.lines[ln], opp);
    s.forEach((c, pos) => {
      if (!c.faceUp) return;
      const d = CARD_DEFS[c.defId];
      if (d.middleText) targets.push({ line: ln, player: opp, pos, card: c });
    });
  }
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Resolve opp middle as Mirror 1", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const fn = MIDDLE_EFFECTS[CARD_DEFS[targets[i].card.defId].key];
  if (!fn) return;
  yield* fn(state, ap, li, card);
});

register(MIDDLE_EFFECTS, "MN02:Mirror:2", function* (state, ap) {
  const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
  const idx: number = yield {
    prompt: "Swap which two of your stacks?",
    options: pairs.map(([a, b]) => `L${a}<->L${b}`).concat(["no swap"]),
    targets: [...pairs, null], optional: false, decider: ap,
  };
  if (idx >= pairs.length) return;
  const [a, b] = pairs[idx];
  const lineA = state.lines[a];
  const lineB = state.lines[b];
  if (ap === 0) {
    const tmp = lineA.p0Stack; lineA.p0Stack = lineB.p0Stack; lineB.p0Stack = tmp;
  } else {
    const tmp = lineA.p1Stack; lineA.p1Stack = lineB.p1Stack; lineB.p1Stack = tmp;
  }
});

register(MIDDLE_EFFECTS, "MN02:Mirror:3", function* (state, ap, li, card) {
  const own = enumerateUncovered(state, { owner: "self", exclude: card, activePlayer: ap });
  if (own.length === 0) return;
  yield* chooseFieldTarget("Flip 1 of your cards", own, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !own[i]) return;
  const t = own[i];
  flipCard(state, t.line, t.player, t.pos);
  const oppIn = enumerateUncovered(state, { owner: "opponent", lineFilter: t.line, activePlayer: ap });
  if (oppIn.length === 0) return;
  yield* chooseFieldTarget("Flip 1 opp card in same line", oppIn, state, ap);
  const j = state.scratch["_last_target_idx"] as number | undefined;
  if (j != null && oppIn[j]) flipCard(state, oppIn[j].line, oppIn[j].player, oppIn[j].pos);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Mirror:4", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

// ----- MN02: Peace ---------------------------------------------------------

register(MIDDLE_EFFECTS, "MN02:Peace:1", function* (state) {
  for (const pl of [0, 1] as PlayerIndex[]) {
    while (state.players[pl].hand.length > 0) discardToTrash(state, pl, 0);
  }
  if (false) yield {} as Choice;
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Peace:1", function* (state, ap) {
  if (state.players[ap].hand.length === 0) drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:Peace:2", function* (state, ap) {
  drawCards(state, ap, 1);
  const hand = state.players[ap].hand;
  if (hand.length === 0) return;
  const opts = hand.map((_, i) => describeHandCard(state, ap, i));
  const hi: number = yield {
    prompt: "Play 1 card face-down",
    options: opts, targets: hand.map((_, i) => i), optional: false, decider: ap,
  };
  if (hi < 0 || hi >= hand.length) return;
  const lidx: number = yield {
    prompt: "Which line?", options: ["0", "1", "2"], targets: [0, 1, 2],
    optional: false, decider: ap,
  };
  if (lidx < 0 || lidx >= 3) return;
  const c = state.players[ap].hand.splice(hi, 1)[0];
  c.faceUp = false;
  lineStack(state.lines[lidx], ap).push(c);
});

register(MIDDLE_EFFECTS, "MN02:Peace:3", function* (state, ap) {
  if (state.players[ap].hand.length > 0) {
    const idx: number = yield {
      prompt: "(optional) Discard 1 first",
      options: ["discard", "skip"], targets: [0, -1], optional: true, decider: ap,
    };
    if (idx === 0) yield* discardN(state, ap, 1);
  }
  const threshold = state.players[ap].hand.length;
  const candidates: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const s = lineStack(state.lines[ln], pl);
      if (s.length === 0) continue;
      const c = s[s.length - 1];
      if (c.isCommitted) continue;
      const v = c.faceUp ? CARD_DEFS[c.defId].value : 2;
      if (v > threshold) candidates.push({ line: ln, player: pl, pos: s.length - 1, card: c });
    }
  }
  if (candidates.length === 0) return;
  yield* chooseFieldTarget(`Flip 1 card with value > ${threshold}`, candidates, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && candidates[i]) flipCard(state, candidates[i].line, candidates[i].player, candidates[i].pos);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:Peace:4", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:Peace:6", function* (state, ap, li, card) {
  if (state.players[ap].hand.length > 1) {
    const s = lineStack(state.lines[li], card.owner);
    if (s.indexOf(card) >= 0) card.faceUp = !card.faceUp;
  }
  if (false) yield {} as Choice;
});

// ----- MN02: Smoke ---------------------------------------------------------

function lineHasFacedown(state: GameState, ln: number): boolean {
  for (const pl of [0, 1] as PlayerIndex[]) {
    for (const c of lineStack(state.lines[ln], pl)) {
      if (!c.faceUp) return true;
    }
  }
  return false;
}

register(MIDDLE_EFFECTS, "MN02:Smoke:0", function* (state, ap) {
  for (let ln = 0; ln < 3; ln++) {
    if (lineHasFacedown(state, ln)) playTopDeckFaceDown(state, ap, ln);
  }
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:Smoke:1", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { owner: "self", exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 of your cards (then may shift)", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const t = targets[i];
  flipCard(state, t.line, t.player, t.pos);
  const dest = [0, 1, 2].filter((l) => l !== t.line);
  const didx: number = yield {
    prompt: "(optional) Shift", options: dest.map(String).concat(["skip"]),
    targets: [...dest, -1], optional: true, decider: ap,
  };
  if (didx === -1 || didx >= dest.length) return;
  const newPos = lineStack(state.lines[t.line], t.player).indexOf(t.card);
  if (newPos >= 0) shiftCard(state, t.line, t.player, newPos, dest[didx]);
});

register(MIDDLE_EFFECTS, "MN02:Smoke:3", function* (state, ap) {
  const hand = state.players[ap].hand;
  if (hand.length === 0) return;
  const eligible = [0, 1, 2].filter((ln) => lineHasFacedown(state, ln));
  if (eligible.length === 0) return;
  const hi: number = yield {
    prompt: "Pick a card to play face-down (in line with face-downs)",
    options: hand.map((_, i) => describeHandCard(state, ap, i)),
    targets: hand.map((_, i) => i), optional: false, decider: ap,
  };
  if (hi < 0 || hi >= hand.length) return;
  const lidx: number = yield {
    prompt: "Which line?", options: eligible.map(String), targets: eligible,
    optional: false, decider: ap,
  };
  if (eligible[lidx] == null) return;
  const c = state.players[ap].hand.splice(hi, 1)[0];
  c.faceUp = false;
  lineStack(state.lines[eligible[lidx]], ap).push(c);
});

register(MIDDLE_EFFECTS, "MN02:Smoke:4", function* (state, ap) {
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const s = lineStack(state.lines[ln], pl);
      for (let pos = 0; pos < s.length - 1; pos++) {
        const c = s[pos];
        if (c.isCommitted || c.faceUp) continue;
        targets.push({ line: ln, player: pl, pos, card: c });
      }
    }
  }
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Shift 1 covered face-down card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== targets[i].line);
  const didx: number = yield {
    prompt: "To which line?", options: dest.map(String), targets: dest,
    optional: false, decider: ap,
  };
  if (dest[didx] != null) shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dest[didx]);
});

// ----- MN02: Time ----------------------------------------------------------

register(MIDDLE_EFFECTS, "MN02:Time:0", function* (state, ap) {
  const trash = state.players[ap].trash;
  if (trash.length > 0) {
    const opts = trash.map((c, i) => `trash[${i}]: ${CARD_DEFS[c.defId].protocol} ${CARD_DEFS[c.defId].value}`);
    const idx: number = yield {
      prompt: "Play 1 card from your trash",
      options: opts, targets: trash.map((_, i) => i), optional: false, decider: ap,
    };
    if (idx >= 0 && idx < trash.length) {
      const c = trash[idx]; // don't remove until we've picked a destination
      const d = CARD_DEFS[c.defId];
      const spirit1 = playerMayPlayAnyLineFaceup(state, ap);
      const psychic1 = oppMustPlayFacedown(state, ap);
      const lineBlocked = [0, 1, 2].map((ln) => oppPlayBlockedInLine(state, ln, ap));
      const lineFdBlocked = [0, 1, 2].map((ln) => oppPlayFacedownBlockedInLine(state, ln, ap));
      const chaos3 = d.protocol === "Chaos" && d.value === 3;
      const corruption0 = d.protocol === "Corruption" && d.value === 0;
      const unrestrictedFu = spirit1 || chaos3 || corruption0;
      const legal: { ln: number; fu: boolean; cross: boolean; label: string }[] = [];
      for (let ln = 0; ln < 3; ln++) {
        if (lineBlocked[ln] || psychic1) continue;
        if (unrestrictedFu || state.players[ap].protocols[ln] === d.protocol) {
          legal.push({ ln, fu: true, cross: false, label: `FU L${ln}` });
        }
      }
      for (let ln = 0; ln < 3; ln++) {
        if (lineBlocked[ln] || lineFdBlocked[ln]) continue;
        legal.push({ ln, fu: false, cross: false, label: `FD L${ln}` });
      }
      if (corruption0) {
        for (let ln = 0; ln < 3; ln++) {
          legal.push({ ln, fu: true, cross: true, label: `FU OPP L${ln}` });
          legal.push({ ln, fu: false, cross: true, label: `FD OPP L${ln}` });
        }
      }
      if (legal.length > 0) {
        const li2: number = yield {
          prompt: "Place where?", options: legal.map((l) => l.label),
          targets: legal, optional: false, decider: ap,
        };
        const pick = legal[li2];
        if (pick) {
          trash.splice(trash.indexOf(c), 1);
          const targetSide: PlayerIndex = pick.cross ? ((ap === 0 ? 1 : 0) as PlayerIndex) : ap;
          if (pick.cross) c.owner = targetSide;
          c.faceUp = pick.fu;
          lineStack(state.lines[pick.ln], targetSide).push(c);
        }
      }
    }
  }
  const ps = state.players[ap];
  ps.deck.push(...ps.trash);
  ps.trash = [];
  const rng = { state: state.rngState };
  const { rngShuffle } = require("./rng");
  rngShuffle(rng, ps.deck);
  state.rngState = rng.state;
});

register(MIDDLE_EFFECTS, "MN02:Time:1", function* (state, ap) {
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const s = lineStack(state.lines[ln], pl);
      for (let pos = 0; pos < s.length - 1; pos++) {
        const c = s[pos];
        if (!c.isCommitted) targets.push({ line: ln, player: pl, pos, card: c });
      }
    }
  }
  if (targets.length > 0) {
    yield* chooseFieldTarget("Flip 1 covered card", targets, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
  }
  // Discard entire deck.
  const ps = state.players[ap];
  while (ps.deck.length > 0) {
    const c = ps.deck.pop()!;
    c.faceUp = true;
    state.players[c.owner].trash.push(c);
  }
});

// Time 2 top (Codex 8/2025 errata): "After you shuffle your deck: Draw 1
// card. Then, you may shift this card."
register(AFTER_SELF_SHUFFLE_EFFECTS, "MN02:Time:2", function* (state, ap, li, card) {
  drawCards(state, ap, 1);
  const s = lineStack(state.lines[li], card.owner);
  const pos = s.indexOf(card);
  if (pos < 0) return;
  const dest = [0, 1, 2].filter((l) => l !== li);
  if (dest.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Shift Time 2",
    options: dest.map(String).concat(["skip"]),
    targets: [...dest, -1], optional: true, decider: ap,
  };
  if (idx >= 0 && idx < dest.length) shiftCard(state, li, card.owner, pos, dest[idx]);
});

register(MIDDLE_EFFECTS, "MN02:Time:2", function* (state, ap) {
  if (state.players[ap].trash.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Shuffle trash into deck",
    options: ["shuffle"], targets: [0], optional: true, decider: ap,
  };
  if (idx === 0) {
    const ps = state.players[ap];
    ps.deck.push(...ps.trash);
    ps.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, ps.deck);
    state.rngState = rng.state;
    (state.scratch as Record<string, unknown>)[`_pending_after_shuffle_by_p${ap}`] = true;
  }
});

register(MIDDLE_EFFECTS, "MN02:Time:3", function* (state, ap, li) {
  const trash = state.players[ap].trash;
  if (trash.length === 0) return;
  const opts = trash.map((c, i) => `trash[${i}]: ${CARD_DEFS[c.defId].protocol} ${CARD_DEFS[c.defId].value}`);
  const idx: number = yield {
    prompt: "Reveal a trash card; play face-down in another line",
    options: opts, targets: trash.map((_, i) => i), optional: false, decider: ap,
  };
  if (idx < 0 || idx >= trash.length) return;
  const c = trash.splice(idx, 1)[0];
  const other = [0, 1, 2].filter((l) => l !== li);
  const didx: number = yield {
    prompt: "Which other line?",
    options: other.map(String), targets: other, optional: false, decider: ap,
  };
  if (other[didx] != null) {
    c.faceUp = false;
    lineStack(state.lines[other[didx]], ap).push(c);
  }
});

register(MIDDLE_EFFECTS, "MN02:Time:4", function* (state, ap) {
  drawCards(state, ap, 2);
  for (let k = 0; k < 2; k++) {
    if (state.players[ap].hand.length === 0) return;
    yield* discardN(state, ap, 1);
  }
});

// ----- MN02: War -----------------------------------------------------------

// War 0 top: "After you refresh: You may flip this card."
register(AFTER_SELF_REFRESH_EFFECTS, "MN02:War:0", function* (state, ap, li, card) {
  const idx: number = yield {
    prompt: "(optional) Flip War 0 now?",
    options: ["flip", "skip"], targets: [0, -1], optional: true, decider: ap,
  };
  if (idx === 0) {
    const s = lineStack(state.lines[li], card.owner);
    if (s.indexOf(card) >= 0) card.faceUp = !card.faceUp;
  }
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:War:0", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Delete 1 card", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:War:1", function* (state, ap) {
  yield* discardOptionalLoop(state, ap, state.players[ap].hand.length);
  refreshPlayer(state, ap);
});

register(MIDDLE_EFFECTS, "MN02:War:2", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:War:2", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  while (state.players[opp].hand.length > 0) discardToTrash(state, opp, 0);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:War:3", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

register(BOTTOM_ON_PLAY_EFFECTS, "MN02:War:3", function* (state, ap) {
  const hand = state.players[ap].hand;
  if (hand.length === 0) return;
  const opts = hand.map((_, i) => describeHandCard(state, ap, i));
  const hi: number = yield {
    prompt: "(optional) Play 1 card face-down",
    options: opts.concat(["skip"]),
    targets: hand.map((_, i) => i).concat([-1]),
    optional: true, decider: ap,
  };
  if (hi === -1 || hi >= hand.length) return;
  const lidx: number = yield {
    prompt: "Which line?", options: ["0", "1", "2"], targets: [0, 1, 2],
    optional: false, decider: ap,
  };
  if (lidx < 0 || lidx >= 3) return;
  const c = state.players[ap].hand.splice(hi, 1)[0];
  c.faceUp = false;
  lineStack(state.lines[lidx], ap).push(c);
});

register(MIDDLE_EFFECTS, "MN02:War:4", function* (state, ap) {
  yield* discardN(state, (ap === 0 ? 1 : 0) as PlayerIndex, 1);
});

// ----- AX02: Assimilation --------------------------------------------------

register(MIDDLE_EFFECTS, "AX02:Assimilation:1", function* (state, ap) {
  yield* discardN(state, ap, 1);
  refreshPlayer(state, ap);
});

register(BOTTOM_ON_PLAY_EFFECTS, "AX02:Assimilation:1", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const psOpp = state.players[opp];
  if (psOpp.deck.length === 0 && psOpp.trash.length > 0) {
    psOpp.deck = psOpp.trash; psOpp.trash = [];
    const rng = { state: state.rngState };
    const { rngShuffle } = require("./rng");
    rngShuffle(rng, psOpp.deck);
    state.rngState = rng.state;
  }
  if (psOpp.deck.length > 0) {
    const c = psOpp.deck.pop()!;
    c.owner = ap;
    c.faceUp = false;
    state.players[ap].hand.push(c);
  }
  const hand = state.players[ap].hand;
  if (hand.length === 0) return;
  const opts = hand.map((_, i) => describeHandCard(state, ap, i));
  const idx: number = yield {
    prompt: "Discard 1 card into opp's trash",
    options: opts, targets: hand.map((_, i) => i),
    optional: false, decider: ap,
  };
  if (idx < 0 || idx >= hand.length) return;
  const c = state.players[ap].hand.splice(idx, 1)[0];
  c.faceUp = true;
  state.players[opp].trash.push(c);
});

register(MIDDLE_EFFECTS, "AX02:Assimilation:4", function* (state, ap) {
  bothPlayersDrawTop(state, ap);
  if (false) yield {} as Choice;
});

// ----- AX02: Diversity -----------------------------------------------------
// Diversity 6 top: "End: If there are not at least 3 different protocols
// on cards in the field, delete this card." Codex says this is an End
// trigger, not continuous. The `checkDiversity6SelfDestruct` continuous
// sweep in helpers.ts remains as a defence-in-depth safety net but the
// rules-correct fire point is the End trigger below.
register(END_EFFECTS, "AX02:Diversity:6", function* (state) {
  checkDiversity6SelfDestruct(state);
  if (false) yield {} as Choice;
});

// ----- AX02: Unity ---------------------------------------------------------

function countUnityInField(state: GameState): number {
  let n = 0;
  for (let ln = 0; ln < 3; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      for (const c of lineStack(state.lines[ln], pl)) {
        if (CARD_DEFS[c.defId].protocol === "Unity") n++;
      }
    }
  }
  return n;
}

register(MIDDLE_EFFECTS, "AX02:Unity:0", function* (state, ap, li, card) {
  if (countUnityInField(state) <= 1) return;
  const idx: number = yield {
    prompt: "Flip 1 card or draw 1?",
    options: ["flip", "draw"], targets: [0, 1], optional: false, decider: ap,
  };
  if (idx === 1) { drawCards(state, ap, 1); return; }
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) { drawCards(state, ap, 1); return; }
  yield* chooseFieldTarget("Flip 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(BOTTOM_FIRST_EFFECTS, "AX02:Unity:0", function* (state, ap, li, card) {
  const idx: number = yield {
    prompt: "First: flip 1 card or draw 1?",
    options: ["flip", "draw"], targets: [0, 1], optional: false, decider: ap,
  };
  if (idx === 1) { drawCards(state, ap, 1); return; }
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) { drawCards(state, ap, 1); return; }
  yield* chooseFieldTarget("Flip 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

register(MIDDLE_EFFECTS, "AX02:Unity:3", function* (state, ap, li, card) {
  if (countUnityInField(state) <= 1) return;
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Flip 1 card", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

// ---------------------------------------------------------------------------
// Sentinels
// ---------------------------------------------------------------------------

export function* uncommitSentinel(card: CardInst): EffectGen {
  card.isCommitted = false;
  if (false) yield {} as Choice;
}

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

function keyForDefId(defId: number): string | null {
  if (defId < 0 || defId >= CARD_DEFS.length) return null;
  return CARD_DEFS[defId].key;
}
export function getMiddleEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (MIDDLE_EFFECTS[k] ?? null);
}
export function getBottomFirstEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (BOTTOM_FIRST_EFFECTS[k] ?? null);
}
export function getBottomOnPlayEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (BOTTOM_ON_PLAY_EFFECTS[k] ?? null);
}
export function getStartEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (START_EFFECTS[k] ?? null);
}
export function getEndEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (END_EFFECTS[k] ?? null);
}
export function getTopTriggerEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (TOP_TRIGGER_EFFECTS[k] ?? null);
}
export function getWhenCoveredEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (WHEN_COVERED_EFFECTS[k] ?? null);
}
export function getAfterClearCacheEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_CLEAR_CACHE_EFFECTS[k] ?? null);
}
export function getAfterSelfDiscardEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_SELF_DISCARD_EFFECTS[k] ?? null);
}
export function getAfterOppDiscardEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_OPP_DISCARD_EFFECTS[k] ?? null);
}
export function getAfterSelfDeleteEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_SELF_DELETE_EFFECTS[k] ?? null);
}
export function getAfterSelfDrawEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_SELF_DRAW_EFFECTS[k] ?? null);
}
export function getAfterSelfShuffleEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_SELF_SHUFFLE_EFFECTS[k] ?? null);
}
export function getAfterSelfRefreshEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_SELF_REFRESH_EFFECTS[k] ?? null);
}
export function getFlipTriggerEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (FLIP_TRIGGER_EFFECTS[k] ?? null);
}
export function getWhenDeletedByCompileEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (WHEN_DELETED_BY_COMPILE_EFFECTS[k] ?? null);
}
