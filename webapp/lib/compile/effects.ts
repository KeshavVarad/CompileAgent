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
  sourceStillActive,
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
export const AFTER_OPP_DRAW_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_OPP_REFRESH_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_OPP_COMPILE_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_ANY_REFRESH_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_OPP_PLAY_IN_LINE_EFFECTS: Record<string, EffectFn> = {};
export const AFTER_SELF_DISCARD_ON_OPP_TURN_EFFECTS: Record<string, EffectFn> = {};
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
  // Render labels from the decider's perspective so face-down identities
  // on the opp's side stay hidden in the prompt (Codex p.5).
  const options = targets.map((t) => describeCard(state, t, decider));
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

// Control component rearrange generator — yielded before compile/refresh
// when ap holds the control component. Codex p.5-6: "first the control
// component is returned to its neutral position and that player may
// rearrange one player's protocols — either theirs or their opponent's
// — then they complete their compile or refresh." Codex p.8: "the
// control component resets to neutral even if you choose not to
// rearrange any protocols."
export function* controlRearrangeGen(state: GameState, ap: PlayerIndex): EffectGen {
  if (state.controlHolder !== ap) return;
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const idx: number = yield {
    prompt: "Control component — rearrange whose protocols? (or skip)",
    options: [
      `yours (P${ap + 1})`,
      `opp's (P${opp + 1})`,
      "skip (no rearrange)",
    ],
    targets: [ap, opp, -1],
    optional: false,
    decider: ap,
  };
  const targetPlayer: PlayerIndex | null = idx === 0 ? ap : idx === 1 ? opp : null;
  if (targetPlayer !== null) {
    const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
    const opts = pairs.map(([a, b]) => `swap P${targetPlayer + 1} L${a}<->L${b}`);
    const pidx: number = yield {
      prompt: `Swap which two of P${targetPlayer + 1}'s protocols?`,
      options: opts,
      targets: pairs,
      optional: false,
      decider: ap,
    };
    if (pidx >= 0 && pidx < pairs.length) {
      const [a, b] = pairs[pidx];
      const ps = state.players[targetPlayer];
      [ps.protocols[a], ps.protocols[b]] = [ps.protocols[b], ps.protocols[a]];
      [ps.compiled[a], ps.compiled[b]] = [ps.compiled[b], ps.compiled[a]];
    }
  }
  // Reset control component to neutral position unconditionally (Codex p.8).
  state.controlHolder = null;
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

register(MIDDLE_EFFECTS, "AX01:Hate:2", function* (state, ap, _li, card) {
  // Delete your highest value uncovered card. Delete opponent's highest
  // value uncovered card. Codex p.12 clarification: if Hate 2 itself is
  // your highest value uncovered card, the first clause deletes it and
  // the second clause "no longer exists and does not trigger."
  for (const who of [ap, (1 - ap) as PlayerIndex]) {
    if (!sourceStillActive(state, card)) return;
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

// Love 1 bottom — "End: You may give 1 card from your hand to your
// opponent. If you do, draw 2 cards." Fires only while uncovered.
register(END_EFFECTS, "AX01:Love:1", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  const hand = state.players[ap].hand;
  if (hand.length === 0) return;
  const opts = hand.map((_, i) => describeHandCard(state, ap, i));
  const idx: number = yield {
    prompt: "Give a card to opponent (optional)? If so pick which",
    options: opts, targets: hand.map((_, i) => i),
    optional: true, decider: ap,
  };
  if (idx === -1 || idx < 0 || idx >= hand.length) return;
  const c = state.players[ap].hand.splice(idx, 1)[0];
  c.owner = (ap === 0 ? 1 : 0) as PlayerIndex;
  state.players[c.owner].hand.push(c);
  drawCards(state, ap, 2);
});

register(MIDDLE_EFFECTS, "AX01:Love:2", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  drawCards(state, opp, 1);
  yield* refreshPlayer(state, ap);
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
  // "Shift 1 of your opponent's covered cards." Default rule on shift
  // is uncovered-only; Darkness 0 explicitly overrides via the
  // "covered" keyword, so we enumerate only cards beneath the top of
  // each opp stack (pos < length - 1 = covered) and exclude any cards
  // mid-commit.
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const targets: FieldTarget[] = [];
  for (let li = 0; li < NUM_LINES; li++) {
    const stack = lineStack(state.lines[li], opp);
    for (let pos = 0; pos < stack.length - 1; pos++) {
      const c = stack[pos];
      if (c.isCommitted) continue;
      targets.push({ line: li, player: opp, pos, card: c });
    }
  }
  yield* chooseFieldTarget("Shift 1 of opponent's covered cards", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const src = targets[i];
  // Codex p.4: a shift must be to a DIFFERENT line on the same side.
  // Exclude the source line from the destination prompt.
  const destLines = [0, 1, 2].filter((l) => l !== src.line);
  const dstIdx: number = yield {
    prompt: "To which line?",
    options: destLines.map((l) => `L${l + 1}`),
    targets: destLines,
    optional: false,
    decider: ap,
  };
  if (destLines[dstIdx] == null) return;
  shiftCard(state, src.line, src.player, src.pos, destLines[dstIdx]);
});

register(MIDDLE_EFFECTS, "MN01:Darkness:1", function* (state, ap, _li, card) {
  // "Flip 1 of your opponent's cards. You may shift that card."
  const targets = enumerateUncovered(state, { owner: "opponent", activePlayer: ap });
  yield* chooseFieldTarget("Flip 1 of opponent's cards", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const tgt = targets[i];
  flipCard(state, tgt.line, tgt.player, tgt.pos);
  // Codex: if Darkness 1 itself leaves play in the flip cascade, the
  // remaining "you may shift that card" stops.
  if (!sourceStillActive(state, card)) return;
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

// Psychic 1 bottom: "Start: Flip this card." Bottom-tier effects only
// resolve while uncovered (Codex), so covering Psychic 1 keeps the top
// "opp must play face-down" persistent effect active indefinitely.
// Mirrors src/compile_engine/effects.py.
register(START_EFFECTS, "MN01:Psychic:1", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  // Flip face-down — deactivates the top lockout. No trigger pushed
  // since this is a deactivation, not an on-flip-up event.
  card.faceUp = false;
  if (false) yield {} as Choice;
});

register(START_EFFECTS, "MN01:Death:1", function* (state, ap, li, card) {
  // Errata: "Start: You may draw 1 card. If you do, delete 1 other card. Then, delete this card."
  // Codex p.2: if the "delete 1 other" sub-instruction has no valid
  // target, the chain still resolves (you don't gain the benefit of
  // the unfulfilled instruction, but "Then, delete this card" still
  // fires). So offer the optional draw chain even when no other delete
  // targets currently exist.
  const deckHas = state.players[ap].deck.length > 0 || state.players[ap].trash.length > 0;
  if (!deckHas) return;
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
  // Delete this card from wherever it is (Codex FAQ p.7: top can self-delete even if covered).
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

// Fire 3 bottom — "End: You may discard 1 card. If you do, flip 1 card."
// Fires only while uncovered.
register(END_EFFECTS, "MN01:Fire:3", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  if (state.players[ap].hand.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Discard 1 to flip 1 card?",
    options: ["accept", "skip"], targets: [0, -1],
    optional: true, decider: ap,
  };
  if (idx !== 0) return;
  yield* discardN(state, ap, 1);
  const flipTargets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (flipTargets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 card", flipTargets, state, ap);
  const j = state.scratch["_last_target_idx"] as number | undefined;
  if (j == null || !flipTargets[j]) return;
  flipCard(state, flipTargets[j].line, flipTargets[j].player, flipTargets[j].pos);
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
  // Draw 2 cards. Shift 1 card either to or from this line.
  // Codex p.3 default targeting: "your cards or your opponent's cards
  // can both be selected" — opp's cards are valid too. Shifts stay on
  // the card's own side; "this line" refers to the line column.
  drawCards(state, ap, 2);
  const candidates = enumerateShiftTargets(state, { owner: "any", activePlayer: ap });
  if (candidates.length === 0) return;
  type Opt = { label: string; t: FieldTarget; dst: number };
  const opts: Opt[] = [];
  for (const t of candidates) {
    const sideTag = t.player === ap ? "yours" : "opp";
    if (t.line === li) {
      for (const dst of [0, 1, 2]) if (dst !== li) opts.push({ label: `FROM L${t.line} → L${dst} (${sideTag}): ${describeCard(state, t, ap)}`, t, dst });
    } else {
      opts.push({ label: `TO L${li} ← L${t.line} (${sideTag}): ${describeCard(state, t, ap)}`, t, dst: li });
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
  // Codex: if Gravity 2 itself leaves play during the flip cascade, the
  // shift clause stops. (exclude=card kept it from being a direct flip
  // target, but a downstream trigger could still remove it.)
  if (!sourceStillActive(state, card)) return;
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

register(MIDDLE_EFFECTS, "MN01:Life:0", function* (state, ap, li, card) {
  // Codex p.9: "If Life 0 gets covered during this process, its middle
  // command stops." Notes lines first; processes one line at a time
  // with consequences before next; short-circuits if Life 0 leaves the
  // field or is no longer uncovered on owner's side.
  const linesToPlay = [0, 1, 2].filter((ln) => lineStack(state.lines[ln], ap).length > 0);
  for (const ln of linesToPlay) {
    const ownerStack = lineStack(state.lines[li], ap);
    if (ownerStack.length === 0 || ownerStack[ownerStack.length - 1] !== card) return;
    playTopDeckFaceDown(state, ap, ln);
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
  // Codex: if Light 0 itself leaves play in the flip cascade (e.g. the
  // flipped card's middle deletes Light 0), the draw clause stops.
  if (!sourceStillActive(state, card)) return;
  const v = flipped.faceUp ? CARD_DEFS[flipped.defId].value : FACE_DOWN_BASE_VALUE;
  drawCards(state, ap, v);
});

// Light 1 bottom — "End: Draw 1 card." Bottom-tier End: only fires
// while the card is face-up + uncovered.
register(END_EFFECTS, "MN01:Light:1", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
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
  // Shift all face-down cards in this line to another line. Codex p.9:
  // "The face-down cards shifted by Light 3 maintain the same relative
  // positioning in their stacks." Iterate bottom→top and re-lookup each
  // card's current position before shifting so the destination accumulates
  // bottom→top in source order.
  const otherLines = [0, 1, 2].filter((l) => l !== li);
  if (otherLines.length === 0) return;
  const didx: number = yield {
    prompt: "Shift all face-down in this line to which other line?",
    options: otherLines.map(String), targets: otherLines, optional: false, decider: ap,
  };
  const dst = otherLines[didx];
  if (dst == null) return;
  for (const pl of [0, 1] as PlayerIndex[]) {
    const fdCards = lineStack(state.lines[li], pl).filter((c) => !c.faceUp);
    for (const c of fdCards) {
      const curS = lineStack(state.lines[li], pl);
      const pos = curS.indexOf(c);
      if (pos >= 0) shiftCard(state, li, pl, pos, dst);
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
// Metal 6 top: "When this card would be covered or flipped: First, delete
// this card." Two trigger paths:
//   - "Covered" — registered as @when_covered, fires before the
//     committed card lands on top.
//   - "Flipped" — preempted directly in flipCard (see helpers.ts). The
//     post-flip FLIP_TRIGGER broadcast filters on c.faceUp, which would
//     silently skip face-up→face-down transitions — and the rule
//     requires deleting BEFORE the flip in any case.
register(WHEN_COVERED_EFFECTS, "MN01:Metal:6", metal6SelfDelete);

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
  // Errata 9/2025: "Flip each other uncovered face-up card." Codex p.9:
  // "this only affects uncovered cards" + owner notes the set first,
  // then processes one at a time. Uses flipCard (not raw face_up
  // assignment) so flip triggers / Ice 4 immunity / when-covered hooks
  // all behave correctly.
  type E = { line: number; player: PlayerIndex; card: CardInst };
  const eligible: E[] = [];
  for (let ln = 0; ln < NUM_LINES; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = lineStack(state.lines[ln], pl);
      if (stack.length === 0) continue;
      const top = stack[stack.length - 1];
      if (top === card || !top.faceUp) continue;
      eligible.push({ line: ln, player: pl, card: top });
    }
  }
  for (const e of eligible) {
    const curS = lineStack(state.lines[e.line], e.player);
    const pos = curS.indexOf(e.card);
    if (pos >= 0) flipCard(state, e.line, e.player, pos);
  }
  if (false) yield {} as Choice;
});

// Plague 4 bottom — "End: Your opponent deletes 1 of their face-down
// cards. You may flip this card." Fires only while uncovered.
// Mirrors src/compile_engine/effects.py: the deletion choice is made
// by the opponent (decider=opp), then the owner may optionally flip
// Plague 4 face-down.
register(END_EFFECTS, "MN01:Plague:4", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < NUM_LINES; ln++) {
    const s = lineStack(state.lines[ln], opp);
    s.forEach((c, pos) => {
      if (!c.faceUp) targets.push({ line: ln, player: opp, pos, card: c });
    });
  }
  if (targets.length > 0) {
    const options = targets.map((t) => describeCard(state, t, ap));
    const idx: number = yield {
      prompt: "Opponent deletes one of their face-down cards",
      options, targets,
      optional: false, decider: opp,
    };
    if (idx >= 0 && idx < targets.length) {
      const t = targets[idx];
      deleteCardFromField(state, t.line, t.player, t.pos);
    }
  }
  // Optional self-flip.
  const flipIdx: number = yield {
    prompt: "(optional) Flip Plague 4 (this card)?",
    options: ["flip", "skip"], targets: [0, -1],
    optional: true, decider: ap,
  };
  if (flipIdx !== 0) return;
  // Re-locate the card in case it moved during the prompt window.
  for (let ln = 0; ln < NUM_LINES; ln++) {
    const s = lineStack(state.lines[ln], ap);
    const idx = s.indexOf(card);
    if (idx >= 0) {
      flipCard(state, ln, ap, idx);
      return;
    }
  }
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

// Psychic 4 bottom — "End: You may return 1 of your opponent's cards.
// If you do, flip this card." Fires only while uncovered.
register(END_EFFECTS, "MN01:Psychic:4", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  const targets = enumerateUncovered(state, { owner: "opponent", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Return 1 of opponent's cards to their hand",
    targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  returnCardToHand(state, targets[i].line, targets[i].player, targets[i].pos);
  // Flip this card.
  for (let ln = 0; ln < NUM_LINES; ln++) {
    const s = lineStack(state.lines[ln], ap);
    const idx = s.indexOf(card);
    if (idx >= 0) {
      flipCard(state, ln, ap, idx);
      return;
    }
  }
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

// Speed 3 bottom — "End: You may shift 1 of your cards. If you do,
// flip this card." Fires only while uncovered.
register(END_EFFECTS, "MN01:Speed:3", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  const targets = enumerateShiftTargets(state, { owner: "self", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Shift 1 of your cards (then flip Speed 3)",
    targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const dest = [0, 1, 2].filter((l) => l !== targets[i].line);
  const didx: number = yield {
    prompt: "To which line?", options: dest.map(String), targets: dest, optional: false, decider: ap,
  };
  if (dest[didx] == null) return;
  shiftCard(state, targets[i].line, targets[i].player, targets[i].pos, dest[didx]);
  // Flip this card.
  for (let ln = 0; ln < NUM_LINES; ln++) {
    const s = lineStack(state.lines[ln], ap);
    const idx = s.indexOf(card);
    if (idx >= 0) {
      flipCard(state, ln, ap, idx);
      return;
    }
  }
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
  // Refresh. Draw 1 card. (Two sentences per Codex p.2 — refresh may
  // yield the control-rearrange prompt before the draw.)
  yield* refreshPlayer(state, ap);
  drawCards(state, ap, 1);
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

register(MIDDLE_EFFECTS, "MN01:Water:1", function* (state, ap, li, card) {
  // Play top of deck face-down in each OTHER line. Codex p.10 parallels
  // Life 0's clarification — process one line at a time, bail if Water 1
  // leaves the field mid-resolution.
  const otherLines = [0, 1, 2].filter((ln) => ln !== li);
  for (const ln of otherLines) {
    if (!lineStack(state.lines[li], ap).includes(card)) return;
    playTopDeckFaceDown(state, ap, ln);
  }
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN01:Water:2", function* (state, ap) {
  // Draw 2 cards. Rearrange your protocols. Codex p.4: "the end state
  // of that rearrangement must be different from the start state."
  drawCards(state, ap, 2);
  const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
  const options = pairs.map(([a, b]) => `swap L${a}<->L${b}`);
  const idx: number = yield {
    prompt: "Rearrange your protocols (must swap a pair)",
    options, targets: pairs, optional: false, decider: ap,
  };
  if (idx >= 0 && idx < pairs.length) {
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

// Chaos 0 bottom — "Start: Draw the top card of your opponent's deck.
// Your opponent draws the top card of your deck." Fires only while
// uncovered.
register(START_EFFECTS, "MN02:Chaos:0", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  bothPlayersDrawTop(state, ap);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:Chaos:1", function* (state, ap) {
  // Rearrange your protocols. Rearrange opp's protocols.
  // Codex p.13: "You must make a change to the protocols of both
  // players." + Codex p.4 "the end state of that rearrangement must be
  // different from the start state" — no "no swap" option offered.
  const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
  for (const who of [ap, (ap === 0 ? 1 : 0) as PlayerIndex] as PlayerIndex[]) {
    const opts = pairs.map(([a, b]) => `swap P${who} L${a}<->L${b}`);
    const idx: number = yield {
      prompt: `Rearrange P${who}'s protocols (must change)`,
      options: opts, targets: pairs, optional: false, decider: ap,
    };
    if (idx >= 0 && idx < pairs.length) {
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

// Chaos 4 bottom — "End: Discard your hand. Draw that many cards."
// Fires only while uncovered.
register(END_EFFECTS, "MN02:Chaos:4", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
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

// Courage 0 bottom — "End: You may discard 1 card. If you do, your
// opponent discards 1 card." Fires only while uncovered.
register(END_EFFECTS, "MN02:Courage:0", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
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

// Courage 2 middle — "Draw 1 card."
register(MIDDLE_EFFECTS, "MN02:Courage:2", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

// Courage 2 bottom — "End: If opp has higher total value in this line,
// draw 1 card." Fires only while uncovered.
register(END_EFFECTS, "MN02:Courage:2", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const { computeLineValue } = require("./helpers");
  if (computeLineValue(state, li, opp) > computeLineValue(state, li, ap)) drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

// Courage 3 bottom — "End: You may shift this card to the line where
// opp has their highest total value." Fires only while uncovered.
// Codex p.9: "Multiple lines can be tied for highest total value. In
// this case, the player chooses."
register(END_EFFECTS, "MN02:Courage:3", function* (state, ap, li, card) {
  const stack0 = lineStack(state.lines[li], ap);
  if (stack0.length === 0 || stack0[stack0.length - 1] !== card) return;
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const { computeLineValue } = require("./helpers");
  const vals = [0, 1, 2].map((i) => computeLineValue(state, i, opp) as number);
  const maxV = Math.max(...vals);
  const dests = [0, 1, 2].filter((i) => vals[i] === maxV && i !== li);
  if (dests.length === 0) return;
  const opts = dests.map((i) => `L${i} (opp value ${vals[i]})`).concat(["skip"]);
  const targets = (dests as number[]).concat([-1]);
  const idx: number = yield {
    prompt: "(optional) Shift Courage 3 to opp's strongest line",
    options: opts, targets, optional: true, decider: ap,
  };
  if (idx === -1 || idx < 0 || idx >= dests.length) return;
  const best = dests[idx];
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

register(MIDDLE_EFFECTS, "MN02:Fear:3", function* (state, ap, li) {
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

register(MIDDLE_EFFECTS, "MN02:Fear:5", function* (state, ap) {
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

// Ice 1 bottom — "After your opponent plays a card in this line: Your
// opponent discards 1 card." Line-scoped at the broadcast site. Fires
// only while uncovered.
register(AFTER_OPP_PLAY_IN_LINE_EFFECTS, "MN02:Ice:1", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
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

// Ice 3 top — "End: If this card is covered, you may shift it." Top-tier
// text is active while face-up regardless of cover; the condition itself
// requires the card to be covered.
register(END_EFFECTS, "MN02:Ice:3", function* (state, ap, _li, card) {
  if (!card.faceUp) return;
  // Re-locate Ice 3 in case it was shifted since play.
  let curLine = -1;
  let curPos = -1;
  for (let ln = 0; ln < 3; ln++) {
    const s = lineStack(state.lines[ln], card.owner);
    const p = s.indexOf(card);
    if (p >= 0) { curLine = ln; curPos = p; break; }
  }
  if (curLine < 0) return;
  const s = lineStack(state.lines[curLine], card.owner);
  if (curPos === s.length - 1) return; // not covered
  const dest = [0, 1, 2].filter((l) => l !== curLine);
  if (dest.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Shift covered Ice 3 to another line",
    options: dest.map(String).concat(["skip"]),
    targets: [...dest, -1], optional: true, decider: ap,
  };
  if (idx === -1 || idx >= dest.length) return;
  shiftCard(state, curLine, card.owner, curPos, dest[idx]);
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
  // Codex FAQ p.7: discarding from top of deck counts as a discard.
  // Flag opp as discarder so after-discard triggers (Plague 1, War 3,
  // Peace 4, etc.) fire.
  const { flagAfterDiscard } = require("./helpers");
  flagAfterDiscard(state, opp);
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

// Mirror 1 bottom — "End: You may resolve the middle command of 1 of
// opp's cards as if it were on this card." Fires only while uncovered.
// Codex p.3 default targeting: only uncovered cards. Codex p.9:
// Mirror 1's bottom is blocked by Fear 0 — we honour this by filtering
// targets through `middleSuppressed` (ap's own Fear 0 / Apathy 2 in the
// target's line nullifies the copied middle).
register(END_EFFECTS, "MN02:Mirror:1", function* (state, ap, li, card) {
  const ownStack = lineStack(state.lines[li], ap);
  if (ownStack.length === 0 || ownStack[ownStack.length - 1] !== card) return;
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    const s = lineStack(state.lines[ln], opp);
    if (s.length === 0) continue;
    const pos = s.length - 1;
    const c = s[pos];  // uncovered = top of stack
    if (!c.faceUp) continue;
    const d = CARD_DEFS[c.defId];
    if (!d.middleText) continue;
    if (middleSuppressed(state, ln, c)) continue;
    targets.push({ line: ln, player: opp, pos, card: c });
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
  // Swap all of your cards in one stack with another of your stacks.
  // Codex p.13: "A stack must have at least 1 card in it to swap." +
  // Codex p.4 "end state must differ" — filter pairs to those where at
  // least one stack is non-empty.
  const allPairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
  const pairs = allPairs.filter(([a, b]) => state.lines[a][ap === 0 ? "p0Stack" : "p1Stack"].length > 0 || state.lines[b][ap === 0 ? "p0Stack" : "p1Stack"].length > 0);
  if (pairs.length === 0) return;
  const idx: number = yield {
    prompt: "Swap which two of your stacks?",
    options: pairs.map(([a, b]) => `L${a}<->L${b}`),
    targets: pairs, optional: false, decider: ap,
  };
  if (idx < 0 || idx >= pairs.length) return;
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
  // Flip 1 of your cards. Flip 1 of opp's cards in the same line.
  // Codex p.13: "If Mirror 3 flips itself first, the second flip doesn't
  // happen." Mirror 3 IS a valid first-flip target; short-circuit the
  // second clause if Mirror 3 went face-down.
  const own = enumerateUncovered(state, { owner: "self", activePlayer: ap });
  if (own.length === 0) return;
  yield* chooseFieldTarget("Flip 1 of your cards", own, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !own[i]) return;
  const t = own[i];
  flipCard(state, t.line, t.player, t.pos);
  // If Mirror 3 self-flipped (now face-down), the second flip doesn't happen.
  if (!card.faceUp) return;
  const oppIn = enumerateUncovered(state, { owner: "opponent", lineFilter: t.line, activePlayer: ap });
  if (oppIn.length === 0) return;
  yield* chooseFieldTarget("Flip 1 opp card in same line", oppIn, state, ap);
  const j = state.scratch["_last_target_idx"] as number | undefined;
  if (j != null && oppIn[j]) flipCard(state, oppIn[j].line, oppIn[j].player, oppIn[j].pos);
});

// Mirror 4 bottom — "After your opponent draws cards: Draw 1 card."
// Fires only while uncovered.
register(AFTER_OPP_DRAW_EFFECTS, "MN02:Mirror:4", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

// ----- MN02: Peace ---------------------------------------------------------

// Peace 1 middle — "Both players discard their hand." Codex p.9: "The
// owner decides which player discards their hand first." Order matters
// because after-discard triggers fire between the two batches.
register(MIDDLE_EFFECTS, "MN02:Peace:1", function* (state, ap) {
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const hSelf = state.players[ap].hand.length;
  const hOpp = state.players[opp].hand.length;
  if (hSelf === 0 && hOpp === 0) return;
  let first: PlayerIndex;
  if (hSelf > 0 && hOpp > 0) {
    const idx: number = yield {
      prompt: "Which player discards their hand first?",
      options: [`P${ap + 1} (you) first`, `P${opp + 1} (opp) first`],
      targets: [ap, opp], optional: false, decider: ap,
    };
    first = (idx === 0 ? ap : opp);
  } else {
    first = hSelf > 0 ? ap : opp;
  }
  const order: PlayerIndex[] = [first, (1 - first) as PlayerIndex];
  for (const pl of order) {
    while (state.players[pl].hand.length > 0) discardToTrash(state, pl, 0);
  }
});

// Peace 1 bottom — "End: If your hand is empty, draw 1 card." Fires
// only while uncovered.
register(END_EFFECTS, "MN02:Peace:1", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
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

// Peace 4 bottom — "After you discard cards during your opponent's
// turn: Draw 1 card." Drain only fires when current_player != discarder,
// so `ap` here is the discarder (Peace 4's owner) on opp's turn. Fires
// only while uncovered.
register(AFTER_SELF_DISCARD_ON_OPP_TURN_EFFECTS, "MN02:Peace:4", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
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
  // Codex: if Smoke 1 itself leaves play during the flip cascade, the
  // optional shift stops.
  if (!sourceStillActive(state, card)) return;
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

// War 0 bottom — "After your opponent draws cards: You may delete 1
// card." Fires only while uncovered.
register(AFTER_OPP_DRAW_EFFECTS, "MN02:War:0", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Delete 1 card (War 0)", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  deleteCardFromField(state, targets[i].line, targets[i].player, targets[i].pos);
});

// War 1 bottom — "After your opponent refreshes: Discard any number of
// cards. Refresh." Fires only while uncovered.
register(AFTER_OPP_REFRESH_EFFECTS, "MN02:War:1", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  yield* discardOptionalLoop(state, ap, state.players[ap].hand.length);
  yield* refreshPlayer(state, ap);
});

register(MIDDLE_EFFECTS, "MN02:War:2", function* (state, ap, li, card) {
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Flip 1 card", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

// War 2 bottom — "After your opponent compiles: Your opponent discards
// their hand." Fires only while uncovered.
register(AFTER_OPP_COMPILE_EFFECTS, "MN02:War:2", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  while (state.players[opp].hand.length > 0) discardToTrash(state, opp, 0);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "MN02:War:3", function* (state, ap) {
  drawCards(state, ap, 1);
  if (false) yield {} as Choice;
});

// War 3 bottom — "After your opponent discards cards: You may play 1
// card face-down." Fires only while uncovered. `ap` here is the card
// owner (broadcast targets the opp-of-discarder side).
register(AFTER_OPP_DISCARD_EFFECTS, "MN02:War:3", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
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
  yield* refreshPlayer(state, ap);
});

// Assimilation 1 bottom — "After a player refreshes: Draw the top card
// of your opponent's deck. Discard 1 card into their trash." Fires only
// while uncovered. Triggered by *either* player's refresh.
register(AFTER_ANY_REFRESH_EFFECTS, "AX02:Assimilation:1", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
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
  // Atypical discard destination — still a discard from ap's hand, so
  // flag for after-discard triggers.
  const { flagAfterDiscard } = require("./helpers");
  flagAfterDiscard(state, ap);
});

register(MIDDLE_EFFECTS, "AX02:Assimilation:4", function* (state, ap) {
  bothPlayersDrawTop(state, ap);
  if (false) yield {} as Choice;
});

// Assimilation 6 bottom — "End: Play the top card of your deck face-down
// on your opponent's side." Bottom-tier End: fires only while uncovered.
register(END_EFFECTS, "AX02:Assimilation:6", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  const ps = state.players[ap];
  if (ps.deck.length === 0) return;
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const idx: number = yield {
    prompt: "Play your deck-top face-down on which opp line?",
    options: ["opp L1", "opp L2", "opp L3"],
    targets: [0, 1, 2],
    optional: false, decider: ap,
  };
  if (idx < 0 || idx > 2) return;
  playTopDeckFaceDown(state, opp, idx as 0 | 1 | 2);
});

// ----- AX02: Diversity -----------------------------------------------------

// Diversity 0 middle: "If 6 different protocols on face-up cards in
// field, flip the Diversity protocol to its compiled side."
register(MIDDLE_EFFECTS, "AX02:Diversity:0", function* (state, ap) {
  const protos = new Set<string>();
  for (let ln = 0; ln < 3; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      for (const c of lineStack(state.lines[ln], pl)) {
        if (c.faceUp) protos.add(CARD_DEFS[c.defId].protocol);
      }
    }
  }
  if (protos.size >= 6) {
    const ps = state.players[ap];
    const slot = ps.protocols.indexOf("Diversity");
    if (slot >= 0) {
      ps.compiled[slot] = true;
      logInfo(state, `P${ap + 1} compiled Diversity via diversity-0 condition (6 protocols on field).`);
    }
  }
  if (false) yield {} as Choice;
});

// Diversity 0 bottom — "End: You may play one non-Diversity card in
// this line." Stubbed as a no-op — would need a new ACTION variant to
// model the off-turn play affordance. Regular plays cover the common
// case. Bottom-tier End: fires only while uncovered.
register(END_EFFECTS, "AX02:Diversity:0", function* (state, ap, li, card) {
  const stack = lineStack(state.lines[li], ap);
  if (stack.length === 0 || stack[stack.length - 1] !== card) return;
  if (false) yield {} as Choice;
});

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
  // "If there is another Unity card in the field, you may flip 1
  // face-up card." Face-up filter mirrors the card text.
  if (countUnityInField(state) <= 1) return;
  const targets = enumerateUncovered(state, { exclude: card, face: "up", activePlayer: ap });
  if (targets.length === 0) return;
  yield* chooseFieldTarget("(optional) Flip 1 face-up card", targets, state, ap, true);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

// ----- AX02: Assimilation 0 / 2, Diversity 1 / 4, Unity 1 / 2 / 4 ----------
// (Diversity 3 top + Unity 1 top + Unity 1 bottom + Ice 4 bottom are
// passive — see computeLineValue, isShiftTargetableWhileCovered,
// unityCardMayBePlayedFaceupInLine, and flipCard respectively.)

function distinctProtocolsInLine(state: GameState, lineIdx: number): number {
  const protos = new Set<string>();
  for (const pl of [0, 1] as PlayerIndex[]) {
    for (const c of lineStack(state.lines[lineIdx], pl)) {
      if (c.faceUp) protos.add(CARD_DEFS[c.defId].protocol);
    }
  }
  return protos.size;
}

function distinctProtocolsInField(state: GameState): number {
  const protos = new Set<string>();
  for (let ln = 0; ln < 3; ln++) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      for (const c of lineStack(state.lines[ln], pl)) {
        if (c.faceUp) protos.add(CARD_DEFS[c.defId].protocol);
      }
    }
  }
  return protos.size;
}

register(MIDDLE_EFFECTS, "AX02:Assimilation:0", function* (state, ap) {
  // M: Put one of your opponent's covered or uncovered field cards
  // directly into your hand. "covered or uncovered" overrides the
  // default uncovered-only targeting (Codex p.3).
  const opp: PlayerIndex = ap === 0 ? 1 : 0;
  const targets: FieldTarget[] = [];
  for (let ln = 0; ln < 3; ln++) {
    const s = lineStack(state.lines[ln], opp);
    s.forEach((c, pos) => {
      if (c.isCommitted) return;
      targets.push({ line: ln, player: opp, pos, card: c });
    });
  }
  if (targets.length === 0) return;
  yield* chooseFieldTarget("Steal 1 opp field card into your hand", targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i == null || !targets[i]) return;
  const t = targets[i];
  const srcStack = lineStack(state.lines[t.line], t.player);
  const wasTop = t.pos === srcStack.length - 1;
  srcStack.splice(t.pos, 1);
  t.card.owner = ap;
  t.card.faceUp = false;
  state.players[ap].hand.push(t.card);
  if (wasTop && srcStack.length && srcStack[srcStack.length - 1].faceUp) {
    state.triggers.push({ kind: "uncover", line: t.line, player: t.player, card: srcStack[srcStack.length - 1] });
  }
  checkDiversity6SelfDestruct(state);
});

register(END_EFFECTS, "AX02:Assimilation:2", function* (state, ap, li, card) {
  // Bottom-tier End: — "Play the top card of your opponent's deck
  // face-down in YOUR stack in this line." Fires only while uncovered.
  const guardStack = lineStack(state.lines[li], ap);
  if (guardStack.length === 0 || guardStack[guardStack.length - 1] !== card) return;
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
  const c = psOpp.deck.pop()!;
  c.owner = ap;
  c.faceUp = false;
  lineStack(state.lines[li], ap).push(c);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "AX02:Diversity:1", function* (state, ap, li, card) {
  // M: Shift 1 uncovered card, then draw N (= distinct face-up
  // protocols on cards in this line).
  const targets = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  if (targets.length > 0) {
    yield* chooseFieldTarget("Shift 1 uncovered card", targets, state, ap);
    const i = state.scratch["_last_target_idx"] as number | undefined;
    if (i != null && targets[i]) {
      const t = targets[i];
      const dests = [0, 1, 2].filter((l) => l !== t.line);
      const didx: number = yield {
        prompt: "To which line?",
        options: dests.map(String), targets: dests,
        optional: false, decider: ap,
      };
      if (didx >= 0 && didx < dests.length) {
        shiftCard(state, t.line, t.player, t.pos, dests[didx]);
      }
    }
  }
  const n = distinctProtocolsInLine(state, li);
  if (n > 0) drawCards(state, ap, n);
});

register(MIDDLE_EFFECTS, "AX02:Diversity:4", function* (state, ap, li, card) {
  // M: Flip 1 uncovered card with a value less than the number of
  // different protocols on face-up cards in the field.
  const threshold = distinctProtocolsInField(state);
  if (threshold <= 0) return;
  const all = enumerateUncovered(state, { exclude: card, activePlayer: ap });
  const targets = all.filter((t) => {
    const v = t.card.faceUp ? CARD_DEFS[t.card.defId].value : FACE_DOWN_BASE_VALUE;
    return v < threshold;
  });
  if (targets.length === 0) return;
  yield* chooseFieldTarget(`Flip 1 uncovered card with value < ${threshold}`, targets, state, ap);
  const i = state.scratch["_last_target_idx"] as number | undefined;
  if (i != null && targets[i]) flipCard(state, targets[i].line, targets[i].player, targets[i].pos);
});

// Unity 1 top — "Start: If this card is covered, you may shift this
// card." Top-tier text is active while face-up regardless of cover.
register(START_EFFECTS, "AX02:Unity:1", function* (state, ap, _li, card) {
  if (!card.faceUp) return;
  let curLine = -1;
  let curPos = -1;
  for (let ln = 0; ln < 3; ln++) {
    const s = lineStack(state.lines[ln], card.owner);
    const p = s.indexOf(card);
    if (p >= 0) { curLine = ln; curPos = p; break; }
  }
  if (curLine < 0) return;
  const s = lineStack(state.lines[curLine], card.owner);
  if (curPos === s.length - 1) return; // not covered
  const dest = [0, 1, 2].filter((l) => l !== curLine);
  if (dest.length === 0) return;
  const idx: number = yield {
    prompt: "(optional) Shift covered Unity 1 to another line",
    options: dest.map(String).concat(["skip"]),
    targets: [...dest, -1], optional: true, decider: ap,
  };
  if (idx === -1 || idx >= dest.length) return;
  shiftCard(state, curLine, card.owner, curPos, dest[idx]);
});

register(MIDDLE_EFFECTS, "AX02:Unity:1", function* (state, ap) {
  // M: If there are 5 or more Unity cards in the field, flip the Unity
  // protocol to the compiled side and delete all cards in that line.
  if (countUnityInField(state) < 5) return;
  const ps = state.players[ap];
  const unityLine = ps.protocols.findIndex((p) => p === "Unity");
  if (unityLine < 0) return;
  ps.compiled[unityLine] = true;
  for (const pl of [0, 1] as PlayerIndex[]) {
    const stack = lineStack(state.lines[unityLine], pl);
    for (const c of stack) {
      c.faceUp = true;
      state.players[c.owner].trash.push(c);
    }
    if (pl === 0) state.lines[unityLine].p0Stack = [];
    else state.lines[unityLine].p1Stack = [];
  }
  checkDiversity6SelfDestruct(state);
  if (false) yield {} as Choice;
});

register(MIDDLE_EFFECTS, "AX02:Unity:2", function* (state, ap) {
  // M: Draw cards equal to the number of Unity cards in the field.
  const n = countUnityInField(state);
  if (n > 0) drawCards(state, ap, n);
  if (false) yield {} as Choice;
});

// Unity 4 top — "End: If your hand is empty, reveal your deck, draw
// all Unity cards from it, then shuffle your deck." Top-tier End:
// fires while face-up regardless of cover.
register(END_EFFECTS, "AX02:Unity:4", function* (state, ap, _li, card) {
  if (!card.faceUp) return;
  if (state.players[ap].hand.length > 0) return;
  const ps = state.players[ap];
  const unityCards = ps.deck.filter((c) => CARD_DEFS[c.defId].protocol === "Unity");
  if (unityCards.length === 0) return;
  logInfo(state, `P${ap + 1} revealed deck: drew ${unityCards.length} Unity card(s)`);
  for (const c of unityCards) {
    const idx = ps.deck.indexOf(c);
    if (idx >= 0) ps.deck.splice(idx, 1);
    ps.hand.push(c);
  }
  const rng = { state: state.rngState };
  const { rngShuffle } = require("./rng");
  const { flagAfterShuffle } = require("./helpers");
  rngShuffle(rng, ps.deck);
  state.rngState = rng.state;
  flagAfterShuffle(state, ap);
  if (false) yield {} as Choice;
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
export function getAfterOppDrawEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_OPP_DRAW_EFFECTS[k] ?? null);
}
export function getAfterOppRefreshEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_OPP_REFRESH_EFFECTS[k] ?? null);
}
export function getAfterOppCompileEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_OPP_COMPILE_EFFECTS[k] ?? null);
}
export function getAfterAnyRefreshEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_ANY_REFRESH_EFFECTS[k] ?? null);
}
export function getAfterOppPlayInLineEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_OPP_PLAY_IN_LINE_EFFECTS[k] ?? null);
}
export function getAfterSelfDiscardOnOppTurnEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (AFTER_SELF_DISCARD_ON_OPP_TURN_EFFECTS[k] ?? null);
}
export function getFlipTriggerEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (FLIP_TRIGGER_EFFECTS[k] ?? null);
}
export function getWhenDeletedByCompileEffect(defId: number): EffectFn | null {
  const k = keyForDefId(defId);
  return k == null ? null : (WHEN_DELETED_BY_COMPILE_EFFECTS[k] ?? null);
}
