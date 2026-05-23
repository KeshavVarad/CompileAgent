/**
 * Atomic helpers + value computation. Ported 1:1 from
 * src/compile_engine/effects.py — same semantics, same trigger queue.
 */

import { CARD_DEFS, safeCardDef } from "./cards";
import { rngShuffle } from "./rng";
import type { Action, CardInst, Choice, GameState, LineState, PlayerIndex } from "./types";
import { FACE_DOWN_BASE_VALUE, NUM_LINES } from "./types";

export function lineStack(line: LineState, player: PlayerIndex): CardInst[] {
  return player === 0 ? line.p0Stack : line.p1Stack;
}

export function setLineStack(line: LineState, player: PlayerIndex, stack: CardInst[]): void {
  if (player === 0) line.p0Stack = stack;
  else line.p1Stack = stack;
}

// ---------------------------------------------------------------------------
// Value computation
// ---------------------------------------------------------------------------

function stackFaceDownBaseValue(state: GameState, lineIdx: number, player: PlayerIndex): number {
  const stack = lineStack(state.lines[lineIdx], player);
  let base = FACE_DOWN_BASE_VALUE;
  for (const c of stack) {
    if (!c.faceUp) continue;
    const d = CARD_DEFS[c.defId];
    if (d.protocol === "Darkness" && d.value === 2) {
      base = Math.max(base, 4);
    }
  }
  return base;
}

function cardsInLineBoth(state: GameState, lineIdx: number): CardInst[] {
  return [...state.lines[lineIdx].p0Stack, ...state.lines[lineIdx].p1Stack];
}

export function computeLineValue(state: GameState, lineIdx: number, player: PlayerIndex): number {
  const stack = lineStack(state.lines[lineIdx], player);
  const opp: PlayerIndex = player === 0 ? 1 : 0;
  const fdValue = stackFaceDownBaseValue(state, lineIdx, player);
  let total = 0;
  for (const c of stack) {
    if (c.faceUp) total += CARD_DEFS[c.defId].value;
    else total += fdValue;
  }

  const fdCountInLine = cardsInLineBoth(state, lineIdx).filter((c) => !c.faceUp).length;
  const oppStack = lineStack(state.lines[lineIdx], opp);
  for (const c of stack) {
    if (!c.faceUp) continue;
    const d = CARD_DEFS[c.defId];
    // Apathy 0 top: +1 per face-down in line (both sides).
    if (d.protocol === "Apathy" && d.value === 0) total += fdCountInLine;
    // Smoke 2 top (MN02): same as Apathy 0.
    if (d.protocol === "Smoke" && d.value === 2) total += fdCountInLine;
    // Mirror 0 top (MN02): +1 per opp card in line.
    if (d.protocol === "Mirror" && d.value === 0) total += oppStack.length;
    // Clarity 0 top (MN02): +1 per card in own hand.
    if (d.protocol === "Clarity" && d.value === 0) total += state.players[player].hand.length;
  }

  // Metal 0 top (opp side): reduces our value by 2.
  for (const c of oppStack) {
    if (!c.faceUp) continue;
    const d = CARD_DEFS[c.defId];
    if (d.protocol === "Metal" && d.value === 0) total -= 2;
  }

  // Diversity 3 top (AX02, own side): +2 if any non-Diversity face-up
  // card is in THIS stack. +2 per Diversity 3 instance.
  const hasNonDivFaceup = stack.some(
    (c) => c.faceUp && CARD_DEFS[c.defId].protocol !== "Diversity",
  );
  if (hasNonDivFaceup) {
    for (const c of stack) {
      if (!c.faceUp) continue;
      const d = CARD_DEFS[c.defId];
      if (d.protocol === "Diversity" && d.value === 3) total += 2;
    }
  }

  return Math.max(total, 0);
}

// ---------------------------------------------------------------------------
// Atomic mutations
// ---------------------------------------------------------------------------

/**
 * Append an "info" entry to the engine event log. Used to make
 * implicit/silent gameplay events visible to the client — e.g. an
 * effect that tries to make the opponent discard but their hand is
 * empty, or a draw that finds both deck and trash exhausted. The Codex
 * (p.2) says card text resolves even when impossible, so the user
 * never sees these as a discrete action; this log surfaces them.
 */
export function logInfo(state: GameState, text: string): void {
  state.log.push({ kind: "info", turn: state.turn, text, timestamp: Date.now() });
}

/** Pre-render a human-readable label for an action against the game
 *  state that produced it. Used by step() to stamp action log entries
 *  with their label BEFORE the action lands, since the hand index used
 *  by play/discard actions resolves to a different card afterwards. */
export function formatActionLabel(state: GameState, action: Action): string {
  const seat = state.currentPlayer;
  const seatLabel = seat === 0
    ? (state.config?.player0Label ?? "P1")
    : (state.config?.player1Label ?? "P2");
  const hand = state.players[seat]?.hand ?? [];
  const c = action.handIndex != null ? hand[action.handIndex] : undefined;
  const def = c ? safeCardDef(c.defId) : null;
  const cardLabel = def && def.defId !== -1
    ? `${def.protocol} ${def.value}`
    : "card";
  switch (action.type) {
    case "DRAFT_PROTOCOL":
      return `${seatLabel} drafted ${action.protocol}`;
    case "PLAY_FACE_UP":
      return `${seatLabel} played ${cardLabel} face-up in L${(action.lineIndex ?? 0) + 1}`;
    case "PLAY_FACE_DOWN":
      return `${seatLabel} played a card face-down in L${(action.lineIndex ?? 0) + 1}`;
    case "REFRESH":
      return `${seatLabel} refreshed`;
    case "COMPILE_LINE":
      return `${seatLabel} compiled L${(action.lineIndex ?? 0) + 1}`;
    case "DISCARD_CARD":
      return `${seatLabel} discarded ${cardLabel}`;
    case "SHIFT_OWN_CARD":
      return `${seatLabel} shifted card → L${(action.choiceIndex ?? 0) + 1}`;
    case "CHOOSE_TARGET":
      return `${seatLabel} picked option ${(action.choiceIndex ?? 0) + 1}`;
    case "SKIP_OPTIONAL":
      return `${seatLabel} skipped`;
    case "NOOP":
      return `${seatLabel} (no action)`;
    default:
      return action.type;
  }
}

// Per-atomic deferred event flags. Set by mutation helpers below; drained
// by the engine after the current effect stack resolves (mirrors the
// Python _drain_pending_after_events).
export function flagAfterDiscard(state: GameState, player: PlayerIndex): void {
  (state.scratch as Record<string, unknown>)[`_pending_after_discard_by_p${player}`] = true;
}
function flagAfterDraw(state: GameState, player: PlayerIndex): void {
  (state.scratch as Record<string, unknown>)[`_pending_after_draw_by_p${player}`] = true;
}
function flagAfterDelete(state: GameState, deleter: PlayerIndex): void {
  (state.scratch as Record<string, unknown>)[`_pending_after_delete_by_p${deleter}`] = true;
}
export function flagAfterShuffle(state: GameState, player: PlayerIndex): void {
  (state.scratch as Record<string, unknown>)[`_pending_after_shuffle_by_p${player}`] = true;
}
export function flagAfterRefresh(state: GameState, player: PlayerIndex): void {
  (state.scratch as Record<string, unknown>)[`_pending_after_refresh_by_p${player}`] = true;
}
export function flagAfterCompile(state: GameState, player: PlayerIndex): void {
  (state.scratch as Record<string, unknown>)[`_pending_after_compile_by_p${player}`] = true;
}
export function flagAfterPlayInLine(state: GameState, player: PlayerIndex, line: number): void {
  // List of (player_who_played, line) so multiple plays within a single
  // effect chain each get their own broadcast.
  const lst = ((state.scratch as Record<string, unknown>)["_pending_after_play_list"] ??= []) as [PlayerIndex, number][];
  lst.push([player, line]);
}
function flagFlip(state: GameState, card: CardInst): void {
  const lst = ((state.scratch as Record<string, unknown>)["_pending_flip_cards"] ??= []) as CardInst[];
  lst.push(card);
}

export function drawCards(state: GameState, player: PlayerIndex, n: number): number {
  if (n <= 0) return 0;
  if (playerCannotDraw(state, player)) {
    logInfo(state, `P${player + 1} could not draw ${n} (Ice 6 is suppressing draws while hand has cards).`);
    return 0;
  }
  const ps = state.players[player];
  let drawn = 0;
  let shuffled = false;
  for (let i = 0; i < n; i++) {
    if (ps.deck.length === 0) {
      if (ps.trash.length === 0) break;
      ps.deck = ps.trash;
      ps.trash = [];
      rngShuffle({ state: state.rngState }, ps.deck);
      shuffled = true;
    }
    const c = ps.deck.pop();
    if (c) {
      ps.hand.push(c);
      drawn++;
    }
  }
  if (drawn > 0) flagAfterDraw(state, player);
  if (shuffled) flagAfterShuffle(state, player);
  if (drawn < n) {
    logInfo(state, `P${player + 1} drew ${drawn} of ${n} cards (deck + trash exhausted).`);
  }
  return drawn;
}

export function discardToTrash(state: GameState, player: PlayerIndex, handIndex: number): CardInst {
  const ps = state.players[player];
  const [c] = ps.hand.splice(handIndex, 1);
  c.faceUp = true;
  state.players[c.owner].trash.push(c);
  flagAfterDiscard(state, player);
  return c;
}

export function deleteCardFromField(
  state: GameState,
  lineIdx: number,
  player: PlayerIndex,
  stackPos: number,
): CardInst {
  const stack = lineStack(state.lines[lineIdx], player);
  const wasTop = stackPos === stack.length - 1;
  const [c] = stack.splice(stackPos, 1);
  c.faceUp = true;
  state.players[c.owner].trash.push(c);
  // Removing the top exposes the under-card → fire its middle on uncover.
  if (wasTop && stack.length && stack[stack.length - 1].faceUp) {
    state.triggers.push({ kind: "uncover", line: lineIdx, player, card: stack[stack.length - 1] });
  }
  // "After you delete cards:" — attribute to whoever is the active
  // player at the time of the delete (mirrors Python).
  flagAfterDelete(state, state.currentPlayer);
  checkDiversity6SelfDestruct(state);
  return c;
}

export function flipCard(
  state: GameState,
  lineIdx: number,
  player: PlayerIndex,
  stackPos: number,
): CardInst {
  const stack = lineStack(state.lines[lineIdx], player);
  const c = stack[stackPos];
  const d = safeCardDef(c.defId);
  // Ice 4: "This card cannot be flipped." Persistent immunity while
  // face-up on the field. Mirrors src/compile_engine/effects.py.
  if (d.key === "MN02:Ice:4" && c.faceUp) {
    logInfo(state, `Flip on ${d.protocol} ${d.value} was blocked (immune).`);
    return c;
  }
  // Metal 6 top: "When this card would be covered or flipped: First,
  // delete this card." Top text is active only while face-up. We
  // preempt here (rather than via the post-flip FLIP_TRIGGER broadcast)
  // because (a) the trigger fires BEFORE the flip ("First, ...") and
  // (b) the broadcast filters on c.faceUp post-flip, which would skip
  // face-up→face-down transitions entirely. Codex p.10: the calling
  // flip is "used up" — no alternate target.
  if (d.key === "MN01:Metal:6" && c.faceUp) {
    deleteCardFromField(state, lineIdx, player, stackPos);
    return c;
  }
  const wasUp = c.faceUp;
  c.faceUp = !c.faceUp;
  if (!wasUp && c.faceUp) {
    if (c.defId === -1) {
      // Record-mode placeholder being revealed. Push reveal_placeholder
      // first; fireNextTrigger handles it by yielding a Choice for identity
      // and re-pushing the face_up trigger so effects fire normally after.
      state.triggers.push({ kind: "reveal_placeholder", line: lineIdx, player, card: c });
    } else {
      state.triggers.push({ kind: "face_up", line: lineIdx, player, card: c });
    }
  }
  // `Flip:` emphasis fires on any flip direction.
  flagFlip(state, c);
  checkDiversity6SelfDestruct(state);
  return c;
}

/**
 * Returns true iff `card` is on the field, face-up, and the uncovered top of
 * its stack. The Compile Codex (Hate 2 clarification p.12, Mirror 3 p.13) is
 * explicit: a card's text stops resolving the moment the source card leaves
 * play — whether by being deleted, flipped face-down, or covered. Use this
 * between clauses of a multi-clause middle whose earlier clause could
 * plausibly remove the source (self-target OR downstream cascade).
 */
export function sourceStillActive(state: GameState, card: CardInst): boolean {
  if (!card.faceUp) return false;
  for (const line of state.lines) {
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = lineStack(line, pl);
      if (stack.length > 0 && stack[stack.length - 1] === card) return true;
    }
  }
  return false;
}

export function shiftCard(
  state: GameState,
  srcLine: number,
  srcPlayer: PlayerIndex,
  srcPos: number,
  dstLine: number,
): CardInst {
  const stackSrc = lineStack(state.lines[srcLine], srcPlayer);
  const wasTop = srcPos === stackSrc.length - 1;
  const [c] = stackSrc.splice(srcPos, 1);
  c.isCommitted = true;
  const stackDst = lineStack(state.lines[dstLine], srcPlayer);
  const soonCovered = stackDst.length ? stackDst[stackDst.length - 1] : null;
  stackDst.push(c);
  // Push triggers in LIFO-reverse-chronological order; see Python shift_card.
  if (c.faceUp && !wasTop) {
    state.triggers.push({ kind: "uncover", line: dstLine, player: srcPlayer, card: c });
  }
  state.triggers.push({ kind: "uncommit", card: c });
  if (soonCovered && soonCovered.faceUp) {
    // We rely on hasWhenCoveredEffect being false for most cards; we still
    // push the trigger so the engine can decide.
    state.triggers.push({
      kind: "when_covered", line: dstLine, player: soonCovered.owner, card: soonCovered,
    });
  }
  // Removing top of src exposes the under-card → fire its middle.
  if (wasTop && stackSrc.length && stackSrc[stackSrc.length - 1].faceUp) {
    state.triggers.push({ kind: "uncover", line: srcLine, player: srcPlayer, card: stackSrc[stackSrc.length - 1] });
  }
  checkDiversity6SelfDestruct(state);
  return c;
}

export function returnCardToHand(
  state: GameState,
  lineIdx: number,
  player: PlayerIndex,
  stackPos: number,
): CardInst {
  const stack = lineStack(state.lines[lineIdx], player);
  const wasTop = stackPos === stack.length - 1;
  const [c] = stack.splice(stackPos, 1);
  c.faceUp = false;
  state.players[c.owner].hand.push(c);
  // Removing the top exposes the under-card → fire its middle.
  if (wasTop && stack.length && stack[stack.length - 1].faceUp) {
    state.triggers.push({ kind: "uncover", line: lineIdx, player, card: stack[stack.length - 1] });
  }
  checkDiversity6SelfDestruct(state);
  return c;
}

export function playTopDeckFaceDown(
  state: GameState,
  player: PlayerIndex,
  lineIdx: number,
): CardInst | null {
  const ps = state.players[player];
  if (ps.deck.length === 0) {
    if (ps.trash.length === 0) return null;
    ps.deck = ps.trash;
    ps.trash = [];
    rngShuffle({ state: state.rngState }, ps.deck);
    if (ps.deck.length === 0) return null;
  }
  const c = ps.deck.pop()!;
  c.faceUp = false;
  lineStack(state.lines[lineIdx], player).push(c);
  return c;
}

/** Generator. A "refresh" — replenishes hand to 5 and flags the
 *  after-refresh event. Codex p.10 (Spirit 0 clarification): when you
 *  refresh as instructed it is a "normal refresh action, including
 *  spending the control component, if applicable" — so we yield the
 *  control-rearrange prompt before drawing when `player` holds the
 *  control component. All callers must `yield* refreshPlayer(...)`. */
export function* refreshPlayer(state: GameState, player: PlayerIndex): Generator<Choice, void, number> {
  if (state.controlHolder === player) {
    // `controlRearrangeGen` lives in effects.ts; require() it lazily to
    // avoid a circular import (effects.ts already pulls from helpers.ts).
    const { controlRearrangeGen } = require("./effects");
    yield* controlRearrangeGen(state, player);
  }
  const ps = state.players[player];
  const need = 5 - ps.hand.length;
  if (need > 0) drawCards(state, player, need);
  flagAfterRefresh(state, player);
}

// ---------------------------------------------------------------------------
// Target enumeration
// ---------------------------------------------------------------------------

export type FieldTarget = {
  line: number;
  player: PlayerIndex;
  pos: number;
  card: CardInst;
};

export type EnumOpts = {
  owner?: "any" | "self" | "opponent";   // default any
  face?: "any" | "up" | "down";          // default any
  exclude?: CardInst | null;
  activePlayer: PlayerIndex;
  lineFilter?: number | null;
};

export function enumerateUncovered(state: GameState, opts: EnumOpts): FieldTarget[] {
  const out: FieldTarget[] = [];
  for (let li = 0; li < NUM_LINES; li++) {
    if (opts.lineFilter != null && li !== opts.lineFilter) continue;
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = lineStack(state.lines[li], pl);
      if (stack.length === 0) continue;
      const c = stack[stack.length - 1];
      if (c.isCommitted) continue;
      if (opts.exclude && c === opts.exclude) continue;
      if (opts.owner === "self" && pl !== opts.activePlayer) continue;
      if (opts.owner === "opponent" && pl === opts.activePlayer) continue;
      if (opts.face === "up" && !c.faceUp) continue;
      if (opts.face === "down" && c.faceUp) continue;
      out.push({ line: li, player: pl, pos: stack.length - 1, card: c });
    }
  }
  return out;
}

export function enumerateAll(state: GameState, opts: EnumOpts): FieldTarget[] {
  const out: FieldTarget[] = [];
  for (let li = 0; li < NUM_LINES; li++) {
    if (opts.lineFilter != null && li !== opts.lineFilter) continue;
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = lineStack(state.lines[li], pl);
      stack.forEach((c, pos) => {
        if (c.isCommitted) return;
        if (opts.exclude && c === opts.exclude) return;
        if (opts.owner === "self" && pl !== opts.activePlayer) return;
        if (opts.owner === "opponent" && pl === opts.activePlayer) return;
        if (opts.face === "up" && !c.faceUp) return;
        if (opts.face === "down" && c.faceUp) return;
        out.push({ line: li, player: pl, pos, card: c });
      });
    }
  }
  return out;
}

function isShiftTargetableWhileCovered(card: CardInst): boolean {
  if (!card.faceUp) return false;
  const d = CARD_DEFS[card.defId];
  return (
    (d.protocol === "Speed" && d.value === 2)
    || (d.protocol === "Spirit" && d.value === 3)
    || (d.protocol === "Unity" && d.value === 1)
  );
}

export function enumerateShiftTargets(state: GameState, opts: EnumOpts): FieldTarget[] {
  const targets = enumerateUncovered(state, opts);
  if (opts.face === "up" || opts.face === "any" || opts.face === undefined) {
    for (let li = 0; li < NUM_LINES; li++) {
      if (opts.lineFilter != null && li !== opts.lineFilter) continue;
      for (const pl of [0, 1] as PlayerIndex[]) {
        if (opts.owner === "self" && pl !== opts.activePlayer) continue;
        if (opts.owner === "opponent" && pl === opts.activePlayer) continue;
        const stack = lineStack(state.lines[li], pl);
        for (let pos = 0; pos < stack.length - 1; pos++) {
          const c = stack[pos];
          if (c.isCommitted) continue;
          if (opts.exclude && c === opts.exclude) continue;
          if (isShiftTargetableWhileCovered(c)) {
            targets.push({ line: li, player: pl, pos, card: c });
          }
        }
      }
    }
  }
  return targets;
}

/** Render a field target as an option label for a Choice prompt.
 *  Face-down cards are private information per Codex p.5: the
 *  perspective player only knows their own face-downs. We hide the
 *  identity of opp's face-down cards from the choice label (the
 *  underlying target object still carries the real card so the engine
 *  resolves correctly — only the displayed string is redacted).
 *  Record mode is exempt because the recorder is transcribing both
 *  sides and already knows what was played. */
export function describeCard(
  state: GameState,
  target: FieldTarget,
  viewer: PlayerIndex | null = null,
): string {
  const recordMode = state.config.mode === "record";
  const ownedByViewer = viewer != null && target.player === viewer;
  const knowsIdentity = target.card.faceUp || ownedByViewer || recordMode;
  const sideLabel = viewer == null
    ? `P${target.player + 1}`
    : ownedByViewer ? "your" : "opp";
  const lane = `L${target.line + 1}`;
  if (!knowsIdentity) {
    return `${lane} ${sideLabel}: face-down (2)`;
  }
  const d = CARD_DEFS[target.card.defId];
  const facing = target.card.faceUp ? "face-up" : "face-down";
  return `${lane} ${sideLabel}: ${d.protocol} ${d.value} (${facing})`;
}

export function describeHandCard(state: GameState, player: PlayerIndex, idx: number): string {
  const c = state.players[player].hand[idx];
  const d = CARD_DEFS[c.defId];
  return `hand[${idx}]: ${d.protocol} ${d.value}`;
}

// ---------------------------------------------------------------------------
// Persistent rule queries (used by game.ts for legal-action filtering)
// ---------------------------------------------------------------------------

export function middleSuppressed(state: GameState, lineIdx: number, card: CardInst): boolean {
  // Apathy 2 — line-local suppression.
  for (const c of cardsInLineBoth(state, lineIdx)) {
    if (c === card || !c.faceUp) continue;
    const d = CARD_DEFS[c.defId];
    if (d.protocol === "Apathy" && d.value === 2) return true;
  }
  // Fear 0 — global suppression of opponent middles during active turn.
  const ap = state.currentPlayer;
  if (card.owner !== ap) {
    for (let li = 0; li < NUM_LINES; li++) {
      for (const cc of lineStack(state.lines[li], ap)) {
        if (!cc.faceUp) continue;
        const d = CARD_DEFS[cc.defId];
        if (d.protocol === "Fear" && d.value === 0) return true;
      }
    }
  }
  return false;
}

export function checkDiversity6SelfDestruct(state: GameState): void {
  // Continuous predicate: while < 3 distinct protocols are on the field, any
  // face-up Diversity 6 self-deletes. Mirrors the Python helper.
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const protos = new Set<string>();
    for (let li = 0; li < NUM_LINES; li++) {
      for (const pl of [0, 1] as PlayerIndex[]) {
        for (const c of lineStack(state.lines[li], pl)) {
          protos.add(CARD_DEFS[c.defId].protocol);
        }
      }
    }
    if (protos.size >= 3) return;
    let found = false;
    for (let li = 0; li < NUM_LINES && !found; li++) {
      for (const pl of [0, 1] as PlayerIndex[]) {
        if (found) break;
        const s = lineStack(state.lines[li], pl);
        for (let pos = 0; pos < s.length; pos++) {
          const c = s[pos];
          if (!c.faceUp) continue;
          const d = CARD_DEFS[c.defId];
          if (d.protocol === "Diversity" && d.value === 6) {
            s.splice(pos, 1);
            c.faceUp = true;
            state.players[c.owner].trash.push(c);
            found = true;
            break;
          }
        }
      }
    }
    if (!found) return;
  }
}

export function playerCannotDraw(state: GameState, player: PlayerIndex): boolean {
  // Ice 6: "If you have any cards in your hand, you cannot draw cards."
  if (state.players[player].hand.length === 0) return false;
  for (let li = 0; li < NUM_LINES; li++) {
    for (const c of lineStack(state.lines[li], player)) {
      if (!c.faceUp) continue;
      const d = CARD_DEFS[c.defId];
      if (d.protocol === "Ice" && d.value === 6) return true;
    }
  }
  return false;
}

export function playerCanCompile(state: GameState, player: PlayerIndex): boolean {
  return !state.players[player].cannotCompileNextTurn;
}

export function oppPlayFacedownBlockedInLine(state: GameState, lineIdx: number, player: PlayerIndex): boolean {
  const opp: PlayerIndex = player === 0 ? 1 : 0;
  const stack = lineStack(state.lines[lineIdx], opp);
  for (const c of stack) {
    if (!c.faceUp) continue;
    const d = CARD_DEFS[c.defId];
    if (d.protocol === "Metal" && d.value === 2) return true;
  }
  return false;
}

export function oppMustPlayFacedown(state: GameState, player: PlayerIndex): boolean {
  const opp: PlayerIndex = player === 0 ? 1 : 0;
  for (let li = 0; li < NUM_LINES; li++) {
    for (const c of lineStack(state.lines[li], opp)) {
      if (!c.faceUp) continue;
      const d = CARD_DEFS[c.defId];
      if (d.protocol === "Psychic" && d.value === 1) return true;
    }
  }
  return false;
}

export function oppPlayBlockedInLine(state: GameState, lineIdx: number, player: PlayerIndex): boolean {
  const opp: PlayerIndex = player === 0 ? 1 : 0;
  const stack = lineStack(state.lines[lineIdx], opp);
  if (stack.length === 0) return false;
  const top = stack[stack.length - 1];
  if (!top.faceUp) return false;
  const d = CARD_DEFS[top.defId];
  return d.protocol === "Plague" && d.value === 0;
}

export function playerMayPlayAnyLineFaceup(state: GameState, player: PlayerIndex): boolean {
  for (let li = 0; li < NUM_LINES; li++) {
    for (const c of lineStack(state.lines[li], player)) {
      if (!c.faceUp) continue;
      const d = CARD_DEFS[c.defId];
      if (d.protocol === "Spirit" && d.value === 1) return true;
    }
  }
  return false;
}

/** Unity 1 bottom (AX02): "Unity cards may be played face-up in this
 *  line." Active while Unity 1 is face-up + uncovered on `player`'s
 *  side of `lineIdx`. Allows face-up play of Unity-protocol cards
 *  without the usual protocol-match restriction. */
export function unityCardMayBePlayedFaceupInLine(
  state: GameState, player: PlayerIndex, lineIdx: number, playedProtocol: string,
): boolean {
  if (playedProtocol !== "Unity") return false;
  const stack = lineStack(state.lines[lineIdx], player);
  if (stack.length === 0) return false;
  const top = stack[stack.length - 1];
  if (!top.faceUp) return false;
  const d = CARD_DEFS[top.defId];
  return d.protocol === "Unity" && d.value === 1;
}

export function playerSkipsCheckCache(state: GameState, player: PlayerIndex): boolean {
  for (let li = 0; li < NUM_LINES; li++) {
    const stack = lineStack(state.lines[li], player);
    if (stack.length === 0) continue;
    const top = stack[stack.length - 1];
    if (!top.faceUp) continue;
    const d = CARD_DEFS[top.defId];
    if (d.protocol === "Spirit" && d.value === 0) return true;
  }
  return false;
}
