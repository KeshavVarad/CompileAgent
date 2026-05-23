/**
 * NN encoder — port of src/compile_engine/nn/encoder.py.
 *
 * Produces the exact same tensor shapes and numeric content the trained
 * model expects, so the in-browser TS engine can drive an ONNX inference
 * in the webapp. Any divergence here from the Python encoder will silently
 * make the bot pick wrong actions; treat as a strict mirror.
 *
 * One-way translation only — the model is read-only at inference time.
 */

import {
  AUX2_PROTOCOLS,
  BASE_PROTOCOLS,
  CARD_DEFS,
  EXPANSION_PROTOCOLS,
  MAIN2_PROTOCOLS,
} from "./cards";
import { computeLineValue, lineStack, playerCanCompile } from "./helpers";
import type { Action, CardInst, GameState, Phase, PlayerIndex } from "./types";
import { FACE_DOWN_BASE_VALUE, NUM_LINES } from "./types";

// Vocabularies / shapes — keep in sync with Python encoder constants.
export const NUM_CARDS = 180;
export const NUM_PROTOCOLS = 30;
export const MAX_STACK = 10;
export const MAX_HAND = 12;
export const MAX_ACTIONS = 32;

const PAD_TOKEN = 0;
const HIDDEN_TOKEN = 1;
const CARD_TOKEN_OFFSET = 2;

export const CARD_VOCAB_SIZE = NUM_CARDS + CARD_TOKEN_OFFSET; // 182
export const PROTO_VOCAB_SIZE = NUM_PROTOCOLS + 1; // 31

// PROTO_LIST is BASE + EXPANSION + MAIN2 + AUX2 in declaration order, then
// id = idx+1 (id=0 reserved for PAD). Mirrors encoder._PROTO_TO_ID.
const PROTO_LIST: readonly string[] = [
  ...BASE_PROTOCOLS,
  ...EXPANSION_PROTOCOLS,
  ...MAIN2_PROTOCOLS,
  ...AUX2_PROTOCOLS,
];
const PROTO_TO_ID = new Map<string, number>(
  PROTO_LIST.map((p, i) => [p, i + 1]),
);

// Phase one-hot: matches Python's IntEnum value -1. TS lacks
// RESOLVING_EFFECT as a distinct Phase, so we synthesise it externally
// (encodeState looks at game.pendingChoice and overrides the phase index).
const PHASE_INDEX: Record<Phase, number> = {
  DRAFT: 0,
  START: 1,
  CHECK_CONTROL: 2,
  CHECK_COMPILE: 3,
  ACTION: 4,
  CHECK_CACHE: 5,
  END: 6,
  // 7 = RESOLVING_EFFECT (Python only; injected externally)
  GAME_OVER: 8,
};
const NUM_PHASES = 9;

// Action-type one-hot order: matches Python ActionType enum declaration.
const ACTION_TYPE_INDEX: Record<string, number> = {
  DRAFT_PROTOCOL: 0,
  PLAY_FACE_UP: 1,
  PLAY_FACE_DOWN: 2,
  REFRESH: 3,
  COMPILE_LINE: 4,
  DISCARD_CARD: 5,
  SHIFT_OWN_CARD: 6,
  CHOOSE_TARGET: 7,
  SKIP_OPTIONAL: 8,
  NOOP: 9,
};
const NUM_ACTION_TYPES = 10;

export const SCALARS_DIM = 8;
// A3: closed-form lookahead deltas per action.
export const ACTION_LOOKAHEAD_DIM = 7;
// Layout (mirror encoder.encode_actions in Python):
//   type_one_hot(10) + hand_idx(1) + src_line(4) + dst_line(4) +
//   choice_idx(1) + target_meta(8) + stated_value(1) +
//   A5 soon_covered_present(1) + A5 soon_covered_face_up(1) +
//   A3 lookahead deltas (7)
export const RAW_ACTION_DIM =
  NUM_ACTION_TYPES + 1 + 4 + 4 + 1 + 8 + 1 + 2 + ACTION_LOOKAHEAD_DIM; // 38

// Prompt categories — used in pending_category one-hot. Mirrors Python's
// PROMPT_CAT_*. Order matters (model expects this exact indexing).
const PROMPT_CAT_NONE = 0;
const PROMPT_CAT_DISCARD_OWN = 1;
const PROMPT_CAT_DISCARD_OPP = 2;
const PROMPT_CAT_FLIP = 3;
const PROMPT_CAT_DELETE = 4;
const PROMPT_CAT_SHIFT = 5;
const PROMPT_CAT_RETURN_OR_STEAL = 6;
const PROMPT_CAT_PICK_LINE = 7;
const PROMPT_CAT_PICK_PROTOCOL = 8;
const PROMPT_CAT_PICK_NUMBER = 9;
const PROMPT_CAT_PLAY_SUB_CARD = 10;
const PROMPT_CAT_OPTIONAL_ACCEPT = 11;
const PROMPT_CAT_REVEAL = 12;
const PROMPT_CAT_GIVE = 13;
const PROMPT_CAT_CONTROL_REARRANGE = 14;
const PROMPT_CAT_REARRANGE_PROTOS = 15;
const PROMPT_CAT_OTHER = 16;
export const NUM_PROMPT_CATEGORIES = 17;

const PENDING_DEPTH_NORM = 8.0; // matches Python _PENDING_DEPTH_NORM

// Cards whose middle/bottom registers a `when_covered` handler. Hard-coded
// because pulling in effects.ts here would balloon the bundle. Keep in
// sync with effects.ts `register(WHEN_COVERED_EFFECTS, …)` callsites.
const WHEN_COVERED_KEYS: ReadonlySet<string> = new Set([
  "MN01:Fire:0",
  "MN01:Metal:6",
]);

// Target-meta codes — mirror Python encoder._TGT_*.
const TGT_NONE = 0;
const TGT_FIELD_CARD = 1;
// const TGT_HAND_CARD = 2; // unused at the encoder layer
const TGT_LINE = 3;
const TGT_INT = 4;
const TGT_STR = 5;
const TGT_SENTINEL = 6;
const TGT_PROTOCOL = 7;

export type EncodedState = {
  field_tokens: BigInt64Array;
  field_flags: Float32Array;          // [3, 2, MAX_STACK, 3] — face_up, committed, position_norm
  field_meta: Float32Array;
  protocols: BigInt64Array;
  hand_tokens: BigInt64Array;
  hand_size: Float32Array;
  trash: Float32Array;
  line_vals: Float32Array;
  scalars: Float32Array;
  phase: Float32Array;
  // A1: pending-choice context. The model uses these even outside of
  // effect resolution (then they're PAD/zero/0.0 respectively).
  pending_card_token: BigInt64Array;  // shape [] — scalar; we use [1] for buffer convenience
  pending_category: Float32Array;     // [NUM_PROMPT_CATEGORIES] one-hot
  pending_depth_norm: Float32Array;   // [1]
};

export type EncodedActions = {
  action_raw: Float32Array;
  action_card_ids: BigInt64Array;
  action_proto_ids: BigInt64Array;
  action_extra_card_ids: BigInt64Array; // A5: soon-covered card token (PAD for non-PLAY)
  action_mask: Uint8Array;              // 0/1 bytes
};

// Bucket a Choice.prompt into one of PROMPT_CAT_*. Mirrors Python's
// encoder._classify_prompt — order-sensitive (earlier branches win).
function classifyPrompt(prompt: string): number {
  if (!prompt) return PROMPT_CAT_NONE;
  const p = prompt.toLowerCase();
  if (p.includes("control component") || (p.includes("rearrange") && p.includes("whose"))) {
    return PROMPT_CAT_CONTROL_REARRANGE;
  }
  if (p.includes("give")) return PROMPT_CAT_GIVE;
  if (p.includes("reveal")) return PROMPT_CAT_REVEAL;
  if (p.includes("state a protocol")) return PROMPT_CAT_PICK_PROTOCOL;
  if (p.includes("state a number")) return PROMPT_CAT_PICK_NUMBER;
  if (p.includes("rearrange") || (p.includes("swap") && p.includes("protocol")) || p.includes("stack")) {
    return PROMPT_CAT_REARRANGE_PROTOS;
  }
  if (p.includes("accept") || (p.includes("optional") && p.includes("?"))) {
    return PROMPT_CAT_OPTIONAL_ACCEPT;
  }
  if (p.includes("play 1 card") || p.includes("play a value-")) {
    return PROMPT_CAT_PLAY_SUB_CARD;
  }
  if (p.includes("which line") || p.includes("to which line")
      || p.includes("choose a line") || p.includes("pick a line")
      || p.includes("select a line")) {
    return PROMPT_CAT_PICK_LINE;
  }
  if (p.includes("discard") && (p.includes("opp") || p.includes("opponent"))) {
    return PROMPT_CAT_DISCARD_OPP;
  }
  if (p.includes("discard")) return PROMPT_CAT_DISCARD_OWN;
  if (p.includes("flip")) return PROMPT_CAT_FLIP;
  if (p.includes("delete")) return PROMPT_CAT_DELETE;
  if (p.includes("shift")) return PROMPT_CAT_SHIFT;
  if (p.includes("return") || p.includes("steal")) return PROMPT_CAT_RETURN_OR_STEAL;
  return PROMPT_CAT_OTHER;
}

// -----------------------------------------------------------------------------
// State encoding
// -----------------------------------------------------------------------------

function cardToken(c: CardInst, perspective: PlayerIndex): number {
  if (c.owner !== perspective && !c.faceUp) return HIDDEN_TOKEN;
  if (c.defId < 0) return HIDDEN_TOKEN; // placeholder safety
  return c.defId + CARD_TOKEN_OFFSET;
}

/** Encode the state from `perspective`'s POV. Output tensors are unbatched
 *  (no leading batch dim); the bot wraps each with [1, ...] before passing
 *  to onnxruntime. */
export function encodeState(
  state: GameState,
  perspective: PlayerIndex,
  options: {
    pendingChoice?: boolean;
    /** Prompt text of the current pending Choice (if any). Used to
     *  classify into PROMPT_CAT_*. */
    pendingPrompt?: string | null;
    /** How deep the pending-effect stack is — drives pending_depth_norm. */
    pendingDepth?: number;
    /** The card whose middle/bottom/etc. produced the current Choice.
     *  Python introspects gen frame locals; in TS we ask the engine to
     *  surface this explicitly via PendingEffect.sourceCard. PAD if
     *  unknown (or system-level generators with no source card). */
    pendingSourceCard?: CardInst | null;
    draftIdx?: number;
    draftScheduleLen?: number;
    decider?: PlayerIndex;
  } = {},
): EncodedState {
  const me = perspective;
  const opp: PlayerIndex = me === 0 ? 1 : 0;

  // field_tokens [3, 2, MAX_STACK]
  const field_tokens = new BigInt64Array(NUM_LINES * 2 * MAX_STACK);
  // field_flags [3, 2, MAX_STACK, 3] — (face_up, committed, position_norm)
  const field_flags = new Float32Array(NUM_LINES * 2 * MAX_STACK * 3);
  // field_meta [3, 2, 3]
  const field_meta = new Float32Array(NUM_LINES * 2 * 3);

  for (let ln = 0; ln < NUM_LINES; ln++) {
    for (const [psIdx, pl] of [[0, me], [1, opp]] as const) {
      const stack = lineStack(state.lines[ln], pl);
      let fu = 0;
      let fd = 0;
      for (let pos = 0; pos < Math.min(stack.length, MAX_STACK); pos++) {
        const c = stack[pos];
        const tok = cardToken(c, me);
        const tokIdx = ln * 2 * MAX_STACK + psIdx * MAX_STACK + pos;
        field_tokens[tokIdx] = BigInt(tok);
        const flagBase = (ln * 2 * MAX_STACK + psIdx * MAX_STACK + pos) * 3;
        field_flags[flagBase + 0] = c.faceUp ? 1.0 : 0.0;
        field_flags[flagBase + 1] = c.isCommitted ? 1.0 : 0.0;
        field_flags[flagBase + 2] = pos / Math.max(1, MAX_STACK - 1);
        if (c.faceUp) fu++; else fd++;
      }
      const depth = Math.min(stack.length, MAX_STACK);
      const base = ln * 2 * 3 + psIdx * 3;
      field_meta[base + 0] = fu;
      field_meta[base + 1] = fd;
      field_meta[base + 2] = depth / MAX_STACK;
    }
  }

  // protocols [2, 3, 2]
  const protocols = new BigInt64Array(2 * NUM_LINES * 2);
  for (const [psIdx, pl] of [[0, me], [1, opp]] as const) {
    const ps = state.players[pl];
    for (let ln = 0; ln < NUM_LINES; ln++) {
      const base = psIdx * NUM_LINES * 2 + ln * 2;
      const proto = ps.protocols[ln];
      protocols[base + 0] = BigInt(proto ? PROTO_TO_ID.get(proto) ?? 0 : 0);
      protocols[base + 1] = BigInt(ps.compiled[ln] ? 1 : 0);
    }
  }

  // hand_tokens [MAX_HAND]
  const hand_tokens = new BigInt64Array(MAX_HAND);
  const myHand = state.players[me].hand;
  for (let i = 0; i < Math.min(myHand.length, MAX_HAND); i++) {
    const c = myHand[i];
    if (c.defId >= 0) hand_tokens[i] = BigInt(c.defId + CARD_TOKEN_OFFSET);
    // Placeholders (record mode) stay as PAD=0; the bot doesn't see
    // identities the engine doesn't have.
  }
  const hand_size = new Float32Array([myHand.length / MAX_HAND]);
  const opp_hand_size = state.players[opp].hand.length / MAX_HAND;

  // trash counts [2, NUM_CARDS]
  const trash = new Float32Array(2 * NUM_CARDS);
  for (const [psIdx, pl] of [[0, me], [1, opp]] as const) {
    for (const c of state.players[pl].trash) {
      if (c.defId >= 0 && c.defId < NUM_CARDS) {
        trash[psIdx * NUM_CARDS + c.defId] += 1;
      }
    }
  }
  for (let i = 0; i < trash.length; i++) trash[i] /= 18.0;

  // line_vals [3, 2]
  const line_vals = new Float32Array(NUM_LINES * 2);
  for (let ln = 0; ln < NUM_LINES; ln++) {
    for (const [psIdx, pl] of [[0, me], [1, opp]] as const) {
      line_vals[ln * 2 + psIdx] = computeLineValue(state, ln, pl) / 20.0;
    }
  }

  // scalars [8]
  const ctrl = state.controlHolder;
  const ctrlFlag = ctrl == null ? 0.0 : ctrl === me ? 1.0 : -1.0;
  const nDraft = Math.max(1, options.draftScheduleLen ?? 0);
  const draftPickIdxNorm = Math.min(options.draftIdx ?? 0, nDraft) / nDraft;
  const decider = options.decider ?? state.currentPlayer;
  const scalars = new Float32Array([
    state.turn / state.config.maxTurns,
    ctrlFlag,
    state.config.includeExpansion ? 1.0 : 0.0,
    state.players[me].cannotCompileNextTurn ? 1.0 : 0.0,
    state.players[opp].cannotCompileNextTurn ? 1.0 : 0.0,
    decider === me ? 1.0 : 0.0,
    opp_hand_size,
    draftPickIdxNorm,
  ]);

  // phase one-hot [9]
  const phase = new Float32Array(NUM_PHASES);
  const phaseIdx = options.pendingChoice ? 7 : PHASE_INDEX[state.phase] ?? 0;
  phase[phaseIdx] = 1.0;

  // A1: pending-choice context. Mirrors Python's encoder._pending_source_*
  // path — but where Python introspects the suspended generator's frame
  // locals, here we rely on the engine to attach `sourceCard` to each
  // PendingEffect and pass it in via options.pendingSourceCard. PAD when
  // unknown (system generators, refresh/control-rearrange/etc.).
  const pending_card_token_buf = new BigInt64Array(1);
  const srcCard = options.pendingSourceCard ?? null;
  if (srcCard && srcCard.defId >= 0) {
    if (srcCard.owner !== me && !srcCard.faceUp) {
      pending_card_token_buf[0] = BigInt(HIDDEN_TOKEN);
    } else {
      pending_card_token_buf[0] = BigInt(srcCard.defId + CARD_TOKEN_OFFSET);
    }
  } else {
    pending_card_token_buf[0] = BigInt(PAD_TOKEN);
  }

  const pending_category = new Float32Array(NUM_PROMPT_CATEGORIES);
  const promptCat = options.pendingChoice && options.pendingPrompt
    ? classifyPrompt(options.pendingPrompt)
    : PROMPT_CAT_NONE;
  pending_category[promptCat] = 1.0;

  const pending_depth_norm = new Float32Array(1);
  const depthRaw = options.pendingChoice ? (options.pendingDepth ?? 0) : 0;
  pending_depth_norm[0] = Math.min(depthRaw, PENDING_DEPTH_NORM) / PENDING_DEPTH_NORM;

  return {
    field_tokens,
    field_flags,
    field_meta,
    protocols,
    hand_tokens,
    hand_size,
    trash,
    line_vals,
    scalars,
    phase,
    pending_card_token: pending_card_token_buf,
    pending_category,
    pending_depth_norm,
  };
}

// -----------------------------------------------------------------------------
// Action encoding
// -----------------------------------------------------------------------------

type ChoiceLike = {
  prompt: string;
  // Targets are opaque — the engine encodes them as tuples (line, player,
  // pos, card) for field-card choices, strings for protocol picks, or
  // ints for line/number choices.
  targets: unknown[];
};

function classifyTarget(
  target: unknown,
  prompt: string,
): { meta: number; defId: number | null; lineIdx: number | null; protoId: number | null } {
  if (Array.isArray(target) && target.length >= 4) {
    const card = target[3] as { defId?: number } | undefined;
    if (card && typeof card.defId === "number") {
      return { meta: TGT_FIELD_CARD, defId: card.defId, lineIdx: target[0] as number, protoId: null };
    }
  }
  if (typeof target === "string") {
    const id = PROTO_TO_ID.get(target);
    if (id != null) return { meta: TGT_PROTOCOL, defId: null, lineIdx: null, protoId: id };
    return { meta: TGT_STR, defId: null, lineIdx: null, protoId: null };
  }
  if (typeof target === "number") {
    if (target === -1) return { meta: TGT_SENTINEL, defId: null, lineIdx: null, protoId: null };
    const p = prompt.toLowerCase();
    if (p.includes("protocol")) {
      return { meta: TGT_PROTOCOL, defId: null, lineIdx: null, protoId: target + 1 };
    }
    if (p.includes("line") && target >= 0 && target <= 2) {
      return { meta: TGT_LINE, defId: null, lineIdx: target, protoId: null };
    }
    return { meta: TGT_INT, defId: null, lineIdx: null, protoId: null };
  }
  if (target == null) return { meta: TGT_NONE, defId: null, lineIdx: null, protoId: null };
  return { meta: TGT_NONE, defId: null, lineIdx: null, protoId: null };
}

function actionCardDefId(
  state: GameState,
  a: Action,
  perspective: PlayerIndex,
  choice: ChoiceLike | null,
): number | null {
  if (a.type === "PLAY_FACE_UP" || a.type === "PLAY_FACE_DOWN" || a.type === "DISCARD_CARD") {
    const hi = a.handIndex;
    if (typeof hi === "number" && hi >= 0 && hi < state.players[perspective].hand.length) {
      return state.players[perspective].hand[hi].defId;
    }
  }
  if (a.type === "SHIFT_OWN_CARD") {
    if (typeof a.lineIndex === "number" && typeof a.handIndex === "number") {
      const stack = lineStack(state.lines[a.lineIndex], perspective);
      if (a.handIndex >= 0 && a.handIndex < stack.length) {
        return stack[a.handIndex].defId;
      }
    }
  }
  if (a.type === "CHOOSE_TARGET" && choice && typeof a.choiceIndex === "number") {
    const idx = a.choiceIndex;
    if (idx >= 0 && idx < choice.targets.length) {
      const cls = classifyTarget(choice.targets[idx], choice.prompt);
      if (cls.defId != null) return cls.defId;
    }
  }
  return null;
}

function actionTargetMeta(
  a: Action,
  choice: ChoiceLike | null,
): { meta: number; lineIdx: number | null; protoId: number | null } {
  if (a.type !== "CHOOSE_TARGET" || !choice) {
    return { meta: TGT_NONE, lineIdx: null, protoId: null };
  }
  const idx = a.choiceIndex;
  if (typeof idx !== "number" || idx < 0 || idx >= choice.targets.length) {
    return { meta: TGT_NONE, lineIdx: null, protoId: null };
  }
  const cls = classifyTarget(choice.targets[idx], choice.prompt);
  return { meta: cls.meta, lineIdx: cls.lineIdx, protoId: cls.protoId };
}

// A5: for PLAY actions, look at what card (if any) is about to be covered.
// Returns (token, faceUpFlag, triggersUnderWhenCovered).
function soonCoveredForPlay(
  state: GameState,
  a: Action,
  perspective: PlayerIndex,
): { token: number; faceUp: number; whenCov: number } {
  if (a.type !== "PLAY_FACE_UP" && a.type !== "PLAY_FACE_DOWN") {
    return { token: PAD_TOKEN, faceUp: 0, whenCov: 0 };
  }
  let lineIdx = a.lineIndex ?? -1;
  let targetSide: PlayerIndex = perspective;
  const NUM = NUM_LINES;
  const ps = state.players[perspective];
  if (typeof a.handIndex === "number" && a.handIndex >= 0 && a.handIndex < ps.hand.length) {
    const c = ps.hand[a.handIndex];
    const d = c.defId >= 0 ? CARD_DEFS[c.defId] : null;
    if (d && d.protocol === "Corruption" && d.value === 0
        && lineIdx >= NUM && lineIdx < 2 * NUM) {
      targetSide = (perspective === 0 ? 1 : 0) as PlayerIndex;
      lineIdx -= NUM;
    }
  }
  if (!(lineIdx >= 0 && lineIdx < NUM)) {
    return { token: PAD_TOKEN, faceUp: 0, whenCov: 0 };
  }
  const stack = lineStack(state.lines[lineIdx], targetSide);
  if (stack.length === 0) return { token: PAD_TOKEN, faceUp: 0, whenCov: 0 };
  const top = stack[stack.length - 1];
  const token = cardToken(top, perspective);
  const faceUp = top.faceUp ? 1 : 0;
  let whenCov = 0;
  if (top.faceUp && top.defId >= 0) {
    const key = CARD_DEFS[top.defId].key;
    if (WHEN_COVERED_KEYS.has(key)) whenCov = 1;
  }
  return { token, faceUp, whenCov };
}

// A3: closed-form lookahead deltas. Mirrors encoder._compute_action_lookahead.
// Returns [would_compile, triggers_under_when_covered, card_value_added_norm,
//          dest_line_diff_post_norm, hand_delta_me_norm, control_resets, is_pure_target].
function computeActionLookahead(
  state: GameState,
  a: Action,
  perspective: PlayerIndex,
  out: Float32Array,
  outOffset: number,
): void {
  // Zero by default (already zero in caller's buffer).
  const me = perspective;
  const opp: PlayerIndex = me === 0 ? 1 : 0;
  if (a.type === "CHOOSE_TARGET") {
    out[outOffset + 6] = 1.0;
    return;
  }
  if (a.type === "PLAY_FACE_UP" || a.type === "PLAY_FACE_DOWN") {
    const ps = state.players[me];
    const hi = a.handIndex;
    if (typeof hi !== "number" || hi < 0 || hi >= ps.hand.length) return;
    const c = ps.hand[hi];
    if (c.defId < 0) return;
    const d = CARD_DEFS[c.defId];
    const faceUp = a.type === "PLAY_FACE_UP";
    let lineIdx = a.lineIndex ?? -1;
    let targetSide: PlayerIndex = me;
    if (d.protocol === "Corruption" && d.value === 0
        && lineIdx >= NUM_LINES && lineIdx < 2 * NUM_LINES) {
      targetSide = opp;
      lineIdx -= NUM_LINES;
    }
    if (lineIdx < 0 || lineIdx >= NUM_LINES) return;
    const valueAdded = faceUp ? d.value : FACE_DOWN_BASE_VALUE;
    out[outOffset + 2] = valueAdded / 12.0;
    const curUs = computeLineValue(state, lineIdx, targetSide);
    const curThem = computeLineValue(state, lineIdx, targetSide === 0 ? 1 : 0);
    const postDiff = targetSide === me
      ? (curUs + valueAdded) - curThem
      : curUs - (curThem + valueAdded);
    out[outOffset + 3] = Math.max(-1.0, Math.min(1.0, postDiff / 20.0));
    if (faceUp && targetSide === me) {
      const postUs = curUs + valueAdded;
      if (postUs >= 10 && postUs > curThem && playerCanCompile(state, me)) {
        out[outOffset + 0] = 1.0;
      }
    }
    out[outOffset + 4] = -1.0 / 5.0;
    const sc = soonCoveredForPlay(state, a, perspective);
    out[outOffset + 1] = sc.whenCov;
    return;
  }
  if (a.type === "REFRESH") {
    const need = Math.max(0, 5 - state.players[me].hand.length);
    out[outOffset + 4] = need / 5.0;
    if (state.controlHolder === me) out[outOffset + 5] = 1.0;
    return;
  }
  if (a.type === "COMPILE_LINE") {
    out[outOffset + 3] = 0.0;
    if (state.controlHolder === me) out[outOffset + 5] = 1.0;
    return;
  }
  if (a.type === "DISCARD_CARD") {
    out[outOffset + 4] = -1.0 / 5.0;
    return;
  }
  if (a.type === "SHIFT_OWN_CARD") {
    const li = a.lineIndex ?? -1;
    if (li < 0 || li >= NUM_LINES) return;
    const stack = lineStack(state.lines[li], me);
    const hi = a.handIndex ?? -1;
    if (hi < 0 || hi >= stack.length) return;
    const c = stack[hi];
    const v = c.faceUp && c.defId >= 0 ? CARD_DEFS[c.defId].value : FACE_DOWN_BASE_VALUE;
    out[outOffset + 2] = v / 12.0;
    const dst = a.choiceIndex ?? -1;
    if (dst >= 0 && dst < NUM_LINES) {
      const curUs = computeLineValue(state, dst, me);
      const curThem = computeLineValue(state, dst, opp);
      const postDiff = (curUs + v) - curThem;
      out[outOffset + 3] = Math.max(-1.0, Math.min(1.0, postDiff / 20.0));
    }
    return;
  }
  // DRAFT_PROTOCOL / SKIP_OPTIONAL / NOOP — leave deltas at zero.
}

export function encodeActions(
  state: GameState,
  legal: Action[],
  perspective: PlayerIndex,
  choice: ChoiceLike | null,
): EncodedActions {
  const raw = new Float32Array(MAX_ACTIONS * RAW_ACTION_DIM);
  const card_ids = new BigInt64Array(MAX_ACTIONS);
  const proto_ids = new BigInt64Array(MAX_ACTIONS);
  const extra_card_ids = new BigInt64Array(MAX_ACTIONS);
  const mask = new Uint8Array(MAX_ACTIONS);

  const n = Math.min(legal.length, MAX_ACTIONS);
  for (let i = 0; i < n; i++) {
    const a = legal[i];
    const rowBase = i * RAW_ACTION_DIM;
    // [0..NUM_ACTION_TYPES) — action-type one-hot
    const aIdx = ACTION_TYPE_INDEX[a.type];
    if (aIdx != null) raw[rowBase + aIdx] = 1.0;
    let off = NUM_ACTION_TYPES;
    // hand_index norm
    raw[rowBase + off] = typeof a.handIndex === "number" && a.handIndex >= 0
      ? a.handIndex / MAX_HAND
      : -1.0;
    off += 1;
    // src line one-hot (+ none slot at index 3)
    if (typeof a.lineIndex === "number" && a.lineIndex >= 0 && a.lineIndex < 3) {
      raw[rowBase + off + a.lineIndex] = 1.0;
    } else {
      raw[rowBase + off + 3] = 1.0;
    }
    off += 4;
    // dst line one-hot (SHIFT_OWN_CARD uses choice_index as dst line)
    if (a.type === "SHIFT_OWN_CARD"
        && typeof a.choiceIndex === "number"
        && a.choiceIndex >= 0 && a.choiceIndex < 3) {
      raw[rowBase + off + a.choiceIndex] = 1.0;
    } else {
      raw[rowBase + off + 3] = 1.0;
    }
    off += 4;
    // choice_index normalised
    raw[rowBase + off] = typeof a.choiceIndex === "number" && a.choiceIndex >= 0
      ? a.choiceIndex / MAX_ACTIONS
      : -1.0;
    off += 1;
    // target meta one-hot (8 slots)
    const tm = actionTargetMeta(a, choice);
    raw[rowBase + off + tm.meta] = 1.0;
    off += 8;
    // stated value norm (for state-a-number choices)
    let statedVal = -1.0;
    if (a.type === "CHOOSE_TARGET" && choice && choice.prompt.toLowerCase().includes("number")) {
      const t = typeof a.choiceIndex === "number" ? choice.targets[a.choiceIndex] : null;
      if (typeof t === "number" && t >= 0 && t <= 6) statedVal = t / 6.0;
    }
    raw[rowBase + off] = statedVal;
    off += 1;

    // A5: soon-covered flags (PLAY actions). The actual card embedding
    // is carried via extra_card_ids and looked up in the model.
    const sc = soonCoveredForPlay(state, a, perspective);
    raw[rowBase + off] = sc.token !== PAD_TOKEN ? 1.0 : 0.0;
    raw[rowBase + off + 1] = sc.faceUp;
    off += 2;

    // A3: closed-form lookahead deltas (7 floats).
    computeActionLookahead(state, a, perspective, raw, rowBase + off);
    off += ACTION_LOOKAHEAD_DIM;

    // card / proto embedding lookup ids
    const defId = actionCardDefId(state, a, perspective, choice);
    if (defId != null && defId >= 0) {
      card_ids[i] = BigInt(defId + CARD_TOKEN_OFFSET);
    } else {
      card_ids[i] = BigInt(PAD_TOKEN);
    }
    if (a.type === "DRAFT_PROTOCOL" && a.protocol) {
      proto_ids[i] = BigInt(PROTO_TO_ID.get(a.protocol) ?? 0);
    } else if (tm.protoId != null) {
      proto_ids[i] = BigInt(tm.protoId);
    }
    // A5: secondary "soon-covered" card token (PAD for non-PLAY actions).
    extra_card_ids[i] = BigInt(sc.token);
    mask[i] = 1;
  }
  return {
    action_raw: raw,
    action_card_ids: card_ids,
    action_proto_ids: proto_ids,
    action_extra_card_ids: extra_card_ids,
    action_mask: mask,
  };
}

// Re-export protocol lookup for tests that want to verify the mapping.
export { PROTO_TO_ID };

// Re-exported so callers can sanity check def_id ordering against Python.
export { CARD_DEFS };
