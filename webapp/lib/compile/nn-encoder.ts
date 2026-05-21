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
import { computeLineValue, lineStack } from "./helpers";
import type { Action, CardInst, GameState, Phase, PlayerIndex } from "./types";
import { NUM_LINES } from "./types";

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
export const RAW_ACTION_DIM =
  NUM_ACTION_TYPES + 1 + 4 + 4 + 1 + 8 + 1; // 29

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
  field_meta: Float32Array;
  protocols: BigInt64Array;
  hand_tokens: BigInt64Array;
  hand_size: Float32Array;
  trash: Float32Array;
  line_vals: Float32Array;
  scalars: Float32Array;
  phase: Float32Array;
};

export type EncodedActions = {
  action_raw: Float32Array;
  action_card_ids: BigInt64Array;
  action_proto_ids: BigInt64Array;
  action_mask: Uint8Array; // 0/1 bytes
};

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
    draftIdx?: number;
    draftScheduleLen?: number;
    decider?: PlayerIndex;
  } = {},
): EncodedState {
  const me = perspective;
  const opp: PlayerIndex = me === 0 ? 1 : 0;

  // field_tokens [3, 2, MAX_STACK]
  const field_tokens = new BigInt64Array(NUM_LINES * 2 * MAX_STACK);
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
        field_tokens[ln * 2 * MAX_STACK + psIdx * MAX_STACK + pos] = BigInt(tok);
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

  return {
    field_tokens,
    field_meta,
    protocols,
    hand_tokens,
    hand_size,
    trash,
    line_vals,
    scalars,
    phase,
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

export function encodeActions(
  state: GameState,
  legal: Action[],
  perspective: PlayerIndex,
  choice: ChoiceLike | null,
): EncodedActions {
  const raw = new Float32Array(MAX_ACTIONS * RAW_ACTION_DIM);
  const card_ids = new BigInt64Array(MAX_ACTIONS);
  const proto_ids = new BigInt64Array(MAX_ACTIONS);
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
    mask[i] = 1;
  }
  return { action_raw: raw, action_card_ids: card_ids, action_proto_ids: proto_ids, action_mask: mask };
}

// Re-export protocol lookup for tests that want to verify the mapping.
export { PROTO_TO_ID };

// Re-exported so callers can sanity check def_id ordering against Python.
export { CARD_DEFS };
