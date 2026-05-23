/**
 * NN bot — runs a trained PolicyValueNet (exported to ONNX) over the
 * current game state and picks the highest-logit legal action. Loaded
 * once per process via a singleton cache; subsequent calls reuse the
 * session.
 *
 * Only used inside server routes (Node runtime). The bot reads the ONNX
 * file via fs from the bundled `public/models/...` path, so the deploy
 * needs `public/models/*` shipped.
 */

import fs from "node:fs";
import path from "node:path";
// Use the WASM build of onnxruntime — no native shared-library dependency,
// so the Vercel serverless bundle stays self-contained. Slightly slower than
// onnxruntime-node, but our model is tiny (~2.4MB) so per-inference latency
// is still milliseconds.
import * as ort from "onnxruntime-web";

import { CURRENT_BOT } from "../bot-config";
import type { Game } from "./game";
import type { Action, PlayerIndex } from "./types";
import { type Bot } from "./bot";
import {
  MAX_ACTIONS,
  encodeActions,
  encodeState,
  type EncodedActions,
  type EncodedState,
} from "./nn-encoder";

let SESSION: ort.InferenceSession | null = null;
let SESSION_PROMISE: Promise<ort.InferenceSession> | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (SESSION) return SESSION;
  if (SESSION_PROMISE) return SESSION_PROMISE;
  // Point onnxruntime-web at the .wasm files we copy into public/ort/ at
  // build time. The WASM single-threaded build keeps things simple — no
  // SharedArrayBuffer / cross-origin isolation requirements.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = false;
  // Read the model bytes via fs (we're in a Node serverless route, not the
  // browser). InferenceSession.create accepts a Uint8Array.
  const file = path.join(
    process.cwd(),
    "public",
    CURRENT_BOT.modelPath.replace(/^\//, ""),
  );
  const bytes = fs.readFileSync(file);
  SESSION_PROMISE = ort.InferenceSession.create(bytes).then((s) => {
    SESSION = s;
    return s;
  });
  return SESSION_PROMISE;
}

function tensorFromState(s: EncodedState): Record<string, ort.Tensor> {
  return {
    field_tokens: new ort.Tensor("int64", s.field_tokens, [1, 3, 2, 10]),
    field_flags:  new ort.Tensor("float32", s.field_flags, [1, 3, 2, 10, 3]),
    field_meta:   new ort.Tensor("float32", s.field_meta, [1, 3, 2, 3]),
    protocols:    new ort.Tensor("int64", s.protocols, [1, 2, 3, 2]),
    hand_tokens:  new ort.Tensor("int64", s.hand_tokens, [1, 12]),
    hand_size:    new ort.Tensor("float32", s.hand_size, [1, 1]),
    trash:        new ort.Tensor("float32", s.trash, [1, 2, 180]),
    line_vals:    new ort.Tensor("float32", s.line_vals, [1, 3, 2]),
    scalars:      new ort.Tensor("float32", s.scalars, [1, 8]),
    phase:        new ort.Tensor("float32", s.phase, [1, 9]),
    // A1: pending-choice context tensors.
    pending_card_token: new ort.Tensor("int64", s.pending_card_token, [1]),
    pending_category:   new ort.Tensor("float32", s.pending_category, [1, s.pending_category.length]),
    pending_depth_norm: new ort.Tensor("float32", s.pending_depth_norm, [1, 1]),
  };
}

function tensorFromActions(a: EncodedActions, rawDim: number): Record<string, ort.Tensor> {
  return {
    action_raw:             new ort.Tensor("float32", a.action_raw, [1, MAX_ACTIONS, rawDim]),
    action_card_ids:        new ort.Tensor("int64", a.action_card_ids, [1, MAX_ACTIONS]),
    action_proto_ids:       new ort.Tensor("int64", a.action_proto_ids, [1, MAX_ACTIONS]),
    action_extra_card_ids:  new ort.Tensor("int64", a.action_extra_card_ids, [1, MAX_ACTIONS]),
    action_mask:            new ort.Tensor("bool", a.action_mask, [1, MAX_ACTIONS]),
  };
}

export class NNBot implements Bot {
  // Note: choose() is synchronous in the Bot interface, but ONNX is async.
  // The caller (server route) must use chooseAsync explicitly.
  choose(_game: Game, _legal: Action[]): Action {
    throw new Error("NNBot.choose is async; use chooseAsync from server routes");
  }

  async chooseAsync(game: Game, legal: Action[]): Promise<Action> {
    const r = await this.evaluateAsync(game, legal);
    return r.action;
  }

  /** Run inference and return the chosen action plus auxiliary info
   *  (value estimate, softmax probabilities for the legal-action set).
   *  Used both for play (chooseAsync) and AI eval of recorded games. */
  async evaluateAsync(
    game: Game,
    legal: Action[],
  ): Promise<NNEvalResult> {
    if (legal.length === 0) throw new Error("no legal actions");

    const state = game.state;
    const decider = game.decider() as PlayerIndex;

    let pendingChoice: { prompt: string; targets: unknown[] } | null = null;
    let pendingSourceCard: import("./types").CardInst | null = null;
    let pendingDepth = 0;
    const eng = game as unknown as {
      pending?: {
        lastChoice?: { prompt: string; targets: unknown[] } | null;
        sourceCard?: import("./types").CardInst | null;
      }[];
    };
    const pendStack = eng.pending;
    if (pendStack && pendStack.length > 0) {
      pendingDepth = pendStack.length;
      const top = pendStack[pendStack.length - 1];
      if (top.lastChoice) {
        pendingChoice = { prompt: top.lastChoice.prompt, targets: top.lastChoice.targets };
      }
      pendingSourceCard = top.sourceCard ?? null;
    }

    const enc = encodeState(state, decider, {
      pendingChoice: pendingChoice != null,
      pendingPrompt: pendingChoice?.prompt ?? null,
      pendingDepth,
      pendingSourceCard,
      draftIdx: state.draftIdx,
      draftScheduleLen: state.draftSchedule.length,
      decider,
    });
    const acts = encodeActions(state, legal, decider, pendingChoice);

    const sess = await getSession();
    const feeds = {
      ...tensorFromState(enc),
      ...tensorFromActions(acts, acts.action_raw.length / MAX_ACTIONS),
    };
    const out = await sess.run(feeds);
    const logits = out["logits"].data as Float32Array;
    const valueArr = out["value"].data as Float32Array;
    const value = valueArr.length > 0 ? valueArr[0] : 0;

    const n = Math.min(legal.length, MAX_ACTIONS);

    // Softmax over legal slots only. (Masked padded slots are large-negative
    // already in the model, so we could softmax over the whole row, but
    // restricting to n keeps the probability mass exactly inside the
    // legal-action set.)
    let maxL = -Infinity;
    for (let i = 0; i < n; i++) if (logits[i] > maxL) maxL = logits[i];
    let sumExp = 0;
    const probs: number[] = new Array(n);
    for (let i = 0; i < n; i++) { probs[i] = Math.exp(logits[i] - maxL); sumExp += probs[i]; }
    for (let i = 0; i < n; i++) probs[i] /= sumExp || 1;

    let best = 0;
    let bestLogit = -Infinity;
    for (let i = 0; i < n; i++) {
      if (logits[i] > bestLogit) { bestLogit = logits[i]; best = i; }
    }

    return {
      action: legal[best],
      actionIndex: best,
      value,
      // Bundled (index, prob) tuples so the caller can render top-N
      // suggestions without re-doing the softmax.
      probabilities: probs,
    };
  }
}

export type NNEvalResult = {
  action: Action;
  actionIndex: number;
  /** Tanh-bounded value head output in [-1, +1]. Positive = good for the
   *  decider, negative = bad. */
  value: number;
  /** Softmax over the legal-action set, parallel to the `legal` input. */
  probabilities: number[];
};

/** Returns true if `strategy` should be backed by NNBot. Any unrecognised
 *  value (including null) falls back to the random bot. */
export function isNNStrategy(strategy: string | null | undefined): boolean {
  return strategy === CURRENT_BOT.id || strategy === "nn";
}
