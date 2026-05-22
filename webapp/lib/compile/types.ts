/**
 * Game state types. Mirrors src/compile_engine/state.py + actions.py.
 *
 * The TS engine auto-resolves card effects via JS generators, exactly like
 * the Python engine: an effect generator may `yield` a `Choice` when it needs
 * a player decision, and the engine sends back the chosen index to resume.
 */

import type { Protocol } from "./cards";

export const NUM_LINES = 3;
export const FACE_DOWN_BASE_VALUE = 2;
export const COMPILE_THRESHOLD = 10;
export const HAND_SIZE_LIMIT = 5;
export const STARTING_HAND = 5;
export const NUM_PROTOCOLS_PER_PLAYER = 3;
export const MAX_EFFECT_STACK_DEPTH = 64;
export const MAX_EFFECT_PUSHES_PER_TURN = 256;

export type PlayerIndex = 0 | 1;

export type CardInst = {
  instId: number;
  defId: number;
  owner: PlayerIndex;
  faceUp: boolean;
  isCommitted: boolean;
};

export type LineState = {
  p0Stack: CardInst[];     // bottom -> top (top = uncovered)
  p1Stack: CardInst[];
};

export type PlayerState = {
  idx: PlayerIndex;
  deck: CardInst[];        // top of deck = end of array
  hand: CardInst[];
  trash: CardInst[];
  protocols: Protocol[];   // 3 once drafted
  compiled: boolean[];     // parallel to protocols
  cannotCompileNextTurn: boolean;
};

export type Phase =
  | "DRAFT"
  | "START"
  | "CHECK_CONTROL"
  | "CHECK_COMPILE"
  | "ACTION"
  | "CHECK_CACHE"
  | "END"
  | "GAME_OVER";

export type GameMode = "play" | "record";

export type GameConfig = {
  includeExpansion: boolean;     // AX01
  includeMain2?: boolean;        // MN02
  includeAux2?: boolean;         // AX02
  /** If set, the draft pool is a random subset of this size sampled from
   *  the union of enabled sets. Undefined = use the full enabled pool.
   *  Mirrors `draft_pool_size` in the Python engine; used to inject
   *  protocol-diversity into RL training. Must be ≥6 (the snake-draft
   *  pick count). */
  draftPoolSize?: number;
  maxTurns: number;
  seed: number;
  /** Display labels for the UI/recorder. */
  player0Label: string;
  player1Label: string;
  /** Per-seat: is this seat played by a human via the UI? */
  human: [boolean, boolean];
  /** Bot strategy when a seat is non-human. */
  botStrategy: "random" | null;
  /** Game mode. "play" = normal interactive; "record" = transcribe a live
   *  game (opponent face-down cards are placeholders until revealed). */
  mode?: GameMode;
  /** In record mode, which seat is the recorder sitting in. Undefined in
   *  play mode. */
  recorderSeat?: 0 | 1;
};

/** Sentinel def_id for cards whose identity is hidden from the recorder. Used
 *  for the opponent's face-down field cards and for cards in the opponent's
 *  hand/deck before they are revealed. */
export const PLACEHOLDER_DEF_ID = -1;

export type ActionType =
  | "DRAFT_PROTOCOL"
  | "PLAY_FACE_UP"
  | "PLAY_FACE_DOWN"
  | "REFRESH"
  | "COMPILE_LINE"
  | "DISCARD_CARD"
  | "SHIFT_OWN_CARD"
  | "CHOOSE_TARGET"
  | "SKIP_OPTIONAL"
  | "NOOP";

export type Action = {
  type: ActionType;
  handIndex?: number;
  lineIndex?: number;
  choiceIndex?: number;
  protocol?: Protocol;
  /** Record-mode only: the def_id the recorder identified for a placeholder
   *  card that just became visible — either an opponent's PLAY_FACE_UP from
   *  hand, or a CHOOSE_TARGET resolution to a "what was that card?" prompt. */
  revealedDefId?: number;
};

export type Choice = {
  prompt: string;
  options: string[];
  /** Opaque per-option payload used by the resuming generator. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  targets: any[];
  optional: boolean;
  decider: PlayerIndex;
};

/** Trigger queue entries — events to fire after the current effect step. */
export type Trigger =
  | { kind: "face_up"; line: number; player: PlayerIndex; card: CardInst }
  | { kind: "uncover"; line: number; player: PlayerIndex; card: CardInst }
  | { kind: "when_covered"; line: number; player: PlayerIndex; card: CardInst }
  | { kind: "uncommit"; card: CardInst }
  /** Record-mode only: a placeholder card was just flipped face-up by an
   *  effect. The engine pauses and asks the recorder for its identity
   *  before proceeding to fire the face_up trigger that follows in LIFO. */
  | { kind: "reveal_placeholder"; line: number; player: PlayerIndex; card: CardInst };

/**
 * One entry in the per-game event log. Two shapes:
 *   - "action": the canonical record of every step() call (used for replay)
 *   - "info":   engine-emitted side-channel commentary for things that
 *               happen but don't show up as discrete actions — e.g. an
 *               effect tries to discard but the target's hand is empty,
 *               or a draw bails because deck+trash are both exhausted.
 *               These exist so the client can surface "tried X but
 *               couldn't because Y" during the bot's animation chain.
 */
export type GameLogEntry =
  | {
      kind: "action";
      turn: number;
      decider: PlayerIndex;
      action: Action;
      /** Pre-computed human-readable label, captured against the state
       *  BEFORE the action lands (so hand-index lookups resolve to the
       *  card that was actually played, not what's there now). Stored
       *  here because the hand changes by render time. */
      label: string;
      timestamp: number;
    }
  | {
      kind: "info";
      turn: number;
      text: string;
      timestamp: number;
    };

export type GameState = {
  config: GameConfig;
  players: [PlayerState, PlayerState];
  lines: LineState[];                  // length NUM_LINES
  currentPlayer: PlayerIndex;
  turn: number;
  phase: Phase;
  controlHolder: PlayerIndex | null;
  winner: PlayerIndex | null;
  triggers: Trigger[];                 // LIFO
  scratch: Record<string, unknown>;
  effectPushesThisTurn: number;
  compiledThisTurn: boolean;
  draftPool: Protocol[];
  draftIdx: number;
  draftSchedule: PlayerIndex[];
  log: GameLogEntry[];
  /** RNG state — Mulberry32 seed for deterministic replay. */
  rngState: number;
};
