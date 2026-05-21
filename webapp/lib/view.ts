/**
 * Server-side projection of a Game into a JSON shape the client UI can render.
 * Keeps the wire payload small and stable across engine refactors.
 */

import { CARD_DEFS, safeCardDef } from "./compile/cards";
import { computeLineValue } from "./compile/helpers";
import type { Game } from "./compile/game";
import type { Action, CardInst, PlayerIndex } from "./compile/types";

export type CardView = {
  instId: number;
  defId: number;
  key: string;
  protocol: string;
  value: number;
  faceUp: boolean;
  isCommitted: boolean;
  topText: string;
  middleText: string;
  bottomText: string;
};

export type LineView = {
  index: number;
  p0Stack: CardView[];
  p1Stack: CardView[];
  p0Value: number;
  p1Value: number;
  p0Protocol: string | null;
  p1Protocol: string | null;
  p0Compiled: boolean;
  p1Compiled: boolean;
};

export type ActionView = Action & {
  label: string;
};

export type ChoiceView = {
  prompt: string;
  options: string[];
  optional: boolean;
  decider: PlayerIndex;
};

export type GameView = {
  config: {
    player0Label: string;
    player1Label: string;
    includeExpansion: boolean;
    seed: number;
    maxTurns: number;
    human: [boolean, boolean];
    botStrategy: string | null;
    mode: "play" | "record";
    recorderSeat: 0 | 1 | null;
  };
  phase: string;
  turn: number;
  currentPlayer: PlayerIndex;
  decider: PlayerIndex;
  winner: PlayerIndex | null;
  controlHolder: PlayerIndex | null;
  isOver: boolean;
  lines: LineView[];
  players: [
    { hand: CardView[]; deckCount: number; trashCount: number; trash: CardView[]; cannotCompileNextTurn: boolean },
    { hand: CardView[]; deckCount: number; trashCount: number; trash: CardView[]; cannotCompileNextTurn: boolean },
  ];
  draft: { pool: string[]; idx: number; schedule: PlayerIndex[] } | null;
  legalActions: ActionView[];
  pendingChoice: ChoiceView | null;
};

function viewCard(c: CardInst): CardView {
  const d = safeCardDef(c.defId);
  return {
    instId: c.instId,
    defId: c.defId,
    key: d.key,
    protocol: d.protocol,
    value: d.value,
    faceUp: c.faceUp,
    isCommitted: c.isCommitted,
    topText: d.topText,
    middleText: d.middleText,
    bottomText: d.bottomText,
  };
}

export function labelAction(game: Game, a: Action): string {
  const st = game.state;
  if (a.type === "DRAFT_PROTOCOL") return `Draft ${a.protocol}`;
  if (a.type === "PLAY_FACE_UP") {
    const c = st.players[st.currentPlayer].hand[a.handIndex!];
    const d = safeCardDef(c.defId);
    if (d.defId === -1 && typeof a.revealedDefId === "number") {
      const revealed = safeCardDef(a.revealedDefId);
      return `Opp plays ${revealed.protocol} ${revealed.value} face-up in line ${a.lineIndex! + 1}`;
    }
    if (d.defId === -1) return `Opp plays face-up in line ${a.lineIndex! + 1} · ?`;
    return `Play ${d.protocol} ${d.value} face-up in line ${a.lineIndex! + 1}`;
  }
  if (a.type === "PLAY_FACE_DOWN") {
    const c = st.players[st.currentPlayer].hand[a.handIndex!];
    const d = safeCardDef(c.defId);
    if (d.defId === -1) return `Opp plays face-down in line ${a.lineIndex! + 1}`;
    return `Play ${d.protocol} ${d.value} face-down in line ${a.lineIndex! + 1}`;
  }
  if (a.type === "REFRESH") return "Refresh";
  if (a.type === "COMPILE_LINE") return `Compile line ${a.lineIndex! + 1}`;
  if (a.type === "DISCARD_CARD") {
    const c = st.players[st.currentPlayer].hand[a.handIndex!];
    const d = safeCardDef(c.defId);
    if (d.defId === -1) return `Opp discards`;
    return `Discard ${d.protocol} ${d.value}`;
  }
  if (a.type === "SHIFT_OWN_CARD") return `Shift card → line ${a.choiceIndex! + 1}`;
  if (a.type === "CHOOSE_TARGET") return `Option ${a.choiceIndex! + 1}`;
  if (a.type === "SKIP_OPTIONAL") return "Skip";
  return a.type;
}

export function viewOfGame(game: Game): GameView {
  const st = game.state;
  const decider = game.decider();
  const legal = game.legalActions();
  const choice = game["pending"] && game["pending"].length > 0
    ? (game["pending"][game["pending"].length - 1] as { lastChoice: { prompt: string; options: string[]; optional: boolean; decider: PlayerIndex } | null }).lastChoice
    : null;

  return {
    config: {
      player0Label: st.config.player0Label,
      player1Label: st.config.player1Label,
      includeExpansion: st.config.includeExpansion,
      seed: st.config.seed,
      maxTurns: st.config.maxTurns,
      human: st.config.human,
      botStrategy: st.config.botStrategy,
      mode: (st.config.mode ?? "play"),
      recorderSeat: (st.config.recorderSeat ?? null) as 0 | 1 | null,
    },
    phase: st.phase,
    turn: st.turn,
    currentPlayer: st.currentPlayer,
    decider,
    winner: st.winner,
    controlHolder: st.controlHolder,
    isOver: game.isOver(),
    lines: st.lines.map((line, i) => ({
      index: i,
      p0Stack: line.p0Stack.map(viewCard),
      p1Stack: line.p1Stack.map(viewCard),
      p0Value: computeLineValue(st, i, 0),
      p1Value: computeLineValue(st, i, 1),
      p0Protocol: st.players[0].protocols[i] ?? null,
      p1Protocol: st.players[1].protocols[i] ?? null,
      p0Compiled: st.players[0].compiled[i] ?? false,
      p1Compiled: st.players[1].compiled[i] ?? false,
    })),
    players: [
      {
        hand: st.players[0].hand.map(viewCard),
        deckCount: st.players[0].deck.length,
        trashCount: st.players[0].trash.length,
        trash: st.players[0].trash.map(viewCard),
        cannotCompileNextTurn: st.players[0].cannotCompileNextTurn,
      },
      {
        hand: st.players[1].hand.map(viewCard),
        deckCount: st.players[1].deck.length,
        trashCount: st.players[1].trash.length,
        trash: st.players[1].trash.map(viewCard),
        cannotCompileNextTurn: st.players[1].cannotCompileNextTurn,
      },
    ],
    draft: st.phase === "DRAFT"
      ? { pool: [...st.draftPool], idx: st.draftIdx, schedule: [...st.draftSchedule] }
      : null,
    legalActions: legal.map((a) => ({ ...a, label: labelAction(game, a) })),
    pendingChoice: choice ? {
      prompt: choice.prompt,
      options: choice.options,
      optional: choice.optional,
      decider: choice.decider,
    } : null,
  };
}
