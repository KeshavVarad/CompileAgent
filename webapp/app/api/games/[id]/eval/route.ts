/**
 * POST /api/games/[id]/eval
 * Body (optional): { perspective?: 0 | 1 }
 *
 * Walks through every action in the saved game's action log; at each
 * decision made by `perspective` (defaults to the recorder's seat in
 * record mode, otherwise 0), runs the NN bot against the same legal set
 * and reports the bot's choice + value + softmax probabilities.
 *
 * Returns one entry per evaluated decision so the UI can show
 * side-by-side "you did X, bot would do Y" with an Elo-style verdict.
 */

import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";

import { getSession } from "@/lib/auth";
import { Game } from "@/lib/compile/game";
import { NNBot } from "@/lib/compile/nn-bot";
import type { Action, PlayerIndex } from "@/lib/compile/types";
import { db, schema } from "@/lib/db";

export const dynamic = "force-dynamic";

type EvalEntry = {
  step: number;
  turn: number;
  phase: string;
  decider: PlayerIndex;
  actual: Action;
  bot: Action;
  value: number;
  /** Top up to 5 action+probability tuples, sorted high → low. */
  topProbabilities: { actionIndex: number; prob: number; action: Action }[];
  agree: boolean;
};

function actionsEqual(a: Action, b: Action): boolean {
  if (a.type !== b.type) return false;
  // Most actions are uniquely identified by (type, handIndex, lineIndex,
  // choiceIndex, protocol). We compare those fields directly.
  if (a.handIndex !== b.handIndex) return false;
  if (a.lineIndex !== b.lineIndex) return false;
  if (a.choiceIndex !== b.choiceIndex) return false;
  if (a.protocol !== b.protocol) return false;
  return true;
}

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!db) return NextResponse.json({ error: "database not configured" }, { status: 503 });
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const { id } = await params;
  const body = await request.json().catch(() => ({}));

  const rows = await db.select().from(schema.games).where(eq(schema.games.id, id)).limit(1);
  const row = rows[0];
  if (!row) return NextResponse.json({ error: "not found" }, { status: 404 });
  if (row.userId !== session.userId) return NextResponse.json({ error: "forbidden" }, { status: 403 });

  const explicitSeat = body.perspective;
  const perspective: PlayerIndex =
    explicitSeat === 0 || explicitSeat === 1
      ? (explicitSeat as PlayerIndex)
      : ((row.recorderSeat ?? 0) as PlayerIndex);

  // Build a fresh game and step through the action log.
  const game = new Game({
    includeExpansion: row.includeExpansion,
    includeMain2: row.includeMain2,
    includeAux2: row.includeAux2,
    maxTurns: row.maxTurns,
    seed: row.seed,
    player0Label: row.player0Label,
    player1Label: row.player1Label,
    human: [row.bot0Strategy === null, row.bot1Strategy === null],
    botStrategy: (row.bot0Strategy ?? row.bot1Strategy) as "random" | null,
    mode: row.mode === "record" ? "record" : "play",
    recorderSeat: row.recorderSeat == null ? undefined : (row.recorderSeat as 0 | 1),
  });
  game.start();

  const bot = new NNBot();
  const entries: EvalEntry[] = [];
  const actions = row.actions as Action[];

  for (let i = 0; i < actions.length; i++) {
    if (game.isOver()) break;
    const decider = game.decider() as PlayerIndex;
    const legal = game.legalActions();
    if (legal.length === 0) break;
    const actual = actions[i];

    if (decider === perspective && legal.length > 1) {
      // legal.length > 1: skip eval when there is no real decision (forced
      // single-action steps add no information).
      const result = await bot.evaluateAsync(game, legal);
      const probs = result.probabilities;
      const top = probs
        .map((p, idx) => ({ actionIndex: idx, prob: p, action: legal[idx] }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);
      entries.push({
        step: i,
        turn: game.state.turn,
        phase: game.state.phase,
        decider,
        actual,
        bot: result.action,
        value: result.value,
        topProbabilities: top,
        agree: actionsEqual(actual, result.action),
      });
    }
    game.step(actual);
  }

  const summary = {
    decisions: entries.length,
    agreementRate: entries.length === 0
      ? 0
      : entries.filter((e) => e.agree).length / entries.length,
    avgBotValue: entries.length === 0
      ? 0
      : entries.reduce((s, e) => s + e.value, 0) / entries.length,
    finalWinner: game.state.winner,
  };

  return NextResponse.json({
    id: row.id,
    perspective,
    entries,
    summary,
  });
}
