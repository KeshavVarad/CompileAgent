/**
 * AI review for a recorded/played game.
 *
 * GET  /api/games/[id]/eval
 *   Returns the *cached* eval result if one was previously computed,
 *   along with the action-count it was computed against. Returns 204
 *   No Content if no eval has been run yet for this game.
 *
 * POST /api/games/[id]/eval
 *   Body (optional): { perspective?: 0 | 1 }
 *   Recomputes the eval from scratch and persists it to the games row
 *   (eval_result + eval_action_count). Walks through every action in
 *   the saved game's action log; at each decision made by
 *   `perspective` (defaults to the recorder's seat in record mode,
 *   otherwise 0), runs the NN bot against the same legal set and
 *   reports the bot's choice + value + softmax probabilities.
 *
 * Returns one entry per evaluated decision so the UI can show
 * side-by-side "you did X, model's top pick was Y" with the bot's
 * full probability distribution over the user's move.
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
  /** The bot's policy probability assigned to the action the user actually
   *  played. Lets the UI say "your move had 12% prob" vs "0.4% prob" rather
   *  than collapsing to a binary agree/differ — the policy is stochastic, so
   *  "differs" alone doesn't mean wrong. */
  actualProb: number;
  /** Convenience: index of `actual` within the bot's top-k distribution
   *  (0 = top pick, 1 = 2nd best, …). -1 if outside the surfaced top-k. */
  actualRank: number;
  /** Probability of the model's top pick — useful for showing how
   *  *concentrated* the policy was at this decision. A 35%-top with a
   *  10%-second is a soft preference; a 95%-top is a sharp call. */
  topProb: number;
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
      const indexed = probs.map((p, idx) => ({ actionIndex: idx, prob: p, action: legal[idx] }));
      const top = [...indexed].sort((a, b) => b.prob - a.prob).slice(0, 5);
      const actualIdx = legal.findIndex((la) => actionsEqual(la, actual));
      const actualProb = actualIdx >= 0 ? probs[actualIdx] : 0;
      // actualRank within the surfaced top-k.
      const actualRank = top.findIndex((t) => t.actionIndex === actualIdx);
      const topProb = top.length > 0 ? top[0].prob : 0;
      entries.push({
        step: i,
        turn: game.state.turn,
        phase: game.state.phase,
        decider,
        actual,
        bot: result.action,
        value: result.value,
        topProbabilities: top,
        actualProb,
        actualRank,
        topProb,
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

  const payload = {
    id: row.id,
    perspective,
    entries,
    summary,
    /** Number of actions in the game log at the time this eval ran.
     *  The UI compares this against the live action count to flag the
     *  cached eval as "stale" once the player makes more moves. */
    actionCount: actions.length,
    /** When the eval finished — surfaced for "last reviewed: 10m ago"
     *  style UI affordances. */
    computedAt: new Date().toISOString(),
  };

  // Persist so the user can reopen the review without recomputing.
  await db
    .update(schema.games)
    .set({ evalResult: payload, evalActionCount: actions.length })
    .where(eq(schema.games.id, id));

  return NextResponse.json(payload);
}

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!db) return NextResponse.json({ error: "database not configured" }, { status: 503 });
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const { id } = await params;

  const rows = await db.select().from(schema.games).where(eq(schema.games.id, id)).limit(1);
  const row = rows[0];
  if (!row) return NextResponse.json({ error: "not found" }, { status: 404 });
  if (row.userId !== session.userId) return NextResponse.json({ error: "forbidden" }, { status: 403 });

  if (!row.evalResult) {
    // 204 No Content: cached eval doesn't exist yet. The client checks
    // for response.ok && response.status !== 204 to decide whether to
    // render a body.
    return new NextResponse(null, { status: 204 });
  }
  // The client wants to compare evalActionCount against (row.actions as
  // unknown[]).length to know if the eval is stale. Surface both.
  const currentActionCount = (row.actions as unknown[]).length;
  return NextResponse.json({
    ...(row.evalResult as Record<string, unknown>),
    actionCount: row.evalActionCount ?? null,
    currentActionCount,
    stale: row.evalActionCount != null && row.evalActionCount < currentActionCount,
  });
}
