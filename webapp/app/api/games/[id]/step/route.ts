/**
 * POST /api/games/[id]/step
 * Body: { action: Action }
 *
 * Applies a human action; if the next decider is a bot, advances bot turns
 * until a human decision is needed (or game over). Persists the appended
 * action list and the result of the game back to the DB.
 */

import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";

import { getSession } from "@/lib/auth";
import type { Action, PlayerIndex } from "@/lib/compile/types";
import { db, schema } from "@/lib/db";
import { autoAdvanceBotWithSnapshots, botForStrategy, gameFromRow } from "@/lib/replay";
import { type GameView, labelAction, viewOfGame } from "@/lib/view";

// Pull "info" entries that the engine added between two log lengths.
// The engine pushes one `kind:"action"` entry per step() plus zero or
// more `kind:"info"` entries from inside effect resolution (e.g.
// "P1 skipped 1 forced discard — hand empty"). We surface the info
// entries so the client can render "tried X but couldn't" alongside
// each animated bot step.
function infoSlice(log: { kind: string; text?: string }[], from: number, to: number): string[] {
  const out: string[] = [];
  for (let i = from; i < to; i++) {
    const e = log[i];
    if (e && e.kind === "info" && typeof e.text === "string") out.push(e.text);
  }
  return out;
}

export const dynamic = "force-dynamic";

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!db) return NextResponse.json({ error: "database not configured" }, { status: 503 });
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const { id } = await params;
  const body = await request.json().catch(() => ({}));
  const action: Action | undefined = body.action;
  if (!action) return NextResponse.json({ error: "missing action" }, { status: 400 });

  const rows = await db.select().from(schema.games).where(eq(schema.games.id, id)).limit(1);
  const row = rows[0];
  if (!row) return NextResponse.json({ error: "not found" }, { status: 404 });
  if (row.userId !== session.userId) return NextResponse.json({ error: "forbidden" }, { status: 403 });

  let game;
  try {
    game = gameFromRow(row);
  } catch (err) {
    // Replay-compat: engine evolved since this game was saved. The user
    // should not be able to keep playing this game — surface a 422 so
    // the client can prompt them to start a new game.
    return NextResponse.json(
      {
        error: "replay_unavailable",
        message: err instanceof Error ? err.message : String(err),
      },
      { status: 422 },
    );
  }
  if (game.isOver()) return NextResponse.json({ error: "game already over" }, { status: 400 });

  // Information-set viewer for label redaction — the seat the human
  // (or recorder) is sitting in. Bot face-down plays in the announce
  // banner should not reveal identity to the human watching.
  const viewer: PlayerIndex | undefined =
    row.recorderSeat != null ? (row.recorderSeat as PlayerIndex)
    : row.bot0Strategy != null && row.bot1Strategy == null ? 1
    : row.bot1Strategy != null && row.bot0Strategy == null ? 0
    : undefined;

  // Apply the player's action.
  const preUserLogLen = game.state.log.length;
  game.step(action);
  const userEvents = infoSlice(game.state.log, preUserLogLen, game.state.log.length);
  // Snapshot the board the moment the user's action (and any same-step
  // effect resolution) finishes — before the bot starts moving. The
  // client renders this first so the player sees what their own action
  // did before Sparkv2's chain plays out.
  const postUserView = viewOfGame(game);
  const newActions: Action[] = [...(row.actions as Action[]), action];

  // Auto-advance bots until a human decision is needed. Capture per-step
  // snapshots so the client can animate the chain instead of jumping
  // straight to the final state.
  const bot0 = botForStrategy(row.bot0Strategy, row.seed + 1);
  const bot1 = botForStrategy(row.bot1Strategy, row.seed + 2);
  type Step = { action: Action; label: string; view: GameView; events: string[] };
  let botSteps: Step[] = [];
  if (bot0 || bot1) {
    const { applied, snapshots } = await autoAdvanceBotWithSnapshots<
      { label: string; preLogLen: number },
      Step
    >(
      game,
      bot0,
      bot1,
      // Pre-step: label the action against the pre-step game state, and
      // record the current log length so we can slice out info events
      // that land during this step.
      (g, a) => ({ label: labelAction(g, a, viewer), preLogLen: g.state.log.length }),
      // Post-step: bundle the post-state view, the captured label, and
      // the info-event delta.
      (a, captured, g) => ({
        action: a,
        label: captured.label,
        view: viewOfGame(g),
        events: infoSlice(g.state.log, captured.preLogLen, g.state.log.length),
      }),
    );
    newActions.push(...applied);
    botSteps = snapshots;
  }

  await db
    .update(schema.games)
    .set({
      actions: newActions,
      updatedAt: new Date(),
      winner: game.isOver() ? game.state.winner : null,
      turnCount: game.state.turn,
      endedAt: game.isOver() ? new Date() : null,
    })
    .where(eq(schema.games.id, id));

  return NextResponse.json({
    id: row.id,
    // `view` is always the final state — the client can fall back to it
    // if the animation is interrupted or postUserView/botSteps are
    // missing (e.g., a future client talking to an older server).
    view: viewOfGame(game),
    postUserView,
    userEvents,
    botSteps,
    totalActions: newActions.length,
  });
}
