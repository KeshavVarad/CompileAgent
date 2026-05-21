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
import type { Action } from "@/lib/compile/types";
import { db, schema } from "@/lib/db";
import { autoAdvanceBotWithSnapshots, botForStrategy, gameFromRow } from "@/lib/replay";
import { type GameView, labelAction, viewOfGame } from "@/lib/view";

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

  const game = gameFromRow(row);
  if (game.isOver()) return NextResponse.json({ error: "game already over" }, { status: 400 });

  // Apply the player's action.
  game.step(action);
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
  type Step = { action: Action; label: string; view: GameView };
  let botSteps: Step[] = [];
  if (bot0 || bot1) {
    const { applied, snapshots } = await autoAdvanceBotWithSnapshots<string, Step>(
      game,
      bot0,
      bot1,
      // Pre-step: label the action against the pre-step game state so
      // card details (which leave the hand once the step lands) are
      // captured in the label.
      (g, a) => labelAction(g, a),
      // Post-step: bundle the post-state view with the captured label.
      (a, label, g) => ({ action: a, label, view: viewOfGame(g) }),
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
    botSteps,
  });
}
