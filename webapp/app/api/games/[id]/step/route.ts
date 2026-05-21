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
import { autoAdvanceBot, botForStrategy, gameFromRow } from "@/lib/replay";
import { viewOfGame } from "@/lib/view";

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
  const newActions: Action[] = [...(row.actions as Action[]), action];

  // Auto-advance bots until a human decision is needed.
  const bot0 = botForStrategy(row.bot0Strategy, row.seed + 1);
  const bot1 = botForStrategy(row.bot1Strategy, row.seed + 2);
  if (bot0 || bot1) {
    // Replay-from-scratch-after-each-step is cheap enough that we just call
    // autoAdvanceBot on the freshly-stepped game; the appended actions are
    // recorded for persistence.
    const applied = await autoAdvanceBot(game, bot0, bot1);
    newActions.push(...applied);
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

  return NextResponse.json({ id: row.id, view: viewOfGame(game) });
}
