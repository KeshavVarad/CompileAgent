/**
 * POST /api/games  — create a new game (returns id + initial view)
 * GET  /api/games  — list saved games (most recent first)
 */

import { NextResponse } from "next/server";
import { and, desc, eq } from "drizzle-orm";

import { getSession } from "@/lib/auth";
import { CURRENT_BOT } from "@/lib/bot-config";
import { Game } from "@/lib/compile/game";
import { db, schema } from "@/lib/db";
import { viewOfGame } from "@/lib/view";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  if (!db) return NextResponse.json({ error: "database not configured" }, { status: 503 });
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const body = await request.json().catch(() => ({}));
  const mode: "play" | "record" = body.mode === "record" ? "record" : "play";
  const recorderSeat: 0 | 1 | null =
    mode === "record" ? (body.recorderSeat === 1 ? 1 : 0) : null;
  const includeExpansion = Boolean(body.includeExpansion ?? false);
  const includeMain2 = Boolean(body.includeMain2 ?? false);
  const includeAux2 = Boolean(body.includeAux2 ?? false);
  const maxTurns = Number(body.maxTurns ?? 200);
  const seed = Number(body.seed ?? Math.floor(Math.random() * 2_000_000_000));
  const player0Label = String(body.player0Label ?? "Player 1");
  const player1Label = String(body.player1Label ?? "Player 2");
  // In record mode there is no bot — the recorder enters every action
  // themselves. In play mode, the opponent is always the current NN bot
  // (random is no longer exposed). We validate against CURRENT_BOT.id so
  // future swaps in bot-config.ts don't need an API edit.
  let bot0Strategy: string | null = mode === "record" ? null : (body.bot0Strategy ?? null);
  let bot1Strategy: string | null = mode === "record" ? null : (body.bot1Strategy ?? CURRENT_BOT.id);
  if (mode === "play") {
    if (bot0Strategy !== null && bot1Strategy !== null) {
      return NextResponse.json({ error: "play mode requires exactly one bot seat" }, { status: 400 });
    }
    const strat = bot0Strategy ?? bot1Strategy;
    if (strat !== CURRENT_BOT.id) {
      return NextResponse.json(
        { error: `unknown bot strategy "${strat}"; expected "${CURRENT_BOT.id}"` },
        { status: 400 },
      );
    }
  }

  const [row] = await db
    .insert(schema.games)
    .values({
      userId: session.userId,
      includeExpansion,
      includeMain2,
      includeAux2,
      maxTurns,
      seed,
      player0Label,
      player1Label,
      bot0Strategy,
      bot1Strategy,
      mode,
      recorderSeat,
      actions: [],
    })
    .returning();

  // Build the live game once to return the initial view.
  const game = new Game({
    includeExpansion, includeMain2, includeAux2, maxTurns, seed,
    player0Label, player1Label,
    human: [bot0Strategy === null, bot1Strategy === null],
    botStrategy: (bot0Strategy ?? bot1Strategy) as "random" | null,
    mode,
    recorderSeat: recorderSeat ?? undefined,
  });
  game.start();

  return NextResponse.json({ id: row.id, view: viewOfGame(game) });
}

export async function GET() {
  if (!db) return NextResponse.json({ games: [] });
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const rows = await db
    .select({
      id: schema.games.id,
      createdAt: schema.games.createdAt,
      endedAt: schema.games.endedAt,
      player0Label: schema.games.player0Label,
      player1Label: schema.games.player1Label,
      bot0Strategy: schema.games.bot0Strategy,
      bot1Strategy: schema.games.bot1Strategy,
      includeExpansion: schema.games.includeExpansion,
      mode: schema.games.mode,
      recorderSeat: schema.games.recorderSeat,
      winner: schema.games.winner,
      turnCount: schema.games.turnCount,
    })
    .from(schema.games)
    .where(eq(schema.games.userId, session.userId))
    .orderBy(desc(schema.games.createdAt))
    .limit(50);
  return NextResponse.json({ games: rows });
}

// Silence unused-import lint when only one of {and, eq} is in play above.
void and;
