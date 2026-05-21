/**
 * GET /api/games/[id]/snapshot?index=N
 *
 * Replays the stored action log up to `index` and returns the game view at
 * that point. Used by the client's Analysis mode to scrub through history.
 *
 * - index=0 → state right after `start()` (no actions applied yet).
 * - index=actions.length → the current live state.
 * - index > actions.length is clamped to actions.length.
 *
 * The response also includes the action that brought the game into this
 * state (`lastAction`/`lastLabel`/`lastEvents`/`actor`), labeled against the
 * pre-action game state so card-name resolution lines up with the player
 * who actually saw that hand.
 */

import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";

import { getSession } from "@/lib/auth";
import { Game } from "@/lib/compile/game";
import type { Action, PlayerIndex } from "@/lib/compile/types";
import { db, schema } from "@/lib/db";
import { labelAction, viewOfGame } from "@/lib/view";

export const dynamic = "force-dynamic";

function infoSlice(log: { kind: string; text?: string }[], from: number, to: number): string[] {
  const out: string[] = [];
  for (let i = from; i < to; i++) {
    const e = log[i];
    if (e && e.kind === "info" && typeof e.text === "string") out.push(e.text);
  }
  return out;
}

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!db) return NextResponse.json({ error: "database not configured" }, { status: 503 });
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const { id } = await params;
  const { searchParams } = new URL(request.url);
  const indexParam = searchParams.get("index");
  if (indexParam == null) return NextResponse.json({ error: "missing index" }, { status: 400 });
  const indexRaw = Number.parseInt(indexParam, 10);
  if (!Number.isFinite(indexRaw) || indexRaw < 0) {
    return NextResponse.json({ error: "invalid index" }, { status: 400 });
  }

  const rows = await db.select().from(schema.games).where(eq(schema.games.id, id)).limit(1);
  const row = rows[0];
  if (!row) return NextResponse.json({ error: "not found" }, { status: 404 });
  if (row.userId !== session.userId) return NextResponse.json({ error: "forbidden" }, { status: 403 });

  const actions = row.actions as Action[];
  const totalActions = actions.length;
  const target = Math.min(indexRaw, totalActions);

  // Information-set viewer: the seat the human (or recorder) actually
  // saw the game from. We label actions and (in future) redact other
  // fields from this seat's perspective so analysis at past steps
  // doesn't reveal info the player didn't have then.
  const viewer: PlayerIndex | undefined =
    row.recorderSeat != null ? (row.recorderSeat as PlayerIndex)
    : row.bot0Strategy != null && row.bot1Strategy == null ? 1
    : row.bot1Strategy != null && row.bot0Strategy == null ? 0
    : undefined;

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
    mode: (row.mode === "record" ? "record" : "play"),
    recorderSeat: row.recorderSeat == null ? undefined : (row.recorderSeat as 0 | 1),
  });
  game.start();

  let lastLabel: string | null = null;
  let lastAction: Action | null = null;
  let lastEvents: string[] = [];
  let actor: PlayerIndex | null = null;
  for (let i = 0; i < target; i++) {
    if (game.isOver()) break;
    const a = actions[i];
    const last = i === target - 1;
    if (last) {
      lastLabel = labelAction(game, a, viewer);
      actor = game.decider();
    }
    const preLogLen = game.state.log.length;
    game.step(a);
    if (last) {
      lastAction = a;
      lastEvents = infoSlice(game.state.log, preLogLen, game.state.log.length);
    }
  }

  return NextResponse.json({
    id: row.id,
    view: viewOfGame(game),
    index: target,
    totalActions,
    lastAction,
    lastLabel,
    lastEvents,
    actor,
  });
}
