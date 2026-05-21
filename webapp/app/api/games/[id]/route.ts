import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";

import { getSession } from "@/lib/auth";
import { db, schema } from "@/lib/db";
import { gameFromRow } from "@/lib/replay";
import { viewOfGame } from "@/lib/view";

export const dynamic = "force-dynamic";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!db) return NextResponse.json({ error: "database not configured" }, { status: 503 });
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const { id } = await params;
  const rows = await db.select().from(schema.games).where(eq(schema.games.id, id)).limit(1);
  const row = rows[0];
  if (!row) return NextResponse.json({ error: "not found" }, { status: 404 });
  if (row.userId !== session.userId) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const game = gameFromRow(row);
  return NextResponse.json({ id: row.id, view: viewOfGame(game), row });
}
