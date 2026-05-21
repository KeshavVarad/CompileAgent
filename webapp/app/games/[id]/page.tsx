import { eq } from "drizzle-orm";
import { notFound, redirect } from "next/navigation";

import { GameClient } from "@/components/game-client";
import { getSession } from "@/lib/auth";
import { db, schema } from "@/lib/db";
import { gameFromRow } from "@/lib/replay";
import { viewOfGame } from "@/lib/view";

export const dynamic = "force-dynamic";

export default async function GamePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const session = await getSession();
  if (!session) redirect(`/login?next=/games/${encodeURIComponent(id)}`);
  if (!db) return <div className="p-12 text-destructive">Database not configured.</div>;
  const rows = await db.select().from(schema.games).where(eq(schema.games.id, id)).limit(1);
  const row = rows[0];
  if (!row) return notFound();
  if (row.userId !== session.userId) {
    return (
      <div className="p-12 text-destructive">
        This game belongs to a different account.
      </div>
    );
  }
  const game = gameFromRow(row);
  const view = viewOfGame(game);
  const totalActions = (row.actions as unknown[]).length;
  return <GameClient gameId={row.id} initialView={view} initialTotalActions={totalActions} />;
}
