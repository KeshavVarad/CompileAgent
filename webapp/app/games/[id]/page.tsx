import { eq } from "drizzle-orm";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";

import { GameClient } from "@/components/game-client";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
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
  const totalActions = (row.actions as unknown[]).length;
  try {
    const game = gameFromRow(row);
    const view = viewOfGame(game);
    return <GameClient gameId={row.id} initialView={view} initialTotalActions={totalActions} />;
  } catch (err) {
    // Replay-compat: the engine has evolved since this game was saved and
    // can no longer reconstruct its exact state from the action log alone.
    // Show a fallback page with metadata + a copy of the action history
    // so the user can still see what happened, rather than a 500.
    const message = err instanceof Error ? err.message : String(err);
    return (
      <UnplayableGame
        id={row.id}
        message={message}
        createdAt={row.createdAt}
        endedAt={row.endedAt}
        winner={row.winner}
        turnCount={row.turnCount}
        player0Label={row.player0Label}
        player1Label={row.player1Label}
        totalActions={totalActions}
      />
    );
  }
}

function UnplayableGame({
  id,
  message,
  createdAt,
  endedAt,
  winner,
  turnCount,
  player0Label,
  player1Label,
  totalActions,
}: {
  id: string;
  message: string;
  createdAt: Date;
  endedAt: Date | null;
  winner: number | null;
  turnCount: number | null;
  player0Label: string;
  player1Label: string;
  totalActions: number;
}) {
  const winnerLabel =
    winner === 0 ? player0Label : winner === 1 ? player1Label : null;
  return (
    <div className="flex-1 mx-auto w-full max-w-3xl px-6 py-10">
      <div className="mb-4">
        <Link
          href="/"
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          ← all games
        </Link>
      </div>
      <Card className="border-destructive/40">
        <CardContent className="py-6 space-y-4">
          <div className="flex items-center gap-2">
            <Badge variant="destructive" className="text-[10px] uppercase tracking-wide">
              Replay unavailable
            </Badge>
            <span className="text-xs font-mono text-muted-foreground">
              game {id.slice(0, 8)}
            </span>
          </div>
          <p className="text-sm">
            This game was played on an earlier engine version and can&apos;t be
            fully reconstructed from its saved action log. The metadata below
            is preserved.
          </p>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground">Players</dt>
            <dd className="font-mono">
              {player0Label} vs {player1Label}
            </dd>
            <dt className="text-muted-foreground">Created</dt>
            <dd className="font-mono">{createdAt.toISOString().slice(0, 10)}</dd>
            <dt className="text-muted-foreground">Ended</dt>
            <dd className="font-mono">
              {endedAt ? endedAt.toISOString().slice(0, 10) : "—"}
            </dd>
            <dt className="text-muted-foreground">Turns played</dt>
            <dd className="font-mono">{turnCount ?? "—"}</dd>
            <dt className="text-muted-foreground">Actions logged</dt>
            <dd className="font-mono">{totalActions}</dd>
            <dt className="text-muted-foreground">Winner</dt>
            <dd className="font-mono">
              {winnerLabel ? `${winnerLabel} (P${(winner as number) + 1})` : "—"}
            </dd>
          </dl>
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
              Engine error detail
            </summary>
            <pre className="mt-2 text-[11px] font-mono whitespace-pre-wrap text-destructive/80">
              {message}
            </pre>
          </details>
        </CardContent>
      </Card>
    </div>
  );
}
