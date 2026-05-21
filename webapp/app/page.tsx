import Link from "next/link";
import { redirect } from "next/navigation";
import { desc, eq } from "drizzle-orm";

import { DeleteGameButton } from "@/components/delete-game-button";
import { LogoutButton } from "@/components/logout-button";
import { NewGameDialog } from "@/components/new-game-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { getSession } from "@/lib/auth";
import { db, schema } from "@/lib/db";

export const dynamic = "force-dynamic";

async function recentGames(userId: string) {
  if (!db) return [];
  return db
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
    .where(eq(schema.games.userId, userId))
    .orderBy(desc(schema.games.createdAt))
    .limit(30);
}

export default async function Home() {
  const session = await getSession();
  if (!session) redirect("/login");
  const games = await recentGames(session.userId).catch(() => []);

  return (
    <div className="flex-1 mx-auto w-full max-w-6xl px-6 py-12">
      <header className="flex items-end justify-between mb-10">
        <div>
          <div className="text-xs text-muted-foreground uppercase tracking-widest font-mono">
            console::compile.recorder
          </div>
          <h1 className="text-4xl font-semibold tracking-tight mt-1">
            Compile · Play & Record
          </h1>
          <p className="text-muted-foreground mt-1 text-sm max-w-xl">
            Play against a bot, or transcribe a live game against another player
            for later AI review.
          </p>
          <div className="mt-3 text-xs font-mono text-muted-foreground flex items-center gap-2">
            <span>signed in as</span>
            <span className="text-foreground">{session.username}</span>
            <LogoutButton />
          </div>
        </div>
        <NewGameDialog />
      </header>

      <Separator className="mb-8" />

      <section>
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="text-lg font-medium">Saved games</h2>
          <div className="text-xs text-muted-foreground font-mono">
            {games.length} record{games.length === 1 ? "" : "s"}
          </div>
        </div>
        {games.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="py-12 text-center text-muted-foreground text-sm">
              No games yet. Start one with the &quot;New game&quot; button.
              {!db && (
                <div className="text-amber-500/90 mt-3 text-xs font-mono">
                  Note: database not configured. Set POSTGRES_URL or DATABASE_URL.
                </div>
              )}
            </CardContent>
          </Card>
        ) : (
          <ScrollArea className="max-h-[60vh]">
            <ul className="grid gap-3 md:grid-cols-2">
              {games.map((g) => {
                const finished = g.endedAt != null;
                const winnerLabel =
                  g.winner == null ? null : g.winner === 0 ? g.player0Label : g.player1Label;
                return (
                  <li key={g.id} className="relative group">
                    <Link href={`/games/${g.id}`} className="block">
                      <Card className="transition-colors group-hover:bg-accent/50">
                        <CardHeader className="pb-2">
                          <div className="flex items-center justify-between gap-3">
                            <div className="font-mono text-xs text-muted-foreground">
                              {new Date(g.createdAt).toLocaleString()}
                            </div>
                            {finished ? (
                              <Badge variant="secondary" className="text-[10px]">
                                {winnerLabel ? `${winnerLabel} won` : "Draw"}
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="text-[10px]">in progress</Badge>
                            )}
                          </div>
                        </CardHeader>
                        <CardContent className="pt-0">
                          <div className="text-sm flex items-center gap-2">
                            {g.mode === "record" && (
                              <Badge variant="destructive" className="text-[10px] uppercase">recording</Badge>
                            )}
                            <span className="font-medium">{g.player0Label}</span>
                            {g.bot0Strategy ? <span className="text-muted-foreground text-xs"> (bot)</span> : null}
                            <span className="text-muted-foreground"> vs </span>
                            <span className="font-medium">{g.player1Label}</span>
                            {g.bot1Strategy ? <span className="text-muted-foreground text-xs"> (bot)</span> : null}
                          </div>
                          <div className="mt-2 flex gap-2 text-xs text-muted-foreground">
                            {g.includeExpansion ? (
                              <Badge variant="secondary" className="text-[10px]">+expansion</Badge>
                            ) : null}
                            {g.turnCount != null ? <span className="font-mono">turn {g.turnCount}</span> : null}
                          </div>
                        </CardContent>
                      </Card>
                    </Link>
                    {/* Sits on top of the link via absolute positioning, with
                        pointer events on; its onClick stops propagation so
                        clicking 'delete' doesn't navigate to the game. */}
                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <DeleteGameButton gameId={g.id} variant="card" />
                    </div>
                  </li>
                );
              })}
            </ul>
          </ScrollArea>
        )}
      </section>

      <footer className="mt-16 text-xs text-muted-foreground font-mono flex items-center gap-4">
        <span>compile-recorder · v0.1</span>
        <Separator orientation="vertical" className="h-3" />
        <Link href="/" className="hover:text-foreground">home</Link>
        <Link href="https://github.com/KeshavVarad/CompileAgent" target="_blank" className="hover:text-foreground">
          source ↗
        </Link>
        <Link href="https://shop.greaterthangames.com/pages/compile" target="_blank" className="hover:text-foreground">
          game rules ↗
        </Link>
      </footer>
    </div>
  );
}
