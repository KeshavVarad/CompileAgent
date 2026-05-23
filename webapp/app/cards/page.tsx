import Link from "next/link";
import { redirect } from "next/navigation";

import { CardsBrowser } from "@/components/cards-browser";
import { Separator } from "@/components/ui/separator";
import { getSession } from "@/lib/auth";

export const dynamic = "force-dynamic";

export default async function CardsPage() {
  const session = await getSession();
  if (!session) redirect("/login");

  return (
    <div className="flex-1 mx-auto w-full max-w-[min(112rem,94vw)] px-6 py-12">
      <header className="flex items-end justify-between mb-8">
        <div>
          <div className="text-xs text-muted-foreground uppercase tracking-widest font-mono">
            console::cards.browser
          </div>
          <h1 className="text-4xl font-semibold tracking-tight mt-1">
            Card reference
          </h1>
          <p className="text-muted-foreground mt-1 text-sm max-w-xl">
            Every card the engine knows about. Click any card to see the full
            text uncropped and file a bug report against it if the in-game
            behavior doesn&apos;t match.
          </p>
        </div>
        <Link
          href="/"
          className="text-xs font-mono text-muted-foreground hover:text-foreground"
        >
          ← back to games
        </Link>
      </header>

      <Separator className="mb-6" />

      <CardsBrowser />
    </div>
  );
}
