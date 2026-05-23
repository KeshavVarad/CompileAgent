"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import type { CardDef } from "@/lib/compile/cards";
import type { GameView } from "@/lib/view";

/**
 * Bug / feature reporter. No backend, no auth token — clicking
 * "Open issue on GitHub" opens a pre-filled new-issue URL in a new tab.
 * GitHub handles auth + posting from there. The pre-fill includes a
 * structured game-context block when the user is on a game page so the
 * maintainer can reproduce without round-tripping for IDs.
 *
 * If you fork this repo, change REPO_URL below to point at your fork.
 */
const REPO_URL = "https://github.com/KeshavVarad/CompileAgent";

type ReportKind = "card-effect" | "bug" | "feature" | "other";

const KIND_OPTIONS: Array<{ value: ReportKind; label: string; labels: string[] }> = [
  { value: "card-effect", label: "Card effect is wrong", labels: ["bug", "card-effect"] },
  { value: "bug", label: "General bug", labels: ["bug"] },
  { value: "feature", label: "Feature request / suggestion", labels: ["enhancement"] },
  { value: "other", label: "Other", labels: ["triage"] },
];

const TITLE_PREFIX: Record<ReportKind, string> = {
  "card-effect": "[Card effect] ",
  bug: "[Bug] ",
  feature: "[Feature] ",
  other: "[Report] ",
};

function buildCardContextMarkdown(card: CardDef | null): string {
  if (!card) return "";
  const tier = (label: string, emp: string, text: string) => {
    if (!emp && !text) return `- **${label}**: —`;
    const emphPart = emp ? `*${emp}* ` : "";
    return `- **${label}**: ${emphPart}${text || "—"}`;
  };
  return [
    "### Card",
    "",
    `- **Card**: ${card.protocol} ${card.value}`,
    `- **Key**: \`${card.key}\``,
    `- **Set**: ${card.setCode}`,
    tier("Top", card.topEmphasis, card.topText),
    tier("Middle", card.middleEmphasis, card.middleText),
    tier("Bottom", card.bottomEmphasis, card.bottomText),
  ].join("\n");
}

function buildGameContextMarkdown(view: GameView | null, gameId: string | null): string {
  if (!view || !gameId) return "";
  const last = view.history.length > 0 ? view.history[view.history.length - 1] : null;
  const lastLabel = last ? `${last.text}${last.actor != null ? ` (P${last.actor + 1})` : ""}` : "—";
  const sets: string[] = ["MN01"];
  if (view.config.includeExpansion) sets.push("AX01");
  // The view doesn't expose includeMain2/Aux2 directly today but that's
  // OK — the seed + game ID are sufficient for a maintainer to fetch the
  // exact state. We surface what the view *does* expose.
  return [
    "### Game context",
    "",
    `- **Game ID**: \`${gameId}\``,
    `- **Seed**: \`${view.config.seed}\``,
    `- **Sets enabled**: ${sets.join(", ")}`,
    `- **Mode**: ${view.config.mode}`,
    `- **Turn**: ${view.turn}`,
    `- **Phase**: ${view.phase}`,
    `- **Current player**: P${view.currentPlayer + 1}`,
    `- **Last log entry**: ${lastLabel}`,
    `- **P1 protocols**: ${view.lines.map((l) => l.p0Protocol ?? "?").join(" / ")}`,
    `- **P2 protocols**: ${view.lines.map((l) => l.p1Protocol ?? "?").join(" / ")}`,
  ].join("\n");
}

export function BugReportDialog({
  view = null,
  gameId = null,
  card = null,
  triggerLabel,
  triggerClassName,
}: {
  view?: GameView | null;
  gameId?: string | null;
  /** When provided, the dialog opens pre-targeted at this card: kind is
   *  forced to "card-effect", the title is seeded with the card name, and
   *  a structured card stanza is appended to the body. Used by the card
   *  browser ([app/cards]) so playtesters can flag a specific card without
   *  re-typing identity details. */
  card?: CardDef | null;
  triggerLabel?: string;
  triggerClassName?: string;
}) {
  const [open, setOpen] = useState(false);
  const [kind, setKind] = useState<ReportKind>(card ? "card-effect" : "card-effect");
  const [title, setTitle] = useState(card ? `${card.protocol} ${card.value} — ` : "");
  const [body, setBody] = useState("");
  const [includeContext, setIncludeContext] = useState(true);

  const kindOpt = KIND_OPTIONS.find((k) => k.value === kind) ?? KIND_OPTIONS[0];
  const canSubmit = title.trim().length > 0;

  function openIssue() {
    const gameCtx = view && gameId && includeContext ? buildGameContextMarkdown(view, gameId) : "";
    const cardCtx = card ? buildCardContextMarkdown(card) : "";
    const fullBody = [
      body.trim(),
      "",
      cardCtx,
      gameCtx,
      "",
      "<sub>Filed via the in-app reporter. Game state, if attached above, is",
      "reproducible via the seed + actions log (visit /api/games/&lt;id&gt;).</sub>",
    ].filter(Boolean).join("\n");
    const params = new URLSearchParams({
      title: `${TITLE_PREFIX[kind]}${title.trim()}`,
      body: fullBody,
      labels: kindOpt.labels.join(","),
    });
    const url = `${REPO_URL}/issues/new?${params.toString()}`;
    window.open(url, "_blank", "noopener,noreferrer");
    setOpen(false);
    // Clear so the next open starts blank — the GitHub tab carries the
    // unsubmitted draft, so we don't lose anything by resetting locally.
    // (For card-prefilled dialogs we re-seed the title on next open via
    // the useState initializer, which is stable across this single mount.)
    setTitle(card ? `${card.protocol} ${card.value} — ` : "");
    setBody("");
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(true)}
        className={triggerClassName ?? "text-[11px]"}
      >
        {triggerLabel ?? "Report bug / feature"}
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-[560px]">
          <DialogHeader>
            <DialogTitle>Report bug or feature</DialogTitle>
            <DialogDescription>
              Opens a pre-filled GitHub issue in a new tab. You&apos;ll review the
              text and click <span className="font-semibold">Submit new issue</span> on
              GitHub to file it. Sign-in to GitHub is required there, but no token
              is needed here.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 mt-2">
            <div className="space-y-1.5">
              <Label htmlFor="kind">Type</Label>
              <Select value={kind} onValueChange={(v) => setKind(v as ReportKind)}>
                <SelectTrigger id="kind"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {KIND_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="title">Title (one line)</Label>
              <Input
                id="title"
                placeholder={
                  kind === "card-effect"
                    ? "e.g. Mirror 1 not blocked by my own Fear 0"
                    : "Short summary"
                }
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                maxLength={180}
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="body">
                Details{" "}
                <span className="text-muted-foreground text-[11px] font-normal">
                  (what happened, what you expected, repro steps)
                </span>
              </Label>
              <textarea
                id="body"
                className="w-full min-h-[120px] rounded-lg border border-input bg-transparent px-2.5 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30 font-mono"
                placeholder={
                  kind === "card-effect"
                    ? "Card name + the specific behavior you saw vs what you expected per the rulebook…"
                    : "What you saw, what you expected, anything else useful…"
                }
                value={body}
                onChange={(e) => setBody(e.target.value)}
              />
            </div>

            {view && gameId && (
              <label className="flex items-start gap-2 text-[12px] cursor-pointer">
                <input
                  type="checkbox"
                  className="mt-1 h-4 w-4 rounded border-input"
                  checked={includeContext}
                  onChange={(e) => setIncludeContext(e.target.checked)}
                />
                <span>
                  Attach game context (game ID, seed, turn, protocols, last log
                  entry).{" "}
                  <span className="text-muted-foreground">
                    Helps the maintainer reproduce. No passwords or hand contents
                    are included.
                  </span>
                </span>
              </label>
            )}

            <div className="flex justify-end gap-2 pt-1">
              <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <Button size="sm" onClick={openIssue} disabled={!canSubmit}>
                Open issue on GitHub
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
