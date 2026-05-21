"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";

/**
 * Tiny "delete" button used on game cards on the home page and on the game
 * page itself. Two-click pattern: opens a confirmation dialog before
 * actually firing the DELETE — accidental clicks on a list of games are
 * an easy way to lose work.
 */
export function DeleteGameButton({
  gameId,
  variant = "card",
  onDeleted,
}: {
  gameId: string;
  /** "card" (compact, inline on a list row) or "page" (label visible, used
   *  on the game-detail page footer). */
  variant?: "card" | "page";
  /** Optional callback after a successful delete — if not provided, we just
   *  router.refresh() in place. The game-detail page uses this to push the
   *  user back home. */
  onDeleted?: () => void;
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [pending, startTransition] = useTransition();

  function submit() {
    startTransition(async () => {
      try {
        const res = await fetch(`/api/games/${gameId}`, { method: "DELETE" });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          toast.error(err.error || `Delete failed (${res.status})`);
          return;
        }
        setOpen(false);
        if (onDeleted) onDeleted();
        else router.refresh();
        toast.success("Game deleted");
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Delete failed");
      }
    });
  }

  // The trigger button lives outside Dialog so we can stop event propagation
  // when it's nested inside a clickable card row (otherwise the click would
  // also navigate to the game page).
  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        className={
          variant === "card"
            ? "h-6 px-2 text-[10px] text-muted-foreground hover:text-destructive"
            : "h-7 text-[11px] text-destructive hover:text-destructive"
        }
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOpen(true);
        }}
        disabled={pending}
      >
        {variant === "card" ? "delete" : "delete game"}
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-[420px]">
          <DialogHeader>
            <DialogTitle>Delete this game?</DialogTitle>
            <DialogDescription>
              The game record, action log, and any AI-eval data are removed
              permanently. This can&apos;t be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setOpen(false)}
              disabled={pending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={submit}
              disabled={pending}
            >
              {pending ? "..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
