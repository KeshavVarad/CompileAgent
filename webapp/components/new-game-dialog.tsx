"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { CURRENT_BOT } from "@/lib/bot-config";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";

type Mode = "play-vs-bot" | "record";

export function NewGameDialog() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [pending, startTransition] = useTransition();
  const [mode, setMode] = useState<Mode>("play-vs-bot");
  const [recorderSeat, setRecorderSeat] = useState<"0" | "1">("0");
  const [includeExpansion, setIncludeExpansion] = useState(false);
  const [includeMain2, setIncludeMain2] = useState(false);
  const [includeAux2, setIncludeAux2] = useState(false);
  const [player0, setPlayer0] = useState("You");
  const [player1, setPlayer1] = useState("Opponent");

  function submit() {
    startTransition(async () => {
      try {
        const isRecord = mode === "record";
        const seatIdx = isRecord ? Number(recorderSeat) : -1;
        const labelMine = player0;
        const labelTheirs = player1;
        const body = isRecord
          ? {
              mode: "record",
              recorderSeat: seatIdx,
              includeExpansion,
              includeMain2,
              includeAux2,
              // Recorder sits on seat `recorderSeat`. The other seat is the
              // opposing player whose hidden info we don't have.
              player0Label: seatIdx === 0 ? labelMine : labelTheirs,
              player1Label: seatIdx === 0 ? labelTheirs : labelMine,
              bot0Strategy: null,
              bot1Strategy: null,
            }
          : {
              mode: "play",
              includeExpansion,
              includeMain2,
              includeAux2,
              player0Label: player0,
              player1Label: CURRENT_BOT.displayLabel,
              bot0Strategy: null,
              bot1Strategy: CURRENT_BOT.id,
            };
        const res = await fetch("/api/games", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          toast.error(err.error || `Failed (${res.status})`);
          return;
        }
        const { id } = await res.json();
        setOpen(false);
        router.push(`/games/${id}`);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Failed to create game");
      }
    });
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger
        className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-6"
      >
        New game
      </DialogTrigger>
      <DialogContent className="sm:max-w-[460px]">
        <DialogHeader>
          <DialogTitle>Start a new game</DialogTitle>
          <DialogDescription>
            Play against a bot, or record a live game for later AI review.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-2">
          <div className="grid gap-2">
            <Label htmlFor="mode">Mode</Label>
            <Select value={mode} onValueChange={(v) => setMode(v as Mode)}>
              <SelectTrigger id="mode"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="play-vs-bot">Play vs {CURRENT_BOT.displayLabel}</SelectItem>
                <SelectItem value="record">Record a live game</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {mode === "record" && (
            <div className="grid gap-2">
              <Label htmlFor="seat">You are…</Label>
              <Select value={recorderSeat} onValueChange={(v) => setRecorderSeat(v as "0" | "1")}>
                <SelectTrigger id="seat"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Player 1 (you draft first)</SelectItem>
                  <SelectItem value="1">Player 2 (you draft second)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-[11px] text-muted-foreground">
                Compile isn&apos;t perfect-info after the draft. Your own hand is fully known
                to the engine; opponent face-down plays are tracked as placeholders until
                an effect reveals them.
              </p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-2">
              <Label htmlFor="p0">Your name</Label>
              <Input id="p0" value={player0} onChange={(e) => setPlayer0(e.target.value)} />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="p1">Opponent name</Label>
              <Input
                id="p1"
                value={mode === "play-vs-bot" ? "Bot (random)" : player1}
                onChange={(e) => setPlayer1(e.target.value)}
                disabled={mode === "play-vs-bot"}
              />
            </div>
          </div>

          <div className="grid gap-1.5 text-sm">
            <label className="flex items-center gap-2">
              <input
                type="checkbox" className="size-4 rounded border-border"
                checked={includeExpansion}
                onChange={(e) => setIncludeExpansion(e.target.checked)}
              />
              Aux 1 (Apathy / Hate / Love)
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox" className="size-4 rounded border-border"
                checked={includeMain2}
                onChange={(e) => setIncludeMain2(e.target.checked)}
              />
              Main 2 (Chaos / Clarity / Corruption / Courage / Fear / Ice / Luck / Mirror / Peace / Smoke / Time / War)
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox" className="size-4 rounded border-border"
                checked={includeAux2}
                onChange={(e) => setIncludeAux2(e.target.checked)}
              />
              Aux 2 (Assimilation / Diversity / Unity)
            </label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => setOpen(false)} disabled={pending}>Cancel</Button>
          <Button onClick={submit} disabled={pending}>{pending ? "Creating..." : "Start"}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
