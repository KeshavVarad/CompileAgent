"use client";

import { useCallback, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { CURRENT_BOT } from "@/lib/bot-config";
import type { Action } from "@/lib/compile/types";

type EvalEntry = {
  step: number;
  turn: number;
  phase: string;
  decider: 0 | 1;
  actual: Action;
  bot: Action;
  value: number;
  topProbabilities: { actionIndex: number; prob: number; action: Action }[];
  agree: boolean;
};

type EvalResponse = {
  id: string;
  perspective: 0 | 1;
  entries: EvalEntry[];
  summary: {
    decisions: number;
    agreementRate: number;
    avgBotValue: number;
    finalWinner: 0 | 1 | null;
  };
};

function describeAction(a: Action): string {
  if (a.type === "DRAFT_PROTOCOL") return `Draft ${a.protocol}`;
  if (a.type === "PLAY_FACE_UP") return `Play hand[${a.handIndex}] face-up → L${(a.lineIndex ?? -1) + 1}`;
  if (a.type === "PLAY_FACE_DOWN") return `Play hand[${a.handIndex}] face-down → L${(a.lineIndex ?? -1) + 1}`;
  if (a.type === "DISCARD_CARD") return `Discard hand[${a.handIndex}]`;
  if (a.type === "REFRESH") return "Refresh";
  if (a.type === "COMPILE_LINE") return `Compile L${(a.lineIndex ?? -1) + 1}`;
  if (a.type === "SHIFT_OWN_CARD") return `Shift line[${(a.lineIndex ?? -1) + 1}].pos[${a.handIndex}] → L${(a.choiceIndex ?? -1) + 1}`;
  if (a.type === "CHOOSE_TARGET") return `Choose option #${(a.choiceIndex ?? -1) + 1}`;
  if (a.type === "SKIP_OPTIONAL") return "Skip";
  return a.type;
}

function valueBar(v: number): { tone: string; pct: number } {
  // v ∈ [-1, +1]. Map to a 0..100 bar; tone shifts emerald/red.
  const pct = Math.max(0, Math.min(100, (v + 1) * 50));
  const tone = v > 0.15 ? "bg-emerald-500/70" : v < -0.15 ? "bg-red-500/70" : "bg-amber-500/70";
  return { tone, pct };
}

export function EvalDialog({ gameId }: { gameId: string }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvalResponse | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/games/${gameId}/eval`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast.error(err.error || `Eval failed (${res.status})`);
        return;
      }
      const data: EvalResponse = await res.json();
      setResult(data);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Eval failed");
    } finally {
      setLoading(false);
    }
  }, [gameId]);

  function openDialog() {
    setOpen(true);
    if (!result) void run();
  }

  return (
    <>
      <Button variant="outline" size="sm" onClick={openDialog} className="text-[11px]">
        AI eval (vs {CURRENT_BOT.displayLabel})
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-[860px] max-h-[88vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>AI review</DialogTitle>
            <DialogDescription>
              {CURRENT_BOT.displayLabel} re-played every decision you made from your seat
              and shows what it would have done instead.
            </DialogDescription>
          </DialogHeader>

          {loading && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              Replaying decisions through the bot… (one inference per choice)
            </div>
          )}

          {!loading && result && (
            <>
              <SummaryRow result={result} />
              <div className="space-y-2 mt-2">
                {result.entries.length === 0 ? (
                  <div className="text-sm text-muted-foreground py-4">
                    No real decisions in this game yet (forced moves are skipped).
                  </div>
                ) : (
                  result.entries.map((e) => <EvalRow key={e.step} entry={e} />)
                )}
              </div>
              <div className="mt-4 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => { setResult(null); void run(); }}>
                  re-run
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

function SummaryRow({ result }: { result: EvalResponse }) {
  const { summary } = result;
  const bar = valueBar(summary.avgBotValue);
  return (
    <div className="rounded-lg border bg-card/50 px-3 py-2 grid grid-cols-3 gap-3 text-xs font-mono">
      <div>
        <div className="text-muted-foreground">decisions</div>
        <div className="text-lg font-semibold">{summary.decisions}</div>
      </div>
      <div>
        <div className="text-muted-foreground">agreement with bot</div>
        <div className="text-lg font-semibold">{(summary.agreementRate * 100).toFixed(0)}%</div>
      </div>
      <div>
        <div className="text-muted-foreground">avg bot value</div>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 rounded bg-muted/40 overflow-hidden">
            <div className={`${bar.tone} h-full`} style={{ width: `${bar.pct}%` }} />
          </div>
          <span className="text-[11px]">{summary.avgBotValue.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}

function EvalRow({ entry }: { entry: EvalEntry }) {
  const bar = valueBar(entry.value);
  return (
    <div className={`rounded border px-3 py-2 text-xs ${entry.agree ? "border-emerald-500/30" : "border-amber-500/30"}`}>
      <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground">
        <span>step {entry.step} · turn {entry.turn} · {entry.phase.toLowerCase().replace("_", " ")}</span>
        <span>{entry.agree ? "✓ matches bot" : "△ differs"}</span>
      </div>
      <div className="mt-1 grid grid-cols-[1fr_1fr] gap-2">
        <div>
          <div className="text-[10px] text-muted-foreground">you</div>
          <div className="text-sm">{describeAction(entry.actual)}</div>
        </div>
        <div>
          <div className="text-[10px] text-muted-foreground">bot suggested</div>
          <div className="text-sm">{describeAction(entry.bot)}</div>
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <div className="flex-1 h-1.5 rounded bg-muted/40 overflow-hidden">
          <div className={`${bar.tone} h-full`} style={{ width: `${bar.pct}%` }} />
        </div>
        <span className="text-[10px] font-mono text-muted-foreground w-20 text-right">
          value {entry.value.toFixed(2)}
        </span>
      </div>
      {!entry.agree && entry.topProbabilities.length > 0 && (
        <div className="mt-2 text-[10px] text-muted-foreground">
          top picks:{" "}
          {entry.topProbabilities.slice(0, 3).map((t, i) => (
            <span key={i} className="font-mono">
              {describeAction(t.action)} ({(t.prob * 100).toFixed(0)}%){i < 2 ? " · " : ""}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
