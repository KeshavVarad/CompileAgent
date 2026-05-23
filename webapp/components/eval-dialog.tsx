"use client";

import { useCallback, useEffect, useState } from "react";
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
  actualProb: number;
  actualRank: number;
  topProb: number;
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
  /** Set on cached GET responses; absent on a fresh POST. */
  actionCount?: number;
  currentActionCount?: number;
  stale?: boolean;
  computedAt?: string;
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

/**
 * Classify the user's move relative to the bot's policy. The bot is
 * stochastic — many decisions have multiple reasonable answers — so we
 * avoid binary "right/wrong" framing and instead say where the user's
 * choice sat in the model's distribution.
 *
 * - `top` (prob in the top 25% of mass with rank 0): the model's modal pick
 * - `reasonable` (rank 0-2 OR prob ≥ 15%): a co-favored option
 * - `unusual` (prob 1-15%): the model would have considered something else
 * - `rare` (prob < 1%): a clearly off-policy line; the most likely "blunder" signal
 */
function classifyMove(e: EvalEntry): { kind: "top" | "reasonable" | "unusual" | "rare"; label: string; tone: string } {
  if (e.agree) return { kind: "top", label: "model's top pick", tone: "text-emerald-400" };
  if (e.actualRank >= 0 && e.actualRank <= 2 && e.actualProb >= 0.15)
    return { kind: "reasonable", label: "co-favored", tone: "text-emerald-300/80" };
  if (e.actualProb >= 0.05)
    return { kind: "unusual", label: "lower-probability", tone: "text-amber-400" };
  return { kind: "rare", label: "off-policy", tone: "text-red-400" };
}

function timeAgo(iso?: string): string | null {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  const ms = Date.now() - then;
  if (ms < 0) return "just now";
  const m = Math.floor(ms / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

export function EvalDialog({ gameId }: { gameId: string }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvalResponse | null>(null);
  // Tracks whether the result on screen came from cache (GET) vs a fresh
  // recompute (POST). Affects the "last reviewed: …" affordance.
  const [fromCache, setFromCache] = useState(false);

  const loadCached = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch(`/api/games/${gameId}/eval`, {
        method: "GET",
        cache: "no-store",
      });
      if (res.status === 204) return false;
      if (!res.ok) return false;
      const data: EvalResponse = await res.json();
      setResult(data);
      setFromCache(true);
      return true;
    } catch {
      return false;
    }
  }, [gameId]);

  const recompute = useCallback(async () => {
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
      setFromCache(false);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Eval failed");
    } finally {
      setLoading(false);
    }
  }, [gameId]);

  // On open: prefer cached. If none, kick off a fresh compute.
  useEffect(() => {
    if (!open || result || loading) return;
    void (async () => {
      const had = await loadCached();
      if (!had) await recompute();
    })();
  }, [open, result, loading, loadCached, recompute]);

  return (
    <>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)} className="text-[11px]">
        AI review (vs {CURRENT_BOT.displayLabel})
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-[860px] max-h-[88vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>AI review</DialogTitle>
            <DialogDescription className="space-y-1">
              <span className="block">
                {CURRENT_BOT.displayLabel} re-played each of your decisions from your seat
                and reports the policy distribution over the legal moves.
              </span>
              <span className="block text-amber-400/90 text-[11px]">
                Heads-up: the model's policy is <strong>stochastic</strong> — for many
                decisions multiple options are reasonable. A move flagged as
                &ldquo;lower-probability&rdquo; or &ldquo;off-policy&rdquo; is the model's
                read, not a verdict. Sparkv3 is a strong amateur, not an oracle, and is
                still wrong on plenty of positions.
              </span>
            </DialogDescription>
          </DialogHeader>

          {loading && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              Replaying decisions through the bot… (one inference per choice)
            </div>
          )}

          {!loading && result && (
            <>
              <SummaryRow result={result} fromCache={fromCache} />
              {result.stale && (
                <div className="mt-2 rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-[11px]">
                  This review was computed when the game had{" "}
                  <span className="font-mono">{result.actionCount}</span> actions; it now
                  has <span className="font-mono">{result.currentActionCount}</span>.
                  Click <span className="font-semibold">re-run</span> below for an updated
                  review of the latest moves.
                </div>
              )}
              <div className="space-y-2 mt-2">
                {result.entries.length === 0 ? (
                  <div className="text-sm text-muted-foreground py-4">
                    No real decisions in this game yet (forced moves are skipped).
                  </div>
                ) : (
                  result.entries.map((e) => <EvalRow key={e.step} entry={e} />)
                )}
              </div>
              <div className="mt-4 flex items-center justify-between gap-3">
                <span className="text-[11px] text-muted-foreground">
                  {fromCache && result.computedAt ? (
                    <>last reviewed {timeAgo(result.computedAt)}</>
                  ) : (
                    "fresh review"
                  )}
                </span>
                <Button variant="ghost" size="sm" onClick={() => void recompute()}>
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

function SummaryRow({ result, fromCache }: { result: EvalResponse; fromCache: boolean }) {
  const { summary } = result;
  const bar = valueBar(summary.avgBotValue);
  return (
    <div className="rounded-lg border bg-card/50 px-3 py-2 grid grid-cols-3 gap-3 text-xs font-mono">
      <div>
        <div className="text-muted-foreground">decisions reviewed</div>
        <div className="text-lg font-semibold">{summary.decisions}</div>
        {fromCache && (
          <div className="text-[9px] text-muted-foreground/60">cached</div>
        )}
      </div>
      <div>
        <div className="text-muted-foreground">picks matching model's mode</div>
        <div className="text-lg font-semibold">{(summary.agreementRate * 100).toFixed(0)}%</div>
        <div className="text-[9px] text-muted-foreground/60">
          modal agreement, not correctness
        </div>
      </div>
      <div>
        <div className="text-muted-foreground">avg model value (your seat)</div>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 rounded bg-muted/40 overflow-hidden">
            <div className={`${bar.tone} h-full`} style={{ width: `${bar.pct}%` }} />
          </div>
          <span className="text-[11px]">{summary.avgBotValue.toFixed(2)}</span>
        </div>
        <div className="text-[9px] text-muted-foreground/60">
          range −1 (losing) … +1 (winning)
        </div>
      </div>
    </div>
  );
}

function EvalRow({ entry }: { entry: EvalEntry }) {
  const bar = valueBar(entry.value);
  const verdict = classifyMove(entry);
  const borderTone =
    verdict.kind === "top" ? "border-emerald-500/30"
      : verdict.kind === "reasonable" ? "border-emerald-300/20"
      : verdict.kind === "unusual" ? "border-amber-500/30"
      : "border-red-500/30";
  return (
    <div className={`rounded border px-3 py-2 text-xs ${borderTone}`}>
      <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground">
        <span>step {entry.step} · turn {entry.turn} · {entry.phase.toLowerCase().replace("_", " ")}</span>
        <span className={verdict.tone}>{verdict.label}</span>
      </div>
      <div className="mt-1 grid grid-cols-[1fr_1fr] gap-2">
        <div>
          <div className="text-[10px] text-muted-foreground">
            you played{" "}
            <span className="font-mono">({(entry.actualProb * 100).toFixed(1)}% model prob)</span>
          </div>
          <div className="text-sm">{describeAction(entry.actual)}</div>
        </div>
        <div>
          <div className="text-[10px] text-muted-foreground">
            model's top pick{" "}
            <span className="font-mono">({(entry.topProb * 100).toFixed(1)}%)</span>
          </div>
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
          full top-k:{" "}
          {entry.topProbabilities.slice(0, 4).map((t, i) => (
            <span key={i} className="font-mono">
              {describeAction(t.action)} <span className="text-muted-foreground/60">({(t.prob * 100).toFixed(0)}%)</span>{i < 3 ? " · " : ""}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
