"use client";

import { useCallback, useEffect, useMemo, useState, useTransition } from "react";
import Link from "next/link";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { CARD_OVERLAP_PX, PlayCard } from "@/components/play-card";
import { DeleteGameButton } from "@/components/delete-game-button";
import { EvalDialog } from "@/components/eval-dialog";
import { defsForProtocol, type Protocol } from "@/lib/compile/cards";
import type { Action } from "@/lib/compile/types";
import type { ActionView, CardView, GameView, LineView } from "@/lib/view";

type Selection =
  | { kind: "hand"; handIndex: number }
  | { kind: "board"; lineIndex: number; stackIndex: number }
  | { kind: "opp-hand"; handIndex: number };

type Props = { gameId: string; initialView: GameView; initialTotalActions: number };

type AnalysisSnapshot = {
  view: GameView;
  index: number;
  totalActions: number;
  lastAction: Action | null;
  lastLabel: string | null;
  lastEvents: string[];
  actor: 0 | 1 | null;
};

type IdentityRequest = {
  /** The action the recorder picked but hasn't submitted yet — it's missing
   *  the identity (revealedDefId) of the placeholder hand card it references. */
  action: ActionView;
  /** Which seat owns the hand card whose identity we're picking. Used to
   *  filter candidate cards to that player's drafted protocols. */
  forSeat: 0 | 1;
};

// Two-phase animation timing for the bot's chain. Each step gets:
//   1. ANNOUNCE: banner shows "Sparkv2 plays X face-up in line 2" before
//      anything visible changes on the board. Holds for `announceMs` so
//      the player has time to read the announcement and look at the
//      current board.
//   2. APPLY: the board updates to the post-step view. Holds for
//      `settleMs` so the player can see what changed, then the next
//      step begins.
// Tuned for "immersive, not instantaneous" — a typical bot turn (1–3
// steps) now takes 4–8 seconds instead of <2. Long chains are still
// bounded by the cap-per-turn budget.
const BOT_STEP_INITIAL_DELAY_MS = 700; // pause after user's action before bot starts
const BOT_END_OF_CHAIN_MS = 2200;       // hold the last announcement before banner fades

type StepTiming = {
  /** How long the announce label sits on screen BEFORE the board
   *  updates to the post-step view. Lets the player read what's about
   *  to happen. */
  announceMs: number;
  /** How long the board sits in the post-step state AFTER the apply
   *  before the next step's announce begins. Lets the player see what
   *  changed. */
  settleMs: number;
};

function timingFor(action: Action): StepTiming {
  switch (action.type) {
    case "COMPILE_LINE":
      return { announceMs: 1400, settleMs: 1500 }; // most dramatic — line clears
    case "PLAY_FACE_UP":
      return { announceMs: 1200, settleMs: 1200 }; // big visual reveal
    case "PLAY_FACE_DOWN":
      return { announceMs: 1000, settleMs: 900 };  // card appears face-down
    case "REFRESH":
      return { announceMs: 900, settleMs: 700 };   // hand refill, no board change
    case "DISCARD_CARD":
      return { announceMs: 900, settleMs: 700 };
    case "SHIFT_OWN_CARD":
      return { announceMs: 1000, settleMs: 900 };
    case "CHOOSE_TARGET":
    case "SKIP_OPTIONAL":
      return { announceMs: 800, settleMs: 600 };   // effect sub-decisions
    case "DRAFT_PROTOCOL":
      return { announceMs: 900, settleMs: 600 };
    default:
      return { announceMs: 1000, settleMs: 800 };
  }
}

/** Convert an imperative action label like "Play Plague 1 face-up in
 *  line 2" into present-tense announcer narration ("plays Plague 1
 *  face-up in line 2"). Keeps the agent's actions reading as a live
 *  play-by-play instead of a stiff command list. */
function toAnnouncerLabel(label: string): string {
  const m = label.match(/^([A-Z][a-z]+)\b(.*)$/);
  if (!m) return label;
  const [, verb, rest] = m;
  const lower = verb.toLowerCase();
  const present = (() => {
    if (lower === "play") return "plays";
    if (lower === "compile") return "compiles";
    if (lower === "refresh") return "refreshes";
    if (lower === "draft") return "drafts";
    if (lower === "discard") return "discards";
    if (lower === "shift") return "shifts";
    if (lower === "skip") return "skips";
    if (lower === "option") return "picks option"; // "Option 3" → "picks option 3"
    return lower; // fallback
  })();
  return `${present}${rest}`;
}

type BotStep = { action: Action; label: string; view: GameView; events: string[] };
type Frame =
  | { kind: "announce"; label: string; events: string[]; announceMs: number }
  | { kind: "apply"; view: GameView; settleMs: number };

export function GameClient({ gameId, initialView, initialTotalActions }: Props) {
  const [liveView, setLiveView] = useState<GameView>(initialView);
  const [totalActions, setTotalActions] = useState<number>(initialTotalActions);
  // Analysis mode lets the player scrub through past actions read-only.
  // `analysisIndex` is the snapshot position (0 .. totalActions); null
  // means we're at the live state and the player can act.
  const [analysisIndex, setAnalysisIndex] = useState<number | null>(null);
  const [analysisSnap, setAnalysisSnap] = useState<AnalysisSnapshot | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const inAnalysis = analysisIndex !== null;
  const [pending, startTransition] = useTransition();
  const [selection, setSelection] = useState<Selection | null>(null);
  const [identityReq, setIdentityReq] = useState<IdentityRequest | null>(null);
  // Currently-playing bot animation. While `botQueue` has items, the
  // client is mid-replay of Sparkv2's chain; user input should be
  // gated. `botLastAction` is the most recent bot action label to
  // surface on the status banner.
  // Queue of frames produced from bot steps. Each bot step yields two
  // frames: an `announce` (banner text appears, board unchanged) and an
  // `apply` (board updates to post-step view). The two-phase rhythm
  // gives the agent's actions a "narrated" feel instead of jumping
  // straight from one board state to the next.
  const [botFrames, setBotFrames] = useState<Frame[]>([]);
  const [botLastLabel, setBotLastLabel] = useState<string | null>(null);
  const [botLastEvents, setBotLastEvents] = useState<string[]>([]);
  const animating = botFrames.length > 0;

  // Drain frames one at a time:
  //   - announce frame: pop after a short startup pause (initial step
  //     only), set the banner label, then schedule the announce → apply
  //     transition after announceMs.
  //   - apply frame: hold for settleMs after the visual change so the
  //     player can see what landed, then advance.
  useEffect(() => {
    if (botFrames.length === 0) return;
    const next = botFrames[0];
    const rest = botFrames.slice(1);
    if (next.kind === "announce") {
      const initial = botLastLabel === null;
      const startup = initial ? BOT_STEP_INITIAL_DELAY_MS : 0;
      const t = setTimeout(() => {
        setBotLastLabel(next.label);
        setBotLastEvents(next.events);
        const tt = setTimeout(() => setBotFrames(rest), next.announceMs);
        // tt cleanup handled by the next render re-entering this effect.
        void tt;
      }, startup);
      return () => clearTimeout(t);
    } else {
      const t = setTimeout(() => {
        setLiveView(next.view);
        const tt = setTimeout(() => setBotFrames(rest), next.settleMs);
        void tt;
      }, 0);
      return () => clearTimeout(t);
    }
  }, [botFrames, botLastLabel]);

  // Clear the "last bot action" label a moment after the chain ends so
  // the banner returns to a normal turn-status message.
  useEffect(() => {
    if (botFrames.length > 0 || botLastLabel === null) return;
    const t = setTimeout(() => {
      setBotLastLabel(null);
      setBotLastEvents([]);
    }, BOT_END_OF_CHAIN_MS);
    return () => clearTimeout(t);
  }, [botFrames, botLastLabel]);

  const submit = useCallback(
    (action: Action) => {
      if (animating) return; // gate input during bot playback
      if (inAnalysis) return; // read-only while scrubbing history
      startTransition(async () => {
        setSelection(null);
        setIdentityReq(null);
        try {
          const res = await fetch(`/api/games/${gameId}/step`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ action }),
          });
          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            toast.error(err.error || `Step failed (${res.status})`);
            return;
          }
          const data = await res.json();
          const postUser = (data.postUserView ?? data.view) as GameView;
          const userEvents = (data.userEvents ?? []) as string[];
          const steps = (data.botSteps ?? []) as BotStep[];
          if (typeof data.totalActions === "number") setTotalActions(data.totalActions);
          // Show the user's-action result immediately, then let the
          // effect animate through the bot's chain. If the user's own
          // action emitted info events (e.g. their card text tried to
          // do something impossible), toast them so the player sees
          // why the visual change didn't match expectations.
          setLiveView(postUser);
          for (const e of userEvents) toast.message(e);
          setBotLastLabel(null);
          setBotLastEvents([]);
          // Expand each bot step into a (announce, apply) frame pair
          // so the banner narrates the move before the board updates.
          const frames: Frame[] = [];
          for (const s of steps) {
            const { announceMs, settleMs } = timingFor(s.action);
            frames.push({
              kind: "announce",
              label: toAnnouncerLabel(s.label),
              events: s.events ?? [],
              announceMs,
            });
            frames.push({ kind: "apply", view: s.view, settleMs });
          }
          setBotFrames(frames);
        } catch (e) {
          toast.error(e instanceof Error ? e.message : "Step failed");
        }
      });
    },
    [gameId, animating, inAnalysis],
  );

  // Fetch the snapshot whenever the player navigates to a new index.
  // Bounded by [0, totalActions]; clamping happens server-side too but
  // the client guards against out-of-range requests.
  useEffect(() => {
    if (analysisIndex === null) return;
    let cancelled = false;
    (async () => {
      setAnalysisLoading(true);
      try {
        const res = await fetch(`/api/games/${gameId}/snapshot?index=${analysisIndex}`);
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          if (!cancelled) toast.error(err.error || `Snapshot failed (${res.status})`);
          return;
        }
        const data = (await res.json()) as AnalysisSnapshot;
        if (cancelled) return;
        setAnalysisSnap(data);
        if (typeof data.totalActions === "number") setTotalActions(data.totalActions);
      } catch (e) {
        if (!cancelled) toast.error(e instanceof Error ? e.message : "Snapshot failed");
      } finally {
        if (!cancelled) setAnalysisLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [gameId, analysisIndex]);

  // Keyboard nav while in analysis mode: ←/→ step, Home/End jump,
  // Esc exits. Bound on window so it works regardless of focus.
  useEffect(() => {
    if (!inAnalysis) return;
    const onKey = (e: KeyboardEvent) => {
      // Don't hijack typing in inputs/textareas.
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setAnalysisIndex((i) => (i === null ? null : Math.max(0, i - 1)));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setAnalysisIndex((i) => (i === null ? null : Math.min(totalActions, i + 1)));
      } else if (e.key === "Home") {
        e.preventDefault();
        setAnalysisIndex(0);
      } else if (e.key === "End") {
        e.preventDefault();
        setAnalysisIndex(totalActions);
      } else if (e.key === "Escape") {
        e.preventDefault();
        setAnalysisIndex(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [inAnalysis, totalActions]);

  // In analysis mode we render the historical snapshot; the live state
  // remains untouched so submit() (disabled anyway) and the bot-chain
  // animation queue stay coherent. Until the first snapshot arrives we
  // keep showing the live view so the layout doesn't blink.
  const view: GameView = inAnalysis && analysisSnap ? analysisSnap.view : liveView;

  // Lock `me` to the seat the local user is actually sitting in:
  //   - record mode: recorderSeat
  //   - play mode: the human seat (the one without a bot)
  //   - fallback: the current decider
  // Previously `me = view.decider` worked because in live play the human
  // is always the decider (server auto-advances bots). In analysis mode
  // we may replay to mid-bot-chain states where the bot is the decider —
  // without this lock, the UI would flip perspective and show the bot's
  // hand. Locking to seat identity preserves the player's POV at every
  // step.
  const mode = view.config.mode ?? "play";
  const me: 0 | 1 = mode === "record" && view.config.recorderSeat != null
    ? (view.config.recorderSeat as 0 | 1)
    : view.config.human[0] ? 0
    : view.config.human[1] ? 1
    : view.decider;
  const oppSeat: 0 | 1 = me === 0 ? 1 : 0;
  const activeDecider = view.decider;
  const recordingOppTurn = mode === "record" && activeDecider !== me;

  const handIndicesWithActions = useMemo(() => {
    const set = new Set<number>();
    if (inAnalysis) return set; // read-only while scrubbing
    // In record mode while the opponent is the active decider, the legal
    // actions are *opponent* actions; the recorder's own hand cards have no
    // legal moves until the engine swings back. Don't highlight them.
    if (recordingOppTurn) return set;
    for (const a of view.legalActions) {
      if (
        (a.type === "PLAY_FACE_UP" || a.type === "PLAY_FACE_DOWN" || a.type === "DISCARD_CARD")
        && typeof a.handIndex === "number"
      ) {
        set.add(a.handIndex);
      }
    }
    return set;
  }, [view.legalActions, recordingOppTurn, inAnalysis]);

  const oppHandIndicesWithActions = useMemo(() => {
    const set = new Set<number>();
    if (inAnalysis) return set;
    if (!recordingOppTurn) return set;
    for (const a of view.legalActions) {
      if (
        (a.type === "PLAY_FACE_UP" || a.type === "PLAY_FACE_DOWN" || a.type === "DISCARD_CARD")
        && typeof a.handIndex === "number"
      ) {
        set.add(a.handIndex);
      }
    }
    return set;
  }, [view.legalActions, recordingOppTurn, inAnalysis]);

  const boardKeysWithActions = useMemo(() => {
    const set = new Set<string>();
    if (inAnalysis) return set;
    for (const a of view.legalActions) {
      if (a.type === "SHIFT_OWN_CARD" && typeof a.lineIndex === "number" && typeof a.handIndex === "number") {
        set.add(`${a.lineIndex}:${a.handIndex}`);
      }
    }
    return set;
  }, [view.legalActions, inAnalysis]);

  const selectionActions = useMemo<ActionView[]>(() => {
    if (!selection || inAnalysis) return [];
    if (selection.kind === "hand" || selection.kind === "opp-hand") {
      return view.legalActions.filter(
        (a) =>
          (a.type === "PLAY_FACE_UP" || a.type === "PLAY_FACE_DOWN" || a.type === "DISCARD_CARD")
          && a.handIndex === selection.handIndex,
      );
    }
    return view.legalActions.filter(
      (a) =>
        a.type === "SHIFT_OWN_CARD"
        && a.lineIndex === selection.lineIndex
        && a.handIndex === selection.stackIndex,
    );
  }, [selection, view.legalActions, inAnalysis]);

  const globalActions = useMemo<ActionView[]>(
    () => inAnalysis ? [] : view.legalActions.filter((a) => a.type === "REFRESH" || a.type === "COMPILE_LINE"),
    [view.legalActions, inAnalysis],
  );

  const canEnterAnalysis = !animating && !pending && totalActions > 0;

  return (
    <div className="flex-1 mx-auto w-full max-w-6xl px-6 py-6">
      <TopBar
        view={view}
        analysisOpen={inAnalysis}
        canEnterAnalysis={canEnterAnalysis}
        onEnterAnalysis={() => setAnalysisIndex(totalActions)}
      />

      {inAnalysis && (
        <AnalysisBar
          index={analysisIndex ?? 0}
          totalActions={totalActions}
          snap={analysisSnap}
          loading={analysisLoading}
          playerLabels={[view.config.player0Label, view.config.player1Label]}
          onIndex={(i) => setAnalysisIndex(Math.max(0, Math.min(totalActions, i)))}
          onExit={() => setAnalysisIndex(null)}
        />
      )}

      {view.draft ? (
        <DraftPanel view={view} onPick={(p) => submit({ type: "DRAFT_PROTOCOL", protocol: p as Protocol })} pending={pending} />
      ) : (
        <>
          <PlayerStrip view={view} side={oppSeat} isMe={false} />
          {mode === "record" && (
            <OppHandRow
              view={view}
              oppSeat={oppSeat}
              selection={selection}
              oppHandIndicesWithActions={oppHandIndicesWithActions}
              onSelect={(s) => setSelection(s)}
            />
          )}
          <Board
            view={view}
            mySeat={me}
            selection={selection}
            boardKeysWithActions={boardKeysWithActions}
            onSelect={(s) => setSelection(s)}
          />
          <TurnStatusBanner
            view={view}
            me={me}
            pending={pending}
            botActionLabel={botLastLabel}
            botEvents={botLastEvents}
            animating={animating}
            queueRemaining={Math.ceil(botFrames.length / 2)}
          />
          <PlayerStrip view={view} side={me} isMe />
          <HandRow
            view={view}
            mySeat={me}
            selection={selection}
            handIndicesWithActions={handIndicesWithActions}
            onSelect={(s) => setSelection(s)}
          />
        </>
      )}

      {!inAnalysis && <ChoiceDialog view={view} onSubmit={submit} pending={pending} />}

      {selection && selectionActions.length > 0 && (
        <SelectionBar
          selection={selection}
          actions={selectionActions}
          pending={pending}
          onSubmit={submit}
          onCancel={() => setSelection(null)}
          /* Record-mode: when the recorder picks an action on a placeholder
             hand card (their own or the opponent's), intercept the submit
             and open the identity picker first. */
          interceptIfPlaceholder={(action) => {
            if (mode !== "record") return false;
            if (selection.kind === "board") return false;
            const seat: 0 | 1 = selection.kind === "opp-hand" ? oppSeat : me;
            const idx = action.handIndex;
            if (typeof idx !== "number") return false;
            const c = view.players[seat].hand[idx];
            if (!c || c.defId !== -1) return false;
            setIdentityReq({ action, forSeat: seat });
            return true;
          }}
        />
      )}

      {!view.draft && !view.isOver && (
        <GlobalActionBar actions={globalActions} pending={pending} onSubmit={submit} />
      )}

      <IdentityPickerDialog
        view={view}
        request={identityReq}
        pending={pending}
        onCancel={() => setIdentityReq(null)}
        onPick={(defId) => {
          if (!identityReq) return;
          submit({ ...identityReq.action, revealedDefId: defId });
        }}
      />

      {!view.draft && (
        <div className="mt-3 flex justify-between items-center">
          <DeleteGameButton
            gameId={gameId}
            variant="page"
            onDeleted={() => { window.location.href = "/"; }}
          />
          <EvalDialog gameId={gameId} />
        </div>
      )}

      {view.isOver && <GameOverBanner view={view} />}
    </div>
  );
}

function TopBar({
  view,
  analysisOpen,
  canEnterAnalysis,
  onEnterAnalysis,
}: {
  view: GameView;
  analysisOpen: boolean;
  canEnterAnalysis: boolean;
  onEnterAnalysis: () => void;
}) {
  const phase = view.phase.replace("_", " ").toLowerCase();
  const mode = view.config.mode ?? "play";
  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-3">
        <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">← all games</Link>
        <Separator orientation="vertical" className="h-4" />
        {mode === "record" && (
          <Badge variant="destructive" className="text-[10px] uppercase tracking-wide">
            recording · seat {(view.config.recorderSeat ?? 0) + 1}
          </Badge>
        )}
        <div className="text-xs font-mono text-muted-foreground">turn {view.turn} · {phase}</div>
        {view.controlHolder != null && (
          <Badge variant="secondary" className="text-[10px]">
            control: P{view.controlHolder + 1}
          </Badge>
        )}
      </div>
      <div className="flex items-center gap-3">
        <Button
          variant={analysisOpen ? "default" : "outline"}
          size="sm"
          className="h-7 text-xs"
          disabled={!canEnterAnalysis || analysisOpen}
          onClick={onEnterAnalysis}
          title="Step through past actions (←/→, Home/End, Esc)"
        >
          {analysisOpen ? "analysis" : "↶ analysis"}
        </Button>
        <span className="text-xs font-mono text-muted-foreground">
          seed {view.config.seed} · max {view.config.maxTurns} turns
          {view.config.includeExpansion ? " · +expansion" : ""}
        </span>
      </div>
    </div>
  );
}

function AnalysisBar({
  index,
  totalActions,
  snap,
  loading,
  playerLabels,
  onIndex,
  onExit,
}: {
  index: number;
  totalActions: number;
  snap: AnalysisSnapshot | null;
  loading: boolean;
  playerLabels: [string, string];
  onIndex: (i: number) => void;
  onExit: () => void;
}) {
  const atStart = index <= 0;
  const atEnd = index >= totalActions;
  // Show the action that brought us into this state. At index=0 there's
  // nothing applied yet, so we show "initial state".
  const actor = snap?.actor;
  const label = snap?.lastLabel;
  const events = snap?.lastEvents ?? [];
  return (
    <Card className="mb-3 border-sky-500/40 bg-sky-500/5">
      <CardContent className="py-3">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <Badge variant="default" className="text-[10px] uppercase shrink-0">analysis</Badge>
            <span className="text-xs font-mono text-muted-foreground shrink-0">
              action {index} / {totalActions}
            </span>
            {index === 0 ? (
              <span className="text-sm text-muted-foreground truncate">
                Initial state — before any actions.
              </span>
            ) : label ? (
              <span className="text-sm truncate">
                <span className="font-mono text-muted-foreground mr-1">
                  {actor != null ? playerLabels[actor] : "?"}:
                </span>
                {label}
              </span>
            ) : loading ? (
              <span className="text-sm text-muted-foreground">loading…</span>
            ) : null}
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" disabled={atStart} onClick={() => onIndex(0)} title="First (Home)">«</Button>
            <Button variant="outline" size="sm" className="h-7 px-2 text-xs" disabled={atStart} onClick={() => onIndex(index - 1)} title="Prev (←)">‹ prev</Button>
            <Button variant="outline" size="sm" className="h-7 px-2 text-xs" disabled={atEnd} onClick={() => onIndex(index + 1)} title="Next (→)">next ›</Button>
            <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" disabled={atEnd} onClick={() => onIndex(totalActions)} title="Last (End)">»</Button>
            <Separator orientation="vertical" className="h-5 mx-1" />
            <Button variant="default" size="sm" className="h-7 px-3 text-xs" onClick={onExit} title="Return to live (Esc)">
              exit
            </Button>
          </div>
        </div>
        {events.length > 0 && (
          <ul className="mt-2 space-y-0.5 text-xs text-muted-foreground">
            {events.map((e, i) => (
              <li key={i}>· {e}</li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function DraftPanel({
  view, onPick, pending,
}: { view: GameView; onPick: (p: string) => void; pending: boolean }) {
  const draft = view.draft!;
  const decider = view.decider;
  return (
    <Card className="mb-6">
      <CardContent className="py-6">
        <div className="text-xs font-mono text-muted-foreground mb-2">
          Draft pick {draft.idx + 1} of {draft.schedule.length} — {decider === 0 ? view.config.player0Label : view.config.player1Label} picks
        </div>
        <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
          {draft.pool.map((p) => (
            <Button key={p} variant="outline" size="lg" disabled={pending} onClick={() => onPick(p)}>
              {p}
            </Button>
          ))}
        </div>
        <div className="mt-6 grid grid-cols-2 gap-4 text-xs">
          <ProtocolSlots label={view.config.player0Label} protos={view.lines.map((l) => l.p0Protocol)} />
          <ProtocolSlots label={view.config.player1Label} protos={view.lines.map((l) => l.p1Protocol)} />
        </div>
      </CardContent>
    </Card>
  );
}

function ProtocolSlots({ label, protos }: { label: string; protos: (string | null)[] }) {
  return (
    <div>
      <div className="text-muted-foreground mb-1">{label}</div>
      <div className="flex gap-2">
        {protos.map((p, i) => (
          <div key={i} className="flex-1 border rounded px-2 py-1 font-mono">
            {p ?? <span className="text-muted-foreground">—</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

function PlayerStrip({ view, side, isMe }: { view: GameView; side: 0 | 1; isMe: boolean }) {
  const ps = view.players[side];
  const label = side === 0 ? view.config.player0Label : view.config.player1Label;
  const isBot = side === 0 ? view.config.botStrategy != null && !view.config.human[0] : view.config.botStrategy != null && !view.config.human[1];
  return (
    <div className={`flex items-center gap-3 py-2 ${isMe ? "" : "opacity-90"}`}>
      <div className="flex items-center gap-2 min-w-32">
        <span className="font-medium">{isMe ? "You" : label}</span>
        {isBot && <Badge variant="secondary" className="text-[10px]">bot</Badge>}
        {view.controlHolder === side && <Badge variant="default" className="text-[10px]">control</Badge>}
      </div>
      <Separator orientation="vertical" className="h-5" />
      <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground">
        <span>hand {ps.hand.length}</span>
        <span>deck {ps.deckCount}</span>
        <span>trash {ps.trashCount}</span>
        {ps.cannotCompileNextTurn && <Badge variant="destructive" className="text-[10px]">no-compile next</Badge>}
      </div>
    </div>
  );
}

function HandRow({
  view,
  mySeat,
  selection,
  handIndicesWithActions,
  onSelect,
}: {
  view: GameView;
  mySeat: 0 | 1;
  selection: Selection | null;
  handIndicesWithActions: Set<number>;
  onSelect: (s: Selection | null) => void;
}) {
  const hand = view.players[mySeat].hand;
  if (hand.length === 0) {
    return (
      <div className="mt-3 rounded-lg border border-dashed bg-muted/20 px-4 py-3 text-xs font-mono text-muted-foreground">
        hand is empty
      </div>
    );
  }
  const selectedHandIndex = selection?.kind === "hand" ? selection.handIndex : -1;
  return (
    <div className="mt-3 rounded-lg border bg-card/40 px-3 py-3">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wide text-muted-foreground">
          your hand
        </span>
        {handIndicesWithActions.size > 0 && (
          <span className="text-[10px] font-mono text-muted-foreground">
            click a card to act
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {hand.map((c, i) => {
          const interactive = handIndicesWithActions.has(i);
          return (
            <PlayCard
              key={c.instId}
              card={c}
              variant="hand"
              selected={selectedHandIndex === i}
              onClick={interactive ? () => onSelect({ kind: "hand", handIndex: i }) : undefined}
              disabled={!interactive}
            />
          );
        })}
      </div>
    </div>
  );
}

function Board({
  view,
  mySeat,
  selection,
  boardKeysWithActions,
  onSelect,
}: {
  view: GameView;
  mySeat: 0 | 1;
  selection: Selection | null;
  boardKeysWithActions: Set<string>;
  onSelect: (s: Selection | null) => void;
}) {
  const oppSeat = (mySeat === 0 ? 1 : 0) as 0 | 1;
  return (
    <Card className="my-3">
      <CardContent className="py-4">
        <div className="grid grid-cols-3 gap-4">
          {view.lines.map((line, idx) => (
            <LineCard
              key={idx}
              line={line}
              mySeat={mySeat}
              oppSeat={oppSeat}
              selection={selection}
              boardKeysWithActions={boardKeysWithActions}
              onSelect={onSelect}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function LineCard({
  line,
  mySeat,
  oppSeat,
  selection,
  boardKeysWithActions,
  onSelect,
}: {
  line: LineView;
  mySeat: 0 | 1;
  oppSeat: 0 | 1;
  selection: Selection | null;
  boardKeysWithActions: Set<string>;
  onSelect: (s: Selection | null) => void;
}) {
  const myProto = mySeat === 0 ? line.p0Protocol : line.p1Protocol;
  const oppProto = oppSeat === 0 ? line.p0Protocol : line.p1Protocol;
  const myValue = mySeat === 0 ? line.p0Value : line.p1Value;
  const oppValue = oppSeat === 0 ? line.p0Value : line.p1Value;
  const myStack = mySeat === 0 ? line.p0Stack : line.p1Stack;
  const oppStack = oppSeat === 0 ? line.p0Stack : line.p1Stack;
  const myCompiled = mySeat === 0 ? line.p0Compiled : line.p1Compiled;
  const oppCompiled = oppSeat === 0 ? line.p0Compiled : line.p1Compiled;

  return (
    <div className="rounded-lg border bg-card overflow-hidden flex flex-col min-h-[480px]">
      <div className="px-3 py-2 bg-muted/40 flex items-center justify-between border-b shrink-0">
        <span className="text-[10px] font-mono text-muted-foreground">L{line.index + 1}</span>
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">{oppProto ?? "—"}</span>
          {oppCompiled && <Badge variant="default" className="text-[10px]">compiled</Badge>}
          <span className={`text-sm font-mono ${oppValue > myValue ? "text-amber-500" : "text-muted-foreground"}`}>{oppValue}</span>
        </div>
      </div>
      {/* Each side gets equal flex space so the centre divider stays at
          the visual midline regardless of how many cards either player
          has. The opp half anchors content to the top (justify-start)
          so opp's stack hangs from the header; the me half anchors to
          the bottom (justify-end) so my stack rises off my footer. */}
      <div className="flex-1 flex flex-col items-stretch min-h-0">
        <div className="flex-1 flex flex-col items-stretch min-h-0 overflow-hidden">
          <Stack
            stack={oppStack}
            owner="opp"
            lineIndex={line.index}
            selection={null}
            boardKeysWithActions={new Set()}
            onSelect={() => {}}
          />
        </div>
        <div className="h-px bg-border shrink-0" />
        <div className="flex-1 flex flex-col items-stretch justify-end min-h-0 overflow-hidden">
          <Stack
            stack={myStack}
            owner="me"
            lineIndex={line.index}
            selection={selection}
            boardKeysWithActions={boardKeysWithActions}
            onSelect={onSelect}
          />
        </div>
      </div>
      <div className="px-3 py-2 bg-muted/40 flex items-center justify-between border-t shrink-0">
        <span className={`text-sm font-mono ${myValue > oppValue ? "text-emerald-500" : "text-muted-foreground"}`}>{myValue}</span>
        <div className="flex items-center gap-2">
          {myCompiled && <Badge variant="default" className="text-[10px]">compiled</Badge>}
          <span className="font-semibold text-sm">{myProto ?? "—"}</span>
        </div>
      </div>
    </div>
  );
}

function Stack({
  stack,
  owner,
  lineIndex,
  selection,
  boardKeysWithActions,
  onSelect,
}: {
  stack: CardView[];
  owner: "me" | "opp";
  lineIndex: number;
  selection: Selection | null;
  boardKeysWithActions: Set<string>;
  onSelect: (s: Selection | null) => void;
}) {
  if (stack.length === 0) {
    return (
      <div className="px-2 py-3 flex items-center justify-center text-[10px] text-muted-foreground/60">
        empty
      </div>
    );
  }
  // engine stack[last] is uncovered. Render in natural stack order
  // (oldest at the visual top, newest/uncovered at the visual bottom).
  // Each later card overlaps the previous card's M+B tiers from below,
  // leaving the previous card's header + top tier exposed — matching the
  // Codex "the value and top command are always visible when covered"
  // rule. zIndex grows with stack position so the newer (covering) card
  // sits in front of the one it covers. Capped at 30 to stay below the
  // modal dialog's z-50.
  const lastIndex = stack.length - 1;
  return (
    <div className="px-2 py-2 flex flex-col items-stretch">
      {stack.map((c, stackIdx) => {
        const interactive = owner === "me" && boardKeysWithActions.has(`${lineIndex}:${stackIdx}`);
        const selected =
          selection?.kind === "board"
          && selection.lineIndex === lineIndex
          && selection.stackIndex === stackIdx;
        return (
          <div
            key={c.instId}
            style={{
              // Pull each later card upward so it overlaps the previous
              // card's M + B + footer (the bottom 112px of the previous
              // card). The previous card's header + top tier stays
              // exposed above this card.
              marginTop: stackIdx === 0 ? 0 : -CARD_OVERLAP_PX,
              // Later cards in front so their full body shows and the
              // overlap region hides the older card's bottom tiers.
              zIndex: 10 + stackIdx,
              position: "relative",
            }}
          >
            <PlayCard
              card={c}
              variant="board"
              uncovered={stackIdx === lastIndex}
              hidden={owner === "opp" && !c.faceUp}
              selected={selected}
              onClick={
                interactive
                  ? () => onSelect({ kind: "board", lineIndex, stackIndex: stackIdx })
                  : undefined
              }
            />
          </div>
        );
      })}
    </div>
  );
}

function ChoiceDialog({
  view, onSubmit, pending,
}: { view: GameView; onSubmit: (a: Action) => void; pending: boolean }) {
  if (!view.pendingChoice) return null;
  const choice = view.pendingChoice;
  return (
    <Card className="my-3 border-amber-500/40">
      <CardContent className="py-4">
        <div className="text-xs font-mono text-muted-foreground mb-2">
          {choice.decider === 0 ? view.config.player0Label : view.config.player1Label} must decide
        </div>
        <div className="text-base font-medium mb-3">{choice.prompt}</div>
        <div className="flex flex-wrap gap-2">
          {choice.options.map((opt, i) => (
            <Button
              key={i}
              variant="outline"
              size="sm"
              disabled={pending}
              onClick={() => onSubmit({ type: "CHOOSE_TARGET", choiceIndex: i })}
            >
              {opt}
            </Button>
          ))}
          {choice.optional && (
            <Button
              variant="ghost"
              size="sm"
              disabled={pending}
              onClick={() => onSubmit({ type: "SKIP_OPTIONAL" })}
            >
              Skip
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function SelectionBar({
  selection,
  actions,
  pending,
  onSubmit,
  onCancel,
  interceptIfPlaceholder,
}: {
  selection: Selection;
  actions: ActionView[];
  pending: boolean;
  onSubmit: (a: Action) => void;
  onCancel: () => void;
  /** Record-mode hook: return true if the click was handled (opened a
   *  picker dialog) and the submit should be skipped. */
  interceptIfPlaceholder?: (action: ActionView) => boolean;
}) {
  const groups = useMemo(() => groupSelectionActions(actions), [actions]);
  const isOpp = selection.kind === "opp-hand";
  const header = selection.kind === "hand"
    ? `Hand card #${selection.handIndex + 1}`
    : selection.kind === "opp-hand"
      ? `Opponent hand card #${selection.handIndex + 1}`
      : `Line ${selection.lineIndex + 1} · stack #${selection.stackIndex + 1}`;
  return (
    <div className={`mt-3 rounded-lg border px-3 py-2.5 ${isOpp ? "border-amber-500/30 bg-amber-500/5" : "border-emerald-500/30 bg-emerald-500/5"}`}>
      <div className="mb-2 flex items-center justify-between">
        <span className={`text-[10px] font-mono uppercase tracking-wide ${isOpp ? "text-amber-200/80" : "text-emerald-200/80"}`}>
          {header}{isOpp ? " — log what they did" : ""}
        </span>
        <Button variant="ghost" size="sm" className="h-6 px-2 text-[11px]" onClick={onCancel} disabled={pending}>
          cancel
        </Button>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-2">
        {groups.map(({ heading, actions }) => (
          <div key={heading} className="flex flex-col gap-1">
            <span className="text-[10px] font-mono uppercase tracking-wide text-muted-foreground">{heading}</span>
            <div className="flex flex-wrap gap-1">
              {actions.map((a, i) => (
                <Button
                  key={`${heading}-${i}`}
                  variant="secondary"
                  size="sm"
                  disabled={pending}
                  onClick={() => {
                    if (interceptIfPlaceholder?.(a)) return;
                    onSubmit(a);
                  }}
                >
                  {selectionActionLabel(a)}
                </Button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function OppHandRow({
  view,
  oppSeat,
  selection,
  oppHandIndicesWithActions,
  onSelect,
}: {
  view: GameView;
  oppSeat: 0 | 1;
  selection: Selection | null;
  oppHandIndicesWithActions: Set<number>;
  onSelect: (s: Selection | null) => void;
}) {
  const hand = view.players[oppSeat].hand;
  const oppLabel = oppSeat === 0 ? view.config.player0Label : view.config.player1Label;
  if (hand.length === 0) return null;
  const selectedHi = selection?.kind === "opp-hand" ? selection.handIndex : -1;
  return (
    <div className="mt-2 rounded-lg border border-dashed border-amber-500/30 bg-amber-500/5 px-3 py-2.5">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wide text-amber-200/80">
          {oppLabel}&apos;s hand · placeholders
        </span>
        {oppHandIndicesWithActions.size > 0 && (
          <span className="text-[10px] font-mono text-muted-foreground">
            click a card to log their move
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {hand.map((c, i) => {
          const interactive = oppHandIndicesWithActions.has(i);
          return (
            <PlayCard
              key={c.instId}
              card={c}
              variant="board"
              hidden={c.defId === -1}
              selected={selectedHi === i}
              onClick={interactive ? () => onSelect({ kind: "opp-hand", handIndex: i }) : undefined}
              disabled={!interactive}
            />
          );
        })}
      </div>
    </div>
  );
}

function IdentityPickerDialog({
  view,
  request,
  pending,
  onCancel,
  onPick,
}: {
  view: GameView;
  request: IdentityRequest | null;
  pending: boolean;
  onCancel: () => void;
  onPick: (defId: number) => void;
}) {
  // Face-up plays must match the line's protocol (modulo Spirit 1 / Chaos 3
  // / Corruption 0 bypasses). Default to filtering candidates to just the
  // line's protocol; the recorder can flip a toggle if a bypass was active.
  const [showAll, setShowAll] = useState(false);
  // Reset the toggle whenever a fresh dialog opens so the default behaviour
  // is consistent across plays.
  const requestKey = request
    ? `${request.forSeat}:${request.action.type}:${request.action.handIndex}:${request.action.lineIndex}`
    : null;
  useEffect(() => { setShowAll(false); }, [requestKey]);

  const seatProtos = useMemo(() => {
    if (!request) return [];
    const seat = request.forSeat;
    return view.lines
      .map((l) => (seat === 0 ? l.p0Protocol : l.p1Protocol))
      .filter((p): p is string => p != null);
  }, [request, view]);

  const lineProtocol = useMemo(() => {
    if (!request || request.action.lineIndex == null) return null;
    const seat = request.forSeat;
    const ln = request.action.lineIndex;
    if (ln < 0 || ln >= view.lines.length) return null;
    const line = view.lines[ln];
    return seat === 0 ? line.p0Protocol : line.p1Protocol;
  }, [request, view]);

  const isFaceUp = request?.action.type === "PLAY_FACE_UP";
  const canFilter = isFaceUp && !!lineProtocol;
  const filtered = canFilter && !showAll;

  const candidates = useMemo(() => {
    if (filtered && lineProtocol) {
      return defsForProtocol(lineProtocol as Protocol);
    }
    return seatProtos.flatMap((p) => defsForProtocol(p as Protocol));
  }, [filtered, lineProtocol, seatProtos]);

  const isOpp = !!request && request.forSeat !== (view.config.recorderSeat ?? 0);
  const a = request?.action;
  const title = isOpp
    ? "What did your opponent play?"
    : "Which card did you play?";
  let subtitle = "";
  if (a) {
    const lineN = (a.lineIndex ?? -1) + 1;
    if (a.type === "PLAY_FACE_UP") {
      subtitle = `${isOpp ? "They played" : "You're playing"} face-up in line ${lineN} (${lineProtocol}).`;
    } else if (a.type === "PLAY_FACE_DOWN") {
      subtitle = `${isOpp ? "They played" : "You're playing"} face-down in line ${lineN}.`;
    } else if (a.type === "DISCARD_CARD") {
      subtitle = `${isOpp ? "They discarded" : "You're discarding"}.`;
    }
  }

  return (
    <Dialog open={request != null} onOpenChange={(o) => { if (!o) onCancel(); }}>
      <DialogContent className="sm:max-w-[640px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{subtitle} Pick the card.</DialogDescription>
        </DialogHeader>
        {canFilter && (
          <div className="flex items-center gap-2 text-[11px] font-mono text-muted-foreground">
            <span>showing {filtered ? `${lineProtocol} cards only` : "all drafted cards"}</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-[11px]"
              onClick={() => setShowAll((v) => !v)}
            >
              {filtered ? "show all (Spirit 1 / Chaos 3 / Corruption 0)" : "filter to line protocol"}
            </Button>
          </div>
        )}
        <div className="grid grid-cols-3 gap-2 max-h-[420px] overflow-y-auto py-2">
          {candidates.map((d) => {
            const cardLike: CardView = {
              instId: -1,
              defId: d.defId,
              key: d.key,
              protocol: d.protocol,
              value: d.value,
              faceUp: true,
              isCommitted: false,
              topText: d.topText,
              middleText: d.middleText,
              bottomText: d.bottomText,
            };
            return (
              <PlayCard
                key={d.defId}
                card={cardLike}
                variant="hand"
                onClick={() => onPick(d.defId)}
                disabled={pending}
              />
            );
          })}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function GlobalActionBar({
  actions,
  pending,
  onSubmit,
}: {
  actions: ActionView[];
  pending: boolean;
  onSubmit: (a: Action) => void;
}) {
  if (actions.length === 0) return null;
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      {actions.map((a, i) => (
        <Button
          key={`global-${i}`}
          variant={a.type === "COMPILE_LINE" ? "default" : "outline"}
          size="sm"
          disabled={pending}
          onClick={() => onSubmit(a)}
        >
          {a.label}
        </Button>
      ))}
    </div>
  );
}

function selectionActionLabel(a: ActionView): string {
  // Strip the redundant card name from labels since the user already chose
  // the card. We want short directional labels ("L1", "L2", "L3", "Discard").
  if (a.type === "PLAY_FACE_UP" && typeof a.lineIndex === "number") return `L${a.lineIndex + 1}`;
  if (a.type === "PLAY_FACE_DOWN" && typeof a.lineIndex === "number") return `L${a.lineIndex + 1}`;
  if (a.type === "SHIFT_OWN_CARD" && typeof a.choiceIndex === "number") return `→ L${a.choiceIndex + 1}`;
  if (a.type === "DISCARD_CARD") return "Discard";
  return a.label;
}

function groupSelectionActions(actions: ActionView[]): { heading: string; actions: ActionView[] }[] {
  const buckets: Record<string, ActionView[]> = {};
  for (const a of actions) {
    const key =
      a.type === "PLAY_FACE_UP" ? "Play face-up"
      : a.type === "PLAY_FACE_DOWN" ? "Play face-down"
      : a.type === "SHIFT_OWN_CARD" ? "Shift to"
      : a.type === "DISCARD_CARD" ? "Discard"
      : a.type;
    (buckets[key] ??= []).push(a);
  }
  const order = ["Play face-up", "Play face-down", "Shift to", "Discard"];
  return order
    .filter((h) => buckets[h]?.length)
    .map((heading) => ({ heading, actions: buckets[heading] }));
}

function TurnStatusBanner({
  view, me, pending, botActionLabel, botEvents, animating, queueRemaining,
}: {
  view: GameView;
  me: 0 | 1;
  pending: boolean;
  botActionLabel: string | null;
  botEvents: string[];
  animating: boolean;
  queueRemaining: number;
}) {
  if (view.isOver || view.draft) return null;
  const decider = view.decider;
  const oppLabel = me === 0 ? view.config.player1Label : view.config.player0Label;
  const phase = view.phase;

  // During bot-chain playback, surface what the opponent is doing right
  // now in big text — that's the whole point of the slow transitions.
  // Any engine-emitted "info" events that landed during the step are
  // shown beneath the label so the player sees skipped effects ("tried
  // to discard but my hand was empty") inline with the action.
  if (animating && botActionLabel) {
    return (
      <Card className="my-3 border-amber-500/40">
        <CardContent className="py-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="text-[10px] uppercase">{oppLabel}</Badge>
              <span className="text-sm font-medium">{botActionLabel}</span>
            </div>
            <span className="text-xs font-mono text-muted-foreground">
              {queueRemaining > 0 ? `${queueRemaining} more step${queueRemaining === 1 ? "" : "s"}…` : "playing chain…"}
            </span>
          </div>
          {botEvents.length > 0 && (
            <ul className="mt-2 space-y-0.5 text-xs text-muted-foreground">
              {botEvents.map((e, i) => (
                <li key={i}>· {e}</li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    );
  }

  // Mid-effect choice prompts take priority — they're displayed in their
  // own modal/bar via ChoiceDialog, so we just label the situation here.
  if (view.pendingChoice && view.pendingChoice.decider === me) {
    return (
      <Card className="my-3 border-emerald-500/40">
        <CardContent className="py-3 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Badge variant="default" className="text-[10px] uppercase">your decision</Badge>
            <span className="text-sm font-medium">{view.pendingChoice.prompt}</span>
          </div>
          <span className="text-xs text-muted-foreground">
            {view.pendingChoice.options.length} option{view.pendingChoice.options.length === 1 ? "" : "s"} ·
            {" "}choose below
          </span>
        </CardContent>
      </Card>
    );
  }

  // Bot is the decider — engine usually auto-advances server-side, so this
  // banner appears only briefly during the network round-trip.
  if (decider !== me) {
    return (
      <Card className="my-3 border-amber-500/30">
        <CardContent className="py-3 flex items-center gap-2">
          <Badge variant="secondary" className="text-[10px] uppercase">opponent</Badge>
          <span className="text-sm">
            {pending ? `${oppLabel} is thinking…` : `Waiting on ${oppLabel}…`}
          </span>
        </CardContent>
      </Card>
    );
  }

  // Decider is me — phase-specific guidance.
  const myHand = view.players[me].hand.length;
  const myDeck = view.players[me].deckCount;
  const compileActions = view.legalActions.filter((a) => a.type === "COMPILE_LINE");
  const compileLines = compileActions.map((a) => a.lineIndex! + 1);

  let label = "your turn";
  let message = "Pick a card to play, or refresh / compile from the action bar.";
  let highlight: "default" | "warn" | "info" = "default";
  if (phase === "CHECK_CACHE") {
    const overshoot = Math.max(0, myHand - 5);
    label = "clear cache";
    message = overshoot > 0
      ? `You have ${myHand} cards in hand. Discard ${overshoot} more to get to 5.`
      : "Cache phase — the engine will move on momentarily.";
    highlight = "warn";
  } else if (phase === "CHECK_COMPILE" || compileLines.length > 0) {
    label = "compile";
    message = `You meet the compile requirements in line ${compileLines.join(", ")}. ` +
      `Per the rules you must compile this turn.`;
    highlight = "info";
  } else if (phase === "ACTION") {
    label = "your turn";
    const canRefresh = view.legalActions.some((a) => a.type === "REFRESH");
    const handBits = myHand === 0 ? "your hand is empty — refresh" : `${myHand} cards in hand`;
    const deckBits = myDeck === 0 ? " (deck empty)" : "";
    message = `${handBits}${deckBits}. Click a card to see plays${canRefresh ? ", or refresh from the action bar" : ""}.`;
  }

  const borderClass =
    highlight === "warn" ? "border-amber-500/50" :
    highlight === "info" ? "border-sky-500/40" :
    "border-emerald-500/40";

  return (
    <Card className={`my-3 ${borderClass}`}>
      <CardContent className="py-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Badge variant="default" className="text-[10px] uppercase">{label}</Badge>
          <span className="text-sm">{message}</span>
        </div>
        <span className="text-xs font-mono text-muted-foreground">
          turn {view.turn} · {phase.toLowerCase().replace("_", " ")}
        </span>
      </CardContent>
    </Card>
  );
}

function GameOverBanner({ view }: { view: GameView }) {
  const winnerLabel =
    view.winner == null
      ? "Draw"
      : view.winner === 0
        ? view.config.player0Label
        : view.config.player1Label;
  return (
    <Card className="mt-4 border-emerald-500/40">
      <CardContent className="py-4 flex items-center justify-between">
        <div>
          <div className="text-xs font-mono text-muted-foreground">game over</div>
          <div className="text-lg font-medium">{winnerLabel} {view.winner != null ? "wins" : ""}</div>
        </div>
        <Link href="/"><Button variant="outline">Back to library</Button></Link>
      </CardContent>
    </Card>
  );
}

