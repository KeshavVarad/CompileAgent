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

type Props = { gameId: string; initialView: GameView };

type IdentityRequest = {
  /** The action the recorder picked but hasn't submitted yet — it's missing
   *  the identity (revealedDefId) of the placeholder hand card it references. */
  action: ActionView;
  /** Which seat owns the hand card whose identity we're picking. Used to
   *  filter candidate cards to that player's drafted protocols. */
  forSeat: 0 | 1;
};

export function GameClient({ gameId, initialView }: Props) {
  const [view, setView] = useState<GameView>(initialView);
  const [pending, startTransition] = useTransition();
  const [selection, setSelection] = useState<Selection | null>(null);
  const [identityReq, setIdentityReq] = useState<IdentityRequest | null>(null);

  const submit = useCallback(
    (action: Action) => {
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
          setView(data.view as GameView);
        } catch (e) {
          toast.error(e instanceof Error ? e.message : "Step failed");
        }
      });
    },
    [gameId],
  );

  // In record mode the recorder's seat is fixed regardless of whose turn it
  // is; the same recorder enters both seats' actions. In play mode, the
  // engine auto-advances bot turns so the decider is always the human.
  const mode = view.config.mode ?? "play";
  const me: 0 | 1 = mode === "record" && view.config.recorderSeat != null
    ? (view.config.recorderSeat as 0 | 1)
    : view.decider;
  const oppSeat: 0 | 1 = me === 0 ? 1 : 0;
  const activeDecider = view.decider;
  const recordingOppTurn = mode === "record" && activeDecider !== me;

  const handIndicesWithActions = useMemo(() => {
    const set = new Set<number>();
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
  }, [view.legalActions, recordingOppTurn]);

  const oppHandIndicesWithActions = useMemo(() => {
    const set = new Set<number>();
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
  }, [view.legalActions, recordingOppTurn]);

  const boardKeysWithActions = useMemo(() => {
    const set = new Set<string>();
    for (const a of view.legalActions) {
      if (a.type === "SHIFT_OWN_CARD" && typeof a.lineIndex === "number" && typeof a.handIndex === "number") {
        set.add(`${a.lineIndex}:${a.handIndex}`);
      }
    }
    return set;
  }, [view.legalActions]);

  const selectionActions = useMemo<ActionView[]>(() => {
    if (!selection) return [];
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
  }, [selection, view.legalActions]);

  const globalActions = useMemo<ActionView[]>(
    () => view.legalActions.filter((a) => a.type === "REFRESH" || a.type === "COMPILE_LINE"),
    [view.legalActions],
  );

  return (
    <div className="flex-1 mx-auto w-full max-w-6xl px-6 py-6">
      <TopBar view={view} />

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

      <ChoiceDialog view={view} onSubmit={submit} pending={pending} />

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

function TopBar({ view }: { view: GameView }) {
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
      <div className="text-xs font-mono text-muted-foreground">
        seed {view.config.seed} · max {view.config.maxTurns} turns
        {view.config.includeExpansion ? " · +expansion" : ""}
      </div>
    </div>
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
    <div className="rounded-lg border bg-card overflow-hidden flex flex-col">
      <div className="px-3 py-2 bg-muted/40 flex items-center justify-between border-b">
        <span className="text-[10px] font-mono text-muted-foreground">L{line.index + 1}</span>
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">{oppProto ?? "—"}</span>
          {oppCompiled && <Badge variant="default" className="text-[10px]">compiled</Badge>}
          <span className={`text-sm font-mono ${oppValue > myValue ? "text-amber-500" : "text-muted-foreground"}`}>{oppValue}</span>
        </div>
      </div>
      <Stack
        stack={oppStack}
        owner="opp"
        lineIndex={line.index}
        selection={null}
        boardKeysWithActions={new Set()}
        onSelect={() => {}}
      />
      <div className="h-px bg-border" />
      <Stack
        stack={myStack}
        owner="me"
        lineIndex={line.index}
        selection={selection}
        boardKeysWithActions={boardKeysWithActions}
        onSelect={onSelect}
      />
      <div className="px-3 py-2 bg-muted/40 flex items-center justify-between border-t">
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
  // engine stack[last] is uncovered; render top-to-bottom with uncovered at top.
  // Cards in the column physically overlap so the upper card covers the lower
  // card's header + top tier, leaving only the M+B tiers of covered cards
  // visible — matching how a real Compile play-stack looks.
  const lastIndex = stack.length - 1;
  return (
    <div className="px-2 py-2 flex flex-col items-stretch">
      {[...stack].reverse().map((c, displayIdx) => {
        const stackIdx = lastIndex - displayIdx;
        const interactive = owner === "me" && boardKeysWithActions.has(`${lineIndex}:${stackIdx}`);
        const selected =
          selection?.kind === "board"
          && selection.lineIndex === lineIndex
          && selection.stackIndex === stackIdx;
        return (
          <div
            key={c.instId}
            style={{
              // Pull each non-top card upward so the card above covers its
              // header + top tier.
              marginTop: displayIdx === 0 ? 0 : -CARD_OVERLAP_PX,
              // Higher card stays in front; lower cards have a lower z so
              // their hidden tier really is behind the upper card.
              zIndex: 100 - displayIdx,
              position: "relative",
            }}
          >
            <PlayCard
              card={c}
              variant="board"
              uncovered={displayIdx === 0}
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

