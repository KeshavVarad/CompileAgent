"use client";

import { useMemo, useState } from "react";

import { BugReportDialog } from "@/components/bug-report-dialog";
import { PlayCard } from "@/components/play-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ALL_SETS, CARD_DEFS, type CardDef, type SetCode } from "@/lib/compile/cards";
import { cn } from "@/lib/utils";
import type { CardView } from "@/lib/view";

const SET_LABELS: Record<SetCode, string> = {
  MN01: "Base",
  AX01: "Expansion",
  MN02: "Main 2",
  AX02: "Aux 2",
};

function toView(d: CardDef): CardView {
  return {
    instId: -1,
    defId: d.defId,
    key: d.key,
    protocol: d.protocol,
    value: d.value,
    faceUp: true,
    isCommitted: false,
    topEmphasis: d.topEmphasis,
    topText: d.topText,
    middleEmphasis: d.middleEmphasis,
    middleText: d.middleText,
    bottomEmphasis: d.bottomEmphasis,
    bottomText: d.bottomText,
  };
}

export function CardsBrowser() {
  const [setFilter, setSetFilter] = useState<SetCode | "ALL">("ALL");
  const [protocolFilter, setProtocolFilter] = useState<string>("ALL");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<CardDef | null>(null);

  // Distinct protocols visible under the current set filter — recomputed
  // when the set changes so the protocol dropdown stays consistent.
  const protocols = useMemo(() => {
    const pool = setFilter === "ALL"
      ? CARD_DEFS
      : CARD_DEFS.filter((c) => c.setCode === setFilter);
    const seen = new Set<string>();
    for (const c of pool) seen.add(c.protocol);
    return Array.from(seen).sort();
  }, [setFilter]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return CARD_DEFS.filter((c) => {
      if (setFilter !== "ALL" && c.setCode !== setFilter) return false;
      if (protocolFilter !== "ALL" && c.protocol !== protocolFilter) return false;
      if (!q) return true;
      return (
        c.protocol.toLowerCase().includes(q) ||
        c.key.toLowerCase().includes(q) ||
        c.topText.toLowerCase().includes(q) ||
        c.middleText.toLowerCase().includes(q) ||
        c.bottomText.toLowerCase().includes(q) ||
        c.topEmphasis.toLowerCase().includes(q) ||
        c.middleEmphasis.toLowerCase().includes(q) ||
        c.bottomEmphasis.toLowerCase().includes(q)
      );
    });
  }, [setFilter, protocolFilter, query]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1.5">
          <div className="text-[10px] font-mono uppercase tracking-wide text-muted-foreground">
            Set
          </div>
          <div className="flex gap-1">
            <FilterChip
              active={setFilter === "ALL"}
              onClick={() => { setSetFilter("ALL"); setProtocolFilter("ALL"); }}
            >
              All
            </FilterChip>
            {ALL_SETS.map((s) => (
              <FilterChip
                key={s}
                active={setFilter === s}
                onClick={() => { setSetFilter(s); setProtocolFilter("ALL"); }}
              >
                {SET_LABELS[s]}
              </FilterChip>
            ))}
          </div>
        </div>

        <div className="space-y-1.5">
          <div className="text-[10px] font-mono uppercase tracking-wide text-muted-foreground">
            Protocol
          </div>
          <div className="flex flex-wrap gap-1 max-w-[640px]">
            <FilterChip
              active={protocolFilter === "ALL"}
              onClick={() => setProtocolFilter("ALL")}
            >
              All
            </FilterChip>
            {protocols.map((p) => (
              <FilterChip
                key={p}
                active={protocolFilter === p}
                onClick={() => setProtocolFilter(p)}
              >
                {p}
              </FilterChip>
            ))}
          </div>
        </div>

        <div className="flex-1 min-w-[200px] space-y-1.5">
          <div className="text-[10px] font-mono uppercase tracking-wide text-muted-foreground">
            Search
          </div>
          <Input
            type="search"
            placeholder="card text, emphasis, protocol…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="h-8 text-xs"
          />
        </div>
      </div>

      <div className="text-xs font-mono text-muted-foreground">
        {filtered.length} card{filtered.length === 1 ? "" : "s"}
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-lg border border-dashed py-12 text-center text-sm text-muted-foreground">
          No cards match these filters.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
          {filtered.map((d) => (
            <PlayCard
              key={d.key}
              card={toView(d)}
              variant="hand"
              onClick={() => setSelected(d)}
            />
          ))}
        </div>
      )}

      <CardDetailDialog
        card={selected}
        onOpenChange={(o) => { if (!o) setSelected(null); }}
      />
    </div>
  );
}

function FilterChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "px-2.5 py-1 rounded-md text-[11px] font-mono border transition-colors",
        active
          ? "bg-foreground text-background border-foreground"
          : "border-input text-foreground/80 hover:bg-accent",
      )}
    >
      {children}
    </button>
  );
}

function CardDetailDialog({
  card,
  onOpenChange,
}: {
  card: CardDef | null;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={card != null} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[520px]">
        {card && (
          <>
            <DialogHeader>
              <div className="flex items-center justify-between gap-3">
                <DialogTitle>
                  {card.protocol} {card.value}
                </DialogTitle>
                <Badge variant="outline" className="font-mono text-[10px]">
                  {card.setCode}
                </Badge>
              </div>
              <DialogDescription className="font-mono text-[11px]">
                {card.key}
                {card.errata && (
                  <span className="ml-2 text-amber-300/80">· {card.errata}</span>
                )}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-2">
              <TierRow label="Top" emphasis={card.topEmphasis} text={card.topText} />
              <TierRow label="Middle" emphasis={card.middleEmphasis} text={card.middleText} />
              <TierRow label="Bottom" emphasis={card.bottomEmphasis} text={card.bottomText} />
            </div>

            {card.keywords.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {card.keywords.map((k) => (
                  <Badge key={k} variant="secondary" className="text-[10px]">
                    {k}
                  </Badge>
                ))}
              </div>
            )}

            <div className="flex justify-end pt-2">
              <BugReportDialog
                card={card}
                triggerLabel="Report bug for this card"
                triggerClassName="text-[11px]"
              />
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

function TierRow({
  label,
  emphasis,
  text,
}: {
  label: string;
  emphasis: string;
  text: string;
}) {
  const empty = !emphasis && !text;
  return (
    <div className="flex gap-3 text-sm rounded-md border border-input/60 px-3 py-2">
      <span className="font-mono text-[11px] uppercase text-muted-foreground w-14 shrink-0 pt-0.5">
        {label}
      </span>
      {empty ? (
        <span className="text-muted-foreground/50 italic text-xs">no effect</span>
      ) : (
        <span className="text-foreground/90 leading-snug">
          {emphasis && (
            <span className="font-semibold text-amber-200/90">{emphasis}</span>
          )}
          {emphasis && text ? " " : ""}
          {text}
        </span>
      )}
    </div>
  );
}
