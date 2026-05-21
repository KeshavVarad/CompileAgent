"use client";

import type { CardView } from "@/lib/view";
import { cn } from "@/lib/utils";

type Variant = "hand" | "board" | "trash";

type Props = {
  card: CardView;
  variant?: Variant;
  /** True if the viewer is not allowed to see the card's identity (opp
   *  face-down on field). Hand cards in record mode use this implicitly via
   *  the placeholder defId=-1. */
  hidden?: boolean;
  /** Field-only: whether this card sits on top of its stack. Toggles which
   *  tier is highlighted as the currently-active effect (T when uncovered,
   *  M when covered). */
  uncovered?: boolean;
  selected?: boolean;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
};

// Fixed dimensions so stacked field cards overlap by a predictable amount.
// Layout: header (28) + 3 tiers (52 each) + footer (8) = 192px total.
// Stack overlap = header + top tier = 80px → covered cards reveal M+B only.
export const CARD_TOTAL_PX = 192;
export const CARD_OVERLAP_PX = 80;

export function PlayCard({
  card,
  variant = "board",
  hidden = false,
  uncovered = true,
  selected = false,
  onClick,
  disabled = false,
  className,
}: Props) {
  const knownIdentity = card.defId >= 0;
  const isBoardFaceDown = variant === "board" && !card.faceUp;
  // Show the card back when the viewer can't see the identity. On the
  // field that's "opp face-down" (hidden=true). In hand it's the
  // record-mode placeholder (defId=-1).
  const showAsBack = (hidden && isBoardFaceDown) || (!knownIdentity && variant !== "board");

  if (showAsBack) {
    return <FaceDownBack variant={variant} onClick={onClick} disabled={disabled} selected={selected} className={className} />;
  }

  const interactive = !!onClick && !disabled;
  const headerLabel = isBoardFaceDown ? "face-down" : card.protocol;
  const headerValue = isBoardFaceDown ? 2 : card.value;

  // Which tier is "active" right now:
  //   board, uncovered → T  (top fires; middle suppressed)
  //   board, covered   → M  (middle fires; top suppressed by overlap)
  //   board, face-down → none
  //   hand             → none (all three shown equally; no position yet)
  const activeTier: "T" | "M" | null =
    variant === "board" && !isBoardFaceDown
      ? (uncovered ? "T" : "M")
      : null;

  return (
    <div
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={interactive ? onClick : undefined}
      onKeyDown={
        interactive
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
      style={{ height: CARD_TOTAL_PX }}
      className={cn(
        "relative flex flex-col rounded-md border bg-card text-card-foreground shadow-sm overflow-hidden select-none transition",
        "w-full max-w-[178px]",
        isBoardFaceDown && "opacity-80",
        card.isCommitted && "ring-1 ring-amber-500/50",
        selected && "ring-2 ring-emerald-500/70",
        interactive && "cursor-pointer hover:border-foreground/40",
        disabled && "opacity-50 cursor-not-allowed",
        className,
      )}
    >
      <div className="h-7 shrink-0 px-2.5 flex items-center justify-between border-b bg-muted/40">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-foreground/90 truncate">
          {headerLabel}
        </span>
        <span
          className={cn(
            "font-mono font-semibold leading-none text-base",
            isBoardFaceDown ? "text-muted-foreground" : valueTone(headerValue),
          )}
        >
          {headerValue}
        </span>
      </div>
      <CardTier label="T" text={card.topText} active={activeTier === "T"} faceDown={isBoardFaceDown} />
      <CardTier label="M" text={card.middleText} active={activeTier === "M"} faceDown={isBoardFaceDown} />
      <CardTier label="B" text={card.bottomText} active={false} faceDown={isBoardFaceDown} />
    </div>
  );
}

function CardTier({
  label,
  text,
  active,
  faceDown,
}: {
  label: "T" | "M" | "B";
  text: string;
  active: boolean;
  faceDown: boolean;
}) {
  return (
    <div
      className={cn(
        "shrink-0 px-2 py-1 text-[10px] leading-snug border-t overflow-hidden",
        "h-[52px]",
        active && "bg-emerald-500/10",
      )}
    >
      <div className="flex gap-1">
        <span
          className={cn(
            "font-mono shrink-0 leading-snug",
            active ? "text-emerald-300/80" : "text-foreground/35",
          )}
        >
          {label}
        </span>
        {faceDown ? (
          <span className="text-muted-foreground/40">—</span>
        ) : text ? (
          <span className="text-foreground/85 line-clamp-3">{text}</span>
        ) : (
          <span className="text-muted-foreground/40">—</span>
        )}
      </div>
    </div>
  );
}

function FaceDownBack({
  variant,
  onClick,
  disabled,
  selected,
  className,
}: {
  variant: Variant;
  onClick?: () => void;
  disabled?: boolean;
  selected?: boolean;
  className?: string;
}) {
  const interactive = !!onClick && !disabled;
  return (
    <div
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={interactive ? onClick : undefined}
      onKeyDown={
        interactive
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
      style={{ height: CARD_TOTAL_PX }}
      className={cn(
        "relative w-full max-w-[178px] rounded-md border border-dashed bg-muted/30 text-muted-foreground",
        "flex flex-col items-center justify-center select-none transition",
        selected && "ring-2 ring-emerald-500/70",
        interactive && "cursor-pointer hover:bg-muted/50 hover:border-foreground/40",
        disabled && "opacity-50 cursor-not-allowed",
        className,
      )}
    >
      <span className="text-[10px] font-mono uppercase tracking-[0.2em] opacity-60">
        {variant === "hand" ? "unknown" : "face-down"}
      </span>
      <span className="mt-1 font-mono text-lg opacity-70">2</span>
    </div>
  );
}

function valueTone(value: number): string {
  if (value >= 5) return "text-amber-400";
  if (value >= 3) return "text-foreground";
  return "text-muted-foreground";
}
