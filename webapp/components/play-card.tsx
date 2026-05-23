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
  /** Field-only: whether this card sits on top of its stack. Currently
   *  unused for active-tier highlighting (top is persistent so we always
   *  highlight T while face-up), but kept on the API so the parent can
   *  still distinguish stack position if it ever needs to. */
  uncovered?: boolean;
  selected?: boolean;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
};

// Fixed dimensions so stacked field cards overlap by a predictable amount.
// Layout: header (28) + 3 tiers (52 each) + footer (8) = 192px total at
// 16px root font. Stack overlap = middle + bottom + footer = 112px → the
// upper card hides the lower card's M + B tiers, leaving header + top
// tier visible. This matches Codex p.13 ("Top Command — Persistent: this
// passive text is never covered") and the rulebook back-cover note
// ("always ensure that the Value and the Top Command are always visible
// when covered"). Values expressed in rem so the cards scale with the
// responsive root font-size set in globals.css.
export const CARD_TOTAL_HEIGHT = "12rem";   // 192px at 16px root
export const CARD_OVERLAP_HEIGHT = "7rem";  // 112px at 16px root

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
  // Any face-down card on the field renders as the solid back, on both
  // sides of the field. The user's own face-down used to render as a
  // regular card with a "face-down" header and empty tier rows, which
  // looked inconsistent with the opponent's face-down (the FaceDownBack
  // pattern). In play mode the user already knows what they put down;
  // in record mode the recorder doesn't need to see their face-down's
  // identity on the board to keep transcribing.
  // Hand-card placeholders (defId=-1) still render as the back.
  const showAsBack = isBoardFaceDown || (!knownIdentity && variant !== "board");
  // `hidden` no longer affects rendering on its own — kept on the prop
  // surface for backwards-compat with callers that pass it.
  void hidden;

  if (showAsBack) {
    return <FaceDownBack variant={variant} onClick={onClick} disabled={disabled} selected={selected} className={className} />;
  }

  const interactive = !!onClick && !disabled;
  const headerLabel = isBoardFaceDown ? "face-down" : card.protocol;
  const headerValue = isBoardFaceDown ? 2 : card.value;

  // Which tiers are "live" right now (Codex p.13 "Card Anatomy"):
  //   T — Persistent: active while card is face-up regardless of cover.
  //   M — Immediate: fires once on play/flip/uncover. No continuous
  //       active state, so never highlighted.
  //   B — Auxiliary: only viable while the card is uncovered. We
  //       highlight it only if the card actually has bottom text;
  //       otherwise the emerald wash on an empty tier is misleading.
  // Both T and B highlights surface persistent effects "on the part of
  // the card where the effect lives" so the player can scan the field
  // and see what's still in play.
  const hasTop = !!card.topText || !!card.topEmphasis;
  const hasBottom = !!card.bottomText || !!card.bottomEmphasis;
  const tHighlighted = variant === "board" && !isBoardFaceDown && hasTop;
  const bHighlighted = variant === "board" && !isBoardFaceDown && uncovered && hasBottom;

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
      style={{ height: CARD_TOTAL_HEIGHT }}
      className={cn(
        // Solid card body — no opacity even when face-down. A face-down
        // card that's been played from the recorder's own seat is fully
        // known to the engine, and it still needs to opaquely cover the
        // bottom of whatever card it's stacked on top of.
        "relative flex flex-col rounded-md border bg-card text-card-foreground shadow-sm overflow-hidden select-none transition",
        "w-full max-w-[11.125rem]",
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
      <CardTier label="T" emphasis={card.topEmphasis} text={card.topText} active={tHighlighted} faceDown={isBoardFaceDown} />
      <CardTier label="M" emphasis={card.middleEmphasis} text={card.middleText} active={false} faceDown={isBoardFaceDown} />
      <CardTier label="B" emphasis={card.bottomEmphasis} text={card.bottomText} active={bHighlighted} faceDown={isBoardFaceDown} />
    </div>
  );
}

function CardTier({
  label,
  emphasis,
  text,
  active,
  faceDown,
}: {
  label: "T" | "M" | "B";
  emphasis: string;
  text: string;
  active: boolean;
  faceDown: boolean;
}) {
  const hasContent = !!emphasis || !!text;
  return (
    <div
      className={cn(
        "shrink-0 px-2 py-1 text-[10px] leading-snug border-t overflow-hidden",
        "h-[3.25rem]",
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
        ) : hasContent ? (
          // Emphasis ("Start:", "End:", "When covered:", etc.) is the
          // Codex-specified trigger label. Render it inline as a bolded
          // prefix so the line-clamp-3 budget is shared with the prose
          // and players can tell trigger from effect at a glance.
          <span className="line-clamp-3 text-foreground/85">
            {emphasis && (
              <span className="font-semibold text-amber-200/90">{emphasis}</span>
            )}
            {emphasis && text ? " " : ""}
            {text}
          </span>
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
      style={{ height: CARD_TOTAL_HEIGHT }}
      className={cn(
        // Solid card body so face-down cards opaquely cover the bottom
        // tiers of the card beneath them in a stack (rather than being a
        // ghosted overlay you can read text through). `bg-muted` gives
        // the back a clear "different from face-up" colour while staying
        // fully opaque.
        "relative w-full max-w-[11.125rem] rounded-md border border-foreground/15 bg-muted text-muted-foreground shadow-sm",
        "flex flex-col items-center justify-center select-none transition",
        selected && "ring-2 ring-emerald-500/70",
        interactive && "cursor-pointer hover:bg-muted/80 hover:border-foreground/40",
        disabled && "opacity-50 cursor-not-allowed",
        className,
      )}
    >
      <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/70">
        {variant === "hand" ? "unknown" : "face-down"}
      </span>
      <span className="mt-1 font-mono text-lg text-foreground/80">2</span>
    </div>
  );
}

function valueTone(value: number): string {
  if (value >= 5) return "text-amber-400";
  if (value >= 3) return "text-foreground";
  return "text-muted-foreground";
}
