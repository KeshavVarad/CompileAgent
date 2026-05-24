# Cross-validation against TheApo/compile

Independent implementations of the same game are the best free
cross-check on rules interpretation. This doc records what's been
verified by comparing our engine to
[TheApo/compile](https://github.com/TheApo/compile) (commit
`da8c2e5`, 2026-04-19), and what hasn't.

When in doubt, the
[Compile Codex PDF](../data/The_Compile_Codex.pdf) is still the
authoritative source. This doc is a confidence map, not a replacement.

## Method

We compared two surfaces:

1. **Card text** — concatenated `emphasis + text` per tier from
   [data/cards.json](../data/cards.json), normalised against their
   [data/cards.ts](https://github.com/TheApo/compile/blob/main/data/cards.ts)
   with HTML wrappers stripped.
2. **Handler semantics** for cards we recently changed or where the
   text disagreed — spot-read against their
   [logic/effects/](https://github.com/TheApo/compile/tree/main/logic/effects)
   and [logic/game/](https://github.com/TheApo/compile/tree/main/logic/game)
   files.

## Coverage of their reference implementation

Their `data/cards.ts` includes:

| Category | Their count | Coverage of overlap with ours |
|---|---|---|
| Main 1 (MN01) | 72 | full overlap with our MN01 |
| Aux 1 (AX01) | 18 | full overlap with our AX01 |
| **Main 2 (MN02)** | **6 (Chaos only)** | only Chaos cross-validated; Clarity / Corruption / Courage / Fear / Ice / Luck / Mirror / Peace / Smoke / Time / War **NOT** cross-validated |
| Aux 2 (AX02) | **0** | **NOT** cross-validated at all |
| Fan-Content (custom) | 12 | their own additions; we don't have these |

Additionally, their `custom_protocols/` folder defines fan-content
versions of some protocols (Corruption, Clarity, Love, Metal, Psychic,
Courage, plus an Anarchy protocol that doesn't exist in official
print). Those don't directly cross-validate the official cards but
provide a useful signal about the *mechanism approach* (e.g., how
they handle return-to-deck redirects).

## Confirmed semantics

Independent author arrived at the same handler logic for these cards,
giving high confidence they're correct:

### Recently-fixed trigger registrations (PR #38)
| Card | Tier | Trigger | Cross-validated? |
|---|---|---|---|
| **MN01:Life:3** | bottom | fires on cover (not on play) | ✓ in their cards.ts + `processReactiveEffects('on_cover', …)` |
| **AX01:Apathy:2** | bottom | fires on cover (not on play) | ✓ same |
| **AX01:Hate:4** | bottom | fires on cover (not on play) | ✓ same |
| **MN02:Clarity:1** | bottom | fires on cover (not on play) | **NOT cross-validated** — Clarity isn't in their MN02 coverage. Codex-only. |

Their `processReactiveEffects(state, 'on_cover', …)` path in
`playResolver.ts` fires before the covering card lands, matching our
`@when_covered` dispatch in
[src/compile_engine/game.py](../src/compile_engine/game.py).

### Contested text → resolved

| Card | Disagreement | Resolution |
|---|---|---|
| **MN02:Chaos:3** | Their text reads "You may play cards without matching protocols" (sounds global); ours reads "This card may be played…" (self-only). | Their HANDLER comments explicitly: *"Chaos-3 uses card_property instead (only affects that card being played, not all cards)"*. Their text is a misleading paraphrase; their actual implementation matches ours (self-only scope). |
| **MN01:Life:0** | Theirs has the bottom "When this card would be covered: First, delete this card."; ours has the top "End: If covered, delete this card." | We applied the Dec 2024 Codex errata (see `ERRATA` dict in [webapp/lib/compile/cards.ts](../webapp/lib/compile/cards.ts) and [src/compile_engine/cards.py](../src/compile_engine/cards.py)). Theirs is pre-errata. Our version is current. |

### Mechanism-level corroboration (not direct card cross-check)

| Topic | Their approach | Our approach |
|---|---|---|
| **Return-to-deck redirect** (Corruption 1) | `hasRedirectReturnToDeckPassive` + redirect inside `actionUtils.ts`. Implemented for their *fan-content* Corruption protocol. | Same approach: `_has_active_corruption_1` + redirect inside `return_card_to_hand`. Official MN02 Corruption 1 (PR #40). |

Their fan-content Corruption isn't the same as the official MN02
Corruption, but the *engineering approach* of intercepting the return
at the helper level (rather than firing a separate handler after the
fact) is the same conclusion we reached independently — gives some
indirect confidence in the PR #40 design.

### Text differences with no behavioral impact

These differences exist but don't represent semantic disagreements:

- **MN01:Death:1 top** — wording ("Then, delete this card." vs ", then delete this card.")
- **MN01:Fire:0 bottom** — wording ("First, draw 1 card. Then, flip 1 other card." vs "First, draw 1 card and flip 1 other card.")
- **MN01:Metal:1 middle** — "on their next turn" vs "next turn"
- **MN01:Plague:2 middle** — their typo "Dicard"
- **MN01:Spirit:1 top** — paraphrase ("they may be played without matching protocols" vs "you can play cards in any line")
- **MN02:Chaos:4 bottom** — wording ("Draw that many cards" vs "Draw the same amount")
- **MN02:Chaos:5 middle** — value-5 standard text variant ("You discard 1 card" vs "Discard 1 card")

## NOT cross-validated by this method

These require Codex-only verification, in-app bug reports from real
play, or future independent implementations to corroborate:

### Most of Main 2 (66 cards)
- All Clarity, Corruption, Courage, Fear, Ice, Luck, Mirror, Peace,
  Smoke, Time, War cards. Only Chaos is in their official data.

### All of Aux 2 (17 cards)
- All Assimilation 0-6
- All Diversity 0-6
- All Unity 0-6 — including Unity 0's "flipped by a Unity card" trigger
  (handled via `cause_card` parameter in our
  [flip_card](../src/compile_engine/effects.py); see PR #40)

### Semantic correctness of complex handlers
The cross-check is data-level + spot-read; we did NOT walk every
handler body line by line. Cards with multi-clause / conditional /
optional behaviour could still have subtle divergences from the
rulebook that this audit didn't surface. Examples whose correctness
ultimately rests on the Codex + our own tests:

- Choice ordering / "First:" timing semantics
- "Highest value" tie-breaking rules
- Hidden-information handling during effect chains
- Self-target prohibitions ("1 of your *other* cards")
- Multi-card targeting ("all face-up cards in this line")

### Channel for surfacing future disagreements
Bug reports filed via the in-app
[BugReportDialog](../webapp/components/bug-report-dialog.tsx) (added
in PR #31) are the canonical inbox for card-correctness issues
caught in play. The fixes typically flow through the same workflow as
PRs #32, #38, #40 — playtester report → handler audit →
single-card fix + regression test.

## Re-running the audit

Clone the reference implementation:

```bash
git clone https://github.com/TheApo/compile.git /tmp/compile-other
```

Then run a small Python diff script that:
1. Parses their `data/cards.ts` with a regex over the card object
   literals (`protocol: "…", value: N, top: "…", middle: "…", bottom:
   "…", … category: "…"`).
2. Strips their HTML wrappers (`<div><span class='emphasis'>…</span>
   …</div>`).
3. Normalises whitespace + acute-accent quote variants.
4. Concatenates our `emphasis + text` per tier and compares.

If the reference repo updates and you want to re-verify, the
behavioural cross-checks with the highest value are searching for
patterns that surfaced fixes here:

- **`onCover` / `whenCovered`** — confirm bottom-emphasis "When this
  card would be covered" cards trigger on cover, not on play.
- **`hasRedirectReturnToDeckPassive` / Corruption-style** — confirm
  the return-redirect happens inside the helper, not as a separate
  handler.
- **`card_property` / `ignore_protocol_matching`** — confirm self-only
  scoping for Chaos 3 (not a global rule).

## Confidence summary

| Area | Confidence | Source |
|---|---|---|
| MN01 base set rules | **High** | Cross-validated against TheApo |
| AX01 expansion rules | **High** | Cross-validated against TheApo |
| MN02 Chaos protocol | **High** | Cross-validated against TheApo |
| MN02 rest (11 protocols) | **Medium** | Codex + integration tests only |
| AX02 (Assimilation/Diversity/Unity) | **Medium** | Codex + integration tests only |
| Dec 2024 errata applied | **High** | We have explicit errata dict; TheApo is pre-errata |
| When-covered trigger timing | **High** | Cross-validated via 3 cards (Life 3, Apathy 2, Hate 4) |
| Return-to-deck redirect mechanism | **Medium-High** | Same approach as TheApo's fan-content; PR #40 logic |
| Unity 0 "flipped by Unity" gating | **Medium** | Codex-only; not in TheApo's coverage |
| Per-card semantic edge cases | **Variable** | Best caught via playtester reports |
