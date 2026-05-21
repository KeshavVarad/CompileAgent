/**
 * Game orchestrator — port of src/compile_engine/game.py.
 *
 * Lifecycle:
 *   const game = new Game(config);
 *   game.start();
 *   while (!game.isOver()) {
 *     const decider = game.decider();
 *     const legal = game.legalActions();
 *     const action = humanOrBotChooses(legal);
 *     game.step(action);
 *   }
 */

import {
  AUX2_SET,
  BASE_SET,
  CARD_DEFS,
  EXPANSION_SET,
  MAIN2_SET,
  defsForProtocol,
  protocolsForSets,
  type Protocol,
  type SetCode,
} from "./cards";
import {
  type EffectFn,
  type EffectGen,
  getAfterClearCacheEffect,
  getAfterOppDiscardEffect,
  getAfterSelfDeleteEffect,
  getAfterSelfDiscardEffect,
  getAfterSelfDrawEffect,
  getAfterSelfRefreshEffect,
  getAfterSelfShuffleEffect,
  getBottomFirstEffect,
  getBottomOnPlayEffect,
  getEndEffect,
  getFlipTriggerEffect,
  getMiddleEffect,
  getStartEffect,
  getTopTriggerEffect,
  getWhenCoveredEffect,
  getWhenDeletedByCompileEffect,
  uncommitSentinel,
} from "./effects";
import {
  computeLineValue,
  discardToTrash,
  drawCards,
  lineStack,
  middleSuppressed,
  oppMustPlayFacedown,
  oppPlayBlockedInLine,
  oppPlayFacedownBlockedInLine,
  playerCanCompile,
  playerMayPlayAnyLineFaceup,
  playerSkipsCheckCache,
} from "./helpers";
import { createRng, rngFloat, rngShuffle } from "./rng";
import type {
  Action,
  CardInst,
  Choice,
  GameConfig,
  GameState,
  PlayerIndex,
  PlayerState,
} from "./types";
import {
  COMPILE_THRESHOLD,
  HAND_SIZE_LIMIT,
  MAX_EFFECT_PUSHES_PER_TURN,
  MAX_EFFECT_STACK_DEPTH,
  NUM_LINES,
  NUM_PROTOCOLS_PER_PLAYER,
  STARTING_HAND,
} from "./types";

type PendingEffect = {
  gen: EffectGen;
  lastChoice: Choice | null;
};

export class Game {
  state: GameState;
  private pending: PendingEffect[] = [];
  private instCounter = 0;

  constructor(config: GameConfig) {
    const rng = createRng(config.seed);
    const enabled = new Set<SetCode>([BASE_SET]);
    if (config.includeExpansion) enabled.add(EXPANSION_SET);
    if (config.includeMain2) enabled.add(MAIN2_SET);
    if (config.includeAux2) enabled.add(AUX2_SET);
    const draftPool: Protocol[] = protocolsForSets(enabled);
    rngShuffle(rng, draftPool);

    const emptyPlayer = (idx: PlayerIndex): PlayerState => ({
      idx,
      deck: [],
      hand: [],
      trash: [],
      protocols: [],
      compiled: [],
      cannotCompileNextTurn: false,
    });

    this.state = {
      config,
      players: [emptyPlayer(0), emptyPlayer(1)],
      lines: Array.from({ length: NUM_LINES }, () => ({ p0Stack: [], p1Stack: [] })),
      currentPlayer: 0,
      turn: 0,
      phase: "DRAFT",
      controlHolder: null,
      winner: null,
      triggers: [],
      scratch: { _engine: this },
      effectPushesThisTurn: 0,
      compiledThisTurn: false,
      draftPool,
      draftIdx: 0,
      draftSchedule: [0, 1, 1, 0, 0, 1],
      log: [],
      rngState: rng.state,
    };
  }

  start(): void { this.drive(); }
  isOver(): boolean { return this.state.phase === "GAME_OVER"; }

  decider(): PlayerIndex {
    if (this.pending.length > 0) {
      const top = this.pending[this.pending.length - 1];
      if (top.lastChoice) return top.lastChoice.decider;
    }
    return this.state.currentPlayer;
  }

  legalActions(): Action[] {
    const st = this.state;
    if (st.phase === "GAME_OVER") return [];

    // Mid-effect choice prompt.
    if (this.pending.length > 0 && this.pending[this.pending.length - 1].lastChoice) {
      const choice = this.pending[this.pending.length - 1].lastChoice!;
      const acts: Action[] = choice.options.map((_, i) => ({
        type: "CHOOSE_TARGET", choiceIndex: i,
      } as Action));
      if (choice.optional) acts.push({ type: "SKIP_OPTIONAL" });
      return acts;
    }

    if (st.phase === "DRAFT") {
      return st.draftPool.map((p) => ({ type: "DRAFT_PROTOCOL", protocol: p } as Action));
    }
    if (st.phase === "CHECK_CACHE") {
      const ps = st.players[st.currentPlayer];
      return ps.hand.map((_, i) => ({ type: "DISCARD_CARD", handIndex: i } as Action));
    }
    if (st.phase === "CHECK_COMPILE") {
      const ap = st.currentPlayer;
      return this.compileableLines(ap).map((ln) => ({
        type: "COMPILE_LINE", lineIndex: ln,
      } as Action));
    }
    if (st.phase === "ACTION") return this.actionPhaseLegal();
    return [{ type: "NOOP" }];
  }

  private actionPhaseLegal(): Action[] {
    const st = this.state;
    const ap = st.currentPlayer;
    const ps = st.players[ap];
    const actions: Action[] = [];
    const spirit1Active = playerMayPlayAnyLineFaceup(st, ap);
    const psychic1Forces = oppMustPlayFacedown(st, ap);
    const lineBlocked = [0, 1, 2].map((ln) => oppPlayBlockedInLine(st, ln, ap));
    const lineFdBlocked = [0, 1, 2].map((ln) => oppPlayFacedownBlockedInLine(st, ln, ap));
    // In record mode the engine doesn't know hand-card identities, so it
    // can't pre-check protocol matching. We enumerate all (hand, line)
    // combinations and let the recorder filter via the identity picker;
    // playCard trusts the recorder's revealedDefId at submission time.
    const recordMode = (st.config.mode ?? "play") === "record";

    for (let hi = 0; hi < ps.hand.length; hi++) {
      const c = ps.hand[hi];
      const d = c.defId === -1 ? null : CARD_DEFS[c.defId];
      const chaos3Self = d?.protocol === "Chaos" && d?.value === 3;
      const corruption0Self = d?.protocol === "Corruption" && d?.value === 0;
      const unrestrictedFu = spirit1Active || chaos3Self || corruption0Self || recordMode;
      for (let ln = 0; ln < NUM_LINES; ln++) {
        if (lineBlocked[ln] && !recordMode) continue;
        if (psychic1Forces && !recordMode) continue;
        if (unrestrictedFu || (d != null && ps.protocols[ln] === d.protocol)) {
          actions.push({ type: "PLAY_FACE_UP", handIndex: hi, lineIndex: ln });
        }
      }
      for (let ln = 0; ln < NUM_LINES; ln++) {
        if ((lineBlocked[ln] || lineFdBlocked[ln]) && !recordMode) continue;
        actions.push({ type: "PLAY_FACE_DOWN", handIndex: hi, lineIndex: ln });
      }
      // Corruption 0: may also be played onto OPPONENT's side. Use line
      // indices NUM_LINES..(2*NUM_LINES-1) → opp lines 0..2.
      if (corruption0Self) {
        for (let ln = 0; ln < NUM_LINES; ln++) {
          actions.push({ type: "PLAY_FACE_UP", handIndex: hi, lineIndex: NUM_LINES + ln });
          actions.push({ type: "PLAY_FACE_DOWN", handIndex: hi, lineIndex: NUM_LINES + ln });
        }
      }
    }

    // Spirit 3 only — Speed 2's "shift even if covered" is a targetability
    // modifier (other shift effects can target it), not a standalone action.
    for (let lnSrc = 0; lnSrc < NUM_LINES; lnSrc++) {
      const stack = lineStack(st.lines[lnSrc], ap);
      stack.forEach((c, pos) => {
        if (!c.faceUp) return;
        const d = CARD_DEFS[c.defId];
        if (!(d.protocol === "Spirit" && d.value === 3)) return;
        for (let lnDst = 0; lnDst < NUM_LINES; lnDst++) {
          if (lnDst === lnSrc) continue;
          actions.push({ type: "SHIFT_OWN_CARD", lineIndex: lnSrc, handIndex: pos, choiceIndex: lnDst });
        }
      });
    }

    if (actions.length === 0) return [{ type: "REFRESH" }];
    actions.push({ type: "REFRESH" });
    return actions;
  }

  step(action: Action): void {
    const st = this.state;
    if (st.phase === "GAME_OVER") throw new Error("Game over");

    st.log.push({
      turn: st.turn, decider: this.decider(), action, timestamp: Date.now(),
    });

    if (this.pending.length > 0 && this.pending[this.pending.length - 1].lastChoice) {
      this.resolveChoice(action);
      this.drive();
      return;
    }

    switch (st.phase) {
      case "DRAFT": this.doDraftPick(action); break;
      case "CHECK_CACHE": this.doClearCache(action); break;
      case "CHECK_COMPILE": this.doCompile(action); break;
      case "ACTION": this.doAction(action); break;
      default: break;
    }
    this.drive();
  }

  /** Drive auto-phases + drain effect stack until a player decision is needed. */
  /** Drain pending effects + triggers until a decision is needed or the
   *  queue empties. Returns true if a decision is needed. */
  private drainPending(): boolean {
    const st = this.state;
    while (true) {
      if (
        st.effectPushesThisTurn >= MAX_EFFECT_PUSHES_PER_TURN &&
        (this.pending.length > 0 || st.triggers.length > 0)
      ) {
        this.pending = [];
        st.triggers = [];
        return false;
      }
      while (this.pending.length > 0) {
        const top = this.pending[this.pending.length - 1];
        if (top.lastChoice) return true;
        if (st.triggers.length > 0) { this.fireNextTrigger(); continue; }
        const res = top.gen.next();
        if (res.done) {
          const idx = this.pending.indexOf(top);
          if (idx >= 0) this.pending.splice(idx, 1);
          continue;
        }
        top.lastChoice = res.value;
        return true;
      }
      if (st.triggers.length > 0) { this.fireNextTrigger(); continue; }
      return false;
    }
  }

  private drive(): void {
    const st = this.state;
    // Drain effects, then any deferred "after X" broadcasts, then drain
    // again. Loop until stable or a decision is needed.
    while (true) {
      if (this.drainPending()) return;
      if (!this.drainPendingAfterEvents()) break;
    }
    // Phase advancement.
    while (true) {
      if (st.phase === "GAME_OVER") return;
      if (st.phase === "DRAFT") return;
      if (st.phase === "START") { if (this.doStartPhase()) return; continue; }
      if (st.phase === "CHECK_CONTROL") { this.doCheckControl(); continue; }
      if (st.phase === "CHECK_COMPILE") {
        if (this.compileableLines(st.currentPlayer).length > 0) return;
        st.phase = "ACTION"; continue;
      }
      if (st.phase === "ACTION") return;
      if (st.phase === "CHECK_CACHE") {
        const ps = st.players[st.currentPlayer];
        if (playerSkipsCheckCache(st, st.currentPlayer)) { st.phase = "END"; continue; }
        if (ps.hand.length <= HAND_SIZE_LIMIT) {
          // Cache phase complete — fire `After you clear cache:` triggers if
          // a discard happened, then drain them before END.
          const ap = st.currentPlayer;
          const fk = `_pending_after_clear_cache_p${ap}`;
          const scratch = st.scratch as Record<string, unknown>;
          if (scratch[fk]) {
            delete scratch[fk];
            this.broadcastAfterClearCache(ap);
            if (this.drainPending()) return;
          }
          st.phase = "END"; continue;
        }
        return;
      }
      if (st.phase === "END") { if (this.doEndPhase()) return; continue; }
    }
  }

  private pushEffect(gen: EffectGen): void {
    const st = this.state;
    if (this.pending.length >= MAX_EFFECT_STACK_DEPTH) return;
    if (st.effectPushesThisTurn >= MAX_EFFECT_PUSHES_PER_TURN) return;
    st.effectPushesThisTurn++;
    this.pending.push({ gen, lastChoice: null });
  }

  private fireNextTrigger(): void {
    const st = this.state;
    const t = st.triggers.pop()!;
    if (t.kind === "uncommit") { t.card.isCommitted = false; return; }
    if (t.kind === "when_covered") {
      const stack = lineStack(st.lines[t.line], t.player);
      if (!stack.includes(t.card) || !t.card.faceUp) return;
      if (stack.indexOf(t.card) === stack.length - 1) return; // not actually covered
      const fn = getWhenCoveredEffect(t.card.defId);
      if (fn) this.pushEffect(fn(st, t.player, t.line, t.card));
      return;
    }
    if (t.kind === "reveal_placeholder") {
      // A placeholder card needs identity-fill from the recorder before its
      // face-up trigger can resolve. Push an effect that yields a Choice;
      // resolveChoice patches the card's defId, then the LIFO drain resumes
      // and the face_up trigger (re-pushed below) fires its real effects.
      this.pushEffect(this.revealPlaceholderEffect(t.card, t.player, t.line));
      st.triggers.push({ kind: "face_up", line: t.line, player: t.player, card: t.card });
      return;
    }
    const stack = lineStack(st.lines[t.line], t.player);
    if (!stack.includes(t.card) || !t.card.faceUp) return;
    if (t.kind === "face_up") { this.enqueueFaceUpTriggers(t.card, t.player, t.line); return; }
    // uncover: middle only
    if (middleSuppressed(st, t.line, t.card)) return;
    if (t.card.defId === -1) return;
    const d = CARD_DEFS[t.card.defId];
    if (!d.middleText) return;
    const mf = getMiddleEffect(t.card.defId);
    if (mf) this.pushEffect(mf(st, t.player, t.line, t.card));
  }

  private *revealPlaceholderEffect(
    card: CardInst, player: PlayerIndex, lineIdx: number,
  ): EffectGen {
    const st = this.state;
    const opp = st.players[player];
    // Candidate identities: cards from the opponent's drafted protocols that
    // haven't been definitively used yet. We don't enforce uniqueness — the
    // recorder is asserting they saw this reveal.
    const candidates: { defId: number; label: string }[] = [];
    for (const proto of opp.protocols) {
      for (const d of defsForProtocol(proto)) {
        candidates.push({ defId: d.defId, label: `${d.protocol} ${d.value}` });
      }
    }
    const idx = (yield {
      prompt: `What was the card just revealed in line ${lineIdx + 1}?`,
      options: candidates.map((c) => c.label),
      targets: candidates.map((c) => c.defId),
      optional: false,
      decider: st.config.recorderSeat ?? player,
    }) as number;
    if (typeof idx === "number" && idx >= 0 && idx < candidates.length) {
      card.defId = candidates[idx].defId;
    }
  }

  private enqueueFaceUpTriggers(c: CardInst, ap: PlayerIndex, lineIdx: number): void {
    // Push in reverse-resolution order so LIFO drain runs top -> bottom_first -> middle.
    const d = CARD_DEFS[c.defId];
    const midFn: EffectFn | null = middleSuppressed(this.state, lineIdx, c) ? null : getMiddleEffect(c.defId);
    if (midFn && d.middleText) this.pushEffect(midFn(this.state, ap, lineIdx, c));
    const bf = getBottomFirstEffect(c.defId);
    if (bf) this.pushEffect(bf(this.state, ap, lineIdx, c));
    const bp = getBottomOnPlayEffect(c.defId);
    if (bp) this.pushEffect(bp(this.state, ap, lineIdx, c));
    const tt = getTopTriggerEffect(c.defId);
    if (tt) this.pushEffect(tt(this.state, ap, lineIdx, c));
  }

  enqueueEnterPlayTriggersSkipMiddle(c: CardInst, ap: PlayerIndex, lineIdx: number): void {
    // Luck 1: "Flip that card, ignoring its middle commands" — fire top and
    // bottom triggers but skip middle.
    const bf = getBottomFirstEffect(c.defId);
    if (bf) this.pushEffect(bf(this.state, ap, lineIdx, c));
    const bp = getBottomOnPlayEffect(c.defId);
    if (bp) this.pushEffect(bp(this.state, ap, lineIdx, c));
    const tt = getTopTriggerEffect(c.defId);
    if (tt) this.pushEffect(tt(this.state, ap, lineIdx, c));
  }

  // ---------------------------------------------------------------- DRAFT

  private doDraftPick(a: Action): void {
    if (a.type !== "DRAFT_PROTOCOL" || !a.protocol) throw new Error("expected DRAFT_PROTOCOL");
    if (!this.state.draftPool.includes(a.protocol)) throw new Error("protocol not in pool");
    const picker = this.state.draftSchedule[this.state.draftIdx];
    this.state.players[picker].protocols.push(a.protocol);
    this.state.players[picker].compiled.push(false);
    this.state.draftPool = this.state.draftPool.filter((p) => p !== a.protocol);
    this.state.draftIdx++;
    if (this.state.draftIdx >= this.state.draftSchedule.length) this.finalizeDraft();
  }

  private finalizeDraft(): void {
    const st = this.state;
    const mode = st.config.mode ?? "play";
    for (const pl of [0, 1] as PlayerIndex[]) {
      const isRecordMode = mode === "record";
      const deck: CardInst[] = [];
      if (isRecordMode) {
        // In record mode neither hand is known to the engine up front: the
        // recorder's own physical shuffle determines which card came up, and
        // the opponent's hand is hidden by the game's rules. Both decks are
        // filled with placeholders (defId=-1); the recorder fills in
        // identities at play / reveal time via the action's revealedDefId.
        const total = st.players[pl].protocols.length * 6;
        for (let i = 0; i < total; i++) {
          deck.push({
            instId: this.instCounter++,
            defId: -1,
            owner: pl,
            faceUp: false,
            isCommitted: false,
          });
        }
      } else {
        for (const proto of st.players[pl].protocols) {
          for (const d of defsForProtocol(proto)) {
            deck.push({
              instId: this.instCounter++,
              defId: d.defId,
              owner: pl,
              faceUp: false,
              isCommitted: false,
            });
          }
        }
      }
      const rng = { state: st.rngState };
      rngShuffle(rng, deck);
      st.rngState = rng.state;
      st.players[pl].deck = deck;
      drawCards(st, pl, STARTING_HAND);
    }
    st.phase = "START";
    st.currentPlayer = 0;
    st.turn = 1;
    st.compiledThisTurn = false;
    st.effectPushesThisTurn = 0;
  }

  // ---------------------------------------------------------------- PHASES

  private doStartPhase(): boolean {
    const st = this.state;
    let anyPushed = false;
    for (let ln = 0; ln < NUM_LINES; ln++) {
      const snapshot = [...lineStack(st.lines[ln], st.currentPlayer)];
      for (const c of snapshot) {
        if (!c.faceUp) continue;
        const fn = getStartEffect(c.defId);
        if (fn) { this.pushEffect(fn(st, st.currentPlayer, ln, c)); anyPushed = true; }
      }
    }
    st.phase = "CHECK_CONTROL";
    return anyPushed;
  }

  private doCheckControl(): void {
    const st = this.state;
    const ap = st.currentPlayer;
    let wonLines = 0;
    for (let ln = 0; ln < NUM_LINES; ln++) {
      if (computeLineValue(st, ln, ap) > computeLineValue(st, ln, ap === 0 ? 1 : 0)) wonLines++;
    }
    if (wonLines >= 2) st.controlHolder = ap;
    st.phase = "CHECK_COMPILE";
  }

  private compileableLines(player: PlayerIndex): number[] {
    const st = this.state;
    if (st.compiledThisTurn) return [];
    if (!playerCanCompile(st, player)) return [];
    const out: number[] = [];
    const opp: PlayerIndex = player === 0 ? 1 : 0;
    for (let ln = 0; ln < NUM_LINES; ln++) {
      const ours = computeLineValue(st, ln, player);
      const theirs = computeLineValue(st, ln, opp);
      if (ours >= COMPILE_THRESHOLD && ours > theirs) out.push(ln);
    }
    return out;
  }

  private doCompile(a: Action): void {
    if (a.type !== "COMPILE_LINE" || a.lineIndex == null) throw new Error("expected COMPILE_LINE");
    const st = this.state;
    const ln = a.lineIndex;
    const line = st.lines[ln];
    // Codex p.11: when compiling, all cards in the line are deleted "at the
    // same time". Cards with a `when_deleted_by_compile` interrupt (Speed 2)
    // fire first and can shift themselves out, surviving the compile.
    const interrupts: Array<{ pl: PlayerIndex; c: CardInst; fn: EffectFn }> = [];
    for (const pl of [0, 1] as PlayerIndex[]) {
      const stack = pl === 0 ? line.p0Stack : line.p1Stack;
      for (const c of stack) {
        if (!c.faceUp || c.defId === -1) continue;
        const fn = getWhenDeletedByCompileEffect(c.defId);
        if (fn) interrupts.push({ pl, c, fn });
      }
    }
    if (interrupts.length === 0) {
      this.compileFinalize(a);
      return;
    }
    const ap = st.currentPlayer;
    // Push finalizer first (drains last), then interrupts in reverse so the
    // first-listed drains first under LIFO.
    this.pushEffect(this.compileFinalizerGen(a));
    for (let i = interrupts.length - 1; i >= 0; i--) {
      const it = interrupts[i];
      this.pushEffect(it.fn(st, ap, ln, it.c));
    }
  }

  private *compileFinalizerGen(a: Action): EffectGen {
    this.compileFinalize(a);
    if (false) yield {} as Choice;
  }

  private compileFinalize(a: Action): void {
    if (a.type !== "COMPILE_LINE" || a.lineIndex == null) throw new Error("expected COMPILE_LINE");
    const st = this.state;
    const ap = st.currentPlayer;
    const opp: PlayerIndex = ap === 0 ? 1 : 0;
    const ln = a.lineIndex;
    const line = st.lines[ln];
    for (const c of line.p0Stack) { c.faceUp = true; st.players[c.owner].trash.push(c); }
    for (const c of line.p1Stack) { c.faceUp = true; st.players[c.owner].trash.push(c); }
    line.p0Stack = []; line.p1Stack = [];
    const { checkDiversity6SelfDestruct } = require("./helpers");
    checkDiversity6SelfDestruct(st);
    if (st.players[ap].compiled[ln]) {
      if (st.players[opp].deck.length > 0) {
        const c = st.players[opp].deck.pop()!;
        c.owner = ap; c.faceUp = false;
        st.players[ap].hand.push(c);
      }
    } else {
      st.players[ap].compiled[ln] = true;
    }
    st.compiledThisTurn = true;
    if (st.players[ap].compiled.every(Boolean) && st.players[ap].compiled.length === NUM_PROTOCOLS_PER_PLAYER) {
      st.winner = ap; st.phase = "GAME_OVER"; return;
    }
    st.phase = "END";
  }

  private doAction(a: Action): void {
    const st = this.state;
    const ap = st.currentPlayer;
    if (a.type === "REFRESH") {
      const ps = st.players[ap];
      const need = STARTING_HAND - ps.hand.length;
      if (need > 0) drawCards(st, ap, need);
      st.phase = "CHECK_CACHE"; return;
    }
    if (a.type === "PLAY_FACE_UP") {
      // Record-mode: when a placeholder hand card is played face-up, the
      // recorder identifies it via `revealedDefId`. Patch the placeholder
      // before `playCard` runs its protocol-match check.
      this.patchPlaceholderFromAction(ap, a);
      this.playCard(ap, a.handIndex!, a.lineIndex!, true);
      st.phase = "CHECK_CACHE"; return;
    }
    if (a.type === "PLAY_FACE_DOWN") {
      // Record-mode: the recorder knows what they're playing face-down, so
      // they fill in the identity at play time. The opponent's face-down
      // plays leave revealedDefId unset and stay as placeholders.
      this.patchPlaceholderFromAction(ap, a);
      this.playCard(ap, a.handIndex!, a.lineIndex!, false);
      st.phase = "CHECK_CACHE"; return;
    }
    if (a.type === "SHIFT_OWN_CARD") {
      // Spirit 3 affordance only.
      const srcLine = a.lineIndex!;
      const srcPos = a.handIndex!;
      const dstLine = a.choiceIndex!;
      const stack = lineStack(st.lines[srcLine], ap);
      const c = stack[srcPos];
      const d = c.defId === -1 ? null : CARD_DEFS[c.defId];
      const valid = c.faceUp && d != null && d.protocol === "Spirit" && d.value === 3;
      if (!valid) throw new Error("invalid SHIFT_OWN_CARD target");
      const { shiftCard } = require("./helpers");
      shiftCard(st, srcLine, ap, srcPos, dstLine);
      st.phase = "CHECK_CACHE"; return;
    }
    throw new Error(`unexpected action in ACTION phase: ${a.type}`);
  }

  playCardForEffect(player: PlayerIndex, handIndex: number, lineIndex: number, faceUp: boolean): void {
    this.playCard(player, handIndex, lineIndex, faceUp);
  }

  private playCard(player: PlayerIndex, handIndex: number, lineIndex: number, faceUp: boolean): void {
    const st = this.state;
    const ps = st.players[player];
    if (handIndex < 0 || handIndex >= ps.hand.length) throw new Error("bad hand index");
    const c = ps.hand.splice(handIndex, 1)[0];
    // Face-up plays require a known def_id (the recorder must patch
    // placeholders via revealedDefId before calling playCard). Face-down
    // plays preserve the placeholder.
    if (faceUp && c.defId === -1) {
      throw new Error("cannot play a placeholder card face-up without revealedDefId");
    }
    const d = c.defId === -1
      ? null
      : CARD_DEFS[c.defId];
    // Corruption 0 may target the opponent's side via lineIndex in [3,6).
    let targetSide: PlayerIndex = player;
    let actualLine = lineIndex;
    const crossSide =
      d != null && d.protocol === "Corruption" && d.value === 0 &&
      lineIndex >= NUM_LINES && lineIndex < 2 * NUM_LINES;
    if (crossSide) {
      targetSide = (player === 0 ? 1 : 0) as PlayerIndex;
      actualLine = lineIndex - NUM_LINES;
    }
    if (faceUp && d != null) {
      const chaos3Self = d.protocol === "Chaos" && d.value === 3;
      const corruption0Self = d.protocol === "Corruption" && d.value === 0;
      const targetProtos = st.players[targetSide].protocols;
      // Record mode is a transcription of a real game — the recorder is
      // asserting they saw the play happen, possibly enabled by an effect
      // the engine doesn't fully reason about (e.g. opp's Spirit 1 at a
      // moment we didn't reconstruct). Skip the protocol-match check for
      // both seats in record mode.
      const recordTrust = (st.config.mode ?? "play") === "record";
      const ok =
        recordTrust ||
        targetProtos[actualLine] === d.protocol ||
        playerMayPlayAnyLineFaceup(st, player) ||
        chaos3Self || corruption0Self;
      if (!ok) throw new Error(`face-up ${d.protocol} must match line protocol`);
      c.faceUp = true;
    } else {
      c.faceUp = faceUp;
    }
    if (crossSide) c.owner = targetSide;
    c.isCommitted = true;
    const target = lineStack(st.lines[actualLine], targetSide);
    const soonCovered = target.length > 0 ? target[target.length - 1] : null;
    target.push(c);
    // Push enter-play effects.
    if (faceUp && d != null) {
      const midFn = middleSuppressed(st, actualLine, c) ? null : getMiddleEffect(c.defId);
      if (midFn && d.middleText) this.pushEffect(midFn(st, player, actualLine, c));
      const bf = getBottomFirstEffect(c.defId);
      if (bf) this.pushEffect(bf(st, player, actualLine, c));
      const bp = getBottomOnPlayEffect(c.defId);
      if (bp) this.pushEffect(bp(st, player, actualLine, c));
      const tt = getTopTriggerEffect(c.defId);
      if (tt) this.pushEffect(tt(st, player, actualLine, c));
    }
    this.pushEffect(uncommitSentinel(c));
    if (soonCovered && soonCovered.faceUp) {
      const wc = getWhenCoveredEffect(soonCovered.defId);
      if (wc) this.pushEffect(wc(st, soonCovered.owner, actualLine, soonCovered));
    }
    // Diversity 6 continuous check (a fresh play may bring its own protocol
    // count up to/keep at 3, but Diversity 6 itself entering a sub-3 field
    // must immediately self-destruct).
    const { checkDiversity6SelfDestruct } = require("./helpers");
    checkDiversity6SelfDestruct(st);
  }

  private doClearCache(a: Action): void {
    if (a.type !== "DISCARD_CARD" || a.handIndex == null) throw new Error("expected DISCARD_CARD");
    // Record-mode: if the recorder is discarding their own placeholder, fill
    // in its identity so the trash stays consistent for any later inspection.
    this.patchPlaceholderFromAction(this.state.currentPlayer, a);
    discardToTrash(this.state, this.state.currentPlayer, a.handIndex);
    // Flag that this CHECK_CACHE phase performed a Clear Cache action so the
    // exit transition broadcasts `After you clear cache:` triggers.
    (this.state.scratch as Record<string, unknown>)[
      `_pending_after_clear_cache_p${this.state.currentPlayer}`
    ] = true;
  }

  private broadcastForSide(
    owner: PlayerIndex,
    getter: (defId: number) => EffectFn | null,
  ): boolean {
    const st = this.state;
    let pushed = false;
    for (let ln = 0; ln < 3; ln++) {
      for (const c of [...lineStack(st.lines[ln], owner)]) {
        if (!c.faceUp || c.defId === -1) continue;
        const fn = getter(c.defId);
        if (fn) { this.pushEffect(fn(st, owner, ln, c)); pushed = true; }
      }
    }
    return pushed;
  }

  private broadcastAfterClearCache(owner: PlayerIndex): boolean {
    return this.broadcastForSide(owner, getAfterClearCacheEffect);
  }

  private drainPendingAfterEvents(): boolean {
    const st = this.state;
    const scratch = st.scratch as Record<string, unknown>;
    let pushedAny = false;
    for (const p of [0, 1] as PlayerIndex[]) {
      const dk = `_pending_after_discard_by_p${p}`;
      if (scratch[dk]) {
        delete scratch[dk];
        pushedAny = this.broadcastForSide(p, getAfterSelfDiscardEffect) || pushedAny;
        pushedAny = this.broadcastForSide((1 - p) as PlayerIndex, getAfterOppDiscardEffect) || pushedAny;
      }
    }
    for (const p of [0, 1] as PlayerIndex[]) {
      const dk = `_pending_after_draw_by_p${p}`;
      if (scratch[dk]) {
        delete scratch[dk];
        pushedAny = this.broadcastForSide(p, getAfterSelfDrawEffect) || pushedAny;
      }
    }
    for (const p of [0, 1] as PlayerIndex[]) {
      const dk = `_pending_after_delete_by_p${p}`;
      if (scratch[dk]) {
        delete scratch[dk];
        pushedAny = this.broadcastForSide(p, getAfterSelfDeleteEffect) || pushedAny;
      }
    }
    for (const p of [0, 1] as PlayerIndex[]) {
      const dk = `_pending_after_shuffle_by_p${p}`;
      if (scratch[dk]) {
        delete scratch[dk];
        pushedAny = this.broadcastForSide(p, getAfterSelfShuffleEffect) || pushedAny;
      }
    }
    for (const p of [0, 1] as PlayerIndex[]) {
      const dk = `_pending_after_refresh_by_p${p}`;
      if (scratch[dk]) {
        delete scratch[dk];
        pushedAny = this.broadcastForSide(p, getAfterSelfRefreshEffect) || pushedAny;
      }
    }
    const flips = scratch["_pending_flip_cards"] as CardInst[] | undefined;
    if (flips && flips.length > 0) {
      delete scratch["_pending_flip_cards"];
      for (const c of flips) {
        for (let ln = 0; ln < 3; ln++) {
          for (const pl of [0, 1] as PlayerIndex[]) {
            const s = lineStack(st.lines[ln], pl);
            if (s.includes(c) && c.faceUp) {
              const fn = getFlipTriggerEffect(c.defId);
              if (fn) { this.pushEffect(fn(st, pl, ln, c)); pushedAny = true; }
              break;
            }
          }
        }
      }
    }
    return pushedAny;
  }

  /** Record-mode helper: when an action carries `revealedDefId`, patch the
   *  placeholder hand card it references so downstream effects / engine
   *  bookkeeping see a known card identity. No-op outside record mode (or
   *  for actions without the field). */
  private patchPlaceholderFromAction(player: PlayerIndex, a: Action): void {
    if (typeof a.revealedDefId !== "number" || a.handIndex == null) return;
    const c = this.state.players[player].hand[a.handIndex];
    if (c && c.defId === -1) c.defId = a.revealedDefId;
  }

  private doEndPhase(): boolean {
    const st = this.state;
    const ap = st.currentPlayer;
    let anyPushed = false;
    for (let ln = 0; ln < NUM_LINES; ln++) {
      const snapshot = [...lineStack(st.lines[ln], ap)];
      for (const c of snapshot) {
        if (!c.faceUp) continue;
        const fn = getEndEffect(c.defId);
        if (fn) { this.pushEffect(fn(st, ap, ln, c)); anyPushed = true; }
      }
    }
    if (anyPushed) {
      if (!st.scratch["end_resolved"]) {
        st.scratch["end_resolved"] = true;
        return true;
      }
    }
    delete st.scratch["end_resolved"];
    // Clear "cannot compile" flag for player whose turn ended.
    st.players[ap].cannotCompileNextTurn = false;
    // Pass turn.
    st.currentPlayer = ap === 0 ? 1 : 0;
    st.turn += 1;
    st.compiledThisTurn = false;
    st.effectPushesThisTurn = 0;
    if (st.turn > st.config.maxTurns) {
      // Leader-wins resolution.
      const p0 = st.players[0].compiled.filter(Boolean).length;
      const p1 = st.players[1].compiled.filter(Boolean).length;
      if (p0 > p1) st.winner = 0; else if (p1 > p0) st.winner = 1;
      st.phase = "GAME_OVER";
      return false;
    }
    st.phase = "START";
    return false;
  }

  private resolveChoice(a: Action): void {
    const top = this.pending[this.pending.length - 1];
    const choice = top.lastChoice;
    if (!choice) throw new Error("no choice pending");
    if (a.type === "SKIP_OPTIONAL") {
      if (!choice.optional) throw new Error("choice is not optional");
      top.lastChoice = null;
      const r = top.gen.next(-1);
      if (r.done) {
        const idx = this.pending.indexOf(top);
        if (idx >= 0) this.pending.splice(idx, 1);
      } else top.lastChoice = r.value;
      return;
    }
    if (a.type !== "CHOOSE_TARGET" || a.choiceIndex == null) throw new Error("expected CHOOSE_TARGET");
    const idx = a.choiceIndex;
    if (idx < 0 || idx >= choice.options.length) throw new Error("choice idx oor");
    top.lastChoice = null;
    const r = top.gen.next(idx);
    if (r.done) {
      const i = this.pending.indexOf(top);
      if (i >= 0) this.pending.splice(i, 1);
    } else top.lastChoice = r.value;
  }
}
