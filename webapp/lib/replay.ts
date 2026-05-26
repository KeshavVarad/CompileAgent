/**
 * Given a stored game row (seed + config + actions[]), reconstruct a live
 * Game by replaying the actions through the engine.
 */

import { RandomBot } from "./compile/bot";
import { REVEAL_HAND_PROMPT } from "./compile/effects";
import { Game } from "./compile/game";
import { NNBot, isNNStrategy } from "./compile/nn-bot";
import type { Action } from "./compile/types";
import type { Game as DbGame } from "./db/schema";

/** Backwards-compat shim. When the engine evolves to surface a pending
 *  Choice that pre-existing action histories don't account for, raw
 *  replay would throw "expected CHOOSE_TARGET". Two flavors handled:
 *
 *  1. Known-skippable display-only Choices (REVEAL_HAND_PROMPT, added by
 *     PR #56 for Psychic 0 / Light 4 / Clarity 1) — auto-resolve with
 *     index 0; just dismisses the human-only pause card.
 *  2. New triggers that older engines silently failed to fire (e.g.
 *     PR #59 made Darkness 1's flip-then-shift fire the post-shift
 *     middle, which older saves don't record a CHOOSE_TARGET for). When
 *     the next saved action isn't a CHOOSE_TARGET/SKIP_OPTIONAL but the
 *     engine is waiting on one, abandon the pending generator — this
 *     matches the older engine where the trigger never fired.
 *
 *  Pass `nextAction` so case 2 can compare against what the saved
 *  history actually has queued. Without it the shim only handles case 1.
 *
 *  Exported so the snapshot route (which has its own replay loop) and
 *  any future replayer can reuse it instead of duplicating the logic. */
export function autoResolveCompatChoices(g: Game, nextAction?: Action): void {
  // `pending` is private on Game; bracket-access matches how
  // view.ts already inspects the pending stack from outside the class.
  while (true) {
    const stack = (g as unknown as { pending: { lastChoice: { prompt: string } | null }[] }).pending;
    const top = stack[stack.length - 1];
    const choice = top?.lastChoice;
    if (!choice) return;
    if (choice.prompt === REVEAL_HAND_PROMPT) {
      g.step({ type: "CHOOSE_TARGET", choiceIndex: 0 });
      continue;
    }
    if (
      nextAction &&
      nextAction.type !== "CHOOSE_TARGET" &&
      nextAction.type !== "SKIP_OPTIONAL"
    ) {
      (g as unknown as { abandonPendingChoice: () => void }).abandonPendingChoice();
      continue;
    }
    return;
  }
}

export function gameFromRow(row: DbGame): Game {
  const g = new Game({
    includeExpansion: row.includeExpansion,
    includeMain2: row.includeMain2,
    includeAux2: row.includeAux2,
    maxTurns: row.maxTurns,
    seed: row.seed,
    player0Label: row.player0Label,
    player1Label: row.player1Label,
    human: [row.bot0Strategy === null, row.bot1Strategy === null],
    botStrategy: (row.bot0Strategy ?? row.bot1Strategy) as "random" | null,
    mode: (row.mode === "record" ? "record" : "play"),
    recorderSeat: row.recorderSeat == null ? undefined : (row.recorderSeat as 0 | 1),
  });
  g.start();
  for (const a of (row.actions as Action[])) {
    if (g.isOver()) break;
    // Heal older action histories before applying the next saved step.
    // See autoResolveCompatChoices for the motivation.
    autoResolveCompatChoices(g, a);
    // Inverse compat: if the saved action is a CHOOSE_TARGET but no
    // pending Choice exists (a later engine fix removed the trigger
    // that originally surfaced this prompt — e.g. PR #60's covered-flip
    // middle suppression), skip the now-stale action.
    if (
      a.type === "CHOOSE_TARGET" &&
      (g as unknown as { pending: unknown[] }).pending.length === 0
    ) {
      continue;
    }
    g.step(a);
  }
  return g;
}

/** Auto-advance the bot's turns (and any forced player NOOPs). Returns the
 *  list of bot/forced actions that were applied (for persistence). NN-backed
 *  bots are async, so this function is async too. */
export async function autoAdvanceBot(
  game: Game,
  seat0: RandomBot | NNBot | null,
  seat1: RandomBot | NNBot | null,
): Promise<Action[]> {
  const { applied } = await autoAdvanceBotWithSnapshots(
    game, seat0, seat1, () => undefined, () => undefined,
  );
  return applied;
}

/** Variant of `autoAdvanceBot` that captures snapshots around each bot
 *  step. The `pre` callback runs against the game BEFORE the step
 *  lands (so it can read about-to-be-played card details), `post`
 *  receives the pre-callback's output plus the game AFTER the step
 *  (so it can serialise a `GameView` of the new state). */
export async function autoAdvanceBotWithSnapshots<P, T>(
  game: Game,
  seat0: RandomBot | NNBot | null,
  seat1: RandomBot | NNBot | null,
  pre: (game: Game, action: Action) => P,
  post: (action: Action, captured: P, game: Game) => T,
): Promise<{ applied: Action[]; snapshots: T[] }> {
  const applied: Action[] = [];
  const snapshots: T[] = [];
  const guard = 500;
  for (let i = 0; i < guard; i++) {
    if (game.isOver()) break;
    const who = game.decider();
    const bot = who === 0 ? seat0 : seat1;
    if (!bot) break; // human's turn — return to UI
    const legal = game.legalActions();
    if (legal.length === 0) break;
    const action = bot instanceof NNBot
      ? await bot.chooseAsync(game, legal)
      : bot.choose(game, legal);
    const captured = pre(game, action);
    game.step(action);
    applied.push(action);
    snapshots.push(post(action, captured, game));
  }
  return { applied, snapshots };
}

/** Instantiate the bot for a stored strategy. Anything we don't recognise
 *  falls back to the random bot (so old games keep replaying). */
export function botForStrategy(strategy: string | null, seed: number): RandomBot | NNBot | null {
  if (strategy == null) return null;
  if (isNNStrategy(strategy)) return new NNBot();
  return new RandomBot(seed);
}
