/**
 * Given a stored game row (seed + config + actions[]), reconstruct a live
 * Game by replaying the actions through the engine.
 */

import { RandomBot } from "./compile/bot";
import { Game } from "./compile/game";
import { NNBot, isNNStrategy } from "./compile/nn-bot";
import type { Action } from "./compile/types";
import type { Game as DbGame } from "./db/schema";

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
  const applied: Action[] = [];
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
    game.step(action);
    applied.push(action);
  }
  return applied;
}

/** Instantiate the bot for a stored strategy. Anything we don't recognise
 *  falls back to the random bot (so old games keep replaying). */
export function botForStrategy(strategy: string | null, seed: number): RandomBot | NNBot | null {
  if (strategy == null) return null;
  if (isNNStrategy(strategy)) return new NNBot();
  return new RandomBot(seed);
}
