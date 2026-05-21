/**
 * Simple bot: picks a uniformly random legal action. Deterministic given a
 * seed so games are replayable.
 */

import type { Game } from "./game";
import { createRng, rngInt, type Rng } from "./rng";
import type { Action } from "./types";

export interface Bot {
  choose(game: Game, legal: Action[]): Action;
}

export class RandomBot implements Bot {
  private rng: Rng;
  constructor(seed: number) {
    this.rng = createRng(seed);
  }
  choose(_game: Game, legal: Action[]): Action {
    if (legal.length === 0) throw new Error("no legal actions");
    return legal[rngInt(this.rng, legal.length)];
  }
}
