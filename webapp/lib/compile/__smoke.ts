/**
 * Smoke check: play a random-vs-random game end-to-end. Run via:
 *   npx tsx lib/compile/__smoke.ts
 */

import { RandomBot } from "./bot";
import { Game } from "./game";

const g = new Game({
  includeExpansion: false,
  maxTurns: 200,
  seed: 1,
  player0Label: "Bot A",
  player1Label: "Bot B",
  human: [false, false],
  botStrategy: "random",
});
g.start();
const bots = [new RandomBot(1), new RandomBot(2)];
let steps = 0;
while (!g.isOver()) {
  const who = g.decider();
  const legal = g.legalActions();
  if (legal.length === 0) break;
  g.step(bots[who].choose(g, legal));
  steps++;
  if (steps > 5000) { console.log("hung"); break; }
}
console.log({ turn: g.state.turn, winner: g.state.winner, phase: g.state.phase, steps });
