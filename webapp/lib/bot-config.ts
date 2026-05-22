/**
 * Single swap point for the website's NN bot.
 *
 * Want to swap the deployed bot? Update `CURRENT_BOT` below to point at a
 * different snapshot (after exporting it to ONNX via
 * `scripts/eval/export_onnx.py`) and re-deploy. Everything else — the
 * game-creation API, replay, the dialog opponent label, the model-card
 * metadata that the UI surfaces — reads from here.
 */

export type BotDescriptor = {
  /** Stable identifier baked into game rows (string column in the DB). */
  id: string;
  /** Human-readable label shown in the UI. */
  displayLabel: string;
  /** Absolute path served by Next at runtime. Place the ONNX in
   *  `webapp/models/` and reference it as `/models/...` — Next ships
   *  `public/` statically and we add `models/` to that. */
  modelPath: string;
  /** Source checkpoint for traceability — never read at runtime. */
  sourceCheckpoint: string;
  /** Training run + iteration the snapshot came from. */
  trainingRun: string;
  trainingIter: number;
  /** Eval summary at ship time — surfaced on the home page. */
  evalSummary: {
    vsRandomWinRate: number;
    vsGreedyWinRate: number;
    elo: number;
    gamesPerMatchup: number;
  };
};

/** The bot currently selected as the "Play vs Bot" opponent. */
export const CURRENT_BOT: BotDescriptor = {
  id: "sparkv3",
  displayLabel: "Sparkv3",
  modelPath: "/models/bot-current.onnx",
  sourceCheckpoint: "runs/20260521-230453-az/snapshot_00020.pt",
  trainingRun: "20260521-230453-az",
  trainingIter: 20,
  evalSummary: {
    // n=200, deterministic argmax both sides (for reproducibility with the
    // historic ladder). Sparkv3 was trained with stochastic self-play +
    // Gumbel root selection; in head-to-head it beats Sparkv2 70.0% and
    // joint-distilled 73.0% (both n=200, p < 1e-7).
    vsRandomWinRate: 0.94,
    vsGreedyWinRate: 0.70,
    elo: 1344, // 3-way ship-ladder vs Sparkv2 + joint-distilled, 100 games/pair
    gamesPerMatchup: 200,
  },
};
