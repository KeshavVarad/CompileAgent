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
  id: "sparkv2",
  displayLabel: "Sparkv2",
  modelPath: "/models/bot-current.onnx",
  sourceCheckpoint: "runs/20260521-040612-loose-policy/snapshot_00500.pt",
  trainingRun: "20260521-040612-loose-policy",
  trainingIter: 500,
  evalSummary: {
    vsRandomWinRate: 0.90,
    vsGreedyWinRate: 0.70,
    elo: 1609,
    gamesPerMatchup: 50,
  },
};
