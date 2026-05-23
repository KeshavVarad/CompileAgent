/**
 * Drizzle ORM schema.
 *
 * `users`: account credentials. Login is by `username`; `passwordHash` is
 *   a scrypt-derived hash (node:crypto, no native deps).
 * `games`: one row per played/recorded game; deterministic seed + config +
 *   ordered action list lets us replay state from (seed, actions). Each
 *   game belongs to exactly one user — access is filtered by `userId` in
 *   every API route.
 */

import { pgTable, uuid, jsonb, timestamp, text, integer, boolean } from "drizzle-orm/pg-core";

export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  username: text("username").notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
});

export const games = pgTable("games", {
  id: uuid("id").primaryKey().defaultRandom(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
  endedAt: timestamp("ended_at", { withTimezone: true }),
  // Owner. The game is only visible to / playable by this user; no sharing
  // for now. (After the DB wipe this is NOT NULL; we leave it nullable in
  // schema so the migration can land before backfill.)
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),
  // Config / labels
  player0Label: text("player0_label").notNull(),
  player1Label: text("player1_label").notNull(),
  includeExpansion: boolean("include_expansion").notNull().default(false),
  includeMain2: boolean("include_main2").notNull().default(false),
  includeAux2: boolean("include_aux2").notNull().default(false),
  maxTurns: integer("max_turns").notNull().default(200),
  seed: integer("seed").notNull(),
  // Bot config (null = human-only / hot-seat)
  bot0Strategy: text("bot0_strategy"),
  bot1Strategy: text("bot1_strategy"),
  // Game mode. "play" = a normal interactive game (vs bot or hot-seat);
  // "record" = the user is transcribing a live game they are playing IRL
  // for later AI review. In record mode, opponent face-down cards are
  // placeholders until revealed.
  mode: text("mode").notNull().default("play"),
  // In record mode, which seat the recorder is sitting in (0 or 1). null
  // when mode = "play".
  recorderSeat: integer("recorder_seat"),
  // Replay log (ordered list of Action objects). Source of truth for state.
  actions: jsonb("actions").$type<unknown[]>().notNull().default([]),
  // Result snapshot (populated when game ends)
  winner: integer("winner"),
  turnCount: integer("turn_count"),
  // Persisted AI review. `evalResult` is the full payload the eval dialog
  // renders. `evalActionCount` records how many actions were in the log
  // when this eval ran — the UI compares it against the live action count
  // to flag the eval as "stale" if the game has progressed since.
  evalResult: jsonb("eval_result").$type<unknown>(),
  evalActionCount: integer("eval_action_count"),
});

export type Game = typeof games.$inferSelect;
export type NewGame = typeof games.$inferInsert;
export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
