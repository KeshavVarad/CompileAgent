/**
 * Database connection. Uses `postgres` driver via Drizzle. Works both with
 * Vercel-provisioned Postgres (env `POSTGRES_URL` or `DATABASE_URL`) and any
 * local Postgres for development.
 */

import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema";

const connectionString =
  process.env.POSTGRES_URL ||
  process.env.DATABASE_URL ||
  process.env.POSTGRES_URL_NON_POOLING ||
  "";

if (!connectionString && typeof window === "undefined" && process.env.NODE_ENV !== "test") {
  // Don't crash at import-time in dev; warn loudly so route handlers fail
  // with a clear message when they try to use the DB without a connection
  // string configured.
  // eslint-disable-next-line no-console
  console.warn(
    "[db] No POSTGRES_URL / DATABASE_URL set. Run `vercel env pull .env.local` " +
      "after linking the project, or set the env var manually for local dev.",
  );
}

// `prepare: false` avoids problems in serverless environments where each
// invocation gets a fresh connection.
const client = connectionString
  ? postgres(connectionString, { prepare: false })
  : null;

export const db = client ? drizzle(client, { schema }) : null;
export { schema };
