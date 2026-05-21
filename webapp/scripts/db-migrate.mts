import { readFileSync } from "node:fs";
import postgres from "postgres";
import { config } from "dotenv";

config({ path: ".env.local" });

const url = process.env.POSTGRES_URL;
if (!url) { console.error("no POSTGRES_URL"); process.exit(1); }

import { readdirSync } from "node:fs";
const client = postgres(url);
for (const file of readdirSync("drizzle").filter((f) => f.endsWith(".sql")).sort()) {
  const sql = readFileSync(`drizzle/${file}`, "utf8");
  try { await client.unsafe(sql); console.log(`applied ${file}`); }
  catch (e) { console.log(`skipped ${file}: ${(e as Error).message.split("\n")[0]}`); }
}
const tables = await client`SELECT tablename FROM pg_tables WHERE schemaname='public'`;
console.log("tables:", tables.map((r: { tablename: string }) => r.tablename));
const cols = await client`SELECT column_name FROM information_schema.columns WHERE table_name='games' ORDER BY ordinal_position`;
console.log("games cols:", cols.map((r: { column_name: string }) => r.column_name).join(", "));
await client.end();
