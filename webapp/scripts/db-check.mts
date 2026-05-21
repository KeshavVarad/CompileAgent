import postgres from "postgres";
import { config } from "dotenv";
config({ path: ".env.local" });

const url = process.env.POSTGRES_URL;
if (!url) { console.error("no POSTGRES_URL"); process.exit(1); }
const client = postgres(url);
const tables = await client`SELECT tablename FROM pg_tables WHERE schemaname='public'`;
console.log("tables:", tables.map((r: { tablename: string }) => r.tablename));
const cols = await client`SELECT column_name FROM information_schema.columns WHERE table_name='games' ORDER BY ordinal_position`;
console.log("games cols:", cols.map((r: { column_name: string }) => r.column_name));
const cnt = await client`SELECT count(*)::int AS n FROM games`;
console.log("rows:", cnt[0]);
await client.end();
