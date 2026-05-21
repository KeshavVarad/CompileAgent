/**
 * POST /api/auth/login
 * Body: { username: string, password: string }
 *
 * Verifies credentials, sets the session cookie, returns the user record.
 * Returns 401 on bad credentials (we don't disclose whether the username
 * exists — same error for both cases).
 */

import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";

import { signSession, verifyPassword, writeSessionCookie } from "@/lib/auth";
import { db, schema } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  if (!db) return NextResponse.json({ error: "database not configured" }, { status: 503 });
  const body = await request.json().catch(() => ({}));
  const username = typeof body.username === "string" ? body.username.trim() : "";
  const password = typeof body.password === "string" ? body.password : "";
  if (!username || !password) {
    return NextResponse.json({ error: "username + password required" }, { status: 400 });
  }

  const rows = await db
    .select({
      id: schema.users.id,
      username: schema.users.username,
      passwordHash: schema.users.passwordHash,
    })
    .from(schema.users)
    .where(eq(schema.users.username, username))
    .limit(1);
  const row = rows[0];
  if (!row || !verifyPassword(password, row.passwordHash)) {
    return NextResponse.json({ error: "invalid credentials" }, { status: 401 });
  }

  const token = await signSession({ userId: row.id, username: row.username });
  await writeSessionCookie(token);
  return NextResponse.json({ user: { id: row.id, username: row.username } });
}
