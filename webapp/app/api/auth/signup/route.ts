/**
 * POST /api/auth/signup
 * Body: { username: string, password: string }
 *
 * Creates a new user, hashes the password, sets the session cookie, and
 * returns the new user's id + username. 409 if the username is taken.
 */

import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";

import {
  hashPassword,
  signSession,
  writeSessionCookie,
} from "@/lib/auth";
import { db, schema } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  if (!db) return NextResponse.json({ error: "database not configured" }, { status: 503 });
  const body = await request.json().catch(() => ({}));
  const username = typeof body.username === "string" ? body.username.trim() : "";
  const password = typeof body.password === "string" ? body.password : "";
  if (username.length < 2 || username.length > 32) {
    return NextResponse.json({ error: "username must be 2–32 chars" }, { status: 400 });
  }
  if (!/^[a-zA-Z0-9_.-]+$/.test(username)) {
    return NextResponse.json(
      { error: "username may only contain letters, digits, '.', '_', '-'" },
      { status: 400 },
    );
  }
  if (password.length < 6) {
    return NextResponse.json({ error: "password must be ≥6 chars" }, { status: 400 });
  }

  // Drizzle doesn't surface a typed unique violation, so we check first
  // (it's a single round-trip extra). Race is fine: the DB unique
  // constraint is the source of truth and would reject duplicates.
  const existing = await db
    .select({ id: schema.users.id })
    .from(schema.users)
    .where(eq(schema.users.username, username))
    .limit(1);
  if (existing[0]) {
    return NextResponse.json({ error: "username already taken" }, { status: 409 });
  }

  const [user] = await db
    .insert(schema.users)
    .values({ username, passwordHash: hashPassword(password) })
    .returning({ id: schema.users.id, username: schema.users.username });

  const token = await signSession({ userId: user.id, username: user.username });
  await writeSessionCookie(token);
  return NextResponse.json({ user });
}
