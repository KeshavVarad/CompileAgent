/**
 * GET /api/auth/me
 *
 * Returns the current session's user, or 401 if no session.
 */

import { NextResponse } from "next/server";

import { getSession } from "@/lib/auth";

export const dynamic = "force-dynamic";

export async function GET() {
  const s = await getSession();
  if (!s) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  return NextResponse.json({ user: { id: s.userId, username: s.username } });
}
