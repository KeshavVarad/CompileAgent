/**
 * POST /api/auth/logout
 *
 * Clears the session cookie. Always 200 (idempotent — calling this with
 * no session is a no-op).
 */

import { NextResponse } from "next/server";

import { clearSessionCookie } from "@/lib/auth";

export const dynamic = "force-dynamic";

export async function POST() {
  await clearSessionCookie();
  return NextResponse.json({ ok: true });
}
