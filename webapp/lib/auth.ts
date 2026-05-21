/**
 * Auth: username/password with scrypt-hashed credentials and a signed-JWT
 * session cookie.
 *
 * Why this stack:
 *   - scrypt comes from node:crypto so we have zero native dependencies
 *     (bcrypt would need a prebuilt binary on Vercel — same headache as
 *     onnxruntime-node).
 *   - jose handles JWT sign/verify in pure JS, works in Node + Edge.
 *
 * Session is stored in an HTTP-only cookie called `session` and renewed
 * on every successful login/signup. Expires after 30 days.
 *
 * The DEV_MODE env var, when set, makes every request appear to come from
 * a fixed dev user — used for local end-to-end testing without going
 * through the login UI. Never set DEV_MODE in production.
 */

import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { cookies } from "next/headers";
import { SignJWT, jwtVerify } from "jose";
import { eq } from "drizzle-orm";

import { db, schema } from "./db";

export const SESSION_COOKIE = "session";
const SESSION_TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days

const SCRYPT_KEYLEN = 64;
const SCRYPT_SALT_BYTES = 16;

// ---------------------------------------------------------------------------
// Passwords
// ---------------------------------------------------------------------------

export function hashPassword(plaintext: string): string {
  if (plaintext.length < 6) throw new Error("password must be ≥6 chars");
  const salt = randomBytes(SCRYPT_SALT_BYTES);
  const hash = scryptSync(plaintext, salt, SCRYPT_KEYLEN);
  return `scrypt$${salt.toString("hex")}$${hash.toString("hex")}`;
}

export function verifyPassword(plaintext: string, stored: string): boolean {
  const parts = stored.split("$");
  if (parts.length !== 3 || parts[0] !== "scrypt") return false;
  const salt = Buffer.from(parts[1], "hex");
  const expected = Buffer.from(parts[2], "hex");
  const actual = scryptSync(plaintext, salt, expected.length);
  // Constant-time compare to avoid timing attacks.
  if (actual.length !== expected.length) return false;
  return timingSafeEqual(actual, expected);
}

// ---------------------------------------------------------------------------
// JWT sessions
// ---------------------------------------------------------------------------

function sessionSecret(): Uint8Array {
  const s = process.env.SESSION_SECRET;
  if (!s || s.length < 32) {
    throw new Error(
      "SESSION_SECRET must be set to a string of ≥32 chars in env. " +
      "Generate one with: node -e \"console.log(crypto.randomBytes(48).toString('hex'))\"",
    );
  }
  return new TextEncoder().encode(s);
}

export type SessionPayload = {
  userId: string;
  username: string;
};

export async function signSession(p: SessionPayload): Promise<string> {
  return await new SignJWT({ username: p.username })
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(p.userId)
    .setIssuedAt()
    .setExpirationTime(Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS)
    .sign(sessionSecret());
}

export async function verifySession(token: string): Promise<SessionPayload | null> {
  try {
    const { payload } = await jwtVerify(token, sessionSecret(), { algorithms: ["HS256"] });
    if (typeof payload.sub !== "string" || typeof payload.username !== "string") return null;
    return { userId: payload.sub, username: payload.username };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Reading + writing the session cookie
// ---------------------------------------------------------------------------

/** Cookie attributes used for the session cookie. */
const COOKIE_ATTRS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  path: "/",
  maxAge: SESSION_TTL_SECONDS,
};

/** Resolve the current session, if any, from the request's cookies.
 *
 * In dev (`DEV_MODE=true`) we always return a fixed dev user without
 * touching the cookie — but only after the dev user actually exists in
 * the configured DB. Caller MUST be inside a server route / RSC.
 */
export async function getSession(): Promise<SessionPayload | null> {
  if (process.env.DEV_MODE === "true") {
    const devUser = await ensureDevUser();
    if (devUser) return { userId: devUser.id, username: devUser.username };
  }
  const jar = await cookies();
  const tok = jar.get(SESSION_COOKIE)?.value;
  if (!tok) return null;
  return await verifySession(tok);
}

/** Write the session cookie for the given user. Caller is responsible for
 *  passing in the freshly-signed token. */
export async function writeSessionCookie(token: string): Promise<void> {
  const jar = await cookies();
  jar.set(SESSION_COOKIE, token, COOKIE_ATTRS);
}

export async function clearSessionCookie(): Promise<void> {
  const jar = await cookies();
  jar.set(SESSION_COOKIE, "", { ...COOKIE_ATTRS, maxAge: 0 });
}

// ---------------------------------------------------------------------------
// DEV_MODE convenience: ensure a stable dev user exists in the active DB.
// ---------------------------------------------------------------------------

const DEV_USERNAME = "dev";
const DEV_PASSWORD = "dev-only-not-secret";

async function ensureDevUser(): Promise<{ id: string; username: string } | null> {
  if (!db) return null;
  const existing = await db
    .select({ id: schema.users.id, username: schema.users.username })
    .from(schema.users)
    .where(eq(schema.users.username, DEV_USERNAME))
    .limit(1);
  if (existing[0]) return existing[0];
  const [created] = await db
    .insert(schema.users)
    .values({ username: DEV_USERNAME, passwordHash: hashPassword(DEV_PASSWORD) })
    .returning({ id: schema.users.id, username: schema.users.username });
  return created;
}
