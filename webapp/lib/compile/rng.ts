/**
 * Deterministic RNG (Mulberry32). Same seed → same sequence, so we can
 * faithfully replay games from (seed, action history) on the server.
 */

export type Rng = { state: number };

export function createRng(seed: number): Rng {
  // Mulberry32 expects a 32-bit unsigned integer.
  return { state: (seed >>> 0) || 1 };
}

export function rngFloat(rng: Rng): number {
  let t = (rng.state += 0x6d2b79f5);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

export function rngInt(rng: Rng, maxExclusive: number): number {
  if (maxExclusive <= 0) return 0;
  return Math.floor(rngFloat(rng) * maxExclusive);
}

export function rngShuffle<T>(rng: Rng, arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rngInt(rng, i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

export function rngChoice<T>(rng: Rng, arr: T[]): T {
  return arr[rngInt(rng, arr.length)];
}
