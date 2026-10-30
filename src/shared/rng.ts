/**
 * Deterministic PRNG (mulberry32). Single 32-bit state, cheap, seedable and
 * serializable so whole-world saves reproduce exactly.
 */
export class Rng {
  private s: number

  constructor(seed = 1) {
    this.s = (seed >>> 0) || 1
  }

  next(): number {
    this.s = (this.s + 0x6d2b79f5) | 0
    let t = this.s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  /** uniform in [a, b) */
  range(a: number, b: number): number {
    return a + (b - a) * this.next()
  }

  /** uniform integer in [0, n) */
  int(n: number): number {
    return Math.floor(this.next() * n)
  }

  chance(p: number): boolean {
    return this.next() < p
  }

  /** Box-Muller gaussian, sigma default 1 */
  gauss(sigma = 1): number {
    const u = 1 - this.next()
    const v = this.next()
    return sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }

  get state(): number {
    return this.s
  }

  set state(value: number) {
    this.s = value >>> 0 || 1
  }
}

/** Stable nonzero pseudo-hash of an integer (used for species colors etc). */
export function hashInt(x: number): number {
  let h = Math.imul(x | 0, 0x9e3779b1)
  h ^= h >>> 15
  h = Math.imul(h, 0x85ebca6b)
  h ^= h >>> 13
  return h >>> 0
}
