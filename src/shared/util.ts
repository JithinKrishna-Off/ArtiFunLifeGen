export const TAU = Math.PI * 2

export function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Wrap angle to [-PI, PI) */
export function wrapPi(a: number): number {
  a = a % TAU
  if (a > Math.PI) a -= TAU
  if (a < -Math.PI) a += TAU
  return a
}

/** Format a number compactly for UI display. */
export function fmt(n: number, digits = 1): string {
  if (!Number.isFinite(n)) return '—'
  if (Math.abs(n) >= 100000) return (n / 1000).toFixed(0) + 'k'
  return n.toFixed(digits)
}

/**
 * Species/lineage color: deterministic from the species id, golden-angle hue
 * walk so nearby ids stay distinct. Species colors are NOT UI status colors.
 */
export function speciesColor(speciesId: number): string {
  return speciesHsl(speciesId, 62, 58)
}

export function speciesHsl(speciesId: number, s: number, l: number): string {
  const hue = (speciesId * 137.508) % 360
  return `hsl(${hue.toFixed(1)} ${s} ${l})`
}

export function speciesHueRotate(speciesId: number): number {
  return (speciesId * 137.508) % 360
}
