/**
 * Evolution controller. MINIMAL working version — owned by the evolution agent
 * (src/evo), which may refine internals but must keep these export signatures.
 */
import { NET_HIDDEN, NET_INPUTS, NET_OUTPUTS, type Genotype, type Network, type SimParams, clampTraits, TRAIT_MIN, TRAIT_MAX } from '../shared/model'
import { Rng, hashInt } from '../shared/rng'

export function createNetwork(rng: Rng, compact = 0.7): Network {
  return {
    w1: new Float32Array(NET_HIDDEN * NET_INPUTS).map(() => rng.gauss(compact)),
    b1: new Float32Array(NET_HIDDEN).map(() => rng.gauss(compact)),
    w2: new Float32Array(NET_OUTPUTS * NET_HIDDEN).map(() => rng.gauss(compact)),
    b2: new Float32Array(NET_OUTPUTS).map(() => rng.gauss(compact))
  }
}

export function randomGenotype(rng: Rng): Genotype {
  return {
    net: createNetwork(rng),
    traits: {
      maxSpeed: rng.range(35, 120),
      efficiency: rng.range(0.7, 1.7),
      sensorRange: rng.range(70, 220),
      metabolism: rng.range(0.75, 1.45),
      lifespan: rng.range(4000, 9500),
      size: rng.range(0.8, 1.35),
      reprThreshold: rng.range(0.6, 0.85),
      reprCost: rng.range(0.26, 0.46)
    }
  }
}

export function cloneGenotype(g: Genotype): Genotype {
  return {
    net: { w1: g.net.w1.slice(), b1: g.net.b1.slice(), w2: g.net.w2.slice(), b2: g.net.b2.slice() },
    traits: { ...g.traits }
  }
}

function spanInNorm(v: number, min: number, max: number): number {
  return (v - min) / (max - min)
}

export function mutateGenotype(parent: Genotype, rng: Rng, params: SimParams): Genotype {
  const g = cloneGenotype(parent)
  const rate = params.mutationRate
  const mag = Math.max(0.05, params.mutationMagnitude)
  const big = params.bigMutationChance
  for (const arr of [g.net.w1, g.net.b1, g.net.w2, g.net.b2]) {
    for (let i = 0; i < arr.length; i++) {
      if (rng.chance(rate)) {
        const step = rng.gauss(0.5 * mag * (rng.chance(big) ? 10 : 1))
        arr[i] = clampN(arr[i] + step, -6, 6)
      }
    }
  }
  const keys = Object.keys(g.traits) as (keyof typeof g.traits)[]
  for (const k of keys) {
    if (rng.chance(rate)) {
      const rel = 1 + rng.gauss(0.06 * mag * (rng.chance(big) ? 10 : 1))
      g.traits[k] = g.traits[k] * rel
    }
  }
  g.traits = clampTraits(g.traits)
  return g
}

function clampN(v: number, a: number, b: number): number {
  return v < a ? a : v > b ? b : v
}

/** Feed-forward: tanh hidden, tanh outputs. Writes into `out`. */
export function think(net: Network, inputs: Float32Array, out: Float32Array): void {
  const hidden = new Float32Array(NET_HIDDEN) as Float32Array & { length: number }
  for (let h = 0; h < NET_HIDDEN; h++) {
    let s = net.b1[h]
    const row = h * NET_INPUTS
    for (let i = 0; i < NET_INPUTS; i++) s += net.w1[row + i] * inputs[i]
    hidden[h] = Math.tanh(s)
  }
  for (let o = 0; o < NET_OUTPUTS; o++) {
    let s = net.b2[o]
    const row = o * NET_HIDDEN
    for (let h = 0; h < NET_HIDDEN; h++) s += net.w2[row + h] * hidden[h]
    out[o] = Math.tanh(s)
  }
}

/** Mean squared deviation over all genome floats, traits normalized by bounds. */
export function genotypeDistance(a: Genotype, b: Genotype): number {
  let sum = 0
  let n = 0
  for (const [x, y] of zip([a.net.w1, b.net.w1], [a.net.b1, b.net.b1], [a.net.w2, b.net.w2], [a.net.b2, b.net.b2])) {
    for (let i = 0; i < x.length; i++) {
      const d = x[i] - y[i]
      sum += d * d
      n++
    }
  }
  const tks = Object.keys(TRAIT_MIN) as (keyof typeof TRAIT_MIN)[]
  for (const k of tks) {
    const range = TRAIT_MAX[k] - TRAIT_MIN[k]
    const d = (a.traits[k] - b.traits[k]) / range
    sum += d * d * 4
    n += 4
  }
  return Math.sqrt(sum / Math.max(1, n))
}

function zip(...pairs: [Float32Array, Float32Array][]): [Float32Array, Float32Array][] {
  return pairs
}

// lineage ids double as species ids; a founder organism that deviates strongly
// from every existing lineage starts a new id (see core integration notes).
export function nextLineageId(existing: Set<number>): number {
  return (hashInt(existing.size + 1) | 1) >>> 0
}

export const TRAIT_BOUNDS = { min: TRAIT_MIN, max: TRAIT_MAX }
