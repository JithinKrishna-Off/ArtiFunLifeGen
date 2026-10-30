import type { SimParams } from './model'

export interface ParamMeta {
  key: keyof SimParams
  label: string
  tip: string
  min: number
  max: number
  step: number
  kind: 'int' | 'float' | 'bool'
  /** live params apply to the running world; others require a new world */
  live: boolean
}

export function defaultSimParams(): SimParams {
  return {
    seed: 1337,
    width: 2400,
    height: 1600,
    wrapping: false,
    initialPopulation: 240,
    maxPopulation: 1200,
    initialFood: 800,
    maxFood: 1300,
    foodRegen: 26,
    foodEnergy: 20,
    foodEnergyVariance: 0.35,
    metabolismBase: 0.55,
    movementCost: 0.011,
    sensingCost: 0.004,
    reproductionCooldown: 240,
    minReproductionAge: 600,
    mutationRate: 0.07,
    mutationMagnitude: 1.0,
    bigMutationChance: 0.02
  }
}

/** Deep copy helpers for params (plain numbers + boolean). */
export function cloneParams(p: SimParams): SimParams {
  return { ...p }
}

export const PARAMS_META: readonly ParamMeta[] = [
  { key: 'width', label: 'World width', tip: 'World extent in units. Changing it builds a new world.', min: 600, max: 6000, step: 100, kind: 'int', live: false },
  { key: 'height', label: 'World height', tip: 'World extent in units. Changing it builds a new world.', min: 600, max: 6000, step: 100, kind: 'int', live: false },
  { key: 'initialPopulation', label: 'Initial population', tip: 'Organisms seeded at world creation.', min: 0, max: 2000, step: 10, kind: 'int', live: false },
  { key: 'maxPopulation', label: 'Population limit', tip: 'Hard cap; reproduction is blocked when reached.', min: 50, max: 4000, step: 50, kind: 'int', live: true },
  { key: 'initialFood', label: 'Initial food', tip: 'Food items seeded at world creation.', min: 0, max: 3000, step: 50, kind: 'int', live: false },
  { key: 'maxFood', label: 'Food cap', tip: 'Maximum simultaneous food items.', min: 100, max: 4000, step: 50, kind: 'int', live: true },
  { key: 'foodRegen', label: 'Food regen / s', tip: 'Food items generated per second across patches.', min: 2, max: 150, step: 1, kind: 'float', live: true },
  { key: 'foodEnergy', label: 'Food energy', tip: 'Mean energy per food item.', min: 5, max: 60, step: 1, kind: 'float', live: true },
  { key: 'foodEnergyVariance', label: 'Food variance', tip: 'Relative variance of per-item energy.', min: 0, max: 1, step: 0.05, kind: 'float', live: true },
  { key: 'metabolismBase', label: 'Metabolism', tip: 'Basal energy drain per second at trait metabolism 1.', min: 0.1, max: 2, step: 0.05, kind: 'float', live: true },
  { key: 'movementCost', label: 'Movement cost', tip: 'Extra drain per second per speed unit at efficiency 1.', min: 0.002, max: 0.05, step: 0.001, kind: 'float', live: true },
  { key: 'sensingCost', label: 'Sensing cost', tip: 'Per-second drain scaled by sensor range.', min: 0, max: 0.02, step: 0.001, kind: 'float', live: true },
  { key: 'reproductionCooldown', label: 'Repro cooldown', tip: 'Minimum steps between reproductions per organism. New world only in presets.', min: 0, max: 1200, step: 20, kind: 'int', live: false },
  { key: 'minReproductionAge', label: 'Min repro age', tip: 'Steps before an organism may first reproduce.', min: 0, max: 2400, step: 50, kind: 'int', live: false },
  { key: 'mutationRate', label: 'Mutation rate', tip: 'Probability per gene of mutating.', min: 0, max: 0.5, step: 0.01, kind: 'float', live: true },
  { key: 'mutationMagnitude', label: 'Mutation size', tip: 'Gaussian sigma multiplier for mutation steps.', min: 0.1, max: 4, step: 0.1, kind: 'float', live: true },
  { key: 'bigMutationChance', label: 'Rare jumps', tip: 'Chance that a mutating gene jumps 10x farther.', min: 0, max: 0.2, step: 0.01, kind: 'float', live: true },
  { key: 'seed', label: 'Random seed', tip: 'Deterministic run identifier. Same seed + same inputs = same outcome.', min: 1, max: 4294967295, step: 1, kind: 'int', live: false },
  { key: 'wrapping', label: 'Wrap edges', tip: 'Toroidal world when on; otherwise edges are walls.', min: 0, max: 1, step: 1, kind: 'bool', live: false }
]
