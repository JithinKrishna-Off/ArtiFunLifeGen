/**
 * Artificial Life — shared data model.
 * This file is the single source of truth for cross-module contracts.
 * Ownership: integration coordinator (src/shared). All modules must conform
 * to these types and the semantics documented here.
 *
 * UNITS: world units (arbitrary, world fits params.width x params.height),
 * energy (abstract units), time in simulation STEPS (fixed 60 steps/second at speed 1).
 */

import { clamp } from './util'

// ------------------------------------------------------------ neural network
export const NET_INPUTS = 11
export const NET_HIDDEN = 10
export const NET_OUTPUTS = 3

/** Sensor vector layout, all values normalized to [-1, 1]. */
export const INPUT_LABELS = [
  'cos food dir',
  'sin food dir',
  'food intensity',
  'cos peer dir',
  'sin peer dir',
  'peer density',
  'hazard intensity',
  'cos hazard dir',
  'sin hazard dir',
  'energy level',
  'age phase'
] as const

/** Output vector layout. */
export const OUTPUT_LABELS = ['turn', 'thrust', 'reproduce gate'] as const

export interface Network {
  /** [NET_HIDDEN * NET_INPUTS] row-major */
  w1: Float32Array
  /** [NET_HIDDEN] */
  b1: Float32Array
  /** [NET_OUTPUTS * NET_HIDDEN] row-major */
  w2: Float32Array
  /** [NET_OUTPUTS] */
  b2: Float32Array
}

// ------------------------------------------------------------- genetics
export interface Traits {
  /** max movement speed, units/second */
  maxSpeed: number
  /** movement drain multiplier (lower is cheaper) */
  efficiency: number
  /** sensor range in world units */
  sensorRange: number
  /** basal metabolism multiplier */
  metabolism: number
  /** maximum lifespan in steps */
  lifespan: number
  /** body radius factor (world units ~ 6 * size); affects metabolism a bit, visuals, collision */
  size: number
  /** fraction of maxEnergy required to reproduce */
  reprThreshold: number
  /** fraction of maxEnergy transferred out of parent per reproduction */
  reprCost: number
}

export const TRAIT_MIN: Traits = {
  maxSpeed: 18,
  efficiency: 0.6,
  sensorRange: 40,
  metabolism: 0.6,
  lifespan: 2400,
  size: 0.65,
  reprThreshold: 0.5,
  reprCost: 0.2
}

export const TRAIT_MAX: Traits = {
  maxSpeed: 150,
  efficiency: 2.4,
  sensorRange: 300,
  metabolism: 1.9,
  lifespan: 14000,
  size: 1.6,
  reprThreshold: 0.95,
  reprCost: 0.6
}

export interface Genotype {
  net: Network
  traits: Traits
}

export function clampTraits(t: Traits): Traits {
  return {
    maxSpeed: clamp(t.maxSpeed, TRAIT_MIN.maxSpeed, TRAIT_MAX.maxSpeed),
    efficiency: clamp(t.efficiency, TRAIT_MIN.efficiency, TRAIT_MAX.efficiency),
    sensorRange: clamp(t.sensorRange, TRAIT_MIN.sensorRange, TRAIT_MAX.sensorRange),
    metabolism: clamp(t.metabolism, TRAIT_MIN.metabolism, TRAIT_MAX.metabolism),
    lifespan: clamp(t.lifespan, TRAIT_MIN.lifespan, TRAIT_MAX.lifespan),
    size: clamp(t.size, TRAIT_MIN.size, TRAIT_MAX.size),
    reprThreshold: clamp(t.reprThreshold, TRAIT_MIN.reprThreshold, TRAIT_MAX.reprThreshold),
    reprCost: clamp(t.reprCost, TRAIT_MIN.reprCost, TRAIT_MAX.reprCost)
  }
}

/** Maximum storable energy; larger bodies store more. */
export function maxEnergyFor(t: Traits): number {
  return 110 + 65 * t.size
}

// ------------------------------------------------------------- environment
export interface Obstacle {
  x: number
  y: number
  r: number
}

export interface Hazard {
  x: number
  y: number
  r: number
  /** energy drained per second while inside */
  dps: number
  /** 0 = steady; else oscillates between 0.15x and 1.85x dps with this period in seconds */
  pulsePeriod: number
  phase: number
}

export interface FoodPatch {
  x: number
  y: number
  r: number
  /** relative share of foodRegen this patch receives */
  richness: number
}

export interface Environment {
  width: number
  height: number
  wrapping: boolean
  obstacles: Obstacle[]
  hazards: Hazard[]
  patches: FoodPatch[]
}

export interface FoodItem {
  id: number
  x: number
  y: number
  energy: number
  alive: boolean
}

// ------------------------------------------------------------- organism
export interface Organism {
  /** stable id, unique within one world */
  id: number
  x: number
  y: number
  /** heading in radians */
  dir: number
  /** current forward speed, units/second */
  speed: number
  energy: number
  maxEnergy: number
  /** steps lived */
  age: number
  generation: number
  traits: Traits
  net: Network
  /** lineage id (lineageId === speciesId; new lineage = new mutation-run founder from initial pop) */
  lineageId: number
  momId: number | null
  dadId: number | null
  offspringCount: number
  alive: boolean
  /** last computed sensor vector (same Float32Array ref mutated each step; do not mutate externally) */
  lastInputs: Float32Array | null
  /** last computed action vector (same ref semantics) */
  lastOutputs: Float32Array | null
  reprCooldown: number
}

export function organismRadius(o: Organism): number {
  return 5.5 * o.traits.size
}

// ------------------------------------------------------------- simulation
export type DeathCause = 'starvation' | 'old age' | 'hazard'

export interface BirthEvent {
  step: number
  childId: number
  parentId: number
  speciesId: number
  x: number
  y: number
  generation: number
}

export interface DeathEvent {
  step: number
  id: number
  speciesId: number
  cause: DeathCause
  x: number
  y: number
  generation: number
  age: number
}

export interface StepEvents {
  births: BirthEvent[]
  deaths: DeathEvent[]
}

export interface SpeciesCount {
  speciesId: number
  count: number
}

export interface SimStats {
  step: number
  population: number
  maxPopulation: number
  /** cumulative since world creation (excluding loaded-save baseline restoration is analytics' job) */
  totalBirths: number
  totalDeaths: number
  /** births/deaths over the last 600 steps (10 sim-seconds) */
  birthsWindow: number
  deathsWindow: number
  avgEnergy: number
  avgAge: number
  avgGeneration: number
  maxGeneration: number
  foodCount: number
  speciesCounts: SpeciesCount[]
  diversityIndex: number
}

export interface SimParams {
  seed: number
  width: number
  height: number
  wrapping: boolean
  initialPopulation: number
  maxPopulation: number
  initialFood: number
  maxFood: number
  /** new food items injected per second across patches (proportional to richness) */
  foodRegen: number
  /** mean energy per food item (variance = foodEnergyVariance) */
  foodEnergy: number
  foodEnergyVariance: number
  /** basal drain per second at metabolism=1 */
  metabolismBase: number
  /** extra drain per second per unit speed at efficiency=1: drain += movementCost * speed * efficiency */
  movementCost: number
  /** drain per second at sensorRange=100: drain += sensingCost * sqrt(range/100) */
  sensingCost: number
  /** steps between reproductions (per organism) */
  reproductionCooldown: number
  /** steps before an organism may first reproduce */
  minReproductionAge: number
  /** probability per gene (weight or trait) of mutating */
  mutationRate: number
  /** gaussian sigma multiplier applied to gene mutation step */
  mutationMagnitude: number
  /** probability a mutating gene receives a rare large mutation (10x magnitude) */
  bigMutationChance: number
}

/** Parameters that can be changed on the running simulation without rebuilding the world. */
export const LIVE_PARAM_KEYS = [
  'maxPopulation',
  'maxFood',
  'foodRegen',
  'foodEnergy',
  'metabolismBase',
  'movementCost',
  'sensingCost',
  'mutationRate',
  'mutationMagnitude',
  'bigMutationChance'
] as const
export type LiveParamKey = (typeof LIVE_PARAM_KEYS)[number]

export type InteractionMode = 'inspect' | 'food' | 'obstacle' | 'patch' | 'hazard' | 'erase'

/**
 * The simulation engine contract. Implemented in src/core/simulation.ts.
 * All public reads must be safe from the render thread while the engine is paused OR between steps.
 */
export interface ISimulation {
  readonly params: SimParams
  readonly env: Environment
  readonly organisms: readonly Organism[]
  readonly foods: readonly FoodItem[]
  readonly stats: SimStats
  /** current simulation step counter */
  readonly stepCount: number
  /** Advance exactly one simulation step; returns what happened. */
  step(): StepEvents
  /** Nearest alive organism to (worldX, worldY) within radius world units, or null. */
  pickOrganism(worldX: number, worldY: number, radius: number): Organism | null
  /** Add n organisms with fresh random founders (empty sandbox, reseeding after extinction). */
  spawnOrganisms(count: number): void
  /** Environment editing. mode-specific; returns false if edit budget/limit hit. */
  addFood(x: number, y: number, energy: number): boolean
  addObstacle(x: number, y: number, r: number): boolean
  addPatch(x: number, y: number, r: number): boolean
  addHazard(x: number, y: number, r: number): boolean
  removeAt(x: number, y: number, r: number): void
  /** Undo the last environment editing operation (one level). */
  undoEnvironmentEdit(): boolean
  /** Record for lineage explorer; bounded recent-history (see core implementation notes). */
  getRecord(id: number): LineageRecord | null
  /** Apply live-capable parameter changes; silently ignores non-live keys. Values are clamped. */
  updateLiveParams(patch: Partial<SimParams>): void
}

export interface LineageRecord {
  id: number
  momId: number | null
  dadId: number | null
  generation: number
  speciesId: number
  bornStep: number
  deathStep: number | null
  cause: DeathCause | null
}
