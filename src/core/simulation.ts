/**
 * Simulation engine. MINIMAL runnable skeleton — owned by the engine agent
 * (src/core), which will replace this file wholesale while keeping the
 * ISimulation contract from src/shared/model.ts.
 */
import {
  type ISimulation, type SimParams, type StepEvents, type Environment, type Organism, type FoodItem,
  type SimStats, type LineageRecord, maxEnergyFor
} from '../shared/model'
import { Rng } from '../shared/rng'
import { TAU } from '../shared/util'
import { randomGenotype, think } from '../evo/nn'

let nextId = 1

export class Simulation implements ISimulation {
  readonly params: SimParams
  readonly env: Environment
  readonly organisms: Organism[] = []
  readonly foods: FoodItem[] = []
  stats: SimStats
  stepCount = 0

  private rng: Rng

  constructor(params: SimParams) {
    this.params = params
    this.env = {
      width: params.width,
      height: params.height,
      wrapping: params.wrapping,
      obstacles: [{ x: params.width * 0.5, y: params.height * 0.5, r: 120 }],
      hazards: [],
      patches: [
        { x: params.width * 0.27, y: params.height * 0.3, r: 320, richness: 1 },
        { x: params.width * 0.73, y: params.height * 0.68, r: 340, richness: 1.2 },
        { x: params.width * 0.5, y: params.height * 0.85, r: 260, richness: 0.8 }
      ]
    }
    this.rng = new Rng(params.seed)
    for (let i = 0; i < params.initialFood; i++) this.spawnFoodRandom()
    for (let i = 0; i < params.initialPopulation; i++) this.spawnOrganismRandom()
    this.stats = this.computeStats()
  }

  private spawnFoodRandom() {
    const p = this.env.patches[this.rng.int(this.env.patches.length)]
    const a = this.rng.next() * TAU
    const r = Math.sqrt(this.rng.next()) * p.r
    this.foods.push({
      id: nextId++,
      x: p.x + Math.cos(a) * r,
      y: p.y + Math.sin(a) * r,
      energy: this.params.foodEnergy,
      alive: true
    })
  }

  private spawnOrganismRandom(): Organism {
    const g = randomGenotype(this.rng)
    const o: Organism = {
      id: nextId++,
      x: this.rng.range(0, this.env.width),
      y: this.rng.range(0, this.env.height),
      dir: this.rng.next() * TAU,
      speed: 0,
      energy: maxEnergyFor(g.traits) * 0.55,
      maxEnergy: maxEnergyFor(g.traits),
      age: 0,
      generation: 0,
      traits: g.traits,
      net: g.net,
      lineageId: this.rng.int(97) + 1,
      momId: null,
      dadId: null,
      offspringCount: 0,
      alive: true,
      lastInputs: new Float32Array(11).fill(0),
      lastOutputs: new Float32Array(3).fill(0),
      reprCooldown: 0
    }
    this.organisms.push(o)
    return o
  }

  step(): StepEvents {
    this.step++
    for (const o of this.organisms) {
      const inp = o.lastInputs!
      inp[9] = (o.energy / o.maxEnergy) * 2 - 1
      inp[10] = Math.min(1, o.age / o.traits.lifespan) * 2 - 1
      think(o.net, inp, o.lastOutputs!)
      const turn = o.lastOutputs![0]
      o.dir += turn * 0.12
      const thrust = Math.max(0, o.lastOutputs![1])
      const target = thrust * o.traits.maxSpeed
      o.speed += (target - o.speed) * 0.3
      o.x += Math.cos(o.dir) * o.speed / 60
      o.y += Math.sin(o.dir) * o.speed / 60
      o.energy -= (this.params.metabolismBase * o.traits.metabolism + this.params.movementCost * o.speed * o.traits.efficiency) / 60
      o.age++
      if (o.energy <= 0) o.alive = false
    }
    for (let i = this.organisms.length - 1; i >= 0; i--) if (!this.organisms[i].alive) this.organisms.splice(i, 1)
    this.stats = this.computeStats()
    return { births: [], deaths: [] }
  }

  private computeStats(): SimStats {
    const n = this.organisms.length
    let e = 0, a = 0, g = 0, maxg = 0
    for (const o of this.organisms) { e += o.energy; a += o.age; g += o.generation; if (o.generation > maxg) maxg = o.generation }
    return {
      step: this.step, population: n, maxPopulation: this.params.maxPopulation,
      totalBirths: 0, totalDeaths: 0, birthsWindow: 0, deathsWindow: 0,
      avgEnergy: n ? e / n : 0, avgAge: n ? a / n : 0, avgGeneration: n ? g / n : 0,
      maxGeneration: maxg, foodCount: this.foods.length, speciesCounts: [], diversityIndex: 0
    }
  }

  pickOrganism(x: number, y: number, radius: number): Organism | null { return null }
  spawnOrganisms(count: number): void { for (let i = 0; i < count; i++) this.spawnOrganismRandom(); this.stats = this.computeStats() }
  addFood(x: number, y: number, energy: number): boolean { this.foods.push({ id: nextId++, x, y, energy, alive: true }); return true }
  addObstacle(x: number, y: number, r: number): boolean { this.env.obstacles.push({ x, y, r }); return true }
  addPatch(x: number, y: number, r: number): boolean { this.env.patches.push({ x, y, r, richness: 1 }); return true }
  addHazard(x: number, y: number, r: number): boolean { this.env.hazards.push({ x, y, r, dps: 4, pulsePeriod: 0, phase: 0 }); return true }
  removeAt(x: number, y: number, r: number): void {}
  undoEnvironmentEdit(): boolean { return false }
  getRecord(id: number): LineageRecord | null { return null }
  updateLiveParams(patch: Partial<SimParams>): void { Object.assign(this.params, patch) }
}
