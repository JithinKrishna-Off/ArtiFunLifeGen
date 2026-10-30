/**
 * World renderer. MINIMAL runnable skeleton — owned by the rendering agent
 * (src/render), which will replace this file wholesale.
 */
import { ISimulation, Organism, InteractionMode } from '../shared/model'
import { speciesColor } from '../shared/util'

export interface RenderHooks {
  onOrganismPicked(o: Organism | null): void
}

export class WorldRenderer {
  readonly canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  zoom = 1
  panX = 0
  panY = 0
  mode: InteractionMode = 'inspect'

  constructor(canvas: HTMLCanvasElement, _hooks: RenderHooks) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.resize()
  }

  resize() {
    const r = this.canvas.getBoundingClientRect()
    this.canvas.width = Math.max(1, Math.round(r.width * devicePixelRatio))
    this.canvas.height = Math.max(1, Math.round(r.height * devicePixelRatio))
  }

  fit(sim: ISimulation) {
    const w = this.canvas.width / devicePixelRatio
    const h = this.canvas.height / devicePixelRatio
    this.zoom = Math.min(w / sim.env.width, h / sim.env.height)
    this.panX = (w - sim.env.width * this.zoom) / 2
    this.panY = (h - sim.env.height * this.zoom) / 2
  }

  screenToWorld(sx: number, sy: number): { x: number; y: number } {
    return { x: (sx - this.panX) / this.zoom, y: (sy - this.panY) / this.zoom }
  }

  draw(sim: ISimulation, _selected: Organism | null, _opts: { trails: boolean; sensors: boolean }) {
    const ctx = this.ctx
    ctx.save()
    ctx.fillStyle = '#0a0f12'
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)
    ctx.setTransform(devicePixelRatio * this.zoom, 0, 0, devicePixelRatio * this.zoom, this.panX * devicePixelRatio, this.panY * devicePixelRatio)
    ctx.fillStyle = '#08131a'
    ctx.fillRect(0, 0, sim.env.width, sim.env.height)
    ctx.fillStyle = 'rgba(150,224,170,0.65)'
    for (const f of sim.foods) ctx.fillRect(f.x - 2, f.y - 2, 4, 4)
    for (const o of sim.organisms) {
      ctx.beginPath()
      ctx.fillStyle = speciesColor(o.lineageId)
      ctx.arc(o.x, o.y, 5.5 * o.traits.size, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.restore()
  }
}
