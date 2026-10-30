import { Simulation } from './core/simulation'
import { WorldRenderer } from './render/renderer'
import { UIController } from './ui/app'
import { AnalyticsManager } from './analytics/analytics'
import { defaultSimParams } from './shared/defaultParams'
import type { ISimulation, Organism } from './shared/model'

const STEP_HZ = 60
const MAX_STEPS_PER_FRAME = 240

let sim: ISimulation | null = null
let renderer: WorldRenderer | null = null
let ui: UIController | null = null
const analytics = new AnalyticsManager()

let running = false
let speed = 1
let selected: Organism | null = null

function root(): HTMLElement {
  const el = document.getElementById('app')
  if (!el) throw new Error('missing #app')
  return el
}

function newWorld() {
  sim = new Simulation(defaultSimParams())
  selected = null
  analytics.reset()
  if (renderer && sim) renderer.fit(sim)
  if (ui && sim) ui.bindInspector(sim)
}

function frame(now: number) {
  requestAnimationFrame(frame)
  const last = frame.last || now
  frame.last = now
  const dt = Math.min(0.25, (now - last) / 1000)
  if (running && sim) {
    let budget = Math.min(MAX_STEPS_PER_FRAME, Math.round(STEP_HZ * speed * dt))
    while (budget-- > 0 && sim) {
      const ev = sim.step()
      analytics.onStep(ev, sim)
    }
  }
  if (renderer && sim) renderer.draw(sim, selected, { trails: false, sensors: false })
  analytics.update()
  if (ui && sim) ui.update(sim)
}
frame.last = 0

function main() {
  ui = new UIController(root(), {
    onPlayPause: () => {
      running = !running
      ui!.setRunning(running)
    },
    onStepOnce: () => {
      if (!running && sim) {
        const ev = sim.step()
        analytics.onStep(ev, sim)
      }
    },
    onReset: newWorld,
    onSpeedChange: (s) => (speed = s),
    onNewWorld: newWorld,
    onSaveWorld: () => {},
    onLoadWorld: () => {}
  })
  const canvas = document.createElement('canvas')
  canvas.style.width = '100%'
  canvas.style.height = '100%'
  canvas.style.display = 'block'
  ui.worldContainer.appendChild(canvas)
  analytics.bind(document.getElementById('app')!)
  renderer = new WorldRenderer(canvas, { onOrganismPicked: (o) => (selected = o) })
  newWorld()
  window.addEventListener('resize', () => renderer && renderer.resize() && sim && renderer.fit(sim))
  requestAnimationFrame(frame)
  document.title = 'Artificial Life'
}

addEventListener('DOMContentLoaded', main)
