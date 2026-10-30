/**
 * Application UI. MINIMAL runnable skeleton — owned by the UI agent (src/ui),
 * which will replace this file wholesale. Integration contract that must
 * survive rewriting: UIController + UIHooks + supported dialogs/toasts.
 */
import { ISimulation } from '../shared/model'

export interface UIHooks {
  onPlayPause(): void
  onStepOnce(): void
  onReset(): void
  onSpeedChange(speed: number): void
  onNewWorld(): void
  onSaveWorld(): void
  onLoadWorld(): void
}

export class UIController {
  worldContainer: HTMLElement
  private popEl!: HTMLElement
  private stepEl!: HTMLElement
  private playBtn!: HTMLButtonElement
  running = false

  constructor(root: HTMLElement, hooks: UIHooks) {
    root.innerHTML = `
      <div class="stub-toolbar">
        <button id="btn-play">Play</button>
        <button id="btn-step">Step</button>
        <button id="btn-reset">Reset</button>
        <input id="speed" type="range" min="0.25" max="8" step="0.25" value="1" />
        <span id="stats-pop">pop 0</span><span id="stats-step">step 0</span>
        <button id="btn-new">New</button>
        <button id="btn-save">Save</button>
        <button id="btn-load">Load</button>
      </div>
      <div id="world"></div>
      <div id="inspector" class="stub-inspector">Inspector</div>
      <style>
        body{margin:0;background:#0a0d11;color:#dde4ea;font-family:system-ui,sans-serif}
        .stub-toolbar{display:flex;gap:8px;align-items:center;padding:8px;background:#11161c}
        button{background:#1a222b;color:#dde4ea;border:1px solid #2a333d;border-radius:6px;padding:6px 12px;cursor:pointer}
        #world{position:fixed;inset:49px 0 0 0}
        #inspector{position:absolute;top:49px;right:0;bottom:0;width:280px;background:#11161c}
        .stub-inspector{padding:12px}
        #app{display:block}
        span{font-family:ui-monospace,monospace;font-size:12px}
      </style>`
    this.worldContainer = root.querySelector('#world')!
    this.popEl = root.querySelector('#stats-pop')!
    this.stepEl = root.querySelector('#stats-step')!
    this.playBtn = root.querySelector('#btn-play')!
    root.querySelector('#btn-play')!.addEventListener('click', hooks.onPlayPause)
    root.querySelector('#btn-step')!.addEventListener('click', hooks.onStepOnce)
    root.querySelector('#btn-reset')!.addEventListener('click', hooks.onReset)
    root.querySelector('#btn-new')!.addEventListener('click', hooks.onNewWorld)
    root.querySelector('#btn-save')!.addEventListener('click', hooks.onSaveWorld)
    root.querySelector('#btn-load')!.addEventListener('click', hooks.onLoadWorld)
    ;(root.querySelector('#speed') as HTMLInputElement).addEventListener('input', (e) => hooks.onSpeedChange(parseFloat((e.target as HTMLInputElement).value)))
  }

  setRunning(running: boolean) {
    this.running = running
    this.playBtn.textContent = running ? 'Pause' : 'Play'
  }

  update(sim: ISimulation) {
    this.popEl.textContent = `pop ${sim.stats.population}`
    this.stepEl.textContent = `step ${sim.stats.step}`
  }

  toast(_msg: string, _kind?: 'info' | 'ok' | 'warning' | 'error') {}
  openDialog(_opts: { title: string; body?: string } & unknown): void {}
  bindInspector(_sim: ISimulation) {}
}
