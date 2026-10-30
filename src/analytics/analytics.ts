/**
 * Analytics + persistence. MINIMAL runnable skeleton — owned by the analytics
 * agent (src/analytics), which will replace this file wholesale.
 */
import { ISimulation, StepEvents } from '../shared/model'

export class AnalyticsManager {
  constructor() {}

  /** Called exactly once per simulation step, in stepping order. */
  onStep(_events: StepEvents, sim: ISimulation) {
    if (sim.stats.step % 20 === 0) this.sample(sim)
  }

  private sample(_sim: ISimulation) {}

  /** Bind chart containers; UI marks targets with data-alchart attributes. */
  bind(_root: HTMLElement) {}

  /** Called every frame; throttles chart redraws internally. */
  update() {}

  /** Reset history (new world / reset). */
  reset() {}

  /** Serialized history for save files. */
  exportHistory(): unknown {
    return null
  }

  importHistory(_data: unknown) {}
}
