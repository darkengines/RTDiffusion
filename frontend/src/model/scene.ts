import { Layer } from './layer'

export interface SceneInit {
  width: number
  height: number
  basePrompt?: string
  baseNegativePrompt?: string
  baseDenoise?: number
  baseCfg?: number
}

/**
 * Ordered stack of Layer instances (bottom-up). Holds the scene-level
 * defaults that engines fall back to where no layer contributes.
 *
 * `version` is the monotonic dirty flag the render loop watches. It is
 * the sum of layer versions plus a private counter bumped on structural
 * mutations (add/remove/reorder/base-* change).
 */
export class Scene {
  width: number
  height: number
  basePrompt: string
  baseNegativePrompt: string
  baseDenoise: number
  baseCfg: number
  private _layers: Layer[]
  private _structureVersion: number

  constructor(init: SceneInit) {
    this.width = init.width
    this.height = init.height
    this.basePrompt = init.basePrompt ?? ''
    this.baseNegativePrompt = init.baseNegativePrompt ?? ''
    this.baseDenoise = init.baseDenoise ?? 1.0
    this.baseCfg = init.baseCfg ?? 1.5
    this._layers = []
    this._structureVersion = 1
  }

  get layers(): readonly Layer[] {
    return this._layers
  }

  /** Monotonic across all mutations (structural + intra-layer).
   *
   * Removed layers' versions are absorbed into ``_structureVersion`` so the
   * total never goes down when a layer is removed.
   */
  get version(): number {
    let v = this._structureVersion
    for (const layer of this._layers) v += layer.version
    return v
  }

  addLayer(layer: Layer): void {
    if (layer.width !== this.width || layer.height !== this.height) {
      throw new Error(
        `Layer ${layer.id} dims ${layer.width}x${layer.height} != scene ${this.width}x${this.height}`,
      )
    }
    if (this._layers.some(l => l.id === layer.id)) {
      throw new Error(`Layer id ${layer.id} already in scene`)
    }
    this._layers.push(layer)
    this._structureVersion += 1
  }

  removeLayer(id: string): void {
    const i = this._layers.findIndex(l => l.id === id)
    if (i < 0) return
    const removed = this._layers[i]
    this._layers.splice(i, 1)
    this._structureVersion += removed.version + 1
  }

  reorderLayer(id: string, toIndex: number): void {
    const from = this._layers.findIndex(l => l.id === id)
    if (from < 0) throw new Error(`Layer ${id} not in scene`)
    const [layer] = this._layers.splice(from, 1)
    const clamped = Math.max(0, Math.min(this._layers.length, toIndex))
    this._layers.splice(clamped, 0, layer)
    this._structureVersion += 1
  }

  setBasePrompt(s: string): void {
    if (this.basePrompt !== s) { this.basePrompt = s; this._structureVersion += 1 }
  }

  setBaseNegativePrompt(s: string): void {
    if (this.baseNegativePrompt !== s) { this.baseNegativePrompt = s; this._structureVersion += 1 }
  }

  setBaseDenoise(v: number): void {
    if (this.baseDenoise !== v) { this.baseDenoise = v; this._structureVersion += 1 }
  }

  setBaseCfg(v: number): void {
    if (this.baseCfg !== v) { this.baseCfg = v; this._structureVersion += 1 }
  }

  findLayer(id: string): Layer | undefined {
    return this._layers.find(l => l.id === id)
  }
}
