import type { ChannelBind, ControlNetConfig, PromptBind, PromptEntry } from './types'

export interface LayerInit {
  id: string
  width: number
  height: number
  visible?: boolean
  rgba?: Uint8ClampedArray
  cfg?: number
  cfgMask?: Uint8ClampedArray | null
  cfgBind?: ChannelBind
  denoise?: number
  denoiseMask?: Uint8ClampedArray | null
  denoiseBind?: ChannelBind
  prompts?: PromptEntry[]
  controlnet?: ControlNetConfig | null
}

const RGBA_DEFAULT_CFG = 1.5
const RGBA_DEFAULT_DENOISE = 1.0

/**
 * One layer of the scene. Stores pixel buffers as Uint8ClampedArray so the
 * model is portable to plain Node (vitest+jsdom doesn't ship canvas APIs).
 * The paint engine wires HTMLCanvasElement -> these buffers at stroke commit
 * time; downstream aggregation reads buffers directly.
 *
 * Every mutator bumps `version` monotonically. The render loop uses
 * `Scene.version` to decide when to re-aggregate and re-send.
 */
export class Layer {
  readonly id: string
  readonly width: number
  readonly height: number
  visible: boolean
  rgba: Uint8ClampedArray
  cfg: number
  cfgMask: Uint8ClampedArray | null
  cfgBind: ChannelBind
  denoise: number
  denoiseMask: Uint8ClampedArray | null
  denoiseBind: ChannelBind
  prompts: PromptEntry[]
  controlnet: ControlNetConfig | null
  version: number

  constructor(init: LayerInit) {
    this.id = init.id
    this.width = init.width
    this.height = init.height
    this.visible = init.visible ?? true

    const pixelCount = init.width * init.height
    this.rgba = init.rgba ?? new Uint8ClampedArray(pixelCount * 4)
    if (this.rgba.length !== pixelCount * 4) {
      throw new Error(`Layer ${init.id}: rgba length ${this.rgba.length} != ${pixelCount * 4}`)
    }

    this.cfg = init.cfg ?? RGBA_DEFAULT_CFG
    this.cfgMask = init.cfgMask ?? null
    this.cfgBind = init.cfgBind ?? 'rgba'
    this.denoise = init.denoise ?? RGBA_DEFAULT_DENOISE
    this.denoiseMask = init.denoiseMask ?? null
    this.denoiseBind = init.denoiseBind ?? 'rgba'
    this.prompts = init.prompts ? [...init.prompts] : []
    this.controlnet = init.controlnet ?? null
    this.version = 1

    this._checkMask('cfgMask', this.cfgMask)
    this._checkMask('denoiseMask', this.denoiseMask)
    for (const p of this.prompts) this._checkMask('prompt.mask', p.mask)
  }

  private _checkMask(name: string, mask: Uint8ClampedArray | null): void {
    if (mask !== null && mask.length !== this.width * this.height) {
      throw new Error(`Layer ${this.id}: ${name} length ${mask.length} != ${this.width * this.height}`)
    }
  }

  private _bump(): void {
    this.version += 1
  }

  /** Public bump for direct buffer mutations (paint engine writes pixels in
   *  place without going through a setter). Pass the channel for hooks; the
   *  channel argument is informational only -- internal version is one int. */
  bumpVersion(_channel?: unknown): void {
    this.version += 1
  }

  setVisible(v: boolean): void {
    if (this.visible !== v) { this.visible = v; this._bump() }
  }

  /** Bulk replace the RGBA buffer; length must match width*height*4. */
  setRgba(buf: Uint8ClampedArray): void {
    if (buf.length !== this.width * this.height * 4) {
      throw new Error(`Layer ${this.id}: setRgba length ${buf.length} != ${this.width * this.height * 4}`)
    }
    this.rgba = buf
    this._bump()
  }

  setCfg(value: number): void {
    if (this.cfg !== value) { this.cfg = value; this._bump() }
  }

  setCfgMask(mask: Uint8ClampedArray | null): void {
    this._checkMask('cfgMask', mask)
    this.cfgMask = mask
    this._bump()
  }

  setCfgBind(bind: ChannelBind): void {
    if (this.cfgBind !== bind) { this.cfgBind = bind; this._bump() }
  }

  setDenoise(value: number): void {
    if (this.denoise !== value) { this.denoise = value; this._bump() }
  }

  setDenoiseMask(mask: Uint8ClampedArray | null): void {
    this._checkMask('denoiseMask', mask)
    this.denoiseMask = mask
    this._bump()
  }

  setDenoiseBind(bind: ChannelBind): void {
    if (this.denoiseBind !== bind) { this.denoiseBind = bind; this._bump() }
  }

  addPrompt(p: { text: string; negative?: string; mask?: Uint8ClampedArray | null; bind?: PromptBind }): number {
    const entry: PromptEntry = {
      text: p.text,
      negative: p.negative ?? '',
      mask: p.mask ?? null,
      bind: p.bind ?? 'rgba',
    }
    this._checkMask('prompt.mask', entry.mask)
    this.prompts.push(entry)
    this._bump()
    return this.prompts.length - 1
  }

  updatePrompt(index: number, partial: Partial<PromptEntry>): void {
    const cur = this.prompts[index]
    if (!cur) throw new Error(`Layer ${this.id}: prompt index ${index} out of range`)
    if (partial.mask !== undefined) this._checkMask('prompt.mask', partial.mask)
    this.prompts[index] = { ...cur, ...partial }
    this._bump()
  }

  removePrompt(index: number): void {
    if (index < 0 || index >= this.prompts.length) {
      throw new Error(`Layer ${this.id}: prompt index ${index} out of range`)
    }
    this.prompts.splice(index, 1)
    this._bump()
  }

  setControlNet(cn: ControlNetConfig | null): void {
    this.controlnet = cn
    this._bump()
  }
}
