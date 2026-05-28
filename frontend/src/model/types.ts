// Phase 2 layer model — pure data types matching the backend wire format
// (see backend/app/render_plan.py). No DOM dependencies; HTMLCanvasElement
// stays at the paint/encoder edge so the model is unit-testable in jsdom.

export type ChannelBind = 'self' | 'prompt' | 'rgba' | 'none'
export type PromptBind = 'self' | 'rgba' | 'none'

export interface PromptEntry {
  text: string
  negative: string
  /** H*W grayscale, 0..255. null means use the bind, no own paint. */
  mask: Uint8ClampedArray | null
  bind: PromptBind
}

export interface ControlNetParams {
  cannyLow?: number
  cannyHigh?: number
  openposeHands?: boolean
  openposeFace?: boolean
  mlsdThrV?: number
  mlsdThrD?: number
  lineartCoarse?: boolean
}

export interface ControlNetConfig {
  model: string
  modelPath: string | null
  scale: number
  start: number
  end: number
  preprocessorParams: ControlNetParams
}

/** Aggregated scene the encoder turns into a wire payload. */
export interface AggregatedScene {
  width: number
  height: number
  /** H*W*4 uint8, Porter-Duff over of all visible layers bottom-up. */
  rgba: Uint8ClampedArray
  /** H*W float32, values in [0, 30] (cfg units). max-aggregated. */
  cfgMap: Float32Array
  /** H*W float32, values in [0, 1]. max-aggregated. */
  denoiseMap: Float32Array
  /** Flat list across all layers, bottom-up; each entry's mask has bindings resolved. */
  prompts: Array<{
    text: string
    negative: string
    /** H*W float32 in [0, 1], or null = whole canvas. */
    mask: Float32Array | null
  }>
  controlnet: Array<{
    layerId: string
    config: ControlNetConfig
  }>
  basePrompt: string
  baseNegativePrompt: string
  baseDenoise: number
  baseCfg: number
}

/** Wire payload — JSON-serialisable, matches schemas.InpaintFrame v2 fields. */
export interface WirePayload {
  width: number
  height: number
  rgba_b64?: string
  cfg_map_b64?: string
  denoise_map_b64?: string
  prompts: Array<{
    text: string
    negative: string
    mask_b64?: string
  }>
  controlnet: Array<{
    layer_id: string
    model: string
    model_path: string | null
    scale: number
    start: number
    end: number
    preprocessor_params: Record<string, number | boolean>
  }>
  base_prompt: string
  base_negative_prompt: string
  base_denoise: number
  base_cfg: number
}
