import type { AggregatedScene, ControlNetParams, WirePayload } from './types'

/** Backend constant -- cfg values are normalised to [0,1] for transport then
 *  multiplied back by CFG_HI on decode. Keep in sync with render_plan.py. */
export const CFG_HI = 30.0

/**
 * PNG encoder edge. Pluggable so the model is testable without a real canvas:
 * vitest+jsdom provides no Canvas2D, so tests use a stub encoder that returns
 * deterministic strings. Production code injects `createCanvasPngEncoder`.
 */
export interface PngEncoder {
  /** Encode an RGBA Uint8ClampedArray (length w*h*4) as base64 PNG-A. */
  encodeRgba(buf: Uint8ClampedArray, w: number, h: number): string
  /** Encode a Float32Array luminance buffer in [0,1] as base64 PNG-L. */
  encodeLuminance(buf: Float32Array, w: number, h: number): string
}

/**
 * Turn the aggregated buffers into the JSON-serialisable wire payload that
 * matches the backend's v2 InpaintFrame fields. Pure; PNG byte production is
 * delegated to `enc` so tests don't need a real canvas.
 */
export function aggregatedToWire(agg: AggregatedScene, enc: PngEncoder): WirePayload {
  const { width, height } = agg
  const pixelCount = width * height

  const cfgNormalised = new Float32Array(pixelCount)
  for (let i = 0; i < pixelCount; i++) {
    cfgNormalised[i] = agg.cfgMap[i] / CFG_HI
  }

  return {
    width,
    height,
    rgba_b64: enc.encodeRgba(agg.rgba, width, height),
    cfg_map_b64: enc.encodeLuminance(cfgNormalised, width, height),
    denoise_map_b64: enc.encodeLuminance(agg.denoiseMap, width, height),
    prompts: agg.prompts.map(p => ({
      text: p.text,
      negative: p.negative,
      mask_b64: p.mask ? enc.encodeLuminance(p.mask, width, height) : undefined,
    })),
    controlnet: agg.controlnet.map(c => ({
      layer_id: c.layerId,
      model: c.config.model,
      model_path: c.config.modelPath,
      scale: c.config.scale,
      start: c.config.start,
      end: c.config.end,
      preprocessor_params: _cnParamsToWire(c.config.preprocessorParams),
    })),
    base_prompt: agg.basePrompt,
    base_negative_prompt: agg.baseNegativePrompt,
    base_denoise: agg.baseDenoise,
    base_cfg: agg.baseCfg,
  }
}

function _cnParamsToWire(p: ControlNetParams): Record<string, number | boolean> {
  const out: Record<string, number | boolean> = {}
  if (p.cannyLow !== undefined) out.canny_low = p.cannyLow
  if (p.cannyHigh !== undefined) out.canny_high = p.cannyHigh
  if (p.openposeHands !== undefined) out.openpose_hands = p.openposeHands
  if (p.openposeFace !== undefined) out.openpose_face = p.openposeFace
  if (p.mlsdThrV !== undefined) out.mlsd_thr_v = p.mlsdThrV
  if (p.mlsdThrD !== undefined) out.mlsd_thr_d = p.mlsdThrD
  if (p.lineartCoarse !== undefined) out.lineart_coarse = p.lineartCoarse
  return out
}

/**
 * Production PNG encoder. Uses an HTMLCanvasElement (provided by
 * `canvasFactory`) + ImageData + toDataURL. The factory exists so callers
 * can reuse a single offscreen canvas across encodes instead of allocating
 * one per frame.
 */
export function createCanvasPngEncoder(canvasFactory: () => HTMLCanvasElement): PngEncoder {
  return {
    encodeRgba(buf, w, h) {
      const c = canvasFactory()
      c.width = w
      c.height = h
      const ctx = c.getContext('2d', { alpha: true, willReadFrequently: true })
      if (!ctx) throw new Error('encodeRgba: 2d context unavailable')
      ctx.clearRect(0, 0, w, h)
      const id = new ImageData(buf as Uint8ClampedArray<ArrayBuffer>, w, h)
      ctx.putImageData(id, 0, 0)
      return c.toDataURL('image/png').split(',', 2)[1]
    },
    encodeLuminance(buf, w, h) {
      const c = canvasFactory()
      c.width = w
      c.height = h
      const ctx = c.getContext('2d', { alpha: false, willReadFrequently: true })
      if (!ctx) throw new Error('encodeLuminance: 2d context unavailable')
      const rgba = new Uint8ClampedArray(w * h * 4)
      for (let i = 0; i < w * h; i++) {
        const clamped = Math.max(0, Math.min(1, buf[i]))
        const v = Math.round(clamped * 255)
        const o = i * 4
        rgba[o] = v
        rgba[o + 1] = v
        rgba[o + 2] = v
        rgba[o + 3] = 255
      }
      const id = new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, w, h)
      ctx.putImageData(id, 0, 0)
      return c.toDataURL('image/png').split(',', 2)[1]
    },
  }
}
