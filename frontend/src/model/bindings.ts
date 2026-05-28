import type { Layer } from './layer'
import type { ChannelBind, PromptBind } from './types'

/**
 * Compute a per-pixel binding alpha in [0,1] for a layer's cfg/denoise channel.
 *
 *   'self'   = own painted mask (255 -> 1.0); if mask is null, whole canvas
 *   'prompt' = per-pixel max across all painted prompt masks on this layer;
 *              prompts with bind != 'self' or mask == null contribute 1.0
 *   'rgba'   = layer's rgba alpha channel
 *   'none'   = whole canvas (returns null as a fast-path sentinel)
 *
 * Returns null when the binding is uniformly 1.0 across the canvas, so the
 * caller can skip per-pixel multiplication.
 */
export function resolveChannelBinding(
  layer: Layer,
  bind: ChannelBind,
  ownMask: Uint8ClampedArray | null,
): Float32Array | null {
  const pixelCount = layer.width * layer.height
  if (bind === 'none') return null

  if (bind === 'self') {
    if (ownMask === null) return null
    return _maskToFloat(ownMask, pixelCount)
  }

  if (bind === 'rgba') {
    const out = new Float32Array(pixelCount)
    const rgba = layer.rgba
    for (let i = 0; i < pixelCount; i++) {
      out[i] = rgba[i * 4 + 3] / 255
    }
    return out
  }

  // 'prompt' — per-pixel max across this layer's prompt attention masks.
  if (layer.prompts.length === 0) return null
  let acc: Float32Array | null = null
  for (const p of layer.prompts) {
    const promptAlpha = resolvePromptBinding(layer, p.bind, p.mask)
    if (promptAlpha === null) {
      // any whole-canvas prompt collapses the max to 1.0 -> sentinel
      return null
    }
    if (acc === null) {
      acc = new Float32Array(promptAlpha)
    } else {
      for (let i = 0; i < pixelCount; i++) {
        if (promptAlpha[i] > acc[i]) acc[i] = promptAlpha[i]
      }
    }
  }
  return acc
}

/**
 * Per-pixel attention alpha in [0,1] for one prompt entry.
 *
 *   'self' = own painted mask; null -> whole canvas
 *   'rgba' = layer's rgba alpha
 *   'none' = whole canvas
 *
 * Returns null when uniform 1.0 across the canvas.
 */
export function resolvePromptBinding(
  layer: Layer,
  bind: PromptBind,
  ownMask: Uint8ClampedArray | null,
): Float32Array | null {
  const pixelCount = layer.width * layer.height
  if (bind === 'none') return null
  if (bind === 'self') {
    if (ownMask === null) return null
    return _maskToFloat(ownMask, pixelCount)
  }
  // 'rgba'
  const out = new Float32Array(pixelCount)
  const rgba = layer.rgba
  for (let i = 0; i < pixelCount; i++) {
    out[i] = rgba[i * 4 + 3] / 255
  }
  return out
}

function _maskToFloat(mask: Uint8ClampedArray, pixelCount: number): Float32Array {
  const out = new Float32Array(pixelCount)
  for (let i = 0; i < pixelCount; i++) out[i] = mask[i] / 255
  return out
}
