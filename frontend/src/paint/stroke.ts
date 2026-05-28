import type { Layer } from '../model/layer'
import type { BrushSettings, PaintChannel, PointerPoint } from './types'

/**
 * Paint a single dab into a layer channel buffer. Pure typed-array math --
 * no Canvas2D, so it runs identically in tests (jsdom) and the browser.
 *
 * For rgba channel: blends src color over existing pixels using the brush
 * radial falloff (1 inside hardness*radius, smoothed to 0 at radius).
 * For cfg/denoise/prompt channels: writes the brush maskValue, taking the
 * per-pixel max with the existing value -- mask channels accumulate, they
 * don't fade out under repeated strokes.
 */
export function paintDab(
  layer: Layer,
  channel: PaintChannel,
  point: PointerPoint,
  brush: BrushSettings,
): void {
  const buf = _bufferForChannel(layer, channel)
  if (!buf) return
  const stride = channel === 'rgba' ? 4 : 1
  const radius = Math.max(0.5, brush.size * Math.max(0.05, point.pressure))
  const r2 = radius * radius
  const hard = Math.max(0, Math.min(1, brush.hardness)) * radius
  const featherDenom = Math.max(1e-3, radius - hard)

  const x0 = Math.max(0, Math.floor(point.x - radius))
  const y0 = Math.max(0, Math.floor(point.y - radius))
  const x1 = Math.min(layer.width, Math.ceil(point.x + radius + 1))
  const y1 = Math.min(layer.height, Math.ceil(point.y + radius + 1))

  for (let y = y0; y < y1; y++) {
    const dy = y - point.y
    const dy2 = dy * dy
    for (let x = x0; x < x1; x++) {
      const dx = x - point.x
      const d2 = dx * dx + dy2
      if (d2 > r2) continue
      const d = Math.sqrt(d2)
      const falloff = d <= hard ? 1 : Math.max(0, 1 - (d - hard) / featherDenom)
      if (falloff <= 0) continue
      const o = (y * layer.width + x) * stride
      if (channel === 'rgba') {
        const a = (brush.color.a / 255) * falloff
        const invA = 1 - a
        buf[o] = buf[o] * invA + brush.color.r * a
        buf[o + 1] = buf[o + 1] * invA + brush.color.g * a
        buf[o + 2] = buf[o + 2] * invA + brush.color.b * a
        buf[o + 3] = Math.min(255, buf[o + 3] + 255 * a)
      } else {
        const v = brush.maskValue * falloff
        if (v > buf[o]) buf[o] = v
      }
    }
  }
  layer.bumpVersion(channel)
}

/**
 * Erase a single dab. For rgba: reduces alpha (and zeros rgb where alpha
 * goes to zero). For mask channels: reduces value back toward zero.
 */
export function eraseDab(
  layer: Layer,
  channel: PaintChannel,
  point: PointerPoint,
  brush: BrushSettings,
): void {
  const buf = _bufferForChannel(layer, channel)
  if (!buf) return
  const stride = channel === 'rgba' ? 4 : 1
  const radius = Math.max(0.5, brush.size * Math.max(0.05, point.pressure))
  const r2 = radius * radius
  const hard = Math.max(0, Math.min(1, brush.hardness)) * radius
  const featherDenom = Math.max(1e-3, radius - hard)

  const x0 = Math.max(0, Math.floor(point.x - radius))
  const y0 = Math.max(0, Math.floor(point.y - radius))
  const x1 = Math.min(layer.width, Math.ceil(point.x + radius + 1))
  const y1 = Math.min(layer.height, Math.ceil(point.y + radius + 1))

  for (let y = y0; y < y1; y++) {
    const dy = y - point.y
    const dy2 = dy * dy
    for (let x = x0; x < x1; x++) {
      const dx = x - point.x
      const d2 = dx * dx + dy2
      if (d2 > r2) continue
      const d = Math.sqrt(d2)
      const falloff = d <= hard ? 1 : Math.max(0, 1 - (d - hard) / featherDenom)
      if (falloff <= 0) continue
      const o = (y * layer.width + x) * stride
      if (channel === 'rgba') {
        const newAlpha = Math.max(0, buf[o + 3] - 255 * falloff)
        buf[o + 3] = newAlpha
        if (newAlpha === 0) {
          buf[o] = 0
          buf[o + 1] = 0
          buf[o + 2] = 0
        }
      } else {
        buf[o] = Math.max(0, buf[o] - brush.maskValue * falloff)
      }
    }
  }
  layer.bumpVersion(channel)
}

function _bufferForChannel(layer: Layer, channel: PaintChannel): Uint8ClampedArray | null {
  if (channel === 'rgba') return layer.rgba
  if (channel === 'cfg') {
    if (layer.cfgMask === null) layer.cfgMask = new Uint8ClampedArray(layer.width * layer.height)
    return layer.cfgMask
  }
  if (channel === 'denoise') {
    if (layer.denoiseMask === null) layer.denoiseMask = new Uint8ClampedArray(layer.width * layer.height)
    return layer.denoiseMask
  }
  // prompt channel
  const p = layer.prompts[channel.index]
  if (!p) return null
  if (p.mask === null) p.mask = new Uint8ClampedArray(layer.width * layer.height)
  return p.mask
}
