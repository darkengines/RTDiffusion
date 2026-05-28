// Bridge: turn the v1 ``editor.exportLayerConditions()`` dict list into a v2
// Scene. Lets the v2 aggregator (max-op cfg/denoise, Porter-Duff rgba,
// per-prompt attention masks) run on top of the existing UI without
// touching canvas-editor.ts or any component template.
//
// The image-decode step is pluggable so vitest+jsdom can stub it; production
// uses ``createDomImageDecoder`` which goes through Image + a hidden canvas.

import { Layer } from './layer'
import { Scene } from './scene'
import type { ChannelBind, PromptBind } from './types'

export interface ImageDecoder {
  /** Decode a data URL into an HxWx4 uint8 buffer. */
  decodeRgba(dataUrl: string, w: number, h: number): Promise<Uint8ClampedArray>
  /** Decode a grayscale (or RGB) data URL into an HxW uint8 buffer (L channel). */
  decodeMask(dataUrl: string, w: number, h: number): Promise<Uint8ClampedArray>
}

/** Subset of the v1 condition dict that we read. */
export interface LegacyLayerCondition {
  layer_id?: string
  region_id?: string
  image?: string
  prompt?: string
  negative_prompt?: string
  cfg?: number | null
  denoise?: number | null
  cfg_mask?: string | null
  denoise_mask?: string | null
  prompt_mask?: string | null
}

export interface SceneFromLegacyOptions {
  width: number
  height: number
  basePrompt: string
  baseNegativePrompt: string
  baseCfg: number
  baseDenoise: number
  /** Default binding for cfg/denoise scalars when no per-pixel mask is set. */
  defaultChannelBind?: ChannelBind
  /** Default binding for prompts when no painted mask is present. */
  defaultPromptBind?: PromptBind
}

/**
 * Group conditions by ``layer_id`` (preserving first-occurrence order),
 * decode each region's image / masks once, then build one Layer per
 * layer_id. Regions sharing a layer_id stack their prompts.
 */
export async function buildSceneFromLayerConditions(
  conditions: readonly LegacyLayerCondition[],
  opts: SceneFromLegacyOptions,
  decoder: ImageDecoder,
): Promise<Scene> {
  const { width, height } = opts
  const scene = new Scene({
    width, height,
    basePrompt: opts.basePrompt,
    baseNegativePrompt: opts.baseNegativePrompt,
    baseCfg: opts.baseCfg,
    baseDenoise: opts.baseDenoise,
  })
  const channelBind: ChannelBind = opts.defaultChannelBind ?? 'rgba'
  const promptBind: PromptBind = opts.defaultPromptBind ?? 'rgba'

  // Group by layer_id, preserving first-seen order so the bottom-up stack
  // matches the v1 condition ordering.
  const order: string[] = []
  const groups = new Map<string, LegacyLayerCondition[]>()
  for (const c of conditions) {
    const id = String(c.layer_id ?? '')
    if (!id) continue
    if (!groups.has(id)) { groups.set(id, []); order.push(id) }
    groups.get(id)!.push(c)
  }

  for (const layerId of order) {
    const regions = groups.get(layerId)!
    // First region with an image wins; the rest contribute prompts only.
    const imgUrl = regions.find(r => r.image)?.image
    if (!imgUrl) continue
    let rgba: Uint8ClampedArray
    try {
      rgba = await decoder.decodeRgba(imgUrl, width, height)
    } catch {
      continue
    }
    // First non-null cfg/denoise scalar across regions wins; bind cfg/den
    // through the layer's own rgba alpha by default so the engine treats
    // them as "active where the layer is visible".
    const firstCfg = regions.find(r => r.cfg != null && r.cfg !== undefined)?.cfg
    const firstDen = regions.find(r => r.denoise != null && r.denoise !== undefined)?.denoise
    // Optional channel masks: first present wins (one per layer in v2).
    const cfgMaskUrl = regions.find(r => r.cfg_mask)?.cfg_mask
    const denMaskUrl = regions.find(r => r.denoise_mask)?.denoise_mask
    const cfgMask = cfgMaskUrl ? await _safeMask(decoder, cfgMaskUrl, width, height) : null
    const denMask = denMaskUrl ? await _safeMask(decoder, denMaskUrl, width, height) : null

    const layer = new Layer({
      id: layerId,
      width,
      height,
      rgba,
      cfg: typeof firstCfg === 'number' ? firstCfg : opts.baseCfg,
      cfgMask,
      cfgBind: cfgMask ? 'self' : channelBind,
      denoise: typeof firstDen === 'number' ? firstDen : opts.baseDenoise,
      denoiseMask: denMask,
      denoiseBind: denMask ? 'self' : channelBind,
    })

    for (const r of regions) {
      const text = (r.prompt ?? '').trim()
      if (!text) continue
      const mask = r.prompt_mask ? await _safeMask(decoder, r.prompt_mask, width, height) : null
      layer.addPrompt({
        text,
        negative: (r.negative_prompt ?? '').trim(),
        mask,
        bind: mask ? 'self' : promptBind,
      })
    }

    scene.addLayer(layer)
  }

  return scene
}

async function _safeMask(d: ImageDecoder, url: string, w: number, h: number): Promise<Uint8ClampedArray | null> {
  try {
    return await d.decodeMask(url, w, h)
  } catch {
    return null
  }
}

/**
 * Production decoder: Image + a single reusable canvas. Pass the same
 * factory to ``createCanvasPngEncoder`` if you want to share one element
 * across the encode + decode paths to keep allocations down.
 */
export function createDomImageDecoder(canvasFactory: () => HTMLCanvasElement = () => document.createElement('canvas')): ImageDecoder {
  return {
    decodeRgba(url, w, h) { return _decode(url, w, h, canvasFactory, true) },
    decodeMask(url, w, h) { return _decode(url, w, h, canvasFactory, false) },
  }
}

async function _decode(url: string, w: number, h: number, canvasFactory: () => HTMLCanvasElement, asRgba: boolean): Promise<Uint8ClampedArray> {
  const img = await _loadImage(url)
  const c = canvasFactory()
  c.width = w
  c.height = h
  const ctx = c.getContext('2d', { alpha: true })
  if (!ctx) throw new Error('decoder: 2d context unavailable')
  ctx.clearRect(0, 0, w, h)
  ctx.drawImage(img, 0, 0, w, h)
  const id = ctx.getImageData(0, 0, w, h)
  if (asRgba) {
    return new Uint8ClampedArray(id.data.buffer.slice(0))
  }
  const out = new Uint8ClampedArray(w * h)
  // Use the luma channel: max of (R alpha) so painted strokes that drew
  // RGB read like a grayscale mask. If the source has alpha-only content
  // (i.e. painted on a transparent backing), alpha is what we want.
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = Math.max(id.data[j], id.data[j + 3])
  }
  return out
}

function _loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = (e) => reject(e instanceof Error ? e : new Error('image load failed'))
    img.src = url
  })
}
