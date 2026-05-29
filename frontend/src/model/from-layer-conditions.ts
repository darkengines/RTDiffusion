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

export type MaskChannelKind = 'cfg' | 'denoise' | 'prompt'

/**
 * Optional callback that lets the caller supply a painted mask directly --
 * bypasses the data-URL decode path. The legacy ``exportLayerConditions``
 * dropped ``prompt_mask`` (always undefined inline) and encoded
 * ``cfg_mask`` in a non-PNG format, so without this hook the v2 path
 * couldn't see those painted masks at all and prompts behaved like
 * unmasked text -> "I painted a concept and it wasn't applied".
 */
export type MaskOverride = (layerId: string, channel: MaskChannelKind) => Uint8ClampedArray | null

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
  /** When set, takes precedence over decoded data URLs for cfg/denoise/prompt masks. */
  maskOverride?: MaskOverride
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
    // Optional channel masks: override callback wins (direct canvas access);
    // else first inline data URL across regions wins.
    let cfgMask = opts.maskOverride?.(layerId, 'cfg') ?? null
    if (!cfgMask) {
      const cfgMaskUrl = regions.find(r => r.cfg_mask)?.cfg_mask
      if (cfgMaskUrl) cfgMask = await _safeMask(decoder, cfgMaskUrl, width, height)
    }
    let denMask = opts.maskOverride?.(layerId, 'denoise') ?? null
    if (!denMask) {
      const denMaskUrl = regions.find(r => r.denoise_mask)?.denoise_mask
      if (denMaskUrl) denMask = await _safeMask(decoder, denMaskUrl, width, height)
    }

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

    // Per-layer prompt mask: shared across regions of the same layer.
    // canvas-editor stores one mask per (layer, channel) regardless of
    // region count -- the override path returns that single canvas. We
    // reuse it for every prompt of the layer.
    const sharedPromptMask = opts.maskOverride?.(layerId, 'prompt') ?? null
    for (const r of regions) {
      const text = (r.prompt ?? '').trim()
      if (!text) continue
      let mask: Uint8ClampedArray | null = sharedPromptMask
      if (!mask && r.prompt_mask) {
        mask = await _safeMask(decoder, r.prompt_mask, width, height)
      }
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
 * Production decoder: Image + a single reusable canvas. Caches decoded
 * buffers keyed by (data-url, width, height) so the same painted layer
 * isn't re-decoded every frame while the user paints elsewhere.
 *
 * The context is created with ``willReadFrequently: true`` because
 * ``getImageData`` runs on every decode -- the browser otherwise prints
 * a perf warning and may keep the canvas on the GPU.
 *
 * ``_loadImage`` has a hard 1500 ms timeout: a malformed or empty data
 * URL would otherwise leave the promise pending forever, freezing the
 * send loop (and consequently the UI's RAF) until the layer is deleted.
 */
const _DECODE_CACHE_LIMIT = 16
const _IMAGE_LOAD_TIMEOUT_MS = 1500

export function createDomImageDecoder(canvasFactory: () => HTMLCanvasElement = () => document.createElement('canvas')): ImageDecoder {
  const cache = new Map<string, Uint8ClampedArray>()
  const get = (key: string) => {
    const v = cache.get(key)
    if (v === undefined) return undefined
    // LRU touch: re-insert.
    cache.delete(key)
    cache.set(key, v)
    return v
  }
  const put = (key: string, value: Uint8ClampedArray) => {
    cache.set(key, value)
    while (cache.size > _DECODE_CACHE_LIMIT) {
      const first = cache.keys().next().value
      if (first === undefined) break
      cache.delete(first)
    }
  }
  const decode = async (url: string, w: number, h: number, asRgba: boolean): Promise<Uint8ClampedArray> => {
    const key = `${asRgba ? 'r' : 'm'}:${w}x${h}:${url}`
    const cached = get(key)
    if (cached) return cached
    const out = await _decode(url, w, h, canvasFactory, asRgba)
    put(key, out)
    return out
  }
  return {
    decodeRgba(url, w, h) { return decode(url, w, h, true) },
    decodeMask(url, w, h) { return decode(url, w, h, false) },
  }
}

async function _decode(url: string, w: number, h: number, canvasFactory: () => HTMLCanvasElement, asRgba: boolean): Promise<Uint8ClampedArray> {
  const img = await _loadImage(url)
  const c = canvasFactory()
  c.width = w
  c.height = h
  const ctx = c.getContext('2d', { alpha: true, willReadFrequently: true })
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
    if (!url || typeof url !== 'string') {
      reject(new Error('image load: empty url'))
      return
    }
    const img = new Image()
    const timer = setTimeout(() => {
      img.src = ''
      reject(new Error('image load: timeout'))
    }, _IMAGE_LOAD_TIMEOUT_MS)
    img.onload = () => { clearTimeout(timer); resolve(img) }
    img.onerror = (e) => {
      clearTimeout(timer)
      reject(e instanceof Error ? e : new Error('image load failed'))
    }
    img.src = url
  })
}
