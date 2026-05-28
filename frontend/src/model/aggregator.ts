import { resolveChannelBinding, resolvePromptBinding } from './bindings'
import type { Scene } from './scene'
import type { AggregatedScene } from './types'

/**
 * Walk visible layers bottom-up and produce flat buffers ready for the wire
 * format. Aggregation rules (see plan §Data Model):
 *
 *   rgba    = Porter-Duff "over" with straight (non-premultiplied) alpha
 *   cfgMap  = max(out, layer.cfg * bindingAlpha) per pixel; initialised to baseCfg
 *   denMap  = max(out, layer.denoise * bindingAlpha) per pixel; init baseDenoise
 *   prompts = flattened list, bottom-up, with bindings resolved to a Float32
 *             attention mask (or null = whole canvas)
 *
 * Pure function. Allocates fresh output buffers; callers may reuse via the
 * `into` overload if perf matters in the render loop.
 */
export function aggregate(scene: Scene): AggregatedScene {
  const { width, height } = scene
  const pixelCount = width * height

  const rgba = new Uint8ClampedArray(pixelCount * 4)
  const cfgMap = new Float32Array(pixelCount)
  cfgMap.fill(scene.baseCfg)
  const denoiseMap = new Float32Array(pixelCount)
  denoiseMap.fill(scene.baseDenoise)

  const prompts: AggregatedScene['prompts'] = []
  const controlnet: AggregatedScene['controlnet'] = []

  for (const layer of scene.layers) {
    if (!layer.visible) continue
    if (layer.width !== width || layer.height !== height) continue

    // rgba: straight-alpha over
    _overInPlace(rgba, layer.rgba, pixelCount)

    // cfg: max with binding-scaled scalar
    const cfgAlpha = resolveChannelBinding(layer, layer.cfgBind, layer.cfgMask)
    _maxScalarInto(cfgMap, layer.cfg, cfgAlpha, pixelCount)

    // denoise: same shape
    const denAlpha = resolveChannelBinding(layer, layer.denoiseBind, layer.denoiseMask)
    _maxScalarInto(denoiseMap, layer.denoise, denAlpha, pixelCount)

    // prompts: flatten with resolved attention masks
    for (const p of layer.prompts) {
      if (!p.text) continue
      const attention = resolvePromptBinding(layer, p.bind, p.mask)
      prompts.push({ text: p.text, negative: p.negative, mask: attention })
    }

    if (layer.controlnet !== null) {
      controlnet.push({ layerId: layer.id, config: layer.controlnet })
    }
  }

  return {
    width,
    height,
    rgba,
    cfgMap,
    denoiseMap,
    prompts,
    controlnet,
    basePrompt: scene.basePrompt,
    baseNegativePrompt: scene.baseNegativePrompt,
    baseDenoise: scene.baseDenoise,
    baseCfg: scene.baseCfg,
  }
}

/** Porter-Duff over: dst = src over dst, with straight alpha. */
function _overInPlace(dst: Uint8ClampedArray, src: Uint8ClampedArray, pixelCount: number): void {
  for (let i = 0; i < pixelCount; i++) {
    const o = i * 4
    const srcA = src[o + 3] / 255
    if (srcA === 0) continue
    const dstA = dst[o + 3] / 255
    const outA = srcA + dstA * (1 - srcA)
    if (outA === 0) continue
    const invSrcA = 1 - srcA
    dst[o] = (src[o] * srcA + dst[o] * dstA * invSrcA) / outA
    dst[o + 1] = (src[o + 1] * srcA + dst[o + 1] * dstA * invSrcA) / outA
    dst[o + 2] = (src[o + 2] * srcA + dst[o + 2] * dstA * invSrcA) / outA
    dst[o + 3] = outA * 255
  }
}

/** acc[i] = max(acc[i], scalar * alpha[i]); alpha=null treated as uniform 1.0. */
function _maxScalarInto(
  acc: Float32Array,
  scalar: number,
  alpha: Float32Array | null,
  pixelCount: number,
): void {
  if (alpha === null) {
    for (let i = 0; i < pixelCount; i++) {
      if (scalar > acc[i]) acc[i] = scalar
    }
    return
  }
  for (let i = 0; i < pixelCount; i++) {
    const v = scalar * alpha[i]
    if (v > acc[i]) acc[i] = v
  }
}
