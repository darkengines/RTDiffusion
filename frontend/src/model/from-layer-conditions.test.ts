import { describe, expect, it } from 'vitest'
import { buildSceneFromLayerConditions, type ImageDecoder, type LegacyLayerCondition } from './from-layer-conditions'

function stubDecoder(): ImageDecoder & {
  rgbaCalls: { url: string; w: number; h: number }[]
  maskCalls: { url: string; w: number; h: number }[]
} {
  const rgbaCalls: any[] = []
  const maskCalls: any[] = []
  return {
    rgbaCalls,
    maskCalls,
    async decodeRgba(url, w, h) {
      rgbaCalls.push({ url, w, h })
      const buf = new Uint8ClampedArray(w * h * 4)
      // colour the buffer by url hash so tests can distinguish layers
      const tag = url.charCodeAt(0) & 0xff
      for (let i = 3; i < buf.length; i += 4) buf[i] = tag  // alpha
      return buf
    },
    async decodeMask(url, w, h) {
      maskCalls.push({ url, w, h })
      const buf = new Uint8ClampedArray(w * h)
      buf.fill(url.charCodeAt(0) & 0xff)
      return buf
    },
  }
}

const baseOpts = {
  width: 4, height: 4,
  basePrompt: 'global',
  baseNegativePrompt: 'ng',
  baseCfg: 1.5,
  baseDenoise: 0.7,
}

describe('buildSceneFromLayerConditions', () => {
  it('empty list yields a scene with no layers but base values set', async () => {
    const s = await buildSceneFromLayerConditions([], baseOpts, stubDecoder())
    expect(s.layers).toHaveLength(0)
    expect(s.basePrompt).toBe('global')
    expect(s.baseCfg).toBe(1.5)
  })

  it('skips conditions without layer_id or image', async () => {
    const conds: LegacyLayerCondition[] = [
      {},
      { layer_id: 'L1' },
      { layer_id: '', image: 'data:image/png;base64,A' },
    ]
    const s = await buildSceneFromLayerConditions(conds, baseOpts, stubDecoder())
    expect(s.layers).toHaveLength(0)
  })

  it('one layer with one prompt region becomes one Layer with one prompt', async () => {
    const conds: LegacyLayerCondition[] = [{
      layer_id: 'L1', region_id: 'R1',
      image: 'A:rgba',
      prompt: 'a cat',
      negative_prompt: 'blurry',
      cfg: 8,
      denoise: 0.4,
    }]
    const dec = stubDecoder()
    const s = await buildSceneFromLayerConditions(conds, baseOpts, dec)
    expect(s.layers).toHaveLength(1)
    const l = s.layers[0]
    expect(l.id).toBe('L1')
    expect(l.cfg).toBe(8)
    expect(l.denoise).toBe(0.4)
    expect(l.cfgBind).toBe('rgba')  // no painted cfg_mask
    expect(l.denoiseBind).toBe('rgba')
    expect(l.prompts).toHaveLength(1)
    expect(l.prompts[0]).toMatchObject({ text: 'a cat', negative: 'blurry', bind: 'rgba' })
    expect(l.prompts[0].mask).toBeNull()
    expect(dec.rgbaCalls).toEqual([{ url: 'A:rgba', w: 4, h: 4 }])
  })

  it('groups multiple regions on same layer_id into one Layer with multiple prompts', async () => {
    const conds: LegacyLayerCondition[] = [
      { layer_id: 'L1', region_id: 'R1', image: 'A:rgba', prompt: 'a cat' },
      { layer_id: 'L1', region_id: 'R2', prompt: 'with hat', prompt_mask: 'A:mask' },
      { layer_id: 'L1', region_id: 'R3', prompt: '' },  // dropped
    ]
    const s = await buildSceneFromLayerConditions(conds, baseOpts, stubDecoder())
    expect(s.layers).toHaveLength(1)
    const l = s.layers[0]
    expect(l.prompts.map(p => p.text)).toEqual(['a cat', 'with hat'])
    expect(l.prompts[0].bind).toBe('rgba')
    expect(l.prompts[1].bind).toBe('self')
    expect(l.prompts[1].mask).not.toBeNull()
  })

  it('first non-null cfg/denoise scalar across regions wins', async () => {
    const conds: LegacyLayerCondition[] = [
      { layer_id: 'L1', image: 'A', prompt: 'x' },
      { layer_id: 'L1', cfg: 12 },
      { layer_id: 'L1', denoise: 0.3 },
    ]
    const s = await buildSceneFromLayerConditions(conds, baseOpts, stubDecoder())
    expect(s.layers[0].cfg).toBe(12)
    expect(s.layers[0].denoise).toBe(0.3)
  })

  it('cfg_mask / denoise_mask flip bind to self when present', async () => {
    const conds: LegacyLayerCondition[] = [
      { layer_id: 'L1', image: 'A', cfg_mask: 'A:cfg', denoise_mask: 'A:den' },
    ]
    const s = await buildSceneFromLayerConditions(conds, baseOpts, stubDecoder())
    const l = s.layers[0]
    expect(l.cfgBind).toBe('self')
    expect(l.denoiseBind).toBe('self')
    expect(l.cfgMask).not.toBeNull()
    expect(l.denoiseMask).not.toBeNull()
  })

  it('preserves layer order by first occurrence (bottom-up)', async () => {
    const conds: LegacyLayerCondition[] = [
      { layer_id: 'B', image: 'B' },
      { layer_id: 'A', image: 'A' },
      { layer_id: 'B', prompt: 'second' },
    ]
    const s = await buildSceneFromLayerConditions(conds, baseOpts, stubDecoder())
    expect(s.layers.map(l => l.id)).toEqual(['B', 'A'])
  })

  it('decoder rgba failure skips the layer', async () => {
    const dec = stubDecoder()
    const original = dec.decodeRgba
    dec.decodeRgba = async (url, w, h) => {
      if (url === 'A:bad') throw new Error('decode failed')
      return original.call(dec, url, w, h)
    }
    const conds: LegacyLayerCondition[] = [
      { layer_id: 'L1', image: 'A:bad' },
      { layer_id: 'L2', image: 'A:ok' },
    ]
    const s = await buildSceneFromLayerConditions(conds, baseOpts, dec)
    expect(s.layers.map(l => l.id)).toEqual(['L2'])
  })

  it('mask decode failure leaves mask null but keeps the prompt', async () => {
    const dec = stubDecoder()
    const original = dec.decodeMask
    dec.decodeMask = async (url, w, h) => {
      if (url === 'A:bad') throw new Error('mask broke')
      return original.call(dec, url, w, h)
    }
    const conds: LegacyLayerCondition[] = [
      { layer_id: 'L1', image: 'A', prompt: 'x', prompt_mask: 'A:bad' },
    ]
    const s = await buildSceneFromLayerConditions(conds, baseOpts, dec)
    expect(s.layers[0].prompts[0].mask).toBeNull()
    expect(s.layers[0].prompts[0].bind).toBe('rgba')  // fell back to default
  })
})
