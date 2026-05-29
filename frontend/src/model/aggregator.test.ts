import { describe, expect, it } from 'vitest'
import { aggregate } from './aggregator'
import { Layer } from './layer'
import { Scene } from './scene'

function makeScene(opts: Partial<ConstructorParameters<typeof Scene>[0]> = {}) {
  return new Scene({ width: 2, height: 1, ...opts })
}

describe('aggregate', () => {
  it('empty scene yields base values everywhere', () => {
    const s = makeScene({ baseCfg: 4, baseDenoise: 0.5 })
    const out = aggregate(s)
    expect(out.rgba.every(v => v === 0)).toBe(true)
    expect(Array.from(out.cfgMap)).toEqual([4, 4])
    expect(Array.from(out.denoiseMap)).toEqual([0.5, 0.5])
    expect(out.prompts).toEqual([])
    expect(out.controlnet).toEqual([])
  })

  it('skips invisible layers', () => {
    const s = makeScene()
    const l = new Layer({ id: 'A', width: 2, height: 1, cfg: 20, denoise: 0.9 })
    l.setVisible(false)
    s.addLayer(l)
    const out = aggregate(s)
    expect(out.cfgMap[0]).toBe(s.baseCfg)
    expect(out.denoiseMap[0]).toBe(s.baseDenoise)
  })

  it('rgba over: opaque top covers transparent bottom', () => {
    const s = makeScene()
    const a = new Layer({
      id: 'A', width: 2, height: 1,
      rgba: new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 0, 0]),
    })
    const b = new Layer({
      id: 'B', width: 2, height: 1,
      rgba: new Uint8ClampedArray([0, 0, 0, 0, 0, 255, 0, 255]),
    })
    s.addLayer(a)
    s.addLayer(b)
    const out = aggregate(s)
    // pixel 0: only A opaque -> red. pixel 1: only B opaque -> green.
    expect(Array.from(out.rgba.slice(0, 4))).toEqual([255, 0, 0, 255])
    expect(Array.from(out.rgba.slice(4, 8))).toEqual([0, 255, 0, 255])
  })

  it('rgba over: top alpha 128 blends with bottom', () => {
    const s = makeScene()
    const a = new Layer({
      id: 'A', width: 2, height: 1,
      rgba: new Uint8ClampedArray([200, 200, 200, 255, 0, 0, 0, 0]),
    })
    const b = new Layer({
      id: 'B', width: 2, height: 1,
      rgba: new Uint8ClampedArray([0, 0, 0, 128, 0, 0, 0, 0]),
    })
    s.addLayer(a)
    s.addLayer(b)
    const out = aggregate(s)
    // pixel 0: black over gray, alpha 0.502 => (200*0.5 + 0*0.5) ~ 100
    const r = out.rgba[0]
    expect(r).toBeGreaterThanOrEqual(99)
    expect(r).toBeLessThanOrEqual(101)
  })

  it('cfg: max with rgba-bound contribution', () => {
    const s = makeScene({ baseCfg: 2 })
    const rgba = new Uint8ClampedArray([0, 0, 0, 255, 0, 0, 0, 0])
    const a = new Layer({ id: 'A', width: 2, height: 1, cfg: 8, cfgBind: 'rgba', rgba })
    s.addLayer(a)
    const out = aggregate(s)
    // pixel 0: max(2, 8*1.0) = 8. pixel 1: max(2, 8*0) = 2.
    expect(out.cfgMap[0]).toBe(8)
    expect(out.cfgMap[1]).toBe(2)
  })

  it('denoise: max takes the strongest layer per pixel', () => {
    const s = makeScene({ baseDenoise: 0.1 })
    const a = new Layer({ id: 'A', width: 2, height: 1, denoise: 0.4, denoiseBind: 'none' })
    const b = new Layer({ id: 'B', width: 2, height: 1, denoise: 0.7, denoiseBind: 'none' })
    s.addLayer(a)
    s.addLayer(b)
    const out = aggregate(s)
    // 'none' covers whole canvas; max(0.1, 0.4, 0.7) = 0.7 everywhere.
    expect(out.denoiseMap[0]).toBeCloseTo(0.7, 4)
    expect(out.denoiseMap[1]).toBeCloseTo(0.7, 4)
  })

  it('denoise with self-painted mask scales scalar by mask', () => {
    const s = makeScene({ baseDenoise: 0.0 })
    const mask = new Uint8ClampedArray([255, 0])
    const a = new Layer({
      id: 'A', width: 2, height: 1, denoise: 0.8,
      denoiseBind: 'self', denoiseMask: mask,
    })
    s.addLayer(a)
    const out = aggregate(s)
    expect(out.denoiseMap[0]).toBeCloseTo(0.8, 4)
    expect(out.denoiseMap[1]).toBe(0)
  })

  it('prompts flattened bottom-up with bindings resolved', () => {
    const s = makeScene()
    const a = new Layer({ id: 'A', width: 2, height: 1 })
    a.addPrompt({ text: 'cat', mask: new Uint8ClampedArray([255, 0]), bind: 'self' })
    const b = new Layer({ id: 'B', width: 2, height: 1 })
    b.addPrompt({ text: 'dog', bind: 'none' })
    b.addPrompt({ text: 'sky' })  // default bind=rgba; layer rgba all zero -> alpha 0
    s.addLayer(a)
    s.addLayer(b)
    const out = aggregate(s)
    expect(out.prompts.map(p => p.text)).toEqual(['cat', 'dog', 'sky'])
    // cat has self mask
    expect(Array.from(out.prompts[0].mask!)).toEqual([1, 0])
    // dog has bind=none -> null (whole canvas)
    expect(out.prompts[1].mask).toBeNull()
    // sky bind=rgba with zero alpha -> [0,0]
    expect(Array.from(out.prompts[2].mask!)).toEqual([0, 0])
  })

  it('empty-text prompts are dropped', () => {
    const s = makeScene()
    const a = new Layer({ id: 'A', width: 2, height: 1 })
    a.addPrompt({ text: '' })
    a.addPrompt({ text: 'real' })
    s.addLayer(a)
    const out = aggregate(s)
    expect(out.prompts.map(p => p.text)).toEqual(['real'])
  })

  it('decoupled channels: painted prompt softmap does NOT change denoise map', () => {
    // The three layer channels (prompt / denoise / cfg) are independent
    // per the user-defined semantics. Painting a prompt softmap drives
    // regional CLIP attention only; it does not modify the denoise map.
    const s = new Scene({ width: 2, height: 1, baseDenoise: 0.2 })
    const layer = new Layer({
      id: 'L', width: 2, height: 1,
      denoise: 0.2, denoiseBind: 'none',
    })
    layer.addPrompt({ text: 'a cat', mask: new Uint8ClampedArray([255, 0]), bind: 'self' })
    s.addLayer(layer)
    const out = aggregate(s)
    // Both pixels stay at scene baseDenoise regardless of where the
    // prompt softmap was painted.
    expect(out.denoiseMap[0]).toBeCloseTo(0.2, 4)
    expect(out.denoiseMap[1]).toBeCloseTo(0.2, 4)
  })

  it('decoupled channels: opaque rgba + painted prompt also leaves denoise alone', () => {
    const s = new Scene({ width: 2, height: 1, baseDenoise: 0.3 })
    const layer = new Layer({
      id: 'L', width: 2, height: 1,
      rgba: new Uint8ClampedArray([200, 0, 0, 255, 0, 0, 0, 0]),
      denoise: 0.3, denoiseBind: 'none',
    })
    layer.addPrompt({ text: 'a cat', mask: new Uint8ClampedArray([255, 0]), bind: 'self' })
    s.addLayer(layer)
    const out = aggregate(s)
    expect(out.denoiseMap[0]).toBeCloseTo(0.3, 4)
  })

  it('decoupled channels: whole-canvas prompt (mask=null) does NOT bump anything', () => {
    // A prompt with no spatial mask must NOT push the denoise map to
    // max across the entire canvas -- that would force the engine to
    // regenerate every pixel, defeating layered composition.
    const s = new Scene({ width: 2, height: 1, baseDenoise: 0.25 })
    const layer = new Layer({ id: 'L', width: 2, height: 1, denoise: 0.25, denoiseBind: 'none' })
    layer.addPrompt({ text: 'global', bind: 'none' })  // mask=null
    s.addLayer(layer)
    const out = aggregate(s)
    expect(out.denoiseMap[0]).toBeCloseTo(0.25, 4)
    expect(out.denoiseMap[1]).toBeCloseTo(0.25, 4)
  })

  it('controlnet emitted per layer that has one', () => {
    const s = makeScene()
    const a = new Layer({
      id: 'A', width: 2, height: 1,
      controlnet: {
        model: 'canny', modelPath: null, scale: 0.9, start: 0.0, end: 0.8,
        preprocessorParams: { cannyLow: 50, cannyHigh: 200 },
      },
    })
    s.addLayer(a)
    s.addLayer(new Layer({ id: 'B', width: 2, height: 1 }))  // no cn
    const out = aggregate(s)
    expect(out.controlnet).toHaveLength(1)
    expect(out.controlnet[0].layerId).toBe('A')
    expect(out.controlnet[0].config.model).toBe('canny')
  })
})
