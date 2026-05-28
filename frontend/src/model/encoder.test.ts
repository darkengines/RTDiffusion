import { describe, expect, it } from 'vitest'
import { aggregate } from './aggregator'
import { aggregatedToWire, CFG_HI, type PngEncoder } from './encoder'
import { Layer } from './layer'
import { Scene } from './scene'

/**
 * Stub encoder for unit tests: returns a deterministic string per buffer so
 * we can assert payload structure without a real canvas. Real PNG bytes are
 * tested manually in the browser (jsdom has no Canvas2D).
 */
function stubEncoder(): PngEncoder & { rgbaCalls: number; lumCalls: number; lastLum: Float32Array | null } {
  let rgbaCalls = 0
  let lumCalls = 0
  let lastLum: Float32Array | null = null
  return {
    rgbaCalls,
    lumCalls,
    lastLum,
    encodeRgba(buf, w, h) {
      this.rgbaCalls += 1
      return `rgba:${w}x${h}:${buf.length}`
    },
    encodeLuminance(buf, w, h) {
      this.lumCalls += 1
      this.lastLum = buf
      return `lum:${w}x${h}:${buf[0].toFixed(4)}`
    },
  }
}

function makeScene(opts: Partial<ConstructorParameters<typeof Scene>[0]> = {}) {
  return new Scene({ width: 2, height: 1, ...opts })
}

describe('aggregatedToWire', () => {
  it('encodes empty scene into base-only payload', () => {
    const s = makeScene({ baseCfg: 4, baseDenoise: 0.5, basePrompt: 'global' })
    const enc = stubEncoder()
    const wire = aggregatedToWire(aggregate(s), enc)
    expect(wire.width).toBe(2)
    expect(wire.height).toBe(1)
    expect(wire.rgba_b64).toBe('rgba:2x1:8')
    expect(wire.base_prompt).toBe('global')
    expect(wire.base_cfg).toBe(4)
    expect(wire.base_denoise).toBe(0.5)
    expect(wire.prompts).toEqual([])
    expect(wire.controlnet).toEqual([])
  })

  it('normalises cfg map by CFG_HI before luminance encoding', () => {
    const s = makeScene({ baseCfg: CFG_HI })  // 30 -> normalised to 1.0
    const enc = stubEncoder()
    aggregatedToWire(aggregate(s), enc)
    // The luminance buffer the encoder saw should be all 1.0 (cfg/CFG_HI).
    expect(enc.lastLum![0]).toBeCloseTo(1.0, 4)
  })

  it('passes denoise through unchanged (already in [0,1])', () => {
    const s = makeScene({ baseCfg: 0, baseDenoise: 0.42 })
    const enc = stubEncoder()
    aggregatedToWire(aggregate(s), enc)
    // Two encodeLuminance calls happen (cfg + denoise). The second is denoise.
    // Inspect lastLum: it'll be denoise (last call).
    expect(enc.lastLum![0]).toBeCloseTo(0.42, 4)
  })

  it('emits one prompts entry per aggregated prompt', () => {
    const s = makeScene()
    const a = new Layer({ id: 'A', width: 2, height: 1 })
    a.addPrompt({ text: 'cat', negative: 'blurry', mask: new Uint8ClampedArray([255, 0]), bind: 'self' })
    a.addPrompt({ text: 'dog', bind: 'none' })
    s.addLayer(a)

    const enc = stubEncoder()
    const wire = aggregatedToWire(aggregate(s), enc)
    expect(wire.prompts).toHaveLength(2)
    expect(wire.prompts[0]).toEqual({
      text: 'cat',
      negative: 'blurry',
      mask_b64: expect.any(String),
    })
    expect(wire.prompts[1]).toEqual({
      text: 'dog',
      negative: '',
      mask_b64: undefined,
    })
  })

  it('emits controlnet entries with snake_case preprocessor keys', () => {
    const s = makeScene()
    s.addLayer(new Layer({
      id: 'A', width: 2, height: 1,
      controlnet: {
        model: 'canny',
        modelPath: 'cn/model.safetensors',
        scale: 0.7, start: 0.0, end: 0.8,
        preprocessorParams: {
          cannyLow: 50, cannyHigh: 180,
          openposeHands: true, openposeFace: false,
          mlsdThrV: 0.2, mlsdThrD: 0.3,
          lineartCoarse: true,
        },
      },
    }))
    const wire = aggregatedToWire(aggregate(s), stubEncoder())
    expect(wire.controlnet).toHaveLength(1)
    const cn = wire.controlnet[0]
    expect(cn.layer_id).toBe('A')
    expect(cn.model).toBe('canny')
    expect(cn.model_path).toBe('cn/model.safetensors')
    expect(cn.scale).toBe(0.7)
    expect(cn.preprocessor_params).toEqual({
      canny_low: 50,
      canny_high: 180,
      openpose_hands: true,
      openpose_face: false,
      mlsd_thr_v: 0.2,
      mlsd_thr_d: 0.3,
      lineart_coarse: true,
    })
  })

  it('omits preprocessor keys when undefined', () => {
    const s = makeScene()
    s.addLayer(new Layer({
      id: 'A', width: 2, height: 1,
      controlnet: {
        model: 'depth', modelPath: null,
        scale: 1, start: 0, end: 1,
        preprocessorParams: {},
      },
    }))
    const wire = aggregatedToWire(aggregate(s), stubEncoder())
    expect(wire.controlnet[0].preprocessor_params).toEqual({})
  })

  it('omits prompt mask_b64 when bind resolves to whole canvas', () => {
    const s = makeScene()
    const a = new Layer({ id: 'A', width: 2, height: 1 })
    a.addPrompt({ text: 'sky', bind: 'none' })
    s.addLayer(a)
    const wire = aggregatedToWire(aggregate(s), stubEncoder())
    expect(wire.prompts[0].mask_b64).toBeUndefined()
  })
})
