import { describe, expect, it } from 'vitest'
import { Layer } from '../model/layer'
import { eraseDab, paintDab } from './stroke'
import type { BrushSettings } from './types'

function layer(w = 8, h = 8) {
  return new Layer({ id: 'L', width: w, height: h })
}

function brush(overrides: Partial<BrushSettings> = {}): BrushSettings {
  return {
    size: 3,
    hardness: 1,
    color: { r: 200, g: 100, b: 50, a: 255 },
    maskValue: 255,
    ...overrides,
  }
}

describe('paintDab', () => {
  it('writes rgba pixels at the dab center', () => {
    const l = layer()
    paintDab(l, 'rgba', { x: 4, y: 4, pressure: 1 }, brush())
    const o = (4 * 8 + 4) * 4
    expect(l.rgba[o]).toBeGreaterThan(100)
    expect(l.rgba[o + 3]).toBe(255)  // fully opaque at center
    expect(l.version).toBeGreaterThan(1)
  })

  it('feathers when hardness < 1', () => {
    const l = layer(16, 16)
    paintDab(l, 'rgba', { x: 8, y: 8, pressure: 1 }, brush({ hardness: 0, size: 4 }))
    const center = (8 * 16 + 8) * 4
    const edge = (8 * 16 + 11) * 4  // ~3 px from center, near radius
    expect(l.rgba[center + 3]).toBe(255)
    // edge should have partial alpha (between 0 and 255)
    expect(l.rgba[edge + 3]).toBeGreaterThan(0)
    expect(l.rgba[edge + 3]).toBeLessThan(255)
  })

  it('clips to layer bounds without throwing', () => {
    const l = layer(4, 4)
    paintDab(l, 'rgba', { x: 0, y: 0, pressure: 1 }, brush({ size: 8 }))
    paintDab(l, 'rgba', { x: 100, y: 100, pressure: 1 }, brush({ size: 8 }))
    // top-left got painted
    expect(l.rgba[3]).toBe(255)  // alpha of pixel 0
  })

  it('cfg channel: lazily allocates cfgMask then max-writes value', () => {
    const l = layer()
    expect(l.cfgMask).toBeNull()
    paintDab(l, 'cfg', { x: 4, y: 4, pressure: 1 }, brush({ maskValue: 200 }))
    expect(l.cfgMask).not.toBeNull()
    const o = 4 * 8 + 4
    expect(l.cfgMask![o]).toBe(200)
  })

  it('max-write semantics: repeated dabs do not exceed maskValue', () => {
    const l = layer()
    paintDab(l, 'denoise', { x: 2, y: 2, pressure: 1 }, brush({ maskValue: 200 }))
    paintDab(l, 'denoise', { x: 2, y: 2, pressure: 1 }, brush({ maskValue: 100 }))
    expect(l.denoiseMask![2 * 8 + 2]).toBe(200)  // max, not overwrite
  })

  it('prompt channel uses the indexed prompt mask', () => {
    const l = layer()
    l.addPrompt({ text: 'cat' })
    paintDab(l, { kind: 'prompt', index: 0 }, { x: 1, y: 1, pressure: 1 }, brush())
    expect(l.prompts[0].mask).not.toBeNull()
    expect(l.prompts[0].mask![1 * 8 + 1]).toBe(255)
  })

  it('out-of-range prompt index is a no-op (no throw)', () => {
    const l = layer()
    expect(() =>
      paintDab(l, { kind: 'prompt', index: 9 }, { x: 1, y: 1, pressure: 1 }, brush()),
    ).not.toThrow()
  })

  it('pressure 0.05 still produces a stroke (clamped)', () => {
    const l = layer()
    paintDab(l, 'rgba', { x: 4, y: 4, pressure: 0.05 }, brush({ size: 6 }))
    const o = (4 * 8 + 4) * 4
    expect(l.rgba[o + 3]).toBeGreaterThan(0)
  })
})

describe('eraseDab', () => {
  it('rgba: reduces alpha and clears rgb when alpha hits 0', () => {
    const l = layer()
    paintDab(l, 'rgba', { x: 4, y: 4, pressure: 1 }, brush())
    eraseDab(l, 'rgba', { x: 4, y: 4, pressure: 1 }, brush())
    const o = (4 * 8 + 4) * 4
    expect(l.rgba[o + 3]).toBe(0)
    expect(l.rgba[o]).toBe(0)
  })

  it('mask channel: subtracts toward zero', () => {
    const l = layer()
    paintDab(l, 'cfg', { x: 4, y: 4, pressure: 1 }, brush({ maskValue: 200 }))
    eraseDab(l, 'cfg', { x: 4, y: 4, pressure: 1 }, brush({ maskValue: 150 }))
    const o = 4 * 8 + 4
    expect(l.cfgMask![o]).toBe(50)
    eraseDab(l, 'cfg', { x: 4, y: 4, pressure: 1 }, brush({ maskValue: 200 }))
    expect(l.cfgMask![o]).toBe(0)
  })
})
