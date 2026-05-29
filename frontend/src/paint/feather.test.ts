import { describe, expect, it } from 'vitest'
import { featherMask } from './feather'

function makeBuf(width: number, height: number, fn: (x: number, y: number) => number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      out[y * width + x] = fn(x, y)
  return out
}

describe('featherMask', () => {
  it('feather=0 returns an identical copy (not the same reference)', () => {
    const src = makeBuf(4, 4, (x, y) => (x + y) * 16)
    const out = featherMask(src, 4, 4, 0)
    expect(Array.from(out)).toEqual(Array.from(src))
    expect(out).not.toBe(src)
  })

  it('positive feather grows the mask outward', () => {
    // 16x16 painted square inside a 32x32 canvas.
    const buf = makeBuf(32, 32, (x, y) => (x >= 8 && x <= 23 && y >= 8 && y <= 23) ? 255 : 0)
    const out = featherMask(buf, 32, 32, 2)
    // Outside-edge pixel (6, 16) was 0; after outward feather should be > 0.
    expect(out[16 * 32 + 6]).toBeGreaterThan(0)
    // Deep interior (16, 16) -- far from edge -- stays approximately full.
    expect(out[16 * 32 + 16]).toBeGreaterThan(200)
  })

  it('negative feather shrinks the mask inward', () => {
    // 24x24 painted square inside a 32x32 canvas (small border).
    const buf = makeBuf(32, 32, (x, y) => (x >= 4 && x <= 27 && y >= 4 && y <= 27) ? 255 : 0)
    const out = featherMask(buf, 32, 32, -2)
    // Inside-edge pixel (5, 16) -- one pixel inside the original edge --
    // should be reduced by inward feather (mask shrunk).
    expect(out[16 * 32 + 5]).toBeLessThan(255)
    // Deep interior (16, 16) stays approximately full.
    expect(out[16 * 32 + 16]).toBeGreaterThan(200)
  })

  it('does not mutate the input', () => {
    const src = makeBuf(4, 4, () => 128)
    const ref = Array.from(src)
    featherMask(src, 4, 4, 3)
    expect(Array.from(src)).toEqual(ref)
  })

  it('large radius does not throw or overflow', () => {
    const src = makeBuf(16, 16, (x, y) => (x === 8 && y === 8) ? 255 : 0)
    // Radius larger than image dimensions -- should clamp index lookups.
    const out = featherMask(src, 16, 16, 64)
    expect(out.length).toBe(256)
  })

  it('fully-opaque source stays approximately opaque after feathering', () => {
    const src = makeBuf(8, 8, () => 255)
    const out = featherMask(src, 8, 8, 4)
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(255)
  })

  it('fully-empty source stays empty', () => {
    const src = makeBuf(8, 8, () => 0)
    const out = featherMask(src, 8, 8, 4)
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(0)
    const outNeg = featherMask(src, 8, 8, -4)
    for (let i = 0; i < outNeg.length; i++) expect(outNeg[i]).toBe(0)
  })
})
