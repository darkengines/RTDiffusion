import { describe, expect, it } from 'vitest'
import { Layer } from '../model/layer'
import { brushTool, eraserTool } from './tools'
import type { BrushSettings } from './types'

function layer() {
  return new Layer({ id: 'L', width: 16, height: 16 })
}

const brush: BrushSettings = {
  size: 2,
  hardness: 1,
  color: { r: 255, g: 0, b: 0, a: 255 },
  maskValue: 255,
}

describe('brushTool', () => {
  it('begin lays a dab, step interpolates between points', () => {
    const l = layer()
    brushTool.begin(l, 'rgba', { x: 2, y: 2, pressure: 1 }, brush)
    brushTool.step(l, 'rgba', { x: 8, y: 2, pressure: 1 }, brush)
    brushTool.end()
    // Pixel at start and end should both be painted (red)
    expect(l.rgba[(2 * 16 + 2) * 4]).toBeGreaterThan(100)
    expect(l.rgba[(2 * 16 + 8) * 4]).toBeGreaterThan(100)
    // And a midpoint should also be painted (interpolation)
    expect(l.rgba[(2 * 16 + 5) * 4]).toBeGreaterThan(100)
  })

  it('step before begin is tolerated (no interpolation)', () => {
    const l = layer()
    brushTool.step(l, 'rgba', { x: 4, y: 4, pressure: 1 }, brush)
    expect(l.rgba[(4 * 16 + 4) * 4]).toBeGreaterThan(100)
  })

  it('end resets internal state', () => {
    const l = layer()
    brushTool.begin(l, 'rgba', { x: 1, y: 1, pressure: 1 }, brush)
    brushTool.end()
    // A subsequent step at a far point should not interpolate from the old
    // last-point; assert by checking that pixels between (1,1) and (12,12)
    // remain blank.
    brushTool.step(l, 'rgba', { x: 12, y: 12, pressure: 1 }, brush)
    expect(l.rgba[(6 * 16 + 6) * 4]).toBe(0)
    expect(l.rgba[(12 * 16 + 12) * 4]).toBeGreaterThan(100)
  })
})

describe('eraserTool', () => {
  it('erases along the stroke path', () => {
    const l = layer()
    brushTool.begin(l, 'rgba', { x: 4, y: 4, pressure: 1 }, brush)
    brushTool.step(l, 'rgba', { x: 12, y: 4, pressure: 1 }, brush)
    brushTool.end()
    // Erase a path that crosses the painted line
    eraserTool.begin(l, 'rgba', { x: 4, y: 4, pressure: 1 }, brush)
    eraserTool.step(l, 'rgba', { x: 12, y: 4, pressure: 1 }, brush)
    eraserTool.end()
    expect(l.rgba[(4 * 16 + 4) * 4 + 3]).toBe(0)
    expect(l.rgba[(4 * 16 + 12) * 4 + 3]).toBe(0)
  })
})
