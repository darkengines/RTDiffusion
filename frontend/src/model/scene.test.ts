import { describe, expect, it } from 'vitest'
import { Layer } from './layer'
import { Scene } from './scene'

function makeScene() {
  return new Scene({ width: 2, height: 2 })
}

describe('Scene', () => {
  it('starts empty with defaults', () => {
    const s = makeScene()
    expect(s.layers).toHaveLength(0)
    expect(s.baseCfg).toBe(1.5)
    expect(s.baseDenoise).toBe(1.0)
    expect(s.basePrompt).toBe('')
  })

  it('addLayer requires matching dimensions', () => {
    const s = makeScene()
    expect(() => s.addLayer(new Layer({ id: 'A', width: 4, height: 4 })))
      .toThrowError(/dims/)
  })

  it('rejects duplicate layer ids', () => {
    const s = makeScene()
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    expect(() => s.addLayer(new Layer({ id: 'A', width: 2, height: 2 })))
      .toThrowError(/already in scene/)
  })

  it('version increases on add/remove/reorder', () => {
    const s = makeScene()
    const v0 = s.version
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    expect(s.version).toBeGreaterThan(v0)
    const v1 = s.version
    s.addLayer(new Layer({ id: 'B', width: 2, height: 2 }))
    expect(s.version).toBeGreaterThan(v1)
    const v2 = s.version
    s.reorderLayer('B', 0)
    expect(s.version).toBeGreaterThan(v2)
    const v3 = s.version
    s.removeLayer('A')
    expect(s.version).toBeGreaterThan(v3)
  })

  it('version aggregates layer mutations', () => {
    const s = makeScene()
    const layer = new Layer({ id: 'A', width: 2, height: 2 })
    s.addLayer(layer)
    const v0 = s.version
    layer.setCfg(7)
    expect(s.version).toBe(v0 + 1)
  })

  it('reorderLayer clamps to bounds', () => {
    const s = makeScene()
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    s.addLayer(new Layer({ id: 'B', width: 2, height: 2 }))
    s.reorderLayer('A', 99)
    expect(s.layers.map(l => l.id)).toEqual(['B', 'A'])
  })

  it('removeLayer is a no-op for unknown id', () => {
    const s = makeScene()
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    const v0 = s.version
    s.removeLayer('nope')
    expect(s.version).toBe(v0)
  })

  it('setBase* only bumps on change', () => {
    const s = makeScene()
    s.setBaseCfg(1.5)  // same
    const v0 = s.version
    s.setBaseCfg(2.0)
    expect(s.version).toBe(v0 + 1)
  })
})
