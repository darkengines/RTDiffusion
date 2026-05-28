import { describe, expect, it } from 'vitest'
import { Layer } from './layer'

function makeLayer(opts: Partial<ConstructorParameters<typeof Layer>[0]> = {}) {
  return new Layer({ id: 'L1', width: 2, height: 2, ...opts })
}

describe('Layer', () => {
  it('allocates a zero rgba buffer when none provided', () => {
    const l = makeLayer()
    expect(l.rgba.length).toBe(16)
    expect(l.rgba.every(v => v === 0)).toBe(true)
    expect(l.version).toBe(1)
  })

  it('rejects rgba with wrong length', () => {
    expect(() => new Layer({ id: 'X', width: 2, height: 2, rgba: new Uint8ClampedArray(10) }))
      .toThrowError(/rgba length/)
  })

  it('rejects channel mask with wrong length', () => {
    expect(() => new Layer({
      id: 'X', width: 2, height: 2,
      cfgMask: new Uint8ClampedArray(3),
    })).toThrowError(/cfgMask length/)
  })

  it('bumps version on each scalar setter', () => {
    const l = makeLayer()
    const v0 = l.version
    l.setCfg(5)
    expect(l.version).toBe(v0 + 1)
    l.setDenoise(0.5)
    expect(l.version).toBe(v0 + 2)
  })

  it('does not bump version when scalar is unchanged', () => {
    const l = makeLayer({ cfg: 4 })
    const v0 = l.version
    l.setCfg(4)
    expect(l.version).toBe(v0)
  })

  it('addPrompt returns index and bumps version', () => {
    const l = makeLayer()
    const v0 = l.version
    const i = l.addPrompt({ text: 'a cat' })
    expect(i).toBe(0)
    expect(l.version).toBe(v0 + 1)
    expect(l.prompts).toHaveLength(1)
    expect(l.prompts[0].text).toBe('a cat')
    expect(l.prompts[0].negative).toBe('')
    expect(l.prompts[0].bind).toBe('rgba')
  })

  it('updatePrompt merges partial updates', () => {
    const l = makeLayer()
    l.addPrompt({ text: 'cat' })
    const v0 = l.version
    l.updatePrompt(0, { text: 'dog', negative: 'blurry' })
    expect(l.version).toBe(v0 + 1)
    expect(l.prompts[0].text).toBe('dog')
    expect(l.prompts[0].negative).toBe('blurry')
    expect(l.prompts[0].bind).toBe('rgba')
  })

  it('updatePrompt out of range throws', () => {
    const l = makeLayer()
    expect(() => l.updatePrompt(0, { text: 'x' })).toThrow()
  })

  it('removePrompt shrinks list', () => {
    const l = makeLayer()
    l.addPrompt({ text: 'a' })
    l.addPrompt({ text: 'b' })
    l.removePrompt(0)
    expect(l.prompts.map(p => p.text)).toEqual(['b'])
  })

  it('setRgba replaces buffer and bumps version', () => {
    const l = makeLayer()
    const buf = new Uint8ClampedArray(16)
    buf[0] = 255
    const v0 = l.version
    l.setRgba(buf)
    expect(l.rgba[0]).toBe(255)
    expect(l.version).toBe(v0 + 1)
  })

  it('setRgba with wrong length throws', () => {
    const l = makeLayer()
    expect(() => l.setRgba(new Uint8ClampedArray(8))).toThrow()
  })

  it('setVisible only bumps when changed', () => {
    const l = makeLayer({ visible: true })
    const v0 = l.version
    l.setVisible(true)
    expect(l.version).toBe(v0)
    l.setVisible(false)
    expect(l.version).toBe(v0 + 1)
  })
})
