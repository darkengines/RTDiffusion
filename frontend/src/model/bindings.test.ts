import { describe, expect, it } from 'vitest'
import { resolveChannelBinding, resolvePromptBinding } from './bindings'
import { Layer } from './layer'

function makeLayer(opts: Partial<ConstructorParameters<typeof Layer>[0]> = {}) {
  // 2x1 layer keeps assertions short.
  return new Layer({ id: 'L', width: 2, height: 1, ...opts })
}

describe('resolvePromptBinding', () => {
  it('bind=none returns null (uniform 1.0)', () => {
    const l = makeLayer()
    expect(resolvePromptBinding(l, 'none', null)).toBeNull()
  })

  it('bind=self with mask decodes to [0,1]', () => {
    const l = makeLayer()
    const mask = new Uint8ClampedArray([0, 255])
    const out = resolvePromptBinding(l, 'self', mask)
    expect(out).not.toBeNull()
    expect(Array.from(out!)).toEqual([0, 1])
  })

  it('bind=self with null mask returns null', () => {
    const l = makeLayer()
    expect(resolvePromptBinding(l, 'self', null)).toBeNull()
  })

  it('bind=rgba reads layer rgba alpha', () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 0, 0, 0, 0, 255])
    const l = makeLayer({ rgba })
    const out = resolvePromptBinding(l, 'rgba', null)
    expect(out).not.toBeNull()
    expect(Array.from(out!)).toEqual([0, 1])
  })
})

describe('resolveChannelBinding', () => {
  it('bind=none returns null', () => {
    const l = makeLayer()
    expect(resolveChannelBinding(l, 'none', null)).toBeNull()
  })

  it('bind=rgba mirrors rgba alpha', () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 128, 0, 0, 0, 64])
    const l = makeLayer({ rgba })
    const out = resolveChannelBinding(l, 'rgba', null)!
    expect(out[0]).toBeCloseTo(128 / 255, 4)
    expect(out[1]).toBeCloseTo(64 / 255, 4)
  })

  it('bind=self with own mask returns its float form', () => {
    const l = makeLayer()
    const mask = new Uint8ClampedArray([255, 128])
    const out = resolveChannelBinding(l, 'self', mask)!
    expect(out[0]).toBe(1)
    expect(out[1]).toBeCloseTo(128 / 255, 4)
  })

  it('bind=prompt takes per-pixel max across prompt masks', () => {
    const l = makeLayer()
    l.addPrompt({ text: 'a', mask: new Uint8ClampedArray([255, 0]), bind: 'self' })
    l.addPrompt({ text: 'b', mask: new Uint8ClampedArray([0, 128]), bind: 'self' })
    const out = resolveChannelBinding(l, 'prompt', null)!
    expect(out[0]).toBe(1)
    expect(out[1]).toBeCloseTo(128 / 255, 4)
  })

  it('bind=prompt collapses to null if any prompt is whole-canvas', () => {
    const l = makeLayer()
    l.addPrompt({ text: 'a', mask: new Uint8ClampedArray([255, 0]), bind: 'self' })
    l.addPrompt({ text: 'b', mask: null, bind: 'none' })  // whole canvas
    expect(resolveChannelBinding(l, 'prompt', null)).toBeNull()
  })

  it('bind=prompt with no prompts returns null', () => {
    const l = makeLayer()
    expect(resolveChannelBinding(l, 'prompt', null)).toBeNull()
  })
})
