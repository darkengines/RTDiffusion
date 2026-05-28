import { describe, expect, it } from 'vitest'
import type { PngEncoder } from '../model/encoder'
import { Layer } from '../model/layer'
import { Scene } from '../model/scene'
import type { WirePayload } from '../model/types'
import { SceneSender, type SendTransport } from './loop'

function stubEncoder(): PngEncoder {
  return {
    encodeRgba: (b, w, h) => `rgba:${w}x${h}:${b.length}`,
    encodeLuminance: (b, w, h) => `lum:${w}x${h}:${b[0].toFixed(2)}`,
  }
}

function captureTransport(): SendTransport & { sent: WirePayload[]; backpressured: boolean } {
  const sent: WirePayload[] = []
  const t: any = {
    backpressured: false,
    sent,
    send(payload: WirePayload) {
      if (this.backpressured) return false
      sent.push(payload)
      return true
    },
  }
  return t
}

function makeScene() {
  return new Scene({ width: 2, height: 2 })
}

describe('SceneSender', () => {
  it('tick sends one payload when scene version advances', () => {
    const s = makeScene()
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    const t = captureTransport()
    const sender = new SceneSender({ scene: s, encoder: stubEncoder(), transport: t })
    expect(sender.tick()).not.toBeNull()
    expect(t.sent).toHaveLength(1)
    expect(sender.lastSentVersion).toBe(s.version)
  })

  it('tick is a no-op when scene version is unchanged', () => {
    const s = makeScene()
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    const t = captureTransport()
    const sender = new SceneSender({ scene: s, encoder: stubEncoder(), transport: t })
    sender.tick()
    expect(sender.tick()).toBeNull()
    expect(t.sent).toHaveLength(1)
  })

  it('does not advance lastSentVersion when transport backpressures', () => {
    const s = makeScene()
    const layer = new Layer({ id: 'A', width: 2, height: 2 })
    s.addLayer(layer)
    const t = captureTransport()
    t.backpressured = true
    const sender = new SceneSender({ scene: s, encoder: stubEncoder(), transport: t })
    expect(sender.tick()).toBeNull()
    expect(sender.lastSentVersion).toBe(-1)
    // unblock + tick again -> sends
    t.backpressured = false
    expect(sender.tick()).not.toBeNull()
    expect(sender.lastSentVersion).toBe(s.version)
  })

  it('mid-stroke mutations collapse to one payload per tick', () => {
    const s = makeScene()
    const layer = new Layer({ id: 'A', width: 2, height: 2 })
    s.addLayer(layer)
    const t = captureTransport()
    const sender = new SceneSender({ scene: s, encoder: stubEncoder(), transport: t })
    // multiple mutations between ticks
    layer.setCfg(2)
    layer.setCfg(3)
    layer.setDenoise(0.4)
    sender.tick()
    expect(t.sent).toHaveLength(1)  // one tick = one send despite 3 mutations
  })

  it('start/stop schedule and cancel the RAF', () => {
    const s = makeScene()
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    const t = captureTransport()
    let rafCalls = 0
    let cancelCalls = 0
    let pendingCb: ((ts: number) => void) | null = null
    const sender = new SceneSender({
      scene: s, encoder: stubEncoder(), transport: t,
      raf: (cb) => { rafCalls++; pendingCb = cb; return 42 },
      cancel: () => { cancelCalls++ },
    })
    sender.start()
    expect(rafCalls).toBe(1)
    expect(sender.running).toBe(true)
    // fire the scheduled tick
    pendingCb!(0)
    expect(t.sent).toHaveLength(1)
    expect(rafCalls).toBe(2)
    sender.stop()
    expect(cancelCalls).toBe(1)
    expect(sender.running).toBe(false)
  })

  it('start is idempotent', () => {
    const s = makeScene()
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    let rafCalls = 0
    const sender = new SceneSender({
      scene: s, encoder: stubEncoder(), transport: captureTransport(),
      raf: () => { rafCalls++; return 1 },
      cancel: () => {},
    })
    sender.start()
    sender.start()
    sender.start()
    expect(rafCalls).toBe(1)
  })

  it('records lastSentAt from the injected clock', () => {
    const s = makeScene()
    s.addLayer(new Layer({ id: 'A', width: 2, height: 2 }))
    const sender = new SceneSender({
      scene: s, encoder: stubEncoder(), transport: captureTransport(),
      now: () => 12345,
    })
    sender.tick()
    expect(sender.lastSentAt).toBe(12345)
  })
})
