import { aggregate } from '../model/aggregator'
import { aggregatedToWire, type PngEncoder } from '../model/encoder'
import type { Scene } from '../model/scene'
import type { WirePayload } from '../model/types'

/**
 * Transport adapter the loop pushes into. `send` returns true when the
 * payload was accepted (queued/sent) and false when the transport is
 * backpressured -- the loop will retry on the next RAF tick.
 */
export interface SendTransport {
  send(payload: WirePayload): boolean
}

export interface SceneSenderOptions {
  scene: Scene
  encoder: PngEncoder
  transport: SendTransport
  /** Optional clock for tests; defaults to performance.now. */
  now?: () => number
  /** Optional RAF scheduler for tests; defaults to requestAnimationFrame. */
  raf?: (cb: (ts: number) => void) => number
  /** Optional cancel; defaults to cancelAnimationFrame. */
  cancel?: (handle: number) => void
}

/**
 * Watches scene.version on each RAF tick and pushes a fresh wire payload
 * through `transport.send` when the scene mutated AND the transport is not
 * backpressured. No debounce, no setInterval -- mid-stroke mutations
 * naturally collapse to one send per frame.
 *
 * The v1 send pipeline used a 33 ms debounce + multiple dirty flags +
 * setInterval, which dropped the first frame whenever the user started
 * drawing inside one debounce window. This loop eliminates that whole
 * class of bug: it's a monotonic-version watcher with explicit backpressure.
 */
export class SceneSender {
  readonly scene: Scene
  private _encoder: PngEncoder
  private _transport: SendTransport
  private _now: () => number
  private _raf: (cb: (ts: number) => void) => number
  private _cancel: (handle: number) => void
  private _handle = 0
  private _running = false
  private _lastSentVersion = -1
  private _lastSentAt = 0

  constructor(opts: SceneSenderOptions) {
    this.scene = opts.scene
    this._encoder = opts.encoder
    this._transport = opts.transport
    this._now = opts.now ?? (() => performance.now())
    this._raf = opts.raf ?? ((cb) => requestAnimationFrame(cb))
    this._cancel = opts.cancel ?? ((h) => cancelAnimationFrame(h))
  }

  /** Last scene version successfully accepted by the transport. */
  get lastSentVersion(): number { return this._lastSentVersion }
  /** Wall-clock time (ms from _now) the last accepted send happened. */
  get lastSentAt(): number { return this._lastSentAt }
  get running(): boolean { return this._running }

  start(): void {
    if (this._running) return
    this._running = true
    this._schedule()
  }

  stop(): void {
    this._running = false
    if (this._handle) {
      this._cancel(this._handle)
      this._handle = 0
    }
  }

  /**
   * Single tick exposed for tests. Returns the wire payload that was sent
   * this tick, or null if nothing to do / backpressured.
   */
  tick(): WirePayload | null {
    if (this.scene.version === this._lastSentVersion) return null
    const payload = aggregatedToWire(aggregate(this.scene), this._encoder)
    const accepted = this._transport.send(payload)
    if (!accepted) return null
    this._lastSentVersion = this.scene.version
    this._lastSentAt = this._now()
    return payload
  }

  private _schedule(): void {
    this._handle = this._raf(() => {
      this._handle = 0
      if (!this._running) return
      this.tick()
      this._schedule()
    })
  }
}
