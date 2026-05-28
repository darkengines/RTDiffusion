import { css, html, LitElement, type PropertyValues } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import type { Layer } from '../model/layer'
import type { Scene } from '../model/scene'
import { brushTool, eraserTool } from './tools'
import type { BrushSettings, PaintChannel, PointerPoint, Tool } from './types'

/**
 * Plain Canvas2D paint host -- no Fabric.js. Owns one display canvas (the
 * full composited scene shown to the user) and writes through to the active
 * layer's pixel buffers via the active tool. The render loop watches
 * `scene.version` and re-paints when it changes.
 *
 * Properties:
 *   scene        — Scene to display. The element observes pointer events on
 *                   the canvas and routes them to the active tool acting on
 *                   `activeLayerId` / `activeChannel`.
 *   activeLayerId — which layer receives paint strokes.
 *   activeChannel — which channel of the layer is painted ('rgba' | 'cfg' |
 *                   'denoise' | { kind: 'prompt'; index }).
 *   activeTool    — 'brush' | 'eraser'.
 *   brush         — current brush settings.
 *
 * Events:
 *   `rtd-scene-changed` (CustomEvent<{ version: number }>) on every commit
 *      so external listeners (eg. the send loop) can react.
 */
@customElement('rtd-canvas-host')
export class RtdCanvasHost extends LitElement {
  static styles = css`
    :host { display: block; position: relative; touch-action: none; }
    canvas { display: block; image-rendering: pixelated; cursor: crosshair; }
  `

  @property({ attribute: false }) scene: Scene | null = null
  @property({ attribute: false }) activeLayerId: string = ''
  @property({ attribute: false }) activeChannel: PaintChannel = 'rgba'
  @property({ attribute: false }) activeTool: 'brush' | 'eraser' = 'brush'
  @property({ attribute: false }) brush: BrushSettings = {
    size: 16,
    hardness: 0.7,
    color: { r: 0, g: 0, b: 0, a: 255 },
    maskValue: 255,
  }

  @state() private _lastRenderedVersion = -1
  private _canvas: HTMLCanvasElement | null = null
  private _ctx: CanvasRenderingContext2D | null = null
  private _activePointerId: number | null = null
  private _rafHandle = 0

  protected firstUpdated(): void {
    this._canvas = this.renderRoot.querySelector('canvas')
    this._ctx = this._canvas?.getContext('2d', { alpha: true }) ?? null
    this._scheduleRender()
  }

  protected updated(changed: PropertyValues): void {
    if (changed.has('scene')) this._scheduleRender()
  }

  disconnectedCallback(): void {
    super.disconnectedCallback()
    if (this._rafHandle) cancelAnimationFrame(this._rafHandle)
  }

  private _scheduleRender(): void {
    if (this._rafHandle) return
    this._rafHandle = requestAnimationFrame(() => {
      this._rafHandle = 0
      this._renderIfDirty()
    })
  }

  private _renderIfDirty(): void {
    if (!this.scene || !this._canvas || !this._ctx) return
    if (this.scene.version === this._lastRenderedVersion) {
      // schedule another check in case the scene gets mutated externally
      this._rafHandle = requestAnimationFrame(() => {
        this._rafHandle = 0
        this._renderIfDirty()
      })
      return
    }
    const { width, height } = this.scene
    if (this._canvas.width !== width) this._canvas.width = width
    if (this._canvas.height !== height) this._canvas.height = height
    this._ctx.clearRect(0, 0, width, height)
    // Composite visible layers bottom-up via globalCompositeOperation=source-over.
    this._ctx.globalCompositeOperation = 'source-over'
    for (const layer of this.scene.layers) {
      if (!layer.visible) continue
      if (layer.width !== width || layer.height !== height) continue
      const id = new ImageData(layer.rgba as Uint8ClampedArray<ArrayBuffer>, width, height)
      // putImageData is the only canvas API that respects raw buffers with
      // straight alpha; drawImage of an ImageBitmap would premultiply.
      const tmp = new OffscreenCanvas(width, height)
      const tctx = tmp.getContext('2d', { alpha: true })
      if (!tctx) continue
      tctx.putImageData(id, 0, 0)
      this._ctx.drawImage(tmp, 0, 0)
    }
    this._lastRenderedVersion = this.scene.version
    this.dispatchEvent(new CustomEvent('rtd-scene-changed', {
      detail: { version: this.scene.version },
      bubbles: true,
      composed: true,
    }))
    // re-arm to catch the next mutation
    this._rafHandle = requestAnimationFrame(() => {
      this._rafHandle = 0
      this._renderIfDirty()
    })
  }

  private _toolFor(): Tool {
    return this.activeTool === 'eraser' ? eraserTool : brushTool
  }

  private _activeLayer(): Layer | null {
    if (!this.scene) return null
    return this.scene.findLayer(this.activeLayerId) ?? null
  }

  private _pointFromEvent(ev: PointerEvent): PointerPoint | null {
    if (!this._canvas || !this.scene) return null
    const rect = this._canvas.getBoundingClientRect()
    const sx = this.scene.width / Math.max(1, rect.width)
    const sy = this.scene.height / Math.max(1, rect.height)
    return {
      x: (ev.clientX - rect.left) * sx,
      y: (ev.clientY - rect.top) * sy,
      pressure: ev.pressure > 0 ? ev.pressure : (ev.pointerType === 'mouse' ? 1 : 0.5),
    }
  }

  private _onPointerDown(ev: PointerEvent): void {
    if (this._activePointerId !== null) return
    const layer = this._activeLayer()
    const point = this._pointFromEvent(ev)
    if (!layer || !point) return
    this._activePointerId = ev.pointerId
    this._canvas?.setPointerCapture(ev.pointerId)
    this._toolFor().begin(layer, this.activeChannel, point, this.brush)
    this._scheduleRender()
    ev.preventDefault()
  }

  private _onPointerMove(ev: PointerEvent): void {
    if (this._activePointerId !== ev.pointerId) return
    const layer = this._activeLayer()
    const point = this._pointFromEvent(ev)
    if (!layer || !point) return
    this._toolFor().step(layer, this.activeChannel, point, this.brush)
    this._scheduleRender()
  }

  private _onPointerEnd(ev: PointerEvent): void {
    if (this._activePointerId !== ev.pointerId) return
    const layer = this._activeLayer()
    if (layer) this._toolFor().end(layer, this.activeChannel, this.brush)
    this._canvas?.releasePointerCapture(ev.pointerId)
    this._activePointerId = null
    this._scheduleRender()
  }

  render() {
    return html`
      <canvas
        @pointerdown=${this._onPointerDown}
        @pointermove=${this._onPointerMove}
        @pointerup=${this._onPointerEnd}
        @pointercancel=${this._onPointerEnd}
      ></canvas>
    `
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'rtd-canvas-host': RtdCanvasHost
  }
}
