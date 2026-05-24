import { LitElement, css, html } from 'lit'
import { customElement, property, query } from 'lit/decorators.js'
import type { DebugSurface } from '../../services/stream.service'

@customElement('rtd-debug-surface')
export class RtdDebugSurface extends LitElement {
  @property({ attribute: false }) surface?: DebugSurface
  @query('canvas') private canvasElement?: HTMLCanvasElement
  @query('video') private videoElement?: HTMLVideoElement

  private raf?: number

  static styles = css`
    :host { display: block; width: 100%; height: 100%; background: #000 }
    canvas, video { width: 100%; height: 100%; object-fit: contain; display: block; background: #000 }
  `

  disconnectedCallback() {
    super.disconnectedCallback()
    if (this.raf !== undefined) window.cancelAnimationFrame(this.raf)
    this.raf = undefined
    if (this.videoElement) this.videoElement.srcObject = null
  }

  protected updated() {
    this.attachSurface()
  }

  render() {
    const surface = this.surface
    if (!surface) return html``
    return surface.kind === 'stream'
      ? html`<video muted autoplay playsinline></video>`
      : html`<canvas></canvas>`
  }

  private attachSurface() {
    const surface = this.surface
    if (!surface) return
    if (surface.kind === 'stream') {
      if (this.raf !== undefined) window.cancelAnimationFrame(this.raf)
      this.raf = undefined
      if (this.videoElement && this.videoElement.srcObject !== surface.stream) {
        this.videoElement.srcObject = surface.stream
        void this.videoElement.play().catch(() => undefined)
      }
      return
    }
    if (this.raf === undefined) this.drawCanvasSurface()
  }

  private drawCanvasSurface = () => {
    const surface = this.surface
    const source = surface?.kind === 'canvas' ? surface.canvas : undefined
    const target = this.canvasElement
    if (source && target) {
      if (target.width !== source.width) target.width = source.width
      if (target.height !== source.height) target.height = source.height
      const context = target.getContext('2d', { alpha: true })
      context?.clearRect(0, 0, target.width, target.height)
      context?.drawImage(source, 0, 0, target.width, target.height)
    }
    this.raf = window.requestAnimationFrame(this.drawCanvasSurface)
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-debug-surface': RtdDebugSurface }
}