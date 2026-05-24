import { LitElement, css, html } from 'lit'
import { customElement, property } from 'lit/decorators.js'

/**
 * DearImGui-style number spinner with drag scrubbing.
 *
 * Events:
 *   rtd-change: CustomEvent<{ value: number }>
 *
 * Usage:
 *   <rtd-number label="CFG" .value=${cfg} min="0" max="30" step="0.1" decimals="1"
 *     @rtd-change=${(e) => cfg = e.detail.value}></rtd-number>
 */
@customElement('rtd-number')
export class RtdNumber extends LitElement {
  @property({ type: String }) label = ''
  @property({ type: Number }) value = 0
  @property({ type: Number }) min = -Infinity
  @property({ type: Number }) max = Infinity
  @property({ type: Number }) step = 1
  @property({ type: Number }) decimals = 0
  @property({ type: Boolean }) disabled = false

  private _drag?: { startX: number; startValue: number; pointerId: number }
  private _editing = false

  static styles = css`
    :host { display: block; }
    .wrap {
      display: grid;
      grid-template-columns: max-content minmax(40px, 1fr) 18px 18px;
      align-items: center;
      height: var(--ctrl-h, 22px);
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius, 2px);
      background: var(--bg-input, #14141a);
      overflow: hidden;
      cursor: ew-resize;
      user-select: none;
    }
    .wrap:focus-within { border-color: var(--border-focus, #4d87c4); }
    .label {
      padding: 0 var(--pad-x, 6px);
      font-size: var(--font-size, 12px);
      color: var(--fg-label, #9898a8);
      white-space: nowrap;
      pointer-events: none;
    }
    input[type=number] {
      min-width: 0;
      border: none;
      background: transparent;
      color: var(--fg, #d4d4d8);
      font-size: var(--font-size, 12px);
      font-family: var(--font-mono, monospace);
      text-align: right;
      padding: 0 2px;
      outline: none;
      cursor: text;
    }
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button { display: none; }
    .btn {
      display: grid; place-items: center;
      width: 18px; height: 100%;
      border: none; background: none;
      color: var(--fg-dim, #7a7a8a);
      font-size: 10px; cursor: pointer;
      padding: 0;
    }
    .btn:hover { background: var(--bg-hover, #2e2e3c); color: var(--fg, #d4d4d8); }
    :host([disabled]) .wrap { opacity: 0.4; pointer-events: none; }
  `

  render() {
    const fmt = this.decimals > 0
      ? this.value.toFixed(this.decimals)
      : String(Math.round(this.value))
    return html`
      <div class="wrap"
        @pointerdown=${this._onPointerDown}
        @pointermove=${this._onPointerMove}
        @pointerup=${this._onPointerUp}
        @pointercancel=${this._onPointerUp}>
        <span class="label">${this.label}</span>
        <input type="number"
          .value=${fmt}
          min=${String(this.min)} max=${String(this.max)} step=${this.step}
          @focus=${() => { this._editing = true }}
          @blur=${() => { this._editing = false }}
          @input=${this._onInputChange}
          @keydown=${(e: KeyboardEvent) => e.stopPropagation()}
          @pointerdown=${(e: PointerEvent) => e.stopPropagation()} />
        <button class="btn" @click=${this._dec} tabindex="-1" type="button">−</button>
        <button class="btn" @click=${this._inc} tabindex="-1" type="button">+</button>
      </div>`
  }

  private _clamp(v: number) {
    return Math.max(this.min, Math.min(this.max, v))
  }

  private _round(v: number) {
    const d = Math.max(0, Math.ceil(-Math.log10(this.step)))
    return Number(v.toFixed(d + 1))
  }

  private _emit(v: number) {
    this.dispatchEvent(new CustomEvent('rtd-change', { detail: { value: this._clamp(this._round(v)) }, bubbles: true, composed: true }))
  }

  private _dec = (e: MouseEvent) => {
    const mult = e.shiftKey ? 10 : e.altKey ? 0.1 : 1
    this._emit(this.value - this.step * mult)
  }

  private _inc = (e: MouseEvent) => {
    const mult = e.shiftKey ? 10 : e.altKey ? 0.1 : 1
    this._emit(this.value + this.step * mult)
  }

  private _onInputChange(e: Event) {
    const v = Number((e.target as HTMLInputElement).value)
    if (Number.isFinite(v)) this._emit(v)
  }

  private _onPointerDown(e: PointerEvent) {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLButtonElement) return
    if (e.button !== 0) return
    e.preventDefault()
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    this._drag = { startX: e.clientX, startValue: this.value, pointerId: e.pointerId }
  }

  private _onPointerMove(e: PointerEvent) {
    if (!this._drag || e.pointerId !== this._drag.pointerId || this._editing) return
    const mult = e.shiftKey ? 10 : e.altKey ? 0.1 : 1
    const delta = (e.clientX - this._drag.startX) * this.step * mult
    this._emit(this._drag.startValue + delta)
  }

  private _onPointerUp(e: PointerEvent) {
    if (!this._drag || e.pointerId !== this._drag.pointerId) return
    ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
    this._drag = undefined
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-number': RtdNumber }
}
