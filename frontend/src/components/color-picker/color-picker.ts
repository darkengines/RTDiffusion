import { LitElement, css, html } from 'lit'
import { customElement } from 'lit/decorators.js'
import { StoreController } from '@nanostores/lit'
import { $canvas } from '../../stores/canvas.store'
import { PALETTE, QUICK_PALETTE } from '../../constants'
import type { ColorSlot, RgbaColor } from '../../types'

function hexToRgba(hex: string): RgbaColor {
  const h = hex.replace('#', '')
  return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16), a: 1 }
}

function rgbaToHex({ r, g, b }: RgbaColor) {
  return `#${[r, g, b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`
}


@customElement('rtd-color-picker')
export class RtdColorPicker extends LitElement {
  private _canvas = new StoreController(this, $canvas)
  plannerMode = 'complementary'

  static styles = css`
    :host { display: block }
    .color-panel { display: flex; flex-direction: column; gap: 8px; padding: 8px }
    .color-stack { display: flex; align-items: flex-end; gap: 8px }
    .chip-group { position: relative; width: 46px; height: 46px; flex-shrink: 0 }
    .chip { position: absolute; width: 32px; height: 32px; border-radius: 5px; border: 2px solid var(--surface, #fff); cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,.2); transition: outline 0.1s }
    .chip.primary { top: 0; left: 0 }
    .chip.secondary { bottom: 0; right: 0 }
    .chip.active { outline: 2px solid var(--accent, #5f6fff); outline-offset: 1px; z-index: 1 }
    .swap-btn { position: absolute; right: 0; bottom: 14px; background: var(--surface, #fff); border: 1px solid var(--border, #ccc); border-radius: 3px; font-size: 10px; cursor: pointer; padding: 1px 3px; z-index: 2 }
    .color-inputs { flex: 1 }
    .native-row { display: flex; gap: 6px; align-items: center; margin-bottom: 6px }
    input[type=color] { width: 36px; height: 28px; border: 1px solid var(--border, #ccc); border-radius: 4px; padding: 0 2px; cursor: pointer; background: none }
    .hex-input { font-family: monospace; font-size: 12px; width: 76px; border: 1px solid var(--border, #ccc); border-radius: 4px; padding: 4px 6px }
    .channel-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px }
    .channel-field { display: flex; flex-direction: column; gap: 2px }
    .channel-field span { font-size: 10px; font-weight: 600; color: var(--label, #666) }
    .channel-field input { font-size: 12px; border: 1px solid var(--border, #ccc); border-radius: 3px; padding: 3px 5px; width: 100%; box-sizing: border-box }
    .palette-label { font-size: 10px; font-weight: 600; color: var(--label, #666); margin-top: 4px }
    .palette-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 3px }
    .swatch { width: 100%; aspect-ratio: 1; border-radius: 3px; cursor: pointer; border: 1px solid transparent; box-sizing: border-box }
    .swatch:hover { border-color: var(--accent, #5f6fff); transform: scale(1.1) }
  `

  render() {
    const { brushColor, secondaryBrushColor, activeColorSlot } = this._canvas.value
    const activeHex = activeColorSlot === 'primary' ? brushColor : secondaryBrushColor
    const rgba = hexToRgba(activeHex)
    return html`<div class="color-panel">
      <div class="color-stack">
        <div class="chip-group">
          <button class=${`chip secondary${activeColorSlot === 'secondary' ? ' active' : ''}`}
            style="background:${secondaryBrushColor}"
            @click=${() => $canvas.setKey('activeColorSlot', 'secondary' as ColorSlot)}></button>
          <button class=${`chip primary${activeColorSlot === 'primary' ? ' active' : ''}`}
            style="background:${brushColor}"
            @click=${() => $canvas.setKey('activeColorSlot', 'primary' as ColorSlot)}></button>
          <button class="swap-btn" title="Swap" @click=${() => {
            $canvas.setKey('brushColor', secondaryBrushColor)
            $canvas.setKey('secondaryBrushColor', brushColor)
          }}>⇄</button>
        </div>
        <div class="color-inputs">
          <div class="native-row">
            <input type="color" .value=${activeHex} @input=${(e: Event) => this._setHex((e.target as HTMLInputElement).value)} />
            <input class="hex-input" spellcheck="false" .value=${activeHex.toUpperCase()}
              @change=${(e: Event) => { const v = (e.target as HTMLInputElement).value; if (/^#[0-9a-f]{6}$/i.test(v)) this._setHex(v) }} />
          </div>
          <div class="channel-grid">
            ${this._channelField('R', rgba.r, 0, 255, (v) => this._setRgba({ ...rgba, r: v }))}
            ${this._channelField('G', rgba.g, 0, 255, (v) => this._setRgba({ ...rgba, g: v }))}
            ${this._channelField('B', rgba.b, 0, 255, (v) => this._setRgba({ ...rgba, b: v }))}
            ${this._channelField('A', Math.round(rgba.a * 100), 0, 100, (v) => this._setRgba({ ...rgba, a: v / 100 }))}
          </div>
        </div>
      </div>
      <div class="palette-label">Quick palette</div>
      <div class="palette-grid">
        ${QUICK_PALETTE.map((c) => html`<button class="swatch" style="background:${c}" title=${c} @click=${() => this._setHex(c)}></button>`)}
      </div>
      <div class="palette-label">Full palette</div>
      <div class="palette-grid">
        ${PALETTE.map((c) => html`<button class="swatch" style="background:${c}" title=${c} @click=${() => this._setHex(c)}></button>`)}
      </div>
    </div>`
  }

  private _channelField(label: string, value: number, min: number, max: number, onChange: (v: number) => void) {
    return html`<div class="channel-field">
      <span>${label}</span>
      <input type="number" .value=${String(value)} min=${min} max=${max}
        @change=${(e: Event) => onChange(Number((e.target as HTMLInputElement).value))} />
    </div>`
  }

  private _setHex(hex: string) {
    const { activeColorSlot } = this._canvas.value
    if (activeColorSlot === 'primary') $canvas.setKey('brushColor', hex)
    else $canvas.setKey('secondaryBrushColor', hex)
  }

  private _setRgba(rgba: RgbaColor) {
    this._setHex(rgbaToHex(rgba))
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-color-picker': RtdColorPicker }
}
