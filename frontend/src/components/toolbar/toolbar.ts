import { LitElement, css, html } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { StoreController } from '@nanostores/lit'
import { $canvas } from '../../stores/canvas.store'
import { $layer } from '../../stores/layer.store'
import { $scene } from '../../stores/scene.store'
import { SIZE_PRESETS, TOOL_DEFS } from '../../constants'
import type { ColorSlot, EditorDest, FillEdgeStrategy, MaskChannel, ShapeDrawMode, ToolMode } from '../../types'
import '../rtd-number/rtd-number'
import '../color-picker/color-picker'

@customElement('rtd-toolbar')
export class RtdToolbar extends LitElement {
  private _canvas = new StoreController(this, $canvas)
  private _layer = new StoreController(this, $layer)
  private _scene = new StoreController(this, $scene)

  @state() private _cpOpen = false
  @state() private _cpX = 0
  @state() private _cpY = 0

  static styles = css`
    :host {
      display: grid; gap: 4px; height: auto; padding: 6px 4px;
      overflow: hidden;
      color: var(--fg, #d4d4d8);
    }
    .toolbar-line { display: flex; align-items: center; align-content: flex-start; gap: 4px; flex-wrap: wrap; min-width: 0; }
    .tool-properties { min-height: 26px; padding-top: 2px; border-top: 1px solid rgba(90,90,122,.35); }
    .context-properties { display: flex; align-items: center; gap: 5px; padding-right: 6px; margin-right: 2px; border-right: 1px solid rgba(90,90,122,.35); }
    .property-group { display: flex; align-items: center; gap: 5px; flex-shrink: 0; font-size: 11px; }
    .property-group label { color: var(--fg-dim, #7a7a8a); white-space: nowrap; }
    .property-group select {
      padding: 2px 4px; font-size: 11px;
      background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c);
      border-radius: 3px; color: var(--fg, #d4d4d8); cursor: pointer;
    }
    .sep { width: 1px; height: 20px; background: var(--border, #3a3a4c); margin: 0 2px; flex-shrink: 0 }
    .tool-group { display: flex; gap: 2px; flex-shrink: 0 }
    .tool-btn {
      background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c);
      border-radius: 4px; cursor: pointer; padding: 4px 7px; font-size: 14px; line-height: 1;
      color: var(--fg, #d4d4d8);
    }
    .tool-btn:hover { background: var(--bg-hover, #2e2e3c); border-color: var(--border-hi, #5a5a7a) }
    .tool-btn.active { background: var(--accent, #4d87c4); color: #fff; border-color: var(--accent, #4d87c4) }
    .dest-switch { display: flex; gap: 2px; flex-shrink: 0 }
    .dest-btn {
      background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c);
      border-radius: 4px; cursor: pointer; padding: 3px 8px; font-size: 11px; font-weight: 500;
      color: var(--fg-dim, #7a7a8a);
    }
    .dest-btn:hover { background: var(--bg-hover, #2e2e3c); color: var(--fg, #d4d4d8) }
    .dest-btn.active { background: var(--accent, #4d87c4); color: #fff; border-color: var(--accent, #4d87c4) }
    .channel-switch { display: flex; gap: 2px; flex-shrink: 0 }
    .channel-btn {
      background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c);
      border-radius: 4px; cursor: pointer; padding: 3px 7px; font-size: 11px; font-weight: 500;
      color: var(--fg-dim, #7a7a8a);
    }
    .channel-btn:hover { background: var(--bg-hover, #2e2e3c); color: var(--fg, #d4d4d8) }
    .channel-btn.active { color: #fff; border-color: var(--channel-color, var(--accent, #4d87c4)); background: var(--channel-color, var(--accent, #4d87c4)) }
    .preset-row { display: flex; align-items: center; gap: 3px; flex-shrink: 0; font-size: 11px; }
    .preset-row label { color: var(--fg-dim, #7a7a8a); white-space: nowrap }
    select.preset-sel {
      padding: 2px 4px; font-size: 11px;
      background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c);
      border-radius: 3px; color: var(--fg, #d4d4d8); cursor: pointer;
    }
    rtd-number { flex-shrink: 0; }
    .mask-channel { display:flex; align-items:center; gap:4px; flex-shrink:0; font-size:11px }
    .mask-channel label { color: var(--fg-dim, #7a7a8a); white-space: nowrap }
    .mask-channel select {
      padding: 2px 4px; font-size: 11px;
      background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c);
      border-radius: 3px; color: var(--fg, #d4d4d8); cursor: pointer;
    }
    .color-stack { position: relative; width: 34px; height: 34px; flex-shrink: 0 }
    .color-chip {
      position: absolute; width: 22px; height: 22px; border-radius: 4px;
      border: 2px solid var(--bg-panel, #24242c); cursor: pointer;
      box-shadow: 0 1px 4px rgba(0,0,0,.35);
    }
    .color-chip.primary { top: 0; left: 0 }
    .color-chip.secondary { bottom: 0; right: 0 }
    .color-chip.active { outline: 2px solid var(--fg-accent, #7aadff); outline-offset: 1px }
    .swap-btn {
      position: absolute; bottom: 0; left: 0; width: 12px; height: 12px;
      background: var(--bg-panel, #24242c); border: 1px solid var(--border, #3a3a4c);
      border-radius: 2px; cursor: pointer; font-size: 8px; padding: 0;
      display: flex; align-items: center; justify-content: center; color: var(--fg, #d4d4d8);
    }
    .cp-popup {
      position: fixed; z-index: 9999;
      border-radius: 4px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      overflow: hidden;
    }
    .cp-backdrop { position: fixed; inset: 0; z-index: 9998; }
  `

  render() {
    const { toolMode, editorDest, activeMaskChannel, brushColor, secondaryBrushColor, activeColorSlot, brushSize, brushHardness, maskBrushValue } = this._canvas.value
    const { stageWidth, stageHeight } = this._scene.value
    const presetKey = `${stageWidth}x${stageHeight}`
    const isCustom = !SIZE_PRESETS.some(p => `${p.width}x${p.height}` === presetKey)

    return html`
      <div class="toolbar-line" role="toolbar" aria-label="Drawing tools">
        <div class="tool-group">
          ${(Object.entries(TOOL_DEFS) as [ToolMode, typeof TOOL_DEFS[ToolMode]][]).map(([mode, def]) => html`
            <button
              class=${`tool-btn${toolMode === mode ? ' active' : ''}`}
              title="${def.label} (${def.shortcut})"
              aria-pressed=${toolMode === mode}
              @click=${() => $canvas.setKey('toolMode', mode)}>
              ${def.icon}
            </button>`)}
        </div>

        <div class="sep"></div>

        <div class="dest-switch" role="group" aria-label="Editor destination">
          <button class=${`dest-btn${editorDest === 'paint' ? ' active' : ''}`}
            @click=${() => $canvas.setKey('editorDest', 'paint' as EditorDest)}>Paint</button>
          <button class=${`dest-btn${editorDest === 'mask' ? ' active' : ''}`}
            @click=${() => {
              $canvas.setKey('editorDest', 'mask' as EditorDest)
              $canvas.setKey('activeMaskChannel', 'color' as MaskChannel)
            }}>Masks</button>
        </div>

        <div class="channel-switch" role="group" aria-label="Mask channel">
          ${this._channelButton('color', 'Color', '#4d87c4', activeMaskChannel)}
          ${this._channelButton('cfg', 'CFG', '#7aadff', activeMaskChannel)}
          ${this._channelButton('denoise', 'Denoise', '#ffb45f', activeMaskChannel)}
        </div>

        <rtd-number style="width:92px" label="Mask" .value=${maskBrushValue} min="0" max="255" step="1" decimals="0"
          @rtd-change=${(e: CustomEvent) => $canvas.setKey('maskBrushValue', Math.max(0, Math.min(255, Math.round(e.detail.value))))}></rtd-number>

        <div class="sep"></div>

        <div class="preset-row">
          <label>Preset</label>
          <select class="preset-sel"
            .value=${isCustom ? 'custom' : presetKey}
            @change=${(e: Event) => {
              const v = (e.target as HTMLSelectElement).value
              const p = SIZE_PRESETS.find(x => `${x.width}x${x.height}` === v)
              if (p) { $scene.setKey('stageWidth', p.width); $scene.setKey('stageHeight', p.height) }
            }}>
            <option value="custom" ?selected=${isCustom}>Custom</option>
            ${SIZE_PRESETS.map(p => html`
              <option value=${`${p.width}x${p.height}`} ?selected=${presetKey === `${p.width}x${p.height}`}>${p.label}</option>`)}
          </select>
        </div>

        <rtd-number style="width:90px" label="W" .value=${stageWidth} min="64" max="4096" step="64" decimals="0"
          @rtd-change=${(e: CustomEvent) => $scene.setKey('stageWidth', Math.max(64, Math.round(e.detail.value / 64) * 64))}></rtd-number>
        <rtd-number style="width:90px" label="H" .value=${stageHeight} min="64" max="4096" step="64" decimals="0"
          @rtd-change=${(e: CustomEvent) => $scene.setKey('stageHeight', Math.max(64, Math.round(e.detail.value / 64) * 64))}></rtd-number>

        <div class="sep"></div>

        <div class="color-stack" aria-label="Color slots">
          <button class=${`color-chip secondary${activeColorSlot === 'secondary' ? ' active' : ''}`}
            style="background:${secondaryBrushColor}"
            title="Secondary color — click to activate, double-click to open picker"
            @click=${() => $canvas.setKey('activeColorSlot', 'secondary' as ColorSlot)}
            @dblclick=${(e: MouseEvent) => this._openPicker(e)}></button>
          <button class=${`color-chip primary${activeColorSlot === 'primary' ? ' active' : ''}`}
            style="background:${brushColor}"
            title="Primary color — click to activate, double-click to open picker"
            @click=${(e: MouseEvent) => { $canvas.setKey('activeColorSlot', 'primary' as ColorSlot); this._openPicker(e) }}></button>
          <button class="swap-btn" title="Swap colors" @click=${() => {
            const c = this._canvas.value
            $canvas.setKey('brushColor', c.secondaryBrushColor)
            $canvas.setKey('secondaryBrushColor', c.brushColor)
          }}>⇄</button>
        </div>
      </div>

      <div class="toolbar-line tool-properties" aria-label="Tool properties">
        ${this._renderToolProperties(toolMode, brushSize, brushHardness)}
      </div>

      ${this._cpOpen ? html`
        <div class="cp-backdrop" @click=${() => { this._cpOpen = false }}></div>
        <div class="cp-popup" style="left:${this._cpX}px;top:${this._cpY}px;--border:#3a3a4c;--surface:#1c1c24;--accent:#4d87c4;--label:#9898a8;background:#1c1c24">
          <rtd-color-picker></rtd-color-picker>
        </div>` : ''}
    `
  }

  private _renderToolProperties(toolMode: ToolMode, brushSize: number, brushHardness: number) {
    const isPaintStroke = toolMode === 'brush' || toolMode === 'eraser'
    const isShape = toolMode === 'line' || toolMode === 'rect' || toolMode === 'ellipse'
    const isSelection = toolMode === 'region' || toolMode === 'lassoSelect' || toolMode === 'ellipseSelect' || toolMode === 'magicWand' || toolMode === 'moveSelection'
    return html`
      ${isSelection ? this._renderSelectionProperties(toolMode) : ''}
      ${toolMode === 'fill' ? html`<div class="context-properties">${this._renderFillProperties()}</div>` : ''}
      ${isShape ? html`<div class="context-properties">${this._renderShapeProperties(toolMode)}</div>` : ''}
      ${toolMode === 'eyedropper' ? html`<div class="context-properties"><div class="property-group"><label>Sample</label><span>composite</span></div></div>` : ''}
      ${toolMode === 'text' ? this._renderTextProperties() : ''}
      ${(isPaintStroke || isShape) ? html`
        <rtd-number style="width:110px" label=${isShape ? 'Stroke' : 'Brush'} .value=${brushSize} min="1" max="4096" step="1" decimals="0"
          @rtd-change=${(e: CustomEvent) => $canvas.setKey('brushSize', Math.max(1, Math.round(e.detail.value)))}></rtd-number>
      ` : ''}
      ${isPaintStroke ? html`
        <rtd-number style="width:100px" label="Hard" .value=${brushHardness} min="0" max="100" step="1" decimals="0"
          @rtd-change=${(e: CustomEvent) => $canvas.setKey('brushHardness', Math.max(0, Math.min(100, Math.round(e.detail.value))))}></rtd-number>
      ` : ''}
    `
  }

  private _renderSelectionProperties(toolMode: ToolMode) {
    const { selectionRect, selectedLayerId } = this._layer.value
    const label = toolMode === 'moveSelection' ? 'Move outline' : toolMode === 'magicWand' ? 'Similar color' : 'New selection'
    return html`<div class="context-properties">
      <div class="property-group"><label>Selection</label><span>${label}</span></div>
      ${toolMode === 'magicWand' ? html`<rtd-number style="width:110px" label="Tolerance" .value=${this._canvas.value.fillTolerance} min="0" max="255" step="1" decimals="0"
        @rtd-change=${(e: CustomEvent) => $canvas.setKey('fillTolerance', Math.max(0, Math.min(255, Math.round(e.detail.value))))}></rtd-number>` : ''}
      ${selectionRect ? html`<button class="tool-btn" title="Create mask from selection" @click=${() => this.dispatchEvent(new CustomEvent('rtd-region-add', { detail: { target: selectedLayerId ? 'layer' : 'scene', layerId: selectedLayerId || undefined }, bubbles: true, composed: true }))}>Create mask</button>` : ''}
    </div>`
  }

  private _renderFillProperties() {
    const { fillEdgeStrategy, fillTolerance } = this._canvas.value
    return html`<div class="property-group">
      <label>Fill edge</label>
      <select .value=${fillEdgeStrategy} @change=${(event: Event) => $canvas.setKey('fillEdgeStrategy', (event.target as HTMLSelectElement).value as FillEdgeStrategy)}>
        <option value="none">Exact</option>
        <option value="fringe">Fringe grow</option>
        <option value="transparent_blend">Transparent blend</option>
      </select>
    </div>
    <rtd-number style="width:110px" label="Tolerance" .value=${fillTolerance} min="0" max="255" step="1" decimals="0"
      @rtd-change=${(e: CustomEvent) => $canvas.setKey('fillTolerance', Math.max(0, Math.min(255, Math.round(e.detail.value))))}></rtd-number>`
  }

  private _renderShapeProperties(toolMode: ToolMode) {
    const { shapeDrawMode } = this._canvas.value
    return html`<div class="property-group">
      <label>${toolMode === 'line' ? 'Line' : 'Shape'}</label>
      <select .value=${shapeDrawMode} ?disabled=${toolMode === 'line'} @change=${(event: Event) => $canvas.setKey('shapeDrawMode', (event.target as HTMLSelectElement).value as ShapeDrawMode)}>
        <option value="fill">Fill</option>
        <option value="stroke">Stroke</option>
        <option value="both">Both</option>
      </select>
    </div>`
  }

  private _renderTextProperties() {
    const { textFontSize } = this._canvas.value
    return html`<div class="context-properties">
      <rtd-number style="width:110px" label="Size" .value=${textFontSize} min="6" max="512" step="1" decimals="0"
        @rtd-change=${(e: CustomEvent) => $canvas.setKey('textFontSize', Math.max(6, Math.min(512, Math.round(e.detail.value))))}></rtd-number>
    </div>`
  }

  private _channelButton(channel: MaskChannel, label: string, color: string, active: MaskChannel) {
    return html`<button class=${`channel-btn${active === channel ? ' active' : ''}`}
      style=${`--channel-color: ${color}`}
      aria-pressed=${active === channel}
      @click=${() => {
        $canvas.setKey('editorDest', 'mask' as EditorDest)
        $canvas.setKey('activeMaskChannel', channel)
      }}>${label}</button>`
  }

  private _openPicker(e: MouseEvent) {
    e.stopPropagation()
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    this._cpX = Math.min(r.left, window.innerWidth - 300)
    this._cpY = r.bottom + 4
    this._cpOpen = true
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-toolbar': RtdToolbar }
}
