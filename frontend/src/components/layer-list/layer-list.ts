import { LitElement, css, html, type TemplateResult } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { StoreController } from '@nanostores/lit'
import { $layer } from '../../stores/layer.store'
import type { LayerItem, RegionItem, RegionSchedule } from '../../types'

function clamp(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback
  return Math.max(min, Math.min(max, value))
}

function hslToHex(hue: number, saturation: number, lightness: number): string {
  const normalizedHue = ((hue % 360) + 360) % 360
  const sat = clamp(saturation, 0, 100, 0) / 100
  const light = clamp(lightness, 0, 100, 50) / 100
  const chroma = (1 - Math.abs(2 * light - 1)) * sat
  const x = chroma * (1 - Math.abs(((normalizedHue / 60) % 2) - 1))
  const match = light - chroma / 2
  const [red, green, blue] =
    normalizedHue < 60 ? [chroma, x, 0]
    : normalizedHue < 120 ? [x, chroma, 0]
    : normalizedHue < 180 ? [0, chroma, x]
    : normalizedHue < 240 ? [0, x, chroma]
    : normalizedHue < 300 ? [x, 0, chroma]
    : [chroma, 0, x]
  const toHex = (v: number) => Math.round(clamp((v + match) * 255, 0, 255, 0)).toString(16).padStart(2, '0')
  return `#${toHex(red)}${toHex(green)}${toHex(blue)}`
}

@customElement('rtd-layer-list')
export class RtdLayerList extends LitElement {
  private _layer = new StoreController(this, $layer)

  @state() private schedulerOpen = true

  static styles = css`
    :host { display: flex; flex-direction: column; height: 100%; overflow: hidden; font-size: 12px; color: var(--fg, #d4d4d8); background: var(--bg-panel, #24242c); }
    .head { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; border-bottom: 1px solid var(--border, #3a3a4c); font-weight: 600; flex-shrink: 0; }
    .head-actions { display: flex; gap: 4px }
    .icon-btn { background: none; border: none; cursor: pointer; font-size: 15px; padding: 2px 5px; border-radius: 3px; line-height: 1; color: var(--fg, #d4d4d8); }
    .icon-btn:hover { background: var(--bg-hover, #2e2e3c) }
    .layer-scheduler { border-bottom: 1px solid var(--border, #3a3a4c); background: var(--bg-header, #1e1e28); flex-shrink: 0; }
    .layer-scheduler summary { display: block; padding: 7px 10px; cursor: pointer; color: var(--fg-dim, #7a7a8a); font-size: 11px; user-select: none; }
    .layer-scheduler summary strong { color: var(--fg, #d4d4d8); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .schedule-timeline { display: grid; gap: 5px; padding: 0 10px 8px; }
    .timeline-axis { display: grid; grid-template-columns: minmax(72px, 0.36fr) minmax(0, 1fr); gap: 8px; align-items: center; min-width: 0; color: var(--fg-dim, #7a7a8a); font-size: 10px; }
    .timeline-axis-line { position: relative; height: 12px; margin-right: 48px; }
    .timeline-axis-line span { position: absolute; top: 0; transform: translateX(-50%); }
    .timeline-axis-line span:first-child { left: 0; transform: none; }
    .timeline-axis-line span:nth-child(2) { left: 50%; }
    .timeline-axis-line span:last-child { right: 0; transform: none; }
    .timeline-row { display: grid; grid-template-columns: minmax(72px, 0.36fr) minmax(0, 1fr); gap: 8px; align-items: center; min-width: 0; }
    .timeline-row > span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; color: var(--fg-label, #9898a8); }
    .timeline-row.muted { opacity: 0.38; }
    .timeline-track { position: relative; height: 26px; overflow: hidden; border: 1px solid var(--border, #3a3a4c); background: linear-gradient(90deg, transparent calc(50% - 1px), var(--border-hi, #5a5a7a) calc(50% - 1px), var(--border-hi, #5a5a7a) calc(50% + 1px), transparent calc(50% + 1px)), var(--bg-input, #14141a); }
    .timeline-lane { position: absolute; inset: 0 48px 0 0; cursor: grab; }
    .timeline-lane:active { cursor: grabbing; }
    .timeline-lane i { position: absolute; top: 6px; bottom: 6px; border-radius: 2px; pointer-events: none; box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.4); }
    .timeline-handle { position: absolute; top: 3px; bottom: 3px; width: 12px; margin-left: -6px; border: 1px solid rgba(255,255,255,.72); border-radius: 2px; background: rgba(20,20,26,.55); cursor: ew-resize; z-index: 2; }
    .timeline-handle:hover { background: rgba(255,255,255,.18); border-color: #fff; }
    .timeline-mode { position: absolute; right: 2px; top: 2px; bottom: 2px; width: 42px; min-width: 42px; padding: 0 2px; font-size: 10px; z-index: 3; }
    .timeline-empty { min-height: 24px; display: grid; place-items: center; color: var(--fg-dim, #7a7a8a); font-size: 11px; }
    .list { flex: 1; overflow-y: auto }
    .item { display: flex; align-items: center; gap: 8px; padding: 5px 10px; cursor: pointer; border-bottom: 1px solid var(--border, #3a3a4c); user-select: none; color: var(--fg, #d4d4d8); }
    .item.selected { background: var(--bg-sel, #1a3060); color: var(--fg-accent, #7aadff); }
    .item:hover:not(.selected) { background: var(--bg-hover, #2e2e3c); }
    .order-badge { width: 48px; flex-shrink: 0; display: grid; gap: 1px; justify-items: center; align-content: center; min-height: 30px; border: 1px solid var(--border, #3a3a4c); border-radius: 3px; background: #151820; color: #b9c3d2; font-size: 10px; line-height: 1; font-variant-numeric: tabular-nums; }
    .order-badge strong { font-size: 12px; color: #f0f5ff; }
    .order-badge em { font-style: normal; font-size: 8px; color: #8e9aad; letter-spacing: 0; }
    .order-badge.bottom { border-color: #47b978; background: #102018; }
    .order-badge.bottom em { color: #7ee0a2; }
    .order-badge.top { border-color: #d09b3f; background: #211a10; }
    .order-badge.top em { color: #f2c067; }
    .thumb { width: 32px; height: 32px; border-radius: 3px; object-fit: cover; background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c); flex-shrink: 0; }
    .main { flex: 1; min-width: 0; display: grid; gap: 3px }
    .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap }
    .mask-chips { display: flex; gap: 3px; min-height: 8px; align-items: center }
    .mask-chip { width: 14px; height: 8px; border-radius: 2px; border: 1px solid rgba(255,255,255,.35); background: var(--mask-color); padding: 0; cursor: pointer }
    .mask-chip:hover { border-color: #fff }
    .toggle-btn { background: none; border: none; cursor: pointer; font-size: 13px; padding: 2px; opacity: 0.6; line-height: 1; color: var(--fg, #d4d4d8); }
    .toggle-btn:hover { opacity: 1 }
    .empty { padding: 16px; text-align: center; color: var(--fg-dim, #7a7a8a); font-size: 11px }
  `

  render() {
    const { layers, selectedLayerId } = this._layer.value
    return html`
      ${this._renderScheduler(layers)}
      <div class="head">
        <span>Layers</span>
        <div class="head-actions">
          <button class="icon-btn" title="Add layer"
            @click=${() => this.dispatchEvent(new CustomEvent('layer-add', { bubbles: true, composed: true }))}>➕</button>
          <button class="icon-btn" title="Delete selected"
            @click=${() => this.dispatchEvent(new CustomEvent('layer-delete', { detail: { id: selectedLayerId }, bubbles: true, composed: true }))}>🗑</button>
          <button class="icon-btn" title="Move layer up"
            @click=${() => this.dispatchEvent(new CustomEvent('layer-move-up', { bubbles: true, composed: true }))}>⬆️</button>
          <button class="icon-btn" title="Move layer down"
            @click=${() => this.dispatchEvent(new CustomEvent('layer-move-down', { bubbles: true, composed: true }))}>⬇️</button>
        </div>
      </div>
      <div class="list">
        ${layers.length
          ? layers.map((layer, index) => this._renderItem(layer, selectedLayerId, index, layers.length))
          : html`<div class="empty">No layers yet</div>`}
      </div>`
  }

  private _scheduleOptions() {
    return [
      { value: 'linear', label: 'Linear' },
      { value: 'ease_in', label: 'Ease in' },
      { value: 'ease_out', label: 'Ease out' },
      { value: 'ease_in_out', label: 'Ease in-out' },
    ] as { value: RegionSchedule; label: string }[]
  }

  private _emitLayerPreset(id: string, patch: { scheduleStart?: number; scheduleEnd?: number; schedule?: RegionSchedule }) {
    this.dispatchEvent(new CustomEvent('rtd-layer-preset', { detail: { id, patch }, bubbles: true, composed: true }))
  }

  private _renderScheduler(layers: LayerItem[]): TemplateResult {
    return html`<details class="layer-scheduler" .open=${this.schedulerOpen} @toggle=${(event: Event) => { this.schedulerOpen = (event.currentTarget as HTMLDetailsElement).open }}>
      <summary><strong>Layer scheduler</strong></summary>
      <div class="schedule-timeline">
        <div class="timeline-axis"><span></span><div class="timeline-axis-line"><span>0</span><span>0.5</span><span>1</span></div></div>
        ${layers.length ? layers.map((layer, index) => this._renderScheduleRow(layer, index)) : html`<div class="timeline-empty">No layers</div>`}
      </div>
    </details>`
  }

  private _renderScheduleRow(layer: LayerItem, index: number): TemplateResult {
    const start = clamp(layer.preset.scheduleStart, 0, 1, 0)
    const end = clamp(layer.preset.scheduleEnd, start, 1, 1)
    const left = Math.round(start * 100)
    const width = Math.max(2, Math.round((end - start) * 100))
    const right = Math.round(end * 100)
    const schedule = layer.preset.schedule === 'auto' ? 'linear' : layer.preset.schedule
    const color = hslToHex((index * 47 + 205) % 360, 78, 54)
    return html`<div class=${layer.visible && layer.preset.samplingEnabled ? 'timeline-row' : 'timeline-row muted'}>
      <span title=${layer.name}>${layer.name}</span>
      <div class="timeline-track">
        <div class="timeline-lane" @pointerdown=${(event: PointerEvent) => this._startTimelineDrag(event, 'move', start, end, (nextStart, nextEnd) => this._emitLayerPreset(layer.id, { scheduleStart: nextStart, scheduleEnd: nextEnd }))}>
          <i style=${`left: ${left}%; width: ${width}%; background: ${color};`}></i>
          <span class="timeline-handle" style=${`left: ${left}%;`} title="Resize start"
            @pointerdown=${(event: PointerEvent) => this._startTimelineDrag(event, 'start', start, end, (nextStart, nextEnd) => this._emitLayerPreset(layer.id, { scheduleStart: nextStart, scheduleEnd: nextEnd }))}></span>
          <span class="timeline-handle" style=${`left: ${right}%;`} title="Resize end"
            @pointerdown=${(event: PointerEvent) => this._startTimelineDrag(event, 'end', start, end, (nextStart, nextEnd) => this._emitLayerPreset(layer.id, { scheduleStart: nextStart, scheduleEnd: nextEnd }))}></span>
        </div>
        <select class="timeline-mode" title="Interpolation" .value=${schedule}
          @pointerdown=${(event: Event) => event.stopPropagation()}
          @click=${(event: Event) => event.stopPropagation()}
          @change=${(event: Event) => this._emitLayerPreset(layer.id, { schedule: (event.target as HTMLSelectElement).value as RegionSchedule })}>
          ${this._scheduleOptions().map((option) => html`<option value=${option.value}>${option.label}</option>`)}
        </select>
      </div>
    </div>`
  }

  private _startTimelineDrag(event: PointerEvent, mode: 'move' | 'start' | 'end', start: number, end: number, onInterval: (start: number, end: number) => void) {
    if ((event.target as HTMLElement).closest('select')) return
    const lane = (event.currentTarget as HTMLElement).closest('.timeline-lane') as HTMLElement | null
    if (!lane) return
    const bounds = lane.getBoundingClientRect()
    const usableWidth = Math.max(1, bounds.width)
    const valueFromEvent = (next: PointerEvent) => clamp((next.clientX - bounds.left) / usableWidth, 0, 1, start)
    const initialValue = valueFromEvent(event)
    const duration = clamp(end - start, 0, 1, 0)
    const initialOffset = clamp(initialValue - start, 0, duration, duration / 2)
    const apply = (next: PointerEvent) => {
      const value = valueFromEvent(next)
      if (mode === 'start') onInterval(Math.min(value, end), end)
      else if (mode === 'end') onInterval(start, Math.max(value, start))
      else {
        const nextStart = duration >= 1 ? 0 : clamp(value - initialOffset, 0, 1 - duration, 0)
        onInterval(nextStart, nextStart + duration)
      }
    }
    const stop = () => {
      window.removeEventListener('pointermove', apply)
      window.removeEventListener('pointerup', stop)
    }
    event.preventDefault()
    event.stopPropagation()
    apply(event)
    window.addEventListener('pointermove', apply)
    window.addEventListener('pointerup', stop, { once: true })
  }

  private _renderItem(layer: LayerItem, selectedId: string, index: number, total: number) {
    const masks = this._layer.value.regions.filter((region) => region.target === 'layer' && region.layerId === layer.id)
    const pass = total - index
    const stack = index === 0 ? 'top' : index === total - 1 ? 'bottom' : ''
    const stackLabel = stack === 'top' ? 'TOP' : stack === 'bottom' ? 'BOTTOM' : 'STACK'
    return html`<div class=${`item${layer.id === selectedId ? ' selected' : ''}`}
      @click=${() => $layer.setKey('selectedLayerId', layer.id)}>
      <span class=${`order-badge ${stack}`} title=${`${stackLabel} layer; server render pass ${pass}/${total}; z_index ${pass - 1}`}>
        <strong>${stack === 'top' || stack === 'bottom' ? stackLabel : `#${pass}`}</strong>
        <em>PASS ${pass}</em>
      </span>
      ${layer.preview
        ? html`<img class="thumb" src=${layer.preview} alt=${layer.name} />`
        : html`<div class="thumb"></div>`}
      <div class="main">
        <span class="name" title=${layer.name}>${layer.name}</span>
        ${masks.length ? html`<div class="mask-chips" title=${`${masks.length} mask${masks.length === 1 ? '' : 's'}`}>
          ${masks.slice(0, 8).map((mask) => this._renderMaskChip(layer, mask))}
        </div>` : ''}
      </div>
      <button class="toggle-btn" title=${layer.visible ? 'Hide layer' : 'Show layer'}
        @click=${(e: Event) => {
          e.stopPropagation()
          this.dispatchEvent(new CustomEvent('layer-toggle', {
            detail: { id: layer.id, visible: !layer.visible },
            bubbles: true, composed: true,
          }))
        }}>
        ${layer.visible ? '👁' : '🙈'}
      </button>
    </div>`
  }

  private _renderMaskChip(layer: LayerItem, mask: RegionItem) {
    return html`<button class="mask-chip" style=${`--mask-color: ${mask.color}`} title=${mask.name || mask.color}
      @click=${(event: Event) => {
        event.stopPropagation()
        $layer.setKey('selectedLayerId', layer.id)
        $layer.setKey('layerPanelTab', 'mask')
        this.dispatchEvent(new CustomEvent('rtd-mask-select', {
          detail: { layerId: layer.id, regionId: mask.id, channel: 'color', color: mask.color },
          bubbles: true,
          composed: true,
        }))
      }}></button>`
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-layer-list': RtdLayerList }
}
