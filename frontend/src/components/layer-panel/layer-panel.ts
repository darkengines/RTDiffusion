import { LitElement, css, html, type TemplateResult } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { StoreController } from '@nanostores/lit'
import { $layer } from '../../stores/layer.store'
import { $scene } from '../../stores/scene.store'
import { $assets } from '../../stores/assets.store'
import type {
  LayerItem,
  MaskChannel,
  RegionItem,
  RegionSchedule,
  RegionTarget,
} from '../../types'
import '../rtd-number/rtd-number'
import '../rtd-combo/rtd-combo'

// ── Helpers ────────────────────────────────────────────────────────────────────

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

function resolvedRegionTiming(
  region: RegionItem,
  fallback?: { start: number; end: number; schedule: RegionSchedule; active: boolean },
) {
  if (region.schedule !== 'auto') return { start: region.start, end: region.end, schedule: region.schedule }
  return { start: fallback?.start ?? 0, end: fallback?.end ?? 1, schedule: fallback?.schedule ?? 'linear' as RegionSchedule, active: fallback?.active ?? true }
}

// ── Component ──────────────────────────────────────────────────────────────────

@customElement('rtd-layer-panel')
export class RtdLayerPanel extends LitElement {
  private _layer = new StoreController(this, $layer)
  private _scene = new StoreController(this, $scene)
  private _assets = new StoreController(this, $assets)

  @state() private fitAlphaValue = 1
  @state() private fillViewportValue = 1
  @state() private maskSchedulerOpen = true

  // ── Utilities ────────────────────────────────────────────────────────────────

  private emit<T>(name: string, detail: T) {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }))
  }

  private autoLayerSchedules(layers: LayerItem[]) {
    const schedules = new Map<string, { start: number; end: number; schedule: RegionSchedule; active: boolean }>()
    if (!layers.length) return schedules
    for (const layer of layers) {
      const schedule = layer.preset.schedule
      schedules.set(layer.id, {
        start: layer.preset.scheduleStart,
        end: layer.preset.scheduleEnd,
        schedule,
        active: true,
      })
    }
    return schedules
  }

  private layerRegionSpecs(layer: LayerItem): RegionItem[] {
    const { regions } = this._layer.value
    const explicit = regions.filter((r) => r.target === 'layer' && r.layerId === layer.id)
    return explicit
  }

  // ── Render helpers ────────────────────────────────────────────────────────────

  private numCtrl(
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    decimals: number,
    onChange: (value: number) => void,
    disabled = false,
  ): TemplateResult {
    return html`<rtd-number
      label=${label}
      .value=${value}
      min=${min}
      max=${max}
      step=${step}
      decimals=${decimals}
      ?disabled=${disabled}
      @rtd-change=${(e: CustomEvent<{ value: number }>) => onChange(e.detail.value)}
    ></rtd-number>`
  }

  private renderRegionForceRow(
    label: 'CFG' | 'Denoise',
    value: number,
    inherited: boolean,
    onValue: (value: number) => void,
    onInherited: (value: boolean) => void,
  ): TemplateResult {
    const isCfg = label === 'CFG'
    return html`<div class="mask-force-row">
      ${this.numCtrl(label, value, isCfg ? 0 : 0, isCfg ? 30 : 0.999, isCfg ? 0.1 : 0.01, isCfg ? 1 : 2, onValue, inherited)}
      <label class="check-field inherit-check">
        <input type="checkbox" .checked=${inherited} @change=${(e: Event) => onInherited((e.target as HTMLInputElement).checked)} />
        <span>inherit</span>
      </label>
    </div>`
  }

  private renderMaskActionControls(layerId: string, channel: Extract<MaskChannel, 'color' | 'cfg' | 'denoise'>, regionId?: string): TemplateResult {
    const fitValue = clamp(this.fitAlphaValue, 0, 1, 1)
    const fillValue = clamp(this.fillViewportValue, 0, 1, 1)
    return html`<div class="button-grid mask-action-grid">
      <button @click=${() => this.emit('rtd-mask-select', { layerId, regionId: regionId ?? `special:${channel}:${layerId}`, channel })}>Paint</button>
      <div class="mask-action">
        <button @click=${() => this.emit('rtd-layer-channel-fit-alpha', { id: layerId, channel, regionId, value: fitValue })}>Fit to RGB alpha</button>
        ${this.numCtrl('', fitValue, 0, 1, 0.01, 2, (v) => { this.fitAlphaValue = clamp(v, 0, 1, this.fitAlphaValue) })}
      </div>
      <div class="mask-action">
        <button @click=${() => this.emit('rtd-mask-fill-viewport', { layerId, channel, regionId, value: fillValue })}>Fill viewport</button>
        ${this.numCtrl('', fillValue, 0, 1, 0.01, 2, (v) => { this.fillViewportValue = clamp(v, 0, 1, this.fillViewportValue) })}
      </div>
      <button @click=${() => this.emit('rtd-mask-clear', { layerId, channel, regionId })}>Clear</button>
    </div>`
  }

  private scheduleOptions() {
    return [
      { value: 'linear', label: 'Linear' },
      { value: 'ease_in', label: 'Ease in' },
      { value: 'ease_out', label: 'Ease out' },
      { value: 'ease_in_out', label: 'Ease in-out' },
    ] as { value: RegionSchedule; label: string }[]
  }

  deviceSelect(label: string, value: string, onChange: (v: string) => void): TemplateResult {
    const devices = this._assets.value.gpuDevices
    return html`<label class="field inline">
      <span>${label}</span>
      <select .value=${value} @change=${(e: Event) => onChange((e.target as HTMLSelectElement).value)}>
        ${devices.map((d) => html`<option value=${d.id}>${d.name}${d.memory_total ? ` (${formatBytes(d.memory_total)})` : ''}</option>`)}
      </select>
    </label>`
  }

  // ── Tabs ──────────────────────────────────────────────────────────────────────

  private renderTabs(layer: LayerItem | undefined): TemplateResult {
    const { layerPanelTab } = this._layer.value
    return html`<div class="context-tabs">
      <button class=${layerPanelTab === 'mask' ? 'active' : ''} @click=${() => $layer.setKey('layerPanelTab', 'mask')}>Masks</button>
      ${layer?.isVideo ? html`<button class=${layerPanelTab === 'video' ? 'active' : ''} @click=${() => $layer.setKey('layerPanelTab', 'video')}>Video</button>` : ''}
    </div>`
  }

  // ── Schedulers ───────────────────────────────────────────────────────────────

  private renderMaskSchedulerBox(layer: LayerItem): TemplateResult {
    const layerTiming = this.autoLayerSchedules([layer]).get(layer.id) ?? { start: 0, end: 1, schedule: 'linear' as RegionSchedule, active: true }
    const rows = this.layerRegionSpecs(layer).map((region, index) => ({
      id: region.id,
      label: region.name || `Mask ${index + 1}`,
      color: region.color || hslToHex((index * 41 + 212) % 360, 70, 54),
      timing: resolvedRegionTiming(region, layerTiming),
      onInterval: (start: number, end: number) => this.emit('rtd-region-update', { id: region.id, patch: { start, end, schedule: region.schedule === 'auto' ? 'linear' as RegionSchedule : region.schedule, inherited: false } }),
      onSchedule: (schedule: RegionSchedule) => this.emit('rtd-region-update', { id: region.id, patch: { schedule, inherited: false } }),
    }))
    return this.renderSchedulerBox(
      'Mask scheduler',
      this.maskSchedulerOpen,
      (open) => { this.maskSchedulerOpen = open },
      rows,
      'No scheduled masks',
    )
  }

  private renderSchedulerBox(
    title: string,
    open: boolean,
    setOpen: (open: boolean) => void,
    rows: {
      id: string
      label: string
      color: string
      timing: { start: number; end: number; schedule: RegionSchedule; active?: boolean }
      onInterval: (start: number, end: number) => void
      onSchedule: (schedule: RegionSchedule) => void
    }[],
    emptyText: string,
  ): TemplateResult {
    return html`<details class="schedule-box" .open=${open} @toggle=${(event: Event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}>
      <summary><strong>${title}</strong></summary>
      <div class="schedule-timeline">
        <div class="timeline-axis"><span></span><div class="timeline-axis-line"><span>0</span><span>0.5</span><span>1</span></div></div>
        ${rows.length ? rows.map((row) => this.renderSchedulerRow(row)) : html`<div class="timeline-empty">${emptyText}</div>`}
      </div>
    </details>`
  }

  private renderSchedulerRow(row: {
    id: string
    label: string
    color: string
    timing: { start: number; end: number; schedule: RegionSchedule; active?: boolean }
    onInterval: (start: number, end: number) => void
    onSchedule: (schedule: RegionSchedule) => void
  }): TemplateResult {
    const start = clamp(row.timing.start, 0, 1, 0)
    const end = clamp(row.timing.end, start, 1, 1)
    const left = Math.round(start * 100)
    const width = Math.max(2, Math.round((end - start) * 100))
    const right = Math.round(end * 100)
    return html`<div class=${row.timing.active === false ? 'timeline-row muted' : 'timeline-row'}>
      <span title=${row.label}>${row.label}</span>
      <div class="timeline-track editable">
        <div class="timeline-lane" @pointerdown=${(event: PointerEvent) => this.startTimelineDrag(event, 'move', start, end, row.onInterval)}>
          <i style=${`left: ${left}%; width: ${width}%; background: ${row.color};`}></i>
          <span class="timeline-handle" style=${`left: ${left}%;`} title="Resize start"
            @pointerdown=${(event: PointerEvent) => this.startTimelineDrag(event, 'start', start, end, row.onInterval)}></span>
          <span class="timeline-handle" style=${`left: ${right}%;`} title="Resize end"
            @pointerdown=${(event: PointerEvent) => this.startTimelineDrag(event, 'end', start, end, row.onInterval)}></span>
        </div>
        <select class="timeline-mode" title="Interpolation" .value=${row.timing.schedule === 'auto' ? 'linear' : row.timing.schedule}
          @pointerdown=${(event: Event) => event.stopPropagation()}
          @click=${(event: Event) => event.stopPropagation()}
          @change=${(event: Event) => row.onSchedule((event.target as HTMLSelectElement).value as RegionSchedule)}>
          ${this.scheduleOptions().map((option) => html`<option value=${option.value}>${option.label}</option>`)}
        </select>
      </div>
    </div>`
  }

  private startTimelineDrag(event: PointerEvent, mode: 'move' | 'start' | 'end', start: number, end: number, onInterval: (start: number, end: number) => void) {
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

  // ── Layer transform ───────────────────────────────────────────────────────────

  renderLayerTransform(layer: LayerItem): TemplateResult {
    const { stageWidth, stageHeight } = this._scene.value
    const t = layer.transform
    if (!t) return html``
    return html`<section class="transform-panel">
      <div class="transform-head"><strong>Transform</strong><span>center anchor</span></div>
      <div class="transform-grid">
        ${this.numCtrl('Center X', t.centerX, 0, stageWidth, 1, 0, (v) => this.emit('rtd-layer-center', { id: layer.id, centerX: v, centerY: t.centerY }))}
        ${this.numCtrl('Center Y', t.centerY, 0, stageHeight, 1, 0, (v) => this.emit('rtd-layer-center', { id: layer.id, centerX: t.centerX, centerY: v }))}
        ${this.numCtrl('Width', t.width, 1, stageWidth * 4, 1, 0, (v) => this.emit('rtd-layer-size', { id: layer.id, width: v, height: t.height }))}
        ${this.numCtrl('Height', t.height, 1, stageHeight * 4, 1, 0, (v) => this.emit('rtd-layer-size', { id: layer.id, width: t.width, height: v }))}
        ${this.numCtrl('Scale', t.scale, 1, 400, 1, 0, (v) => this.emit('rtd-layer-scale', { id: layer.id, scale: v }))}
        ${this.numCtrl('Angle', t.angle, -180, 180, 1, 0, (v) => this.emit('rtd-layer-angle', { id: layer.id, angle: v }))}
      </div>
      <div class="button-grid">
        <button @click=${() => this.emit('rtd-layer-center-viewport', { id: layer.id })}>Center</button>
        <button @click=${() => this.emit('rtd-layer-fit-viewport', { id: layer.id })}>Fit viewport</button>
      </div>
    </section>`
  }

  // ── Sampling controls ─────────────────────────────────────────────────────────

  renderSamplingControls(config: {
    prompt: string
    negativePrompt: string
    modelPath?: string
    loraPaths?: string[]
    strength?: number
    cfg: number
    steps: number
    sampler: string
    scheduler: string
    onPrompt: (v: string) => void
    onNegativePrompt: (v: string) => void
    onModelPath?: (v: string) => void
    onToggleLora?: (v: string) => void
    onStrength?: (v: number) => void
    onCfg: (v: number) => void
    onSteps: (v: number) => void
    onSampler: (v: string) => void
    onScheduler: (v: string) => void
  }): TemplateResult {
    const { catalog } = this._assets.value
    const models = catalog.models
    const loras = catalog.loras

    const selectedModelLabel = models.find((m) => m.path === (config.modelPath ?? ''))?.name ?? 'Default'
    const selectedLoraNames = (config.loraPaths ?? [])
      .map((p) => loras.find((l) => l.path === p)?.name ?? p)
      .slice(0, 2)
    const loraCount = (config.loraPaths ?? []).length

    return html`<div class="sampling-controls">
      <label class="field">
        <span>Prompt</span>
        <textarea rows="4" .value=${config.prompt} @input=${(e: Event) => config.onPrompt((e.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field">
        <span>Negative prompt</span>
        <textarea rows="3" .value=${config.negativePrompt} @input=${(e: Event) => config.onNegativePrompt((e.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      ${config.onModelPath ? html`
        <div class="popup-picker field">
          <button class="picker-trigger" type="button" @click=${(e: MouseEvent) =>
            this._openModelPicker(e, config.modelPath ?? '', config.onModelPath!)}>
            <strong>${selectedModelLabel}</strong><span>Model</span>
          </button>
        </div>
      ` : ''}
      <div class="sampling-grid">
        ${config.onStrength ? this.numCtrl('Denoise', config.strength ?? this._scene.value.strength, 0, 0.999, 0.01, 2, config.onStrength) : ''}
        ${this.numCtrl('CFG', config.cfg, 0, 30, 0.1, 1, config.onCfg)}
        ${this.numCtrl('Steps', config.steps, 1, 40, 1, 0, config.onSteps)}
      </div>
      ${this._renderSamplerPicker(config.sampler, config.onSampler)}
      ${this._renderSchedulerPicker(config.scheduler, config.onScheduler)}
      ${config.onToggleLora ? html`
        <div class="popup-picker lora-picker field">
          <button class="picker-trigger" type="button" @click=${(e: MouseEvent) =>
            this._openLoraPicker(e, config.loraPaths ?? [], config.onToggleLora!)}>
            <strong>${loraCount ? selectedLoraNames.join(', ') : 'None'}</strong>
            <span>${loraCount > 2 ? `+${loraCount - 2} LoRAs` : 'LoRAs'}</span>
          </button>
        </div>
      ` : ''}
    </div>`
  }

  private _samplerOptions = [
    { value: 'euler', label: 'Euler' },
    { value: 'euler_ancestral', label: 'Euler Ancestral' },
    { value: 'dpm_2', label: 'DPM 2' },
    { value: 'dpm_2_ancestral', label: 'DPM 2 Ancestral' },
    { value: 'dpm_pp_2m', label: 'DPM++ 2M' },
    { value: 'dpm_pp_2m_sde', label: 'DPM++ 2M SDE' },
    { value: 'dpm_pp_3m_sde', label: 'DPM++ 3M SDE' },
    { value: 'ddim', label: 'DDIM' },
    { value: 'uni_pc', label: 'UniPC' },
    { value: 'lcm', label: 'LCM' },
    { value: 'tcd', label: 'TCD' },
  ]

  private _schedulerOptions = [
    { value: 'simple', label: 'Simple' },
    { value: 'karras', label: 'Karras' },
    { value: 'exponential', label: 'Exponential' },
    { value: 'sgm_uniform', label: 'SGM Uniform' },
    { value: 'beta', label: 'Beta' },
    { value: 'linear', label: 'Linear' },
    { value: 'cosine', label: 'Cosine' },
  ]

  private _renderSamplerPicker(value: string, onChange: (v: string) => void): TemplateResult {
    const label = this._samplerOptions.find((o) => o.value === value)?.label ?? value
    return html`<div class="popup-picker field">
      <button class="picker-trigger" type="button" @click=${(e: MouseEvent) =>
        this._openOptionPickerDialog(e, 'Sampler', value, this._samplerOptions.map(o => ({ value: o.value, label: o.label })), onChange)}>
        <strong>${label}</strong><span>Sampler</span>
      </button>
    </div>`
  }

  private _renderSchedulerPicker(value: string, onChange: (v: string) => void): TemplateResult {
    const label = this._schedulerOptions.find((o) => o.value === value)?.label ?? value
    return html`<div class="popup-picker field">
      <button class="picker-trigger" type="button" @click=${(e: MouseEvent) =>
        this._openOptionPickerDialog(e, 'Scheduler', value, this._schedulerOptions.map(o => ({ value: o.value, label: o.label })), onChange)}>
        <strong>${label}</strong><span>Scheduler</span>
      </button>
    </div>`
  }

  // Inline picker state
  private _pickerAnchorRect?: DOMRect
  private _pickerType: 'options' | 'loras' | null = null
  private _pickerLabel = ''
  private _pickerValue = ''
  private _pickerOptions: { value: string; label: string; meta?: string }[] = []
  private _pickerOnSelect?: (v: string) => void
  private _pickerLoraSelected: string[] = []
  private _pickerOnToggle?: (v: string) => void
  private _pickerSearch = ''

  private _openModelPicker(e: MouseEvent, currentValue: string, onSelect: (v: string) => void) {
    const { catalog } = this._assets.value
    const options = catalog.models.map((m) => ({
      value: m.path,
      label: `${m.preferred ? '* ' : ''}${m.name}`,
      meta: m.size ? formatBytes(m.size) : undefined,
    }))
    this._openOptionPickerDialog(e, 'Model', currentValue, options, onSelect)
  }

  private _openOptionPickerDialog(
    e: MouseEvent,
    label: string,
    value: string,
    options: { value: string; label: string; meta?: string }[],
    onSelect: (v: string) => void,
  ) {
    this._pickerAnchorRect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    this._pickerType = 'options'
    this._pickerLabel = label
    this._pickerValue = value
    this._pickerOptions = options
    this._pickerOnSelect = onSelect
    this._pickerSearch = ''
    this.requestUpdate()
  }

  private _openLoraPicker(e: MouseEvent, selected: string[], onToggle: (v: string) => void) {
    this._pickerAnchorRect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    this._pickerType = 'loras'
    this._pickerLabel = 'LoRAs'
    this._pickerLoraSelected = [...selected]
    this._pickerOnToggle = onToggle
    this._pickerSearch = ''
    this.requestUpdate()
  }

  private _closePicker() {
    this._pickerType = null
    this.requestUpdate()
  }

  private _pickerPosition() {
    const rect = this._pickerAnchorRect
    if (!rect) return { left: 0, top: 0, width: 300, maxHeight: 400 }
    const vp = 8
    const gap = 6
    const preferredWidth = this._pickerLabel === 'Model' ? 520 : this._pickerLabel === 'LoRAs' ? 400 : 320
    const width = Math.min(window.innerWidth - vp * 2, preferredWidth)
    const left = Math.min(Math.max(vp, rect.left), Math.max(vp, window.innerWidth - width - vp))
    const below = window.innerHeight - rect.bottom - vp - gap
    const above = rect.top - vp - gap
    const openBelow = below >= 200 || below >= above
    const maxHeight = Math.max(180, Math.min(520, openBelow ? below : above))
    const top = openBelow ? rect.bottom + gap : Math.max(vp, rect.top - maxHeight - gap)
    return { left, top, width, maxHeight }
  }

  private _renderPicker(): TemplateResult {
    if (!this._pickerType) return html``
    const pos = this._pickerPosition()
    const popoverStyle = [
      `position:fixed`,
      `inset:${pos.top}px auto auto ${pos.left}px`,
      `width:${pos.width}px`,
      `max-height:${pos.maxHeight}px`,
      `z-index:9999`,
      `background:#1c1c24`,
      `border:1px solid #5a5a7a`,
      `border-radius:4px`,
      `box-shadow:0 16px 48px rgba(0,0,0,0.6)`,
      `display:grid`,
      `grid-template-rows:auto auto minmax(0,1fr)`,
      `gap:6px`,
      `padding:8px`,
      `box-sizing:border-box`,
      `overflow:hidden`,
    ].join(';')

    const head = html`<div style="display:grid;grid-template-columns:1fr max-content;gap:8px;align-items:center">
      <strong style="color:#d4d4d8;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${this._pickerLabel}</strong>
      <button style="width:22px;min-width:22px;min-height:22px;padding:0;background:#2e2e3c;border:1px solid #3a3a4c;color:#d4d4d8;border-radius:2px" @click=${this._closePicker}>×</button>
    </div>`

    const search = html`<input placeholder=${`Search ${this._pickerLabel}…`} .value=${this._pickerSearch}
      @input=${(e: Event) => { this._pickerSearch = (e.target as HTMLInputElement).value; this.requestUpdate() }}
      style="width:100%;max-width:none;background:#14141a;border:1px solid #3a3a4c;color:#d4d4d8;border-radius:2px;padding:3px 6px;font-size:12px;box-sizing:border-box" />`

    if (this._pickerType === 'options') {
      const lower = this._pickerSearch.toLowerCase()
      const filtered = lower
        ? this._pickerOptions.filter((o) => o.label.toLowerCase().includes(lower) || o.value.toLowerCase().includes(lower))
        : this._pickerOptions
      const list = html`<div style="overflow:auto;overscroll-behavior:contain;display:grid;gap:2px">
        ${filtered.map((opt) => {
          const isActive = opt.value === this._pickerValue
          const btnStyle = [
            'display:grid;grid-template-columns:1fr max-content;gap:8px;align-items:center',
            'min-height:26px;padding:3px 7px;text-align:left;width:100%;border-radius:2px',
            isActive
              ? 'background:#1a3060;color:#7aadff;border:1px solid #4d87c4'
              : 'background:#1e1e28;color:#d4d4d8;border:1px solid #3a3a4c',
          ].join(';')
          return html`<button style=${btnStyle}
            @click=${() => { this._pickerOnSelect?.(opt.value); this._closePicker() }}>
            <span style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${opt.label}</span>
            ${opt.meta ? html`<small style="color:#7a7a8a;font-size:11px">${opt.meta}</small>` : ''}
          </button>`
        })}
      </div>`
      return html`<div style=${popoverStyle} @click=${(e: Event) => e.stopPropagation()}>${head}${search}${list}</div>
        <div style="position:fixed;inset:0;z-index:9998" @click=${this._closePicker}></div>`
    }

    // Loras picker
    const { catalog } = this._assets.value
    const lower = this._pickerSearch.toLowerCase()
    const filtered = lower
      ? catalog.loras.filter((l) => l.name.toLowerCase().includes(lower) || l.path.toLowerCase().includes(lower))
      : catalog.loras
    const list = html`<div style="overflow:auto;overscroll-behavior:contain;display:grid;gap:2px">
      ${filtered.map((lora) => {
        const isSelected = this._pickerLoraSelected.includes(lora.path)
        const rowStyle = `display:grid;grid-template-columns:auto 1fr;gap:6px;align-items:center;cursor:pointer;padding:3px 6px;border-radius:2px;${isSelected ? 'background:#1a3060;color:#7aadff' : 'color:#d4d4d8'}`
        return html`<label style=${rowStyle}>
          <input type="checkbox" .checked=${isSelected}
            @change=${() => {
              this._pickerOnToggle?.(lora.path)
              this._pickerLoraSelected = isSelected
                ? this._pickerLoraSelected.filter((p) => p !== lora.path)
                : [...this._pickerLoraSelected, lora.path]
              this.requestUpdate()
            }} />
          <span style="overflow-wrap:anywhere;font-size:12px">${lora.name}</span>
        </label>`
      })}
    </div>`
    return html`<div style=${popoverStyle} @click=${(e: Event) => e.stopPropagation()}>${head}${search}${list}</div>
      <div style="position:fixed;inset:0;z-index:9998" @click=${this._closePicker}></div>`
  }

  // ── Transparency controls ────────────────────────────────────────────────────

  private renderTransparencyControls(config: {
    enabled: boolean
    onEnabled: (v: boolean) => void
    label: string
    method?: string
    onMethod?: (v: string) => void
    mode: string
    onMode: (v: string) => void
    color: string
    onColor: (v: string) => void
    tolerance: number
    onTolerance: (v: number) => void
    alphaBlur: number
    onAlphaBlur: (v: number) => void
    alphaThreshold: number
    onAlphaThreshold: (v: number) => void
  }): TemplateResult {
    return html`<section class="advanced-sampling">
      <label class="check-field">
        <input type="checkbox" .checked=${config.enabled} @change=${(e: Event) => config.onEnabled((e.target as HTMLInputElement).checked)} />
        <span>${config.label}</span>
      </label>
      ${config.enabled ? html`
        ${config.onMethod ? html`<label class="field inline">
          <span>Alpha tech</span>
          <select .value=${config.method ?? 'auto'} @change=${(e: Event) => config.onMethod?.((e.target as HTMLSelectElement).value)}>
            <option value="auto">Auto VRAM</option>
            <option value="fast">Fast alpha</option>
            <option value="layerdiffuse">LayerDiffuse</option>
          </select>
        </label>` : ''}
        <label class="field inline">
          <span>Alpha source</span>
          <select .value=${config.mode} @change=${(e: Event) => config.onMode((e.target as HTMLSelectElement).value)}>
            <option value="border">Sample border</option>
            <option value="color">Key color</option>
          </select>
        </label>
        ${config.mode === 'color' ? html`<label class="field inline"><span>Key color</span><input type="color" .value=${config.color} @input=${(e: Event) => config.onColor((e.target as HTMLInputElement).value)} /></label>` : ''}
        ${this.numCtrl('Tolerance', config.tolerance, 0, 255, 1, 0, (v) => config.onTolerance(clamp(v, 0, 255, config.tolerance)))}
        ${this.numCtrl('Alpha blur', config.alphaBlur, 0, 24, 0.5, 1, (v) => config.onAlphaBlur(clamp(v, 0, 24, config.alphaBlur)))}
        ${this.numCtrl('Alpha floor', config.alphaThreshold, 0, 255, 1, 0, (v) => config.onAlphaThreshold(Math.round(clamp(v, 0, 255, config.alphaThreshold))))}
      ` : ''}
    </section>`
  }

  renderLayerTransparencyControls(): TemplateResult {
    const { transparentBackground, transparentMethod, transparentMode, transparentColor, transparentTolerance, transparentAlphaBlur, transparentAlphaThreshold } = this._layer.value
    return this.renderTransparencyControls({
      enabled: transparentBackground,
      onEnabled: (v) => $layer.setKey('transparentBackground', v),
      label: 'Transparent variation',
      method: transparentMethod,
      onMethod: (v) => $layer.setKey('transparentMethod', v),
      mode: transparentMode,
      onMode: (v) => $layer.setKey('transparentMode', v),
      color: transparentColor,
      onColor: (v) => $layer.setKey('transparentColor', v),
      tolerance: transparentTolerance,
      onTolerance: (v) => $layer.setKey('transparentTolerance', v),
      alphaBlur: transparentAlphaBlur,
      onAlphaBlur: (v) => $layer.setKey('transparentAlphaBlur', v),
      alphaThreshold: transparentAlphaThreshold,
      onAlphaThreshold: (v) => $layer.setKey('transparentAlphaThreshold', v),
    })
  }

  // ── Mask tab ──────────────────────────────────────────────────────────────────

  private renderLayerMaskPanel(layer: LayerItem): TemplateResult {
    return html`<div class="layer-tab mask-tab layer-mask-tab">
      ${this.renderMaskSchedulerBox(layer)}
      ${this.renderRgbaMaskCard(layer)}
      ${this.renderSpecialMaskCard(layer, 'cfg')}
      ${this.renderSpecialMaskCard(layer, 'denoise')}
      ${this.renderRegionsPanel('layer', layer.id)}
    </div>`
  }

  private renderRgbaMaskCard(layer: LayerItem): TemplateResult {
    const sc = this._scene.value
    const promptInherited = layer.preset.rgbaPromptInherited ?? true
    const negativePromptInherited = layer.preset.rgbaNegativePromptInherited ?? true
    const denoiseInherited = layer.preset.rgbaDenoiseInherited ?? true
    const cfgInherited = layer.preset.rgbaCfgInherited ?? true
    const weightInherited = layer.preset.rgbaWeightInherited ?? true
    const blendingInherited = layer.preset.rgbaBlendingInherited ?? true
    const blendingEnabled = blendingInherited ? true : (layer.preset.rgbaMaskBlendingEnabled ?? false)
    const blendingRadius = blendingInherited ? 32 : (layer.preset.rgbaMaskBlendingRadius ?? 32)
    const blendingStrength = blendingInherited ? 1 : (layer.preset.rgbaMaskBlendingStrength ?? 1)
    return html`<section class="region-card rgba-mask-card">
      <div class="region-head">
        <button class="region-color region-color-button rgba-mask-icon" style="--region-color: #63d297" title="Paint RGBA mask"
          @click=${() => this.emit('rtd-mask-select', { layerId: layer.id, channel: 'color' as MaskChannel })}></button>
        <input .value=${'RGBA mask'} disabled />
      </div>
      <div class="region-meta">Spatial RGBA input mask · move, scale, rotate, paint, or load image pixels</div>
      <label class="field">
        <span>Prompt</span>
        <textarea rows="2" .value=${promptInherited ? sc.prompt : layer.preset.prompt} ?disabled=${promptInherited}
          @input=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { prompt: (e.target as HTMLTextAreaElement).value, rgbaPromptInherited: false } })}></textarea>
      </label>
      <label class="check-field inherit-check">
        <input type="checkbox" .checked=${promptInherited}
          @change=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaPromptInherited: (e.target as HTMLInputElement).checked } })} />
        <span>inherit prompt</span>
      </label>
      <label class="field">
        <span>Negative prompt</span>
        <textarea rows="2" .value=${negativePromptInherited ? sc.negativePrompt : layer.preset.negativePrompt} ?disabled=${negativePromptInherited}
          @input=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { negativePrompt: (e.target as HTMLTextAreaElement).value, rgbaNegativePromptInherited: false } })}></textarea>
      </label>
      <label class="check-field inherit-check">
        <input type="checkbox" .checked=${negativePromptInherited}
          @change=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaNegativePromptInherited: (e.target as HTMLInputElement).checked } })} />
        <span>inherit negative</span>
      </label>
      <div class="mask-force-grid">
        ${this.numCtrl('Denoise', denoiseInherited ? sc.strength : layer.preset.strength, 0, 0.999, 0.01, 2,
          (v) => this.emit('rtd-layer-preset', { id: layer.id, patch: { strength: clamp(v, 0, 0.999, layer.preset.strength), rgbaDenoiseInherited: false } }), denoiseInherited)}
        ${this.numCtrl('CFG', cfgInherited ? sc.cfg : (layer.preset.layerCfg ?? layer.preset.cfg), 0, 30, 0.1, 1,
          (v) => this.emit('rtd-layer-preset', { id: layer.id, patch: { layerCfg: clamp(v, 0, 30, layer.preset.layerCfg ?? layer.preset.cfg), rgbaCfgInherited: false } }), cfgInherited)}
        ${this.numCtrl('Weight', weightInherited ? 1 : layer.preset.conditionWeight, 0, 4, 0.05, 2,
          (v) => this.emit('rtd-layer-preset', { id: layer.id, patch: { conditionWeight: clamp(v, 0, 4, layer.preset.conditionWeight), rgbaWeightInherited: false } }), weightInherited)}
      </div>
      <div class="mask-force-grid inherit-grid">
        <label class="check-field inherit-check"><input type="checkbox" .checked=${denoiseInherited}
          @change=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaDenoiseInherited: (e.target as HTMLInputElement).checked } })} /><span>inherit denoise</span></label>
        <label class="check-field inherit-check"><input type="checkbox" .checked=${cfgInherited}
          @change=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaCfgInherited: (e.target as HTMLInputElement).checked } })} /><span>inherit CFG</span></label>
        <label class="check-field inherit-check"><input type="checkbox" .checked=${weightInherited}
          @change=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaWeightInherited: (e.target as HTMLInputElement).checked } })} /><span>inherit weight</span></label>
      </div>
      <div class="button-grid mask-action-grid rgba-mask-actions">
        <button @click=${() => this.emit('rtd-rgba-mask-load', { layerId: layer.id })}>Load image</button>
        <button @click=${() => this.emit('rtd-mask-select', { layerId: layer.id, channel: 'color' as MaskChannel })}>Paint</button>
      </div>
      ${this.renderMaskActionControls(layer.id, 'color')}
      <label class="check-field">
        <input type="checkbox" .checked=${blendingEnabled} ?disabled=${blendingInherited}
          @change=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaMaskBlendingEnabled: (e.target as HTMLInputElement).checked, rgbaBlendingInherited: false } })} />
        <span>Blending</span>
      </label>
      <label class="check-field inherit-check">
        <input type="checkbox" .checked=${blendingInherited}
          @change=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaBlendingInherited: (e.target as HTMLInputElement).checked } })} />
        <span>inherit blending</span>
      </label>
      ${blendingEnabled ? html`<div class="resolution-grid">
        ${this.numCtrl('Radius', blendingRadius, -160, 160, 1, 0,
          (v) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaMaskBlendingRadius: Math.round(clamp(v, -160, 160, blendingRadius)), rgbaBlendingInherited: false } }), blendingInherited)}
        ${this.numCtrl('Strength', blendingStrength, 0, 4, 0.05, 2,
          (v) => this.emit('rtd-layer-preset', { id: layer.id, patch: { rgbaMaskBlendingStrength: clamp(v, 0, 4, blendingStrength), rgbaBlendingInherited: false } }), blendingInherited)}
      </div>` : ''}
    </section>`
  }

  private renderSpecialMaskCard(layer: LayerItem, channel: Extract<MaskChannel, 'cfg' | 'denoise'>): TemplateResult {
    const label = channel === 'cfg' ? 'CFG mask' : 'Denoise mask'
    const color = channel === 'cfg' ? '#7aadff' : '#ffb45f'
    const description = channel === 'cfg'
      ? 'Layer-wide fine tuning · transparent by default · overrides regular CFG weights'
      : 'Layer-wide fine tuning · transparent by default · overrides regular denoise weights'
    const enabledKey = channel === 'cfg' ? 'cfgMaskBlendingEnabled' : 'denoiseMaskBlendingEnabled'
    const radiusKey = channel === 'cfg' ? 'cfgMaskBlendingRadius' : 'denoiseMaskBlendingRadius'
    const strengthKey = channel === 'cfg' ? 'cfgMaskBlendingStrength' : 'denoiseMaskBlendingStrength'
    const blendingEnabled = layer.preset[enabledKey] ?? false
    const blendingRadius = layer.preset[radiusKey] ?? 32
    const blendingStrength = layer.preset[strengthKey] ?? 1
    return html`<section class="region-card special-mask-card">
      <div class="region-head">
        <button class="region-color region-color-button" style=${`--region-color: ${color}`}
          title=${`Paint ${label}`}
          @click=${() => this.emit('rtd-mask-select', { layerId: layer.id, regionId: `special:${channel}:${layer.id}`, channel })}></button>
        <input .value=${label} disabled />
      </div>
      <div class="region-meta">${description}</div>
      ${this.renderMaskActionControls(layer.id, channel)}
      <label class="check-field">
        <input type="checkbox" .checked=${blendingEnabled}
          @change=${(e: Event) => this.emit('rtd-layer-preset', { id: layer.id, patch: { [enabledKey]: (e.target as HTMLInputElement).checked } })} />
        <span>Blending</span>
      </label>
      ${blendingEnabled ? html`<div class="resolution-grid">
        ${this.numCtrl('Radius', blendingRadius, -160, 160, 1, 0,
          (v) => this.emit('rtd-layer-preset', { id: layer.id, patch: { [radiusKey]: Math.round(clamp(v, -160, 160, blendingRadius)) } }))}
        ${this.numCtrl('Strength', blendingStrength, 0, 4, 0.05, 2,
          (v) => this.emit('rtd-layer-preset', { id: layer.id, patch: { [strengthKey]: clamp(v, 0, 4, blendingStrength) } }))}
      </div>` : ''}
    </section>`
  }

  // ── Regions panel ─────────────────────────────────────────────────────────────

  private renderRegionsPanel(target: RegionTarget, layerId?: string): TemplateResult {
    const { regions } = this._layer.value
    const targetRegions = regions.filter((r) => r.target === target && (target === 'scene' || r.layerId === layerId))

    return html`<div class="layer-tab regions-tab">
      ${targetRegions.length
        ? html`<div class="region-list">
            ${targetRegions.map((region, index) => html`<section class="region-card">
              <div class="region-head">
                <input class="region-color" type="color" .value=${region.color}
                  title="Use mask color"
                  @click=${() => target === 'layer'
                    ? this.emit('rtd-mask-select', { layerId: region.layerId ?? layerId ?? '', regionId: region.id, channel: 'color' as MaskChannel, color: region.color })
                    : this.emit('rtd-region-use-color', { color: region.color })}
                  @input=${(e: Event) => this.emit('rtd-region-color', { id: region.id, color: (e.target as HTMLInputElement).value })} />
                <input .value=${region.name || this.promptMaskDisplayName(region, layerId, index)}
                  @input=${(e: Event) => this.emit('rtd-region-update', { id: region.id, patch: { name: (e.target as HTMLInputElement).value } })} />
                <button title="Delete mask" @click=${() => this.emit('rtd-region-delete', { id: region.id })}>×</button>
              </div>
              <div class="region-meta">${region.color} · ${region.shape} · ${Math.round(region.rect.width)} x ${Math.round(region.rect.height)}</div>
              ${target === 'layer' ? this.renderMaskActionControls(region.layerId ?? layerId ?? '', 'color', region.id) : ''}
              <label class="field">
                <span>Prompt</span>
                <textarea rows="2" .value=${region.prompt}
                  @input=${(e: Event) => this.emit('rtd-region-update', { id: region.id, patch: { prompt: (e.target as HTMLTextAreaElement).value, inherited: false } })}></textarea>
              </label>
              <label class="field">
                <span>Negative prompt</span>
                <textarea rows="2" .value=${region.negativePrompt}
                  @input=${(e: Event) => this.emit('rtd-region-update', { id: region.id, patch: { negativePrompt: (e.target as HTMLTextAreaElement).value, inherited: false } })}></textarea>
              </label>
              <label class="check-field">
                <input type="checkbox" .checked=${region.blendingEnabled ?? true}
                  @change=${(e: Event) => this.emit('rtd-region-update', { id: region.id, patch: { blendingEnabled: (e.target as HTMLInputElement).checked, inherited: false } })} />
                <span>Blending</span>
              </label>
              ${region.blendingEnabled ?? true ? html`
                <div class="resolution-grid">
                  ${this.numCtrl('Radius', region.blendingRadius ?? 32, -160, 160, 1, 0,
                    (v) => this.emit('rtd-region-update', { id: region.id, patch: { blendingRadius: Math.round(clamp(v, -160, 160, region.blendingRadius ?? 32)), inherited: false } }))}
                  ${this.numCtrl('Strength', region.blendingStrength ?? 1, 0, 4, 0.05, 2,
                    (v) => this.emit('rtd-region-update', { id: region.id, patch: { blendingStrength: clamp(v, 0, 4, region.blendingStrength ?? 1), inherited: false } }))}
                </div>
              ` : ''}
              ${target === 'layer' ? html`
                <div class="mask-force-grid">
                  ${this.renderRegionForceRow(
                    'CFG',
                    region.regionCfg ?? this._scene.value.cfg,
                    region.cfgMaskInherited ?? true,
                    (v) => this.emit('rtd-region-update', { id: region.id, patch: { regionCfg: clamp(v, 0, 30, region.regionCfg ?? this._scene.value.cfg), inherited: false, cfgMaskInherited: false } }),
                    (value) => this.emit('rtd-region-update', { id: region.id, patch: { cfgMaskInherited: value } }),
                  )}
                  ${this.renderRegionForceRow(
                    'Denoise',
                    region.denoise ?? this._scene.value.strength,
                    region.denoiseMaskInherited ?? true,
                    (v) => this.emit('rtd-region-update', { id: region.id, patch: { denoise: clamp(v, 0, 0.999, region.denoise ?? this._scene.value.strength), inherited: false, denoiseMaskInherited: false } }),
                    (value) => this.emit('rtd-region-update', { id: region.id, patch: { denoiseMaskInherited: value } }),
                  )}
                </div>
              ` : ''}
            </section>`)}
          </div>`
        : html`<div class="variation-empty empty">No named masks yet</div>`}
    </div>`
  }

  private promptMaskDisplayName(region: RegionItem, layerId: string | undefined, index: number): string {
    const typed = region.name?.trim()
    if (typed) return typed
    const prompt = region.prompt?.trim()
    if (prompt) return prompt.slice(0, 16)
    const layerIndex = Math.max(0, this._layer.value.layers.findIndex((layer) => layer.id === layerId))
    return `Layer ${layerIndex >= 0 ? layerIndex + 1 : index + 1}`
  }

  // ── Video tab ─────────────────────────────────────────────────────────────────

  private renderVideoLayerPanel(): TemplateResult {
    return html`<div class="layer-tab">
      <div class="empty">Video controls are managed by the canvas. Use the video file button in the toolbar to load a video layer.</div>
    </div>`
  }

  // ── Root render ───────────────────────────────────────────────────────────────

  override render() {
    const { layers, selectedLayerId, layerPanelTab } = this._layer.value
    const layer = layers.find((l) => l.id === selectedLayerId && l.selected) ?? layers.find((l) => l.selected)

    if (!layer) {
      return html`
        <div class="context-tabs-wrap">
          ${this.renderTabs(undefined)}
        </div>
        <div class="context-body">
          <div class="empty layer-empty">Select a layer or mask</div>
        </div>
        ${this._renderPicker()}
      `
    }

    let body: TemplateResult
    if (layerPanelTab === 'video' && layer.isVideo) {
      body = this.renderVideoLayerPanel()
    } else {
      body = this.renderLayerMaskPanel(layer)
    }

    return html`
      ${this.renderTabs(layer)}
      <div class="context-body">
        ${body}
      </div>
      ${this._renderPicker()}
    `
  }

  // ── Styles ────────────────────────────────────────────────────────────────────

  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      height: 100%;
      background: var(--bg-panel, #24242c);
      color: var(--fg, #d4d4d8);
      font-size: var(--font-size, 12px);
      overflow-y: auto;
      min-height: 0;
    }

    .context-tabs-wrap {
      flex-shrink: 0;
    }

    .context-tabs {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
      gap: 2px;
      padding: 6px 6px 0;
      flex-shrink: 0;
    }

    .context-tabs button {
      min-height: 24px;
      padding: 2px 4px;
      color: var(--fg-label, #9898a8);
      background: var(--bg-input, #14141a);
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius, 2px);
      font-size: 12px;
      cursor: pointer;
    }

    .context-tabs button:hover {
      background: var(--bg-hover, #2e2e3c);
      color: var(--fg, #d4d4d8);
    }

    .context-tabs button.active {
      background: var(--bg-sel, #1a3060);
      color: var(--fg-accent, #7aadff);
      border-color: var(--border-focus, #4d87c4);
    }

    .context-body {
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      overscroll-behavior: contain;
      padding: 6px;
      box-sizing: border-box;
      flex: 1;
    }

    .context-body textarea {
      min-height: 48px;
      max-height: 58px;
    }

    .layer-tab {
      display: grid;
      gap: 10px;
      min-height: 0;
    }

    .layer-empty {
      padding: 10px;
      border: 1px dashed #3a3f45;
      border-radius: 6px;
    }

    .mask-tab,
    .regions-tab {
      display: grid;
      gap: 8px;
    }

    .source-tab {
      gap: 8px;
    }

    .source-root-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 4px;
    }

    .source-root-row input {
      width: 100%;
    }

    .source-root-row .icon-btn {
      padding: 0 8px;
      font-size: 14px;
      line-height: 1;
      cursor: pointer;
      white-space: nowrap;
    }

    .transform-panel {
      display: grid;
      gap: 8px;
      min-width: 0;
      padding: 7px;
      border: 1px solid var(--border, #3a3a4c);
      background: var(--bg-header, #1e1e28);
      border-radius: var(--radius, 2px);
    }

    .transform-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 6px;
      align-items: center;
      min-width: 0;
      color: var(--fg, #d4d4d8);
    }

    .transform-head span {
      color: var(--fg-dim, #7a7a8a);
      font-size: 11px;
      white-space: nowrap;
    }

    .transform-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 5px;
      min-width: 0;
    }

    .resolution-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }

    .mask-force-grid {
      display: grid;
      gap: 5px;
    }

    .mask-force-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 8px;
      align-items: center;
    }

    .mask-force-row .inherit-check {
      white-space: nowrap;
      font-size: 12px;
    }

    .sampling-controls {
      display: grid;
      gap: 8px;
    }

    .sampling-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
    }

    .advanced-sampling {
      display: grid;
      gap: 6px;
      padding: 6px 0;
      border-top: 1px solid rgba(36, 36, 93, 0.35);
      border-bottom: 1px solid rgba(36, 36, 93, 0.35);
    }

    .check-field {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 6px;
      align-items: center;
      font-size: 13px;
    }

    .field {
      display: grid;
      gap: 4px;
    }

    .field.inline {
      grid-template-columns: minmax(72px, max-content) minmax(0, 1fr);
      align-items: center;
      gap: 8px;
    }

    .field span {
      font-size: 11px;
      color: var(--fg-label, #9898a8);
    }

    .field label {
      font-size: 11px;
    }

    label.field,
    label.field.inline {
      font-size: 12px;
    }

    input, select, textarea {
      background: #1e1e2e;
      border: 1px solid #3a3a5a;
      border-radius: 2px;
      color: var(--fg, #d4d4d8);
      font-size: 12px;
      padding: 3px 6px;
      box-sizing: border-box;
      width: 100%;
      min-width: 0;
    }

    input[type="checkbox"] {
      width: auto;
      min-width: auto;
    }

    input[type="color"] {
      padding: 1px 2px;
      height: 28px;
    }

    textarea {
      resize: vertical;
    }

    button {
      min-height: 26px;
      padding: 3px 8px;
      background: #2a2a3e;
      border: 1px solid #3a3a5a;
      border-radius: 2px;
      color: var(--fg, #d4d4d8);
      cursor: pointer;
      font-size: 12px;
      width: 100%;
    }

    button:hover {
      background: #3a3a5a;
    }

    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    .button-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
      gap: 4px;
    }

    .mask-action-grid {
      grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));
      align-items: stretch;
    }

    .mask-action {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 58px;
      min-width: 0;
    }

    .mask-action button {
      border-top-right-radius: 0;
      border-bottom-right-radius: 0;
    }

    .mask-action rtd-number {
      min-width: 0;
    }

    .small-btn {
      min-height: 20px;
      font-size: 11px;
      padding: 1px 6px;
      width: auto;
    }

    .empty {
      color: #8c9498;
      font-size: 13px;
    }

    .region-list {
      display: grid;
      gap: 8px;
    }

    .region-card {
      display: grid;
      gap: 7px;
      padding: 8px;
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius-lg, 4px);
      background: var(--bg-header, #1e1e28);
      color: var(--fg, #d4d4d8);
    }

    .region-card input,
    .region-card select,
    .region-card textarea {
      background: var(--bg-input, #14141a);
      color: var(--fg, #d4d4d8);
      border-color: var(--border, #3a3a4c);
    }

    .region-card button {
      background: var(--bg-active, #383848);
      color: var(--fg, #d4d4d8);
      border-color: var(--border-hi, #5a5a7a);
    }

    .region-card button:hover {
      background: var(--bg-hover, #2e2e3c);
    }

    .region-card .check-field {
      color: var(--fg, #d4d4d8);
    }

    .region-card .field span {
      color: var(--fg-label, #9898a8);
    }

    .region-head {
      display: grid;
      grid-template-columns: 28px minmax(0, 1fr) max-content;
      gap: 6px;
      align-items: center;
    }

    .region-head input {
      min-width: 0;
    }

    .region-color {
      width: 28px;
      min-width: 28px;
      height: 28px;
      min-height: 28px;
      padding: 2px;
    }

    .region-card .region-color-button {
      background: var(--region-color);
      border: 1px solid #8b94a3;
      cursor: pointer;
    }

    .region-card .region-color-button:hover {
      background: var(--region-color);
      border-color: #c8d1e0;
    }

    .rgba-mask-card .rgba-mask-icon,
    .rgba-mask-card .rgba-mask-icon:hover {
      background: #63d297;
    }

    .region-meta {
      font-size: 11px;
      color: var(--fg-dim, #7a7a8a);
    }

    .inherited-region-card {
      border-color: var(--border-hi, #5a5a7a);
    }

    .mask-actions {
      display: grid;
    }

    .interval-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }

    .advanced-section {
      border: none;
      padding: 0;
    }

    .advanced-section summary {
      cursor: pointer;
      color: var(--fg-dim, #7a7a8a);
      font-size: 11px;
      padding: 3px 0;
      user-select: none;
    }

    .advanced-body {
      display: grid;
      gap: 6px;
      padding-top: 6px;
    }

    .schedule-box {
      display: grid;
      min-width: 0;
      border: 1px solid var(--border, #3a3a4c);
      background: var(--bg-header, #1e1e28);
      border-radius: var(--radius, 2px);
      color: var(--fg, #d4d4d8);
    }

    .schedule-box summary {
      display: block;
      min-width: 0;
      padding: 7px;
      cursor: pointer;
      color: var(--fg-dim, #7a7a8a);
      font-size: 11px;
      user-select: none;
    }

    .schedule-box summary strong {
      color: var(--fg, #d4d4d8);
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .schedule-timeline {
      display: grid;
      gap: 5px;
      min-width: 0;
      padding: 0 7px 7px;
    }

    .timeline-axis {
      display: grid;
      grid-template-columns: minmax(92px, 0.34fr) minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      min-width: 0;
      color: var(--fg-dim, #7a7a8a);
      font-size: 10px;
    }

    .timeline-axis-line {
      position: relative;
      height: 12px;
      margin-right: 48px;
    }

    .timeline-axis-line span {
      position: absolute;
      top: 0;
      transform: translateX(-50%);
    }

    .timeline-axis-line span:first-child {
      left: 0;
      transform: none;
    }

    .timeline-axis-line span:nth-child(2) {
      left: 50%;
    }

    .timeline-axis-line span:last-child {
      right: 0;
      transform: none;
    }

    .timeline-head,
    .timeline-row {
      display: grid;
      grid-template-columns: minmax(92px, 0.34fr) minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      min-width: 0;
    }

    .timeline-head {
      grid-template-columns: minmax(72px, 0.34fr) 1fr max-content max-content;
      color: var(--fg-dim, #7a7a8a);
      font-size: 11px;
    }

    .timeline-row > span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
      color: var(--fg-label, #9898a8);
    }

    .timeline-row.muted {
      opacity: 0.38;
    }

    .timeline-track {
      position: relative;
      height: 26px;
      overflow: hidden;
      border: 1px solid var(--border, #3a3a4c);
      background:
        linear-gradient(90deg, transparent calc(50% - 1px), var(--border-hi, #5a5a7a) calc(50% - 1px), var(--border-hi, #5a5a7a) calc(50% + 1px), transparent calc(50% + 1px)),
        var(--bg-input, #14141a);
    }

    .timeline-lane {
      position: absolute;
      inset: 0 48px 0 0;
      cursor: grab;
    }

    .timeline-lane:active {
      cursor: grabbing;
    }

    .timeline-lane i {
      position: absolute;
      top: 6px;
      bottom: 6px;
      border-radius: 2px;
      pointer-events: none;
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.4);
    }

    .timeline-handle {
      position: absolute;
      top: 3px;
      bottom: 3px;
      width: 12px;
      margin-left: -6px;
      border: 1px solid rgba(255,255,255,.72);
      border-radius: 2px;
      background: rgba(20,20,26,.55);
      cursor: ew-resize;
      z-index: 2;
    }

    .timeline-handle:hover {
      background: rgba(255,255,255,.18);
      border-color: #fff;
    }

    .timeline-mode {
      position: absolute;
      right: 2px;
      top: 2px;
      bottom: 2px;
      width: 42px;
      min-width: 42px;
      padding: 0 2px;
      font-size: 10px;
      z-index: 3;
    }

    .timeline-empty {
      min-height: 24px;
      display: grid;
      place-items: center;
      color: var(--fg-dim, #7a7a8a);
      font-size: 11px;
    }

    .source-preview {
      display: grid;
      place-items: center;
      min-height: 120px;
      border: 1px solid #24245d;
      background-color: #b8bdc4;
      background-image:
        linear-gradient(45deg, #8f969f 25%, transparent 25%),
        linear-gradient(-45deg, #8f969f 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #8f969f 75%),
        linear-gradient(-45deg, transparent 75%, #8f969f 75%);
      background-position: 0 0, 0 8px, 8px -8px, -8px 0;
      background-size: 16px 16px;
      overflow: hidden;
    }

    .source-preview img {
      max-width: 100%;
      max-height: 180px;
      object-fit: contain;
    }

    .source-select {
      min-height: 112px;
      font-size: 12px;
    }

    .generator-tab {
      gap: 8px;
    }

    .generator-controls {
      display: grid;
      gap: 8px;
    }

    .generator-actions {
      display: grid;
      gap: 6px;
    }

    .generator-action-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      align-items: center;
    }

    .variation-dock {
      border-top: 1px solid #3a3a5a;
      padding-top: 8px;
    }

    .variation-list {
      display: grid;
      gap: 6px;
      grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    }

    .variation {
      display: grid;
      gap: 3px;
      padding: 4px;
      border: 1px solid #3a3a5a;
      background: #1e1e2e;
      cursor: pointer;
      text-align: center;
    }

    .variation.selected {
      border-color: #3f77ff;
      background: #1a2040;
    }

    .variation img {
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
    }

    .variation-label {
      font-size: 10px;
      color: #8898aa;
    }

    .variation-seed {
      font-size: 10px;
      color: #7a8898;
    }

    .variation-empty {
      padding: 16px;
      text-align: center;
    }

    .popup-picker {
      position: relative;
      width: 100%;
      min-width: 0;
    }

    .picker-trigger {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 10px;
      align-items: center;
      width: 100%;
      max-width: 100%;
      min-height: var(--ctrl-h, 22px);
      padding: var(--pad-y, 3px) var(--pad-x, 6px);
      box-sizing: border-box;
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius, 2px);
      color: var(--fg, #d4d4d8);
      background: var(--bg-input, #14141a);
      cursor: pointer;
      text-align: left;
    }

    .picker-trigger:hover {
      border-color: var(--border-hi, #5a5a7a);
      background: var(--bg-hover, #2e2e3c);
    }

    .picker-trigger strong {
      font-size: 12px;
      font-weight: 600;
      text-align: left;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--fg, #d4d4d8);
    }

    .picker-trigger span {
      color: var(--fg-dim, #7a7a8a);
      font-size: 11px;
      white-space: nowrap;
    }

    h2 {
      margin: 0;
      font-size: 13px;
      font-weight: 600;
    }

    .section-header {
      font-weight: 600;
      font-size: 12px;
      color: var(--fg, #d4d4d8);
    }

    .controlnet-panel {
      gap: 8px;
    }

    details {
      border: none;
    }

    summary {
      cursor: pointer;
    }
  `
}

// ── Utility ───────────────────────────────────────────────────────────────────

function formatBytes(bytes: number | undefined): string {
  if (!bytes) return ''
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`
  return `${bytes} B`
}

