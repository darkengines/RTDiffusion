import { LitElement, css, html } from 'lit'
import { customElement, property } from 'lit/decorators.js'
import type { AssetItem, SamplingControlsConfig } from '../../types'
import { SAMPLERS, SCHEDULERS } from '../../constants'

@customElement('rtd-sampling-controls')
export class RtdSamplingControls extends LitElement {
  @property({ type: Object }) config!: SamplingControlsConfig
  @property({ type: Array }) models: AssetItem[] = []
  @property({ type: Array }) loras: AssetItem[] = []

  static styles = css`
    :host { display: block }
    .sampling-controls { display: flex; flex-direction: column; gap: 8px }
    .field { display: flex; flex-direction: column; gap: 4px; font-size: 12px }
    .field span { font-weight: 500; color: var(--label-color, #555) }
    .field textarea, .field select, .field input { font-family: inherit; font-size: 12px; border: 1px solid var(--border, #ccc); border-radius: 4px; padding: 4px 6px; background: var(--input-bg, #fff) }
    .field textarea { resize: vertical }
    .sampling-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px }
    .number-field { display: flex; flex-direction: column; gap: 2px; font-size: 12px }
    .number-field span { font-weight: 500; color: var(--label-color, #555) }
    .number-field input[type=number] { font-family: inherit; font-size: 12px; border: 1px solid var(--border, #ccc); border-radius: 4px; padding: 4px 6px; background: var(--input-bg, #fff); width: 100%; box-sizing: border-box }
    .picker-row { display: flex; gap: 6px; align-items: center; font-size: 12px }
    .picker-row label { flex: 1 }
  `

  render() {
    if (!this.config) return html``
    const c = this.config
    return html`<div class="sampling-controls">
      <label class="field">
        <span>Prompt</span>
        <textarea rows="4" .value=${c.prompt}
          @input=${(e: Event) => c.onPrompt((e.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field">
        <span>Negative prompt</span>
        <textarea rows="3" .value=${c.negativePrompt}
          @input=${(e: Event) => c.onNegativePrompt((e.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      ${c.onModelPath ? html`<label class="field">
        <span>Model</span>
        <select .value=${c.modelPath ?? ''} @change=${(e: Event) => c.onModelPath!((e.target as HTMLSelectElement).value)}>
          ${this.models.map((m) => html`<option value=${m.path}>${m.preferred ? '* ' : ''}${m.name}</option>`)}
        </select>
      </label>` : ''}
      <div class="sampling-grid">
        ${c.onStrength ? this._numberField('Denoise', c.strength ?? 0, 0, 0.999, 0.01, 2, c.onStrength) : ''}
        ${this._numberField('CFG', c.cfg, 0, 30, 0.1, 1, c.onCfg)}
        ${this._numberField('Steps', c.steps, 1, 40, 1, 0, c.onSteps)}
      </div>
      <label class="field">
        <span>Sampler</span>
        <select .value=${c.sampler} @change=${(e: Event) => c.onSampler((e.target as HTMLSelectElement).value)}>
          ${SAMPLERS.map((s) => html`<option value=${s.value}>${s.label}</option>`)}
        </select>
      </label>
      <label class="field">
        <span>Scheduler</span>
        <select .value=${c.scheduler} @change=${(e: Event) => c.onScheduler((e.target as HTMLSelectElement).value)}>
          ${SCHEDULERS.map((s) => html`<option value=${s.value}>${s.label}</option>`)}
        </select>
      </label>
      ${c.onToggleLora ? html`<div class="field">
        <span>LoRAs</span>
        <div>
          ${this.loras.map((lora) => html`<label style="display:flex;gap:6px;align-items:center;font-size:12px">
            <input type="checkbox"
              .checked=${(c.loraPaths ?? []).includes(lora.path)}
              @change=${() => c.onToggleLora!(lora.path)} />
            ${lora.name}
          </label>`)}
        </div>
      </div>` : ''}
    </div>`
  }

  private _numberField(label: string, value: number, min: number, max: number, step: number, decimals: number, onChange: (v: number) => void) {
    return html`<div class="number-field">
      <span>${label}</span>
      <input type="number" .value=${value.toFixed(decimals)} min=${min} max=${max} step=${step}
        @change=${(e: Event) => onChange(parseFloat((e.target as HTMLInputElement).value))} />
    </div>`
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-sampling-controls': RtdSamplingControls }
}
