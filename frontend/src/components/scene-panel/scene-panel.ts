import { LitElement, css, html, type TemplateResult } from 'lit'
import { customElement } from 'lit/decorators.js'
import { StoreController } from '@nanostores/lit'

import { $scene } from '../../stores/scene.store'
import { $stream } from '../../stores/stream.store'
import { $assets } from '../../stores/assets.store'
import { SAMPLERS, SCHEDULERS } from '../../constants'
import type {
  AssetItem,
  LatentReuseNoiseMode,
  ScenePanelTab,
  SeedMode,
  SeedRotationMode,
  StreamMotionMode,
  StreamRuntimePreset,
  StreamVaeMode,
} from '../../types'
import '../../components/rtd-number/rtd-number'
import '../../components/rtd-combo/rtd-combo'

// ── Helpers ───────────────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number, fallback: number) {
  if (!Number.isFinite(value)) return fallback
  return Math.max(min, Math.min(max, value))
}


function isStreamPresetLora(path: string) {
  return /lcm-lora-sdxl|pytorch_lora_weights\.safetensors|sdxl-lightning|sdxl_lightning/i.test(
    path.replace(/\\/g, '/'),
  )
}

function findLoraAsset(loras: AssetItem[], pattern: RegExp) {
  return loras.find((a) => pattern.test(`${a.name} ${a.path}`))?.path
}

function findSdxlTurboModel(models: AssetItem[]) {
  const norm = (a: AssetItem) => a.path.replace(/\\/g, '/').toLowerCase()
  return (
    models.find((a) => norm(a).endsWith('/sdxl-turbo') || norm(a).endsWith('/sd_xl_turbo'))?.path ??
    models.find((a) => a.name.toLowerCase() === 'sdxl-turbo')?.path ??
    models.find((a) => /stabilityai\/sdxl-turbo/i.test(`${a.name} ${a.path}`))?.path
  )
}

function formatBytes(bytes: number | undefined) {
  if (!bytes) return ''
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`
  return `${bytes} B`
}

// ── Component ──────────────────────────────────────────────────────────────────

@customElement('rtd-scene-panel')
export class RtdScenePanel extends LitElement {
  private _scene = new StoreController(this, $scene)
  private _stream = new StoreController(this, $stream)
  private _assets = new StoreController(this, $assets)

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

  static styles = css`
    :host {
      display: flex; flex-direction: column; height: 100%; overflow: hidden;
      font-size: var(--font-size, 12px);
      color: var(--fg, #d4d4d8);
      color-scheme: dark;
      background: var(--bg-panel, #24242c);
    }
    .context-body { flex: 1; overflow-y: auto; overflow-x: hidden; padding: 8px; }
    .layer-tab { display: flex; flex-direction: column; gap: var(--gap, 4px) }
    .field { display: flex; flex-direction: column; gap: 3px }
    .field > span, .field > label > span {
      font-size: 11px; font-weight: 600; color: var(--fg-label, #9898a8);
    }
    .field textarea, .field select,
    .field input[type=text], .field input[type=number], .field input[type=color] {
      font-family: inherit; font-size: var(--font-size, 12px);
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius, 2px); padding: var(--pad-y, 3px) var(--pad-x, 6px);
      background: var(--bg-input, #14141a);
      color: var(--fg, #d4d4d8);
      box-sizing: border-box;
    }
    .field textarea { resize: vertical; width: 100% }
    .field input[type=color] { padding: 1px 3px; width: 48px; height: 28px; cursor: pointer }
    .field.inline { display: grid; grid-template-columns: max-content minmax(0, 1fr); align-items: center; gap: var(--gap-lg, 8px) }
    .field.inline > span { white-space: nowrap }
    .field.inline select, .field.inline input { min-width: 0 }
    .check-field { display: flex; align-items: center; gap: 6px; cursor: pointer; color: var(--fg, #d4d4d8) }
    .seed-rotation-row { display: grid; grid-template-columns: max-content minmax(84px, 1fr) max-content; align-items: center; gap: 6px }
    .seed-rotation-row rtd-number { min-width: 84px }
    .sampling-controls { display: flex; flex-direction: column; gap: var(--gap, 4px) }
    .sampling-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--gap, 4px) }
    .popup-picker { position: relative }
    .picker-trigger {
      display: flex; flex-direction: column; align-items: flex-start; width: 100%;
      padding: var(--pad-y, 3px) var(--pad-x, 6px);
      background: var(--bg-input, #14141a);
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius, 2px); cursor: pointer; text-align: left;
      color: var(--fg, #d4d4d8);
    }
    .picker-trigger strong {
      font-size: var(--font-size, 12px); font-weight: 500; overflow: hidden;
      text-overflow: ellipsis; white-space: nowrap; max-width: 100%;
    }
    .picker-trigger span { font-size: 10px; color: var(--fg-dim, #7a7a8a) }
    .picker-trigger:hover { background: var(--bg-hover, #2e2e3c); border-color: var(--border-hi, #5a5a7a) }
    .advanced-sampling { display: flex; flex-direction: column; gap: var(--gap, 4px) }
    .button-grid { display: flex; gap: var(--gap, 4px); flex-wrap: wrap }
    .button-grid button {
      flex: 1; padding: 5px 10px;
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius, 2px); background: var(--bg-active, #383848);
      color: var(--fg, #d4d4d8); cursor: pointer; font-size: var(--font-size, 12px); font-weight: 500;
    }
    .button-grid button:hover { background: var(--bg-hover, #2e2e3c); border-color: var(--border-hi, #5a5a7a) }
    .button-grid button.stream {
      background: var(--accent, #4d87c4); color: var(--accent-fg, #ffffff);
      border-color: var(--accent, #4d87c4);
    }
    .button-grid button.stream:hover { filter: brightness(1.15) }
    .button-grid button:disabled { opacity: 0.45; cursor: not-allowed }
    .stream-status-line {
      font-size: 11px; padding: 3px 8px;
      background: var(--bg-header, #1e1e28);
      border-radius: var(--radius, 2px); color: var(--fg-dim, #7a7a8a);
      border: 1px solid var(--border, #3a3a4c);
    }
    .empty { font-size: 11px; color: var(--fg-dim, #7a7a8a); padding: 4px 2px; line-height: 1.4 }
    .resolution-grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap, 4px) }
    details.advanced-section {
      border-top: 1px solid var(--border, #3a3a4c); padding-top: 2px;
    }
    details.advanced-section summary {
      cursor: pointer; font-size: 11px; font-weight: 600;
      color: var(--fg-dim, #7a7a8a); padding: 4px 0; user-select: none;
    }
    details.advanced-section summary:hover { color: var(--fg, #d4d4d8) }
    .advanced-body { display: flex; flex-direction: column; gap: var(--gap, 4px); padding-top: 4px }
    .render-strategy-section .region-meta { font-size: 10px; color: var(--fg-dim, #7a7a8a); line-height: 1.4 }
    .schedule-timeline {
      display: flex; flex-direction: column; gap: 2px;
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius, 2px); padding: 4px 6px;
      background: var(--bg-header, #1e1e28);
    }
    .timeline-head {
      display: grid; grid-template-columns: 80px 1fr 1fr 1fr;
      font-size: 10px; color: var(--fg-dim, #7a7a8a); margin-bottom: 2px;
    }
    .timeline-row { display: grid; grid-template-columns: 80px 1fr; gap: 4px; align-items: center }
    .timeline-row span { font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--fg-label, #9898a8) }
    .timeline-track {
      position: relative; height: 8px;
      background: var(--border, #3a3a4c); border-radius: var(--radius, 2px);
    }
    .timeline-track i { position: absolute; top: 0; height: 100%; border-radius: var(--radius, 2px); min-width: 3px }
    .timeline-row.muted { opacity: 0.45 }
    .timeline-empty { font-size: 10px; color: var(--fg-dim, #7a7a8a); padding: 2px }
  `

  render() {
    return html`
      <div class="context-body">
        ${this._renderTab()}
      </div>
      ${this._renderPicker()}`
  }

  private _renderTab() {
    const { scenePanelTab } = this._stream.value
    if (scenePanelTab === 'streamdiffusion') return this._renderStreamDiffusionPanel()
    if (scenePanelTab === 'z-image') return this._renderZImagePanel()
    return this._renderSdxlPanel()
  }

  // ── SDXL tab ─────────────────────────────────────────────────────────────────

  private _renderSdxlPanel() {
    return html`<div class="layer-tab scene-tab">
      ${this._renderRendererStatus('sdxl')}
      <slot name="region-timeline"></slot>
      ${this._renderSceneGenerationControls()}
      <section class="advanced-sampling">
        ${this._renderTritonToggle()}
      </section>
      ${this._renderRenderStrategyPanel()}
      <div class="button-grid">
        <button class="stream" @click=${this._dispatchToggleStream}>
          ${this._stream.value.isStreaming ? 'Stop scene' : 'Generate scene'}
        </button>
      </div>
    </div>`
  }

  // ── Z-Image tab ───────────────────────────────────────────────────────────────

  private _renderZImagePanel() {
    return html`<div class="layer-tab scene-tab">
      ${this._renderRendererStatus('z-image')}
      <slot name="region-timeline"></slot>
      ${this._renderSceneGenerationControls()}
      <section class="advanced-sampling">
        ${this._renderTritonToggle()}
      </section>
      <div class="button-grid">
        <button class="stream" @click=${this._dispatchToggleStream}>
          ${this._stream.value.isStreaming ? 'Stop Z-Image' : 'Start Z-Image'}
        </button>
      </div>
    </div>`
  }

  // ── StreamDiffusion tab ───────────────────────────────────────────────────────

  private _renderStreamDiffusionPanel() {
    const st = this._stream.value
    const capability = this._rendererCapability('streamdiffusion')
    const native = capability?.native === true
    return html`<div class="layer-tab scene-tab">
      ${this._renderRendererStatus('streamdiffusion')}
      <slot name="region-timeline"></slot>
      <div class="empty" style="font-size:0.82em;line-height:1.4;padding:4px 2px 0;opacity:0.75">
        StreamDiffusion runs inference on every frame at 10–30 fps. Draw on the canvas to guide the output.
        Works with any SDXL model — pick a preset that matches your model, set a prompt, and click Start.
      </div>
      ${this._renderSamplingControls()}
      ${this._renderStreamSeedControls()}
      <section class="advanced-sampling">
        <label class="field inline" title="Choose a preset matching your model type.">
          <span>Model preset</span>
          <select .value=${st.runtimePreset}
            @change=${(e: Event) => this._applyStreamRuntimePreset((e.target as HTMLSelectElement).value)}>
            <option value="diffusers">SDXL (any model, slower)</option>
            <option value="lcm-lora-sdxl">SDXL + LCM LoRA (fast)</option>
            <option value="sdxl-lightning-4step">SDXL Lightning 4-step</option>
            <option value="sdxl-turbo">SDXL Turbo (fastest)</option>
          </select>
        </label>
        <label class="field inline">
          <span>Denoising</span>
          <select @change=${(e: Event) => {
            const v = (e.target as HTMLSelectElement).value
            if (v !== 'custom') $stream.setKey('timestepIndices', v)
          }}>
            <option value="45" ?selected=${st.timestepIndices === '45'}>Ultra-fast · 1 step</option>
            <option value="32,45" ?selected=${st.timestepIndices === '32,45'}>Fast · 2 steps</option>
            <option value="16,32,45" ?selected=${st.timestepIndices === '16,32,45'}>Balanced · 3 steps (recommended)</option>
            <option value="8,16,32,45" ?selected=${st.timestepIndices === '8,16,32,45'}>Quality · 4 steps (slower)</option>
            <option value="16,32" ?selected=${st.timestepIndices === '16,32'}>Detail-preserving · 2 steps</option>
            <option value="custom" ?selected=${!['45','32,45','16,32,45','8,16,32,45','16,32'].includes(st.timestepIndices)}>Custom…</option>
          </select>
        </label>
        <label class="field inline">
          <span>Indices</span>
          <input type="text" .value=${st.timestepIndices}
            @change=${(e: Event) => $stream.setKey('timestepIndices', (e.target as HTMLInputElement).value.trim())}
            style="width:100px" />
        </label>
        <rtd-number label="Frame buffer" .value=${st.frameBufferSize} min="1" max="4" step="1" decimals="0"
          @rtd-change=${(e: CustomEvent) => $stream.setKey('frameBufferSize', Math.round(clamp(e.detail.value, 1, 4, st.frameBufferSize)))}></rtd-number>
        <rtd-number label="Max passes" .value=${st.maxPasses} min="1" max="128" step="1" decimals="0"
          @rtd-change=${(e: CustomEvent) => $stream.setKey('maxPasses', Math.round(clamp(e.detail.value, 1, 128, st.maxPasses)))}></rtd-number>
        <label class="field inline">
          <span>Guidance mode</span>
          <select .value=${st.cfgType}
            @change=${(e: Event) => $stream.setKey('cfgType', (e.target as HTMLSelectElement).value)}>
            <option value="none">None (unconditioned)</option>
            <option value="self">Self-negative RCFG (recommended)</option>
            <option value="initialize">Onetime-negative RCFG</option>
            <option value="full">Full CFG (slowest)</option>
          </select>
        </label>
        ${this._renderOutputTransportControl()}
        <label class="field inline" title="Tiny VAE is faster; full VAE is usually sharper.">
          <span>VAE</span>
          <select .value=${st.vaeMode}
            @change=${(e: Event) => {
              const v = (e.target as HTMLSelectElement).value
              if (this._validStreamVaeMode(v)) $stream.setKey('vaeMode', v)
            }}>
            <option value="auto">Auto (env default)</option>
            <option value="tiny">Tiny VAE (faster)</option>
            <option value="full">Full VAE (higher quality)</option>
          </select>
        </label>
        ${this._renderSessionDirectoryControl()}
        ${this._renderTritonToggle()}
        <label class="field inline">
          <span>Auto-motion</span>
          <select .value=${st.motionMode}
            @change=${(e: Event) => {
              const v = (e.target as HTMLSelectElement).value
              if (this._validStreamMotionMode(v)) $stream.setKey('motionMode', v)
            }}>
            <option value="none">None (static)</option>
            <option value="sway">Sway</option>
            <option value="orbit">Orbit</option>
            <option value="push">Push</option>
            <option value="zoom">Zoom</option>
          </select>
        </label>
        <rtd-number label="Motion intensity" .value=${st.motionIntensity} min="0" max="2" step="0.01" decimals="2"
          @rtd-change=${(e: CustomEvent) => $stream.setKey('motionIntensity', clamp(e.detail.value, 0, 2, st.motionIntensity))}></rtd-number>
        <rtd-number label="Motion speed" .value=${st.motionSpeed} min="0.1" max="4" step="0.05" decimals="2"
          @rtd-change=${(e: CustomEvent) => $stream.setKey('motionSpeed', clamp(e.detail.value, 0.1, 4, st.motionSpeed))}></rtd-number>
        <div class="empty">Runtime: ${capability?.runtime ?? 'checking'}${native ? '' : ' (fallback)'}</div>
      </section>
      <div class="button-grid">
        <button class="stream" @click=${this._dispatchToggleStream} ?disabled=${st.status === 'connecting'}>
          ${st.status === 'connecting' ? 'Connecting…'
            : st.isStreaming ? 'Stop stream'
            : native ? 'Start StreamDiffusion' : 'Start (fallback mode)'}
        </button>
        ${st.isStreaming ? html`<div class="stream-status-line">
          ${st.status === 'connecting' ? '⏳ Connecting…'
            : (st.displayFps === 0 && st.fps === 0) ? '⚙ Loading model…'
            : `● Streaming – ${(st.displayFps || st.fps).toFixed(0)} FPS`}
        </div>` : ''}
      </div>
    </div>`
  }

  // ── Scene generation controls ─────────────────────────────────────────────────

  private _renderSceneGenerationControls() {
    const sc = this._scene.value
    const st = this._stream.value
    return html`
      ${this._renderDeviceSelect('Realtime GPU', st.realtimeDevice, (v) => $stream.setKey('realtimeDevice', v))}
      ${this._renderOutputTransportControl()}
      ${this._renderSessionDirectoryControl()}
      ${this._renderSamplingControls()}
      ${this._renderRealtimeSeedAndReuseControls()}
      ${this._renderTransparencyControls({
        enabled: sc.transparentBackground,
        onEnabled: (v) => $scene.setKey('transparentBackground', v),
        label: 'Transparent background',
        mode: sc.transparentMode,
        onMode: (v) => $scene.setKey('transparentMode', v),
        color: sc.transparentColor,
        onColor: (v) => $scene.setKey('transparentColor', v),
        tolerance: sc.transparentTolerance,
        onTolerance: (v) => $scene.setKey('transparentTolerance', v),
        alphaBlur: sc.transparentAlphaBlur,
        onAlphaBlur: (v) => $scene.setKey('transparentAlphaBlur', v),
        alphaThreshold: sc.transparentAlphaThreshold,
        onAlphaThreshold: (v) => $scene.setKey('transparentAlphaThreshold', v),
      })}`
  }

  private _renderRenderStrategyPanel() {
    const sc = this._scene.value
    return html`<details class="advanced-section render-strategy-section" open>
      <summary>Rendering engine</summary>
      <div class="advanced-body">
        <label class="field inline">
          <span>Strategy</span>
          <select .value=${sc.renderStrategy} @change=${(e: Event) => $scene.setKey('renderStrategy', (e.target as HTMLSelectElement).value as 'single' | 'per_layer' | 'tiled')}>
            <option value="single">Single pass</option>
            <option value="per_layer">Per layer detail pass</option>
            <option value="tiled">Tiled canvas pass</option>
          </select>
        </label>
        ${sc.renderStrategy === 'per_layer' ? html`
          <div class="region-meta">Each prompt mask is rendered as its own full-detail crop over the accumulated scene.</div>
          <rtd-number label="Layer padding" .value=${sc.layerBboxPadding} min="0" max="512" step="8" decimals="0"
            @rtd-change=${(e: CustomEvent) => $scene.setKey('layerBboxPadding', Math.round(clamp(e.detail.value, 0, 512, sc.layerBboxPadding)))}></rtd-number>
        ` : ''}
        ${sc.renderStrategy === 'tiled' ? html`
          <label class="field inline">
            <span>Tile grid</span>
            <select .value=${String(sc.tileDivisions)} @change=${(e: Event) => $scene.setKey('tileDivisions', Number((e.target as HTMLSelectElement).value))}>
              <option value="1">1 x 1</option>
              <option value="2">2 x 2</option>
              <option value="3">3 x 3</option>
              <option value="4">4 x 4</option>
            </select>
          </label>
          <rtd-number label="Tile overlap" .value=${sc.tileOverlap} min="0" max="512" step="8" decimals="0"
            @rtd-change=${(e: CustomEvent) => $scene.setKey('tileOverlap', Math.round(clamp(e.detail.value, 0, 512, sc.tileOverlap)))}></rtd-number>
        ` : ''}
      </div>
    </details>`
  }

  private _renderOutputTransportControl() {
    const st = this._stream.value
    return html`<label class="field inline" title="Video uses WebRTC codec compression; image sends per-frame encoded output.">
      <span>Output transport</span>
      <select .value=${st.outputTransport}
        @change=${(e: Event) => {
          const v = (e.target as HTMLSelectElement).value
          if (v === 'video' || v === 'image') $stream.setKey('outputTransport', v)
        }}>
        <option value="video">WebRTC video stream</option>
        <option value="image">Per-frame PNG image</option>
      </select>
    </label>`
  }

  // ── Sampling controls ─────────────────────────────────────────────────────────

  private _renderSamplingControls() {
    const sc = this._scene.value
    const assets = this._assets.value
    return html`<div class="sampling-controls" style="display:flex;flex-direction:column;gap:6px">
      <label class="field">
        <span>Prompt</span>
        <textarea rows="4" .value=${sc.prompt}
          @input=${(e: Event) => $scene.setKey('prompt', (e.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field">
        <span>Negative prompt</span>
        <textarea rows="3" .value=${sc.negativePrompt}
          @input=${(e: Event) => $scene.setKey('negativePrompt', (e.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      ${this._renderModelPicker(sc.selectedModel, assets.catalog.models)}
      <div class="sampling-grid">
        <rtd-number label="Denoise" .value=${sc.strength} min="0" max="0.999" step="0.01" decimals="2"
          title=${'Per-pixel inpaint strength. 1.0 = engine starts from pure noise (any painted RGBA/sketch is ignored as a starting image; only prompts drive the output). Lower values preserve more of the source image: ~0.85 keeps silhouettes, ~0.6 keeps shapes, ~0.4 only refines fine details. Override per pixel by painting the layer\'s Denoise mask.'}
          @rtd-change=${(e: CustomEvent) => $scene.setKey('strength', clamp(e.detail.value, 0, 0.999, sc.strength))}></rtd-number>
        <rtd-number label="CFG" .value=${sc.cfg} min="0" max="30" step="0.1" decimals="1"
          title=${'Classifier-free guidance scale. Higher = engine follows the prompt more strictly. ~6-9 is typical for SDXL. Override per pixel by painting the layer\'s CFG mask.'}
          @rtd-change=${(e: CustomEvent) => $scene.setKey('cfg', clamp(e.detail.value, 0, 30, sc.cfg))}></rtd-number>
        <rtd-number label="Steps" .value=${sc.steps} min="1" max="128" step="1" decimals="0"
          title=${'Number of denoising steps per frame. More steps = higher quality but slower; 20-30 is typical for SDXL, 4-8 for Turbo/LCM.'}
          @rtd-change=${(e: CustomEvent) => $scene.setKey('steps', Math.round(clamp(e.detail.value, 1, 128, sc.steps)))}></rtd-number>
      </div>
      <div class="field inline">
        <span>Sampler</span>
        <rtd-combo .value=${sc.sampler}
          .options=${SAMPLERS.map((s) => ({ label: s.label, value: s.value }))}
          @rtd-change=${(e: CustomEvent) => $scene.setKey('sampler', e.detail.value)}></rtd-combo>
      </div>
      <div class="field inline">
        <span>Scheduler</span>
        <rtd-combo .value=${sc.scheduler}
          .options=${SCHEDULERS.map((s) => ({ label: s.label, value: s.value }))}
          @rtd-change=${(e: CustomEvent) => $scene.setKey('scheduler', e.detail.value)}></rtd-combo>
      </div>
      ${this._renderLoraPicker(sc.selectedLoras, assets.catalog.loras)}
    </div>`
  }

  private _renderModelPicker(selectedPath: string, models: AssetItem[]) {
    const selected = models.find((m) => m.path === selectedPath)
    return html`<div class="popup-picker field">
      <button class="picker-trigger" type="button"
        @click=${(e: MouseEvent) => this._openModelDialog(e, selectedPath, models)}>
        <strong>${selected?.name ?? (selectedPath ? selectedPath.split(/[/\\]/).at(-1) : 'Default')}</strong>
        <span>Model${selected?.size ? ` · ${formatBytes(selected.size)}` : ''}</span>
      </button>
    </div>`
  }

  private _renderTritonToggle() {
    // Shared across StreamDiffusion / SDXL / Z-Image panels: all three backends
    // route through ``torch.compile`` for their UNet (StreamDiffusion gates on
    // ``stream_triton_compile``; SDXL/Z-Image read the same key via the engine
    // config). Disabling skips the 30–120 s first-run compile cost.
    const st = this._stream.value
    return html`<label class="field inline" title="Enable torch.compile (Triton) for faster inference. Disable to skip the 30-120s compile cost on first start.">
      <span>Triton compile</span>
      <input type="checkbox" .checked=${st.tritonCompile}
        @change=${(e: Event) => $stream.setKey('tritonCompile', (e.target as HTMLInputElement).checked)} />
    </label>`
  }

  private _renderSessionDirectoryControl() {
    const st = this._stream.value
    return html`<label class="field">
      <span>📁 Session directory</span>
      <input type="text" placeholder="empty = default runtime paths" .value=${st.sessionDirectory}
        @change=${(e: Event) => $stream.setKey('sessionDirectory', (e.target as HTMLInputElement).value.trim())} />
    </label>`
  }

  private _renderLoraPicker(selectedPaths: string[], loras: AssetItem[]) {
    const selectedNames = loras.filter((l) => selectedPaths.includes(l.path)).map((l) => l.name)
    const empty = loras.length === 0
    return html`<div class="popup-picker lora-picker field">
      <button class="picker-trigger" type="button"
        ?disabled=${empty}
        title=${empty ? 'No LoRAs discovered in your assets folder' : 'Pick LoRAs to apply'}
        @click=${(e: MouseEvent) => !empty && this._openLoraDialog(e, selectedPaths)}>
        <strong>${empty ? 'No LoRAs available' : (selectedNames.length ? selectedNames.slice(0, 2).join(', ') : 'None')}</strong>
        <span>${empty ? 'LoRAs' : (selectedNames.length > 2 ? `+${selectedNames.length - 2} LoRAs` : 'LoRAs')}</span>
      </button>
    </div>`
  }

  private _openModelDialog(e: MouseEvent, current: string, models: AssetItem[]) {
    this._pickerAnchorRect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    this._pickerType = 'options'
    this._pickerLabel = 'Model'
    this._pickerValue = current
    this._pickerOptions = models.map((m) => ({
      value: m.path,
      label: `${m.preferred ? '* ' : ''}${m.name}`,
      meta: m.size ? formatBytes(m.size) : undefined,
    }))
    this._pickerOnSelect = (v) => $scene.setKey('selectedModel', v)
    this._pickerSearch = ''
    this.requestUpdate()
  }

  private _openLoraDialog(e: MouseEvent, selected: string[]) {
    this._pickerAnchorRect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    this._pickerType = 'loras'
    this._pickerLabel = 'LoRAs'
    this._pickerLoraSelected = [...selected]
    this._pickerOnToggle = (path) => {
      const current = $scene.get().selectedLoras
      $scene.setKey('selectedLoras', current.includes(path)
        ? current.filter((p) => p !== path)
        : [...current, path])
    }
    this._pickerSearch = ''
    this.requestUpdate()
  }

  private _closePicker = () => {
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

  // ── Seed and reuse controls ───────────────────────────────────────────────────

  private _renderStreamSeedControls() {
    const sc = this._scene.value
    return html`<section class="advanced-sampling">
      <label class="field inline">
        <span>Seed mode</span>
        <select .value=${sc.seedMode}
          @change=${(e: Event) => {
            const v = (e.target as HTMLSelectElement).value
            if (this._validSeedMode(v)) $scene.setKey('seedMode', v as SeedMode)
          }}>
          <option value="fixed">Fixed</option>
          <option value="random">Random</option>
          <option value="increment">Increment</option>
          <option value="decrement">Decrement</option>
        </select>
      </label>
      <div class="seed-rotation-row">
        <label class="check-field">
          <input type="checkbox" .checked=${sc.seedRotationMode === 'interval'}
            @change=${(e: Event) => $scene.setKey('seedRotationMode', (e.target as HTMLInputElement).checked ? 'interval' : 'off' as SeedRotationMode)} />
          <span>Rotation every</span>
        </label>
        <rtd-number label="ms" .value=${sc.seedRotationIntervalMs} min="1" max="600000" step="100" decimals="0"
          ?disabled=${sc.seedRotationMode !== 'interval'}
          @rtd-change=${(e: CustomEvent) => $scene.setKey('seedRotationIntervalMs', Math.round(clamp(e.detail.value, 1, 600000, sc.seedRotationIntervalMs || 1000)))}></rtd-number>
        <label class="check-field">
          <input type="checkbox" .checked=${sc.seedRotationMode === 'frame'}
            @change=${(e: Event) => $scene.setKey('seedRotationMode', (e.target as HTMLInputElement).checked ? 'frame' : 'off' as SeedRotationMode)} />
          <span>Rotation every frame</span>
        </label>
      </div>
      <label class="field inline">
        <span>Seed</span>
        <input type="number" min="0" max="4294967295" step="1" placeholder="auto"
          .value=${sc.seed}
          @input=${(e: Event) => $scene.setKey('seed', (e.target as HTMLInputElement).value)} />
      </label>
    </section>`
  }

  private _renderRealtimeSeedAndReuseControls() {
    const sc = this._scene.value
    return html`<section class="advanced-sampling">
      <label class="field inline">
        <span>Seed mode</span>
        <select .value=${sc.seedMode}
          @change=${(e: Event) => {
            const v = (e.target as HTMLSelectElement).value
            if (this._validSeedMode(v)) $scene.setKey('seedMode', v as SeedMode)
          }}>
          <option value="fixed">Fixed</option>
          <option value="random">Random</option>
          <option value="increment">Increment</option>
          <option value="decrement">Decrement</option>
        </select>
      </label>
      <div class="seed-rotation-row">
        <label class="check-field">
          <input type="checkbox" .checked=${sc.seedRotationMode === 'interval'}
            @change=${(e: Event) => $scene.setKey('seedRotationMode', (e.target as HTMLInputElement).checked ? 'interval' : 'off' as SeedRotationMode)} />
          <span>Rotation every</span>
        </label>
        <rtd-number label="ms" .value=${sc.seedRotationIntervalMs} min="1" max="600000" step="100" decimals="0"
          ?disabled=${sc.seedRotationMode !== 'interval'}
          @rtd-change=${(e: CustomEvent) => $scene.setKey('seedRotationIntervalMs', Math.round(clamp(e.detail.value, 1, 600000, sc.seedRotationIntervalMs || 1000)))}></rtd-number>
        <label class="check-field">
          <input type="checkbox" .checked=${sc.seedRotationMode === 'frame'}
            @change=${(e: Event) => $scene.setKey('seedRotationMode', (e.target as HTMLInputElement).checked ? 'frame' : 'off' as SeedRotationMode)} />
          <span>Rotation every frame</span>
        </label>
      </div>
      <label class="field inline">
        <span>Seed</span>
        <input type="number" min="0" max="4294967295" step="1" placeholder="auto"
          .value=${sc.seed}
          @input=${(e: Event) => $scene.setKey('seed', (e.target as HTMLInputElement).value)} />
      </label>
      <label class="check-field">
        <input type="checkbox" .checked=${sc.reuseLatent}
          @change=${(e: Event) => $scene.setKey('reuseLatent', (e.target as HTMLInputElement).checked)} />
        <span>Reuse latent</span>
      </label>
      <rtd-number label="Reuse denoise" .value=${sc.reuseLatentDenoise} min="0" max="0.999" step="0.01" decimals="2"
        @rtd-change=${(e: CustomEvent) => $scene.setKey('reuseLatentDenoise', clamp(e.detail.value, 0, 0.999, sc.reuseLatentDenoise))}></rtd-number>
      <label class="field inline">
        <span>Perturbator</span>
        <select .value=${sc.reuseLatentNoiseMode} ?disabled=${!sc.reuseLatent}
          @change=${(e: Event) => {
            const v = (e.target as HTMLSelectElement).value
            if (this._validLatentReuseNoiseMode(v)) $scene.setKey('reuseLatentNoiseMode', v as LatentReuseNoiseMode)
          }}>
          <option value="none">None</option>
          <option value="fixed">Fixed noise</option>
          <option value="random">Random noise</option>
        </select>
      </label>
      <rtd-number label="Perturb noise" .value=${sc.reuseLatentNoise} min="0" max="1" step="0.01" decimals="2"
        @rtd-change=${(e: CustomEvent) => $scene.setKey('reuseLatentNoise', clamp(e.detail.value, 0, 1, sc.reuseLatentNoise))}></rtd-number>
      <div class="button-grid">
        <button type="button" ?disabled=${!sc.reuseLatent}
          @click=${() => this.dispatchEvent(new CustomEvent('rtd:restart-latent-reuse', { bubbles: true, composed: true }))}>
          Restart reuse
        </button>
      </div>
    </section>`
  }

  // ── Transparency controls ─────────────────────────────────────────────────────

  private _renderTransparencyControls(cfg: {
    enabled: boolean; onEnabled: (v: boolean) => void; label: string
    mode: string; onMode: (v: string) => void
    color: string; onColor: (v: string) => void
    tolerance: number; onTolerance: (v: number) => void
    alphaBlur: number; onAlphaBlur: (v: number) => void
    alphaThreshold: number; onAlphaThreshold: (v: number) => void
  }) {
    return html`<section class="advanced-sampling">
      <label class="check-field">
        <input type="checkbox" .checked=${cfg.enabled}
          @change=${(e: Event) => cfg.onEnabled((e.target as HTMLInputElement).checked)} />
        <span>${cfg.label}</span>
      </label>
      ${cfg.enabled ? html`
        <label class="field inline">
          <span>Alpha source</span>
          <select .value=${cfg.mode} @change=${(e: Event) => cfg.onMode((e.target as HTMLSelectElement).value)}>
            <option value="border">Sample border</option>
            <option value="color">Key color</option>
          </select>
        </label>
        ${cfg.mode === 'color' ? html`
          <label class="field inline">
            <span>Key color</span>
            <input type="color" .value=${cfg.color} @input=${(e: Event) => cfg.onColor((e.target as HTMLInputElement).value)} />
          </label>` : ''}
        <rtd-number label="Tolerance" .value=${cfg.tolerance} min="0" max="255" step="1" decimals="0"
          @rtd-change=${(e: CustomEvent) => cfg.onTolerance(Math.round(clamp(e.detail.value, 0, 255, cfg.tolerance)))}></rtd-number>
        <rtd-number label="Alpha blur" .value=${cfg.alphaBlur} min="0" max="24" step="0.5" decimals="1"
          @rtd-change=${(e: CustomEvent) => cfg.onAlphaBlur(clamp(e.detail.value, 0, 24, cfg.alphaBlur))}></rtd-number>
        <rtd-number label="Alpha floor" .value=${cfg.alphaThreshold} min="0" max="255" step="1" decimals="0"
          @rtd-change=${(e: CustomEvent) => cfg.onAlphaThreshold(Math.round(clamp(e.detail.value, 0, 255, cfg.alphaThreshold)))}></rtd-number>
      ` : ''}
    </section>`
  }

  // ── Device select ─────────────────────────────────────────────────────────────

  private _renderDeviceSelect(label: string, value: string, onChange: (v: string) => void) {
    const devices = this._assets.value.gpuDevices
    return html`<label class="field inline">
      <span>${label}</span>
      <select .value=${value} @change=${(e: Event) => onChange((e.target as HTMLSelectElement).value)}>
        ${devices.map((d) => html`<option value=${d.id}>${d.name}${d.memory_total ? ` (${formatBytes(d.memory_total)})` : ''}</option>`)}
      </select>
    </label>`
  }

  // ── Renderer status ───────────────────────────────────────────────────────────

  private _rendererCapability(renderer: ScenePanelTab) {
    return this._assets.value.rendererCapabilities?.renderers?.[renderer]
  }

  private _renderRendererStatus(renderer: ScenePanelTab) {
    const cap = this._rendererCapability(renderer)
    if (!cap) return html`<div class="empty">Renderer status pending</div>`
    const state = cap.configured ? (cap.native ? 'native' : 'fallback') : 'missing'
    const flags = [cap.streamable ? 'streamable' : '', cap.realtime ? 'realtime' : '', cap.transport].filter(Boolean).join(' / ')
    return html`<div class="empty">${state} | ${cap.runtime} | ${flags}<br />${cap.message}</div>`
  }

  // ── Stream runtime preset ─────────────────────────────────────────────────────

  private _applyStreamRuntimePreset(value: string) {
    if (!this._isValidStreamRuntimePreset(value)) return
    const isStreaming = this._stream.value.isStreaming
    if (isStreaming) {
      this.dispatchEvent(new CustomEvent('rtd:stop-stream', { bubbles: true, composed: true }))
    }
    const catalog = this._assets.value.catalog
    $stream.setKey('runtimePreset', value as StreamRuntimePreset)
    const currentLoras = this._scene.value.selectedLoras
    if (value === 'diffusers') {
      $scene.setKey('selectedLoras', currentLoras.filter((p) => !isStreamPresetLora(p)))
      return
    }
    if (value === 'lcm-lora-sdxl') {
      const loraPath = findLoraAsset(catalog.loras, /lcm.*sdxl|lcm-lora-sdxl/i)
        ?? 'D:/comfyui/comfy/models/loras/lcm-lora-sdxl/pytorch_lora_weights.safetensors'
      $scene.setKey('selectedLoras', [...currentLoras.filter((p) => !isStreamPresetLora(p)), loraPath])
      $scene.setKey('sampler', 'lcm')
      $scene.setKey('scheduler', 'simple')
      $scene.setKey('steps', 4)
      $scene.setKey('cfg', 1.2)
      $scene.setKey('strength', 0.7)
      $stream.setKey('timestepIndices', '16,32')
      return
    }
    if (value === 'sdxl-lightning-4step') {
      const loraPath = findLoraAsset(catalog.loras, /lightning.*4step|sdxl_lightning_4step/i)
        ?? 'D:/comfyui/comfy/models/loras/SDXL-Lightning/sdxl_lightning_4step_lora.safetensors'
      $scene.setKey('selectedLoras', [...currentLoras.filter((p) => !isStreamPresetLora(p)), loraPath])
      $scene.setKey('scheduler', 'euler')
      $scene.setKey('steps', 4)
      $scene.setKey('cfg', 1.0)
      $scene.setKey('strength', 0.6)
      $stream.setKey('timestepIndices', '16,32')
      return
    }
    // sdxl-turbo
    const turbo = findSdxlTurboModel(catalog.models)
      ?? 'D:/comfyui/comfy/models/diffusers/sdxl-turbo'
    $scene.setKey('selectedModel', turbo)
    $scene.setKey('selectedLoras', currentLoras.filter((p) => !isStreamPresetLora(p)))
    $scene.setKey('sampler', 'euler')
    $scene.setKey('scheduler', 'simple')
    $scene.setKey('steps', 1)
    $scene.setKey('cfg', 1.0)
    $scene.setKey('strength', 0.5)
    $stream.setKey('timestepIndices', '32,45')
  }

  // ── Event dispatch ────────────────────────────────────────────────────────────

  private _dispatchToggleStream = () => {
    this.dispatchEvent(new CustomEvent('rtd:toggle-stream', { bubbles: true, composed: true }))
  }

  // ── Validators ────────────────────────────────────────────────────────────────

  private _isValidStreamRuntimePreset(v: string): v is StreamRuntimePreset {
    return ['diffusers', 'lcm-lora-sdxl', 'sdxl-lightning-4step', 'sdxl-turbo'].includes(v)
  }

  private _validStreamMotionMode(v: string): v is StreamMotionMode {
    return ['none', 'sway', 'orbit', 'push', 'zoom'].includes(v)
  }

  private _validStreamVaeMode(v: string): v is StreamVaeMode {
    return ['auto', 'tiny', 'full'].includes(v)
  }

  private _validSeedMode(v: string): v is SeedMode {
    return ['fixed', 'random', 'increment', 'decrement'].includes(v)
  }

  private _validLatentReuseNoiseMode(v: string): v is LatentReuseNoiseMode {
    return ['none', 'fixed', 'random'].includes(v)
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-scene-panel': RtdScenePanel }
}
