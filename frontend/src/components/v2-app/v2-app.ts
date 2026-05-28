import { css, html, LitElement } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { Layer } from '../../model/layer'
import { Scene } from '../../model/scene'
import { createCanvasPngEncoder } from '../../model/encoder'
import type { ChannelBind, PromptBind } from '../../model/types'
import '../../paint/canvas-host'
import { SceneSender, type SendTransport } from '../../stream/loop'

/**
 * Minimal v2 app shell. Demonstrates the full new pipeline end-to-end:
 * Scene + Layer model -> canvas-host paint -> aggregator -> encoder -> SceneSender.
 *
 * Intentionally bare -- the real UI gets ported on top of this scaffold.
 * No Fabric.js, no nanostore: the Scene's monotonic version is the single
 * source of reactivity; the component subscribes via canvas-host's
 * rtd-scene-changed event to drive its own re-renders.
 */
@customElement('rtd-v2-app')
export class RtdV2App extends LitElement {
  static styles = css`
    :host { display: grid; grid-template-columns: 240px 1fr; height: 100%;
            color: #ddd; background: #1a1a1a; font: 13px/1.4 system-ui; }
    aside { padding: 12px; border-right: 1px solid #333; overflow: auto; }
    main { display: grid; place-items: center; padding: 12px; }
    rtd-canvas-host { width: 100%; max-width: 768px; aspect-ratio: 1;
                      background: repeating-conic-gradient(#444 0% 25%, #555 0% 50%) 0/16px 16px;
                      border: 1px solid #333; }
    h3 { margin: 12px 0 6px; font-size: 12px; text-transform: uppercase;
         letter-spacing: 0.05em; color: #888; }
    button { background: #2a2a2a; color: #ddd; border: 1px solid #444;
             padding: 4px 8px; border-radius: 3px; cursor: pointer; }
    button:hover { background: #333; }
    button.active { background: #4a6; color: #fff; border-color: #4a6; }
    .layer-row { display: flex; gap: 4px; align-items: center; padding: 4px;
                 border-radius: 3px; cursor: pointer; }
    .layer-row.active { background: #2d3e2d; }
    label { display: block; margin: 4px 0; }
    input[type="range"], input[type="text"], input[type="number"], select {
      width: 100%; background: #222; color: #ddd; border: 1px solid #333;
      padding: 2px 4px; border-radius: 2px;
    }
    .row { display: flex; gap: 6px; align-items: center; }
    .row > * { flex: 1; }
    .status { color: #888; font-size: 11px; margin-top: 8px; }
  `

  @state() private _scene = this._createInitialScene()
  @state() private _activeLayerId = ''
  @state() private _activeChannel: 'rgba' | 'cfg' | 'denoise' | { kind: 'prompt'; index: number } = 'rgba'
  @state() private _activeTool: 'brush' | 'eraser' = 'brush'
  @state() private _brushSize = 24
  @state() private _brushHardness = 0.8
  @state() private _brushColor = '#cccccc'
  @state() private _brushMaskValue = 255
  @state() private _sentCount = 0
  @state() private _lastSentAt = 0

  private _sender: SceneSender | null = null

  connectedCallback(): void {
    super.connectedCallback()
    if (this._scene.layers.length > 0) {
      this._activeLayerId = this._scene.layers[0].id
    }
  }

  disconnectedCallback(): void {
    super.disconnectedCallback()
    this._sender?.stop()
  }

  private _createInitialScene(): Scene {
    const s = new Scene({ width: 512, height: 512, baseCfg: 1.5, baseDenoise: 0.6 })
    s.addLayer(new Layer({ id: 'layer-1', width: 512, height: 512 }))
    return s
  }

  private get _activeLayer(): Layer | null {
    return this._scene.findLayer(this._activeLayerId) ?? null
  }

  private _onSceneChanged(): void {
    // Trigger re-render so the layer list updates after paint mutations.
    this.requestUpdate()
  }

  private _addLayer(): void {
    const id = `layer-${this._scene.layers.length + 1}-${Date.now().toString(36)}`
    this._scene.addLayer(new Layer({ id, width: this._scene.width, height: this._scene.height }))
    this._activeLayerId = id
    this.requestUpdate()
  }

  private _removeLayer(id: string): void {
    this._scene.removeLayer(id)
    if (this._activeLayerId === id) {
      this._activeLayerId = this._scene.layers[0]?.id ?? ''
    }
    this.requestUpdate()
  }

  private _toggleVisible(layer: Layer): void {
    layer.setVisible(!layer.visible)
    this.requestUpdate()
  }

  private _addPrompt(): void {
    const l = this._activeLayer
    if (!l) return
    l.addPrompt({ text: '', bind: 'rgba' })
    this.requestUpdate()
  }

  private _startSender(): void {
    if (this._sender) return
    // captureStream-friendly factory: one offscreen canvas reused per encode
    const cvs = document.createElement('canvas')
    const encoder = createCanvasPngEncoder(() => cvs)
    // Demo transport: counts payloads + logs latest size. Real wiring lives
    // in stream.service.ts (out of scope for phase 6 minimum viable).
    const transport: SendTransport = {
      send: (payload) => {
        this._sentCount += 1
        this._lastSentAt = performance.now()
        if (this._sentCount === 1) {
          // eslint-disable-next-line no-console
          console.info('v2 send sample', {
            width: payload.width,
            height: payload.height,
            promptCount: payload.prompts.length,
            controlnetCount: payload.controlnet.length,
            rgbaBytes: payload.rgba_b64?.length ?? 0,
          })
        }
        this.requestUpdate()
        return true
      },
    }
    this._sender = new SceneSender({ scene: this._scene, encoder, transport })
    this._sender.start()
  }

  private _stopSender(): void {
    this._sender?.stop()
    this._sender = null
    this.requestUpdate()
  }

  private _channelLabel(channel: typeof this._activeChannel): string {
    if (channel === 'rgba') return 'RGBA'
    if (channel === 'cfg') return 'CFG mask'
    if (channel === 'denoise') return 'Denoise mask'
    const l = this._activeLayer
    const p = l?.prompts[channel.index]
    return `Prompt ${channel.index}: ${p?.text || '(empty)'}`
  }

  private _channelOptions() {
    const channels: Array<typeof this._activeChannel> = ['rgba', 'cfg', 'denoise']
    const l = this._activeLayer
    if (l) for (let i = 0; i < l.prompts.length; i++) channels.push({ kind: 'prompt', index: i })
    return channels
  }

  private _renderLayerList() {
    return html`
      <h3>Layers</h3>
      ${this._scene.layers.map(l => html`
        <div class="layer-row ${l.id === this._activeLayerId ? 'active' : ''}"
             @click=${() => { this._activeLayerId = l.id }}>
          <button @click=${(e: Event) => { e.stopPropagation(); this._toggleVisible(l) }}
                  title=${l.visible ? 'hide' : 'show'}>${l.visible ? '👁' : '·'}</button>
          <span style="flex:1">${l.id}</span>
          <button @click=${(e: Event) => { e.stopPropagation(); this._removeLayer(l.id) }} title="remove">✕</button>
        </div>
      `)}
      <button @click=${this._addLayer} style="width:100%; margin-top:4px;">+ add layer</button>
    `
  }

  private _renderChannelControls() {
    const l = this._activeLayer
    if (!l) return html`<p class="status">No active layer.</p>`
    return html`
      <h3>Active channel</h3>
      <select @change=${(e: Event) => {
        const v = (e.target as HTMLSelectElement).value
        if (v === 'rgba' || v === 'cfg' || v === 'denoise') this._activeChannel = v
        else this._activeChannel = { kind: 'prompt', index: Number(v.slice('prompt:'.length)) }
      }}>
        ${this._channelOptions().map(c => {
          const value = typeof c === 'string' ? c : `prompt:${c.index}`
          const selected = (typeof c === 'string' && c === this._activeChannel) ||
                           (typeof c === 'object' && typeof this._activeChannel === 'object' &&
                            c.index === this._activeChannel.index)
          return html`<option value=${value} ?selected=${selected}>${this._channelLabel(c)}</option>`
        })}
      </select>
      ${this._activeChannel === 'cfg' || this._activeChannel === 'denoise' ? html`
        <label>${this._activeChannel} scalar
          <input type="number" min="0" max=${this._activeChannel === 'cfg' ? '30' : '1'}
                 step="0.1" .value=${String(this._activeChannel === 'cfg' ? l.cfg : l.denoise)}
                 @input=${(e: Event) => {
                   const v = Number((e.target as HTMLInputElement).value)
                   if (this._activeChannel === 'cfg') l.setCfg(v)
                   else l.setDenoise(v)
                   this.requestUpdate()
                 }} />
        </label>
        <label>${this._activeChannel} bind
          <select @change=${(e: Event) => {
            const v = (e.target as HTMLSelectElement).value as ChannelBind
            if (this._activeChannel === 'cfg') l.setCfgBind(v)
            else l.setDenoiseBind(v)
            this.requestUpdate()
          }}>
            ${(['self', 'prompt', 'rgba', 'none'] as ChannelBind[]).map(b => {
              const cur = this._activeChannel === 'cfg' ? l.cfgBind : l.denoiseBind
              return html`<option value=${b} ?selected=${b === cur}>${b}</option>`
            })}
          </select>
        </label>
      ` : ''}
    `
  }

  private _renderPrompts() {
    const l = this._activeLayer
    if (!l) return ''
    return html`
      <h3>Prompts</h3>
      ${l.prompts.map((p, i) => html`
        <div style="border:1px solid #333; padding:4px; margin-bottom:4px; border-radius:3px;">
          <input type="text" .value=${p.text} placeholder="prompt text"
                 @input=${(e: Event) => { l.updatePrompt(i, { text: (e.target as HTMLInputElement).value }); this.requestUpdate() }} />
          <input type="text" .value=${p.negative} placeholder="negative"
                 @input=${(e: Event) => { l.updatePrompt(i, { negative: (e.target as HTMLInputElement).value }); this.requestUpdate() }} />
          <div class="row" style="margin-top:4px;">
            <select @change=${(e: Event) => {
              l.updatePrompt(i, { bind: (e.target as HTMLSelectElement).value as PromptBind })
              this.requestUpdate()
            }}>
              ${(['self', 'rgba', 'none'] as PromptBind[]).map(b =>
                html`<option value=${b} ?selected=${b === p.bind}>bind: ${b}</option>`)}
            </select>
            <button @click=${() => { l.removePrompt(i); this.requestUpdate() }}>✕</button>
          </div>
        </div>
      `)}
      <button @click=${this._addPrompt} style="width:100%;">+ add prompt</button>
    `
  }

  private _renderBrushControls() {
    return html`
      <h3>Brush</h3>
      <div class="row">
        <button class=${this._activeTool === 'brush' ? 'active' : ''}
                @click=${() => { this._activeTool = 'brush' }}>brush</button>
        <button class=${this._activeTool === 'eraser' ? 'active' : ''}
                @click=${() => { this._activeTool = 'eraser' }}>eraser</button>
      </div>
      <label>size: ${this._brushSize}
        <input type="range" min="1" max="128" step="1" .value=${String(this._brushSize)}
               @input=${(e: Event) => { this._brushSize = Number((e.target as HTMLInputElement).value) }} />
      </label>
      <label>hardness: ${this._brushHardness.toFixed(2)}
        <input type="range" min="0" max="1" step="0.05" .value=${String(this._brushHardness)}
               @input=${(e: Event) => { this._brushHardness = Number((e.target as HTMLInputElement).value) }} />
      </label>
      ${this._activeChannel === 'rgba' ? html`
        <label>color
          <input type="color" .value=${this._brushColor}
                 @input=${(e: Event) => { this._brushColor = (e.target as HTMLInputElement).value }} />
        </label>
      ` : html`
        <label>mask value: ${this._brushMaskValue}
          <input type="range" min="0" max="255" step="1" .value=${String(this._brushMaskValue)}
                 @input=${(e: Event) => { this._brushMaskValue = Number((e.target as HTMLInputElement).value) }} />
        </label>
      `}
    `
  }

  private _brushSettings() {
    const c = this._brushColor
    const r = parseInt(c.slice(1, 3), 16)
    const g = parseInt(c.slice(3, 5), 16)
    const b = parseInt(c.slice(5, 7), 16)
    return {
      size: this._brushSize,
      hardness: this._brushHardness,
      color: { r, g, b, a: 255 },
      maskValue: this._brushMaskValue,
    }
  }

  render() {
    return html`
      <aside>
        <h3>Scene</h3>
        <label>base prompt
          <input type="text" .value=${this._scene.basePrompt}
                 @input=${(e: Event) => { this._scene.setBasePrompt((e.target as HTMLInputElement).value); this.requestUpdate() }} />
        </label>
        <label>base cfg: ${this._scene.baseCfg.toFixed(2)}
          <input type="range" min="0" max="20" step="0.1" .value=${String(this._scene.baseCfg)}
                 @input=${(e: Event) => { this._scene.setBaseCfg(Number((e.target as HTMLInputElement).value)); this.requestUpdate() }} />
        </label>
        <label>base denoise: ${this._scene.baseDenoise.toFixed(2)}
          <input type="range" min="0" max="1" step="0.01" .value=${String(this._scene.baseDenoise)}
                 @input=${(e: Event) => { this._scene.setBaseDenoise(Number((e.target as HTMLInputElement).value)); this.requestUpdate() }} />
        </label>
        ${this._renderLayerList()}
        ${this._renderChannelControls()}
        ${this._renderPrompts()}
        ${this._renderBrushControls()}
        <h3>Send loop</h3>
        <div class="row">
          ${this._sender
            ? html`<button @click=${this._stopSender}>stop</button>`
            : html`<button @click=${this._startSender}>start</button>`}
        </div>
        <div class="status">
          version: ${this._scene.version}<br>
          sent: ${this._sentCount}${this._lastSentAt
            ? html` &middot; ${Math.round(performance.now() - this._lastSentAt)} ms ago` : ''}
        </div>
      </aside>
      <main>
        <rtd-canvas-host
          .scene=${this._scene}
          .activeLayerId=${this._activeLayerId}
          .activeChannel=${this._activeChannel}
          .activeTool=${this._activeTool}
          .brush=${this._brushSettings()}
          @rtd-scene-changed=${this._onSceneChanged}
        ></rtd-canvas-host>
      </main>
    `
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'rtd-v2-app': RtdV2App
  }
}
