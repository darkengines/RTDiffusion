import { LitElement, css, html } from 'lit'
import { customElement } from 'lit/decorators.js'
import { StoreController } from '@nanostores/lit'
import { $stream } from '../../stores/stream.store'
import { getDebugChannelData, getDebugSurfaces, type DebugSurface } from '../../services/stream.service'
import '../debug-surface/debug-surface'

@customElement('rtd-debug-overlay')
export class RtdDebugOverlay extends LitElement {
  private _stream = new StoreController(this, $stream)

  static styles = css`
    :host { display: block; width: 100%; height: 100% }
    .debug-overlay { display: flex; flex-direction: column; width: 100%; height: 100%; background: #111; color: #eee; overflow: hidden }
    .debug-overlay-empty { justify-content: center; align-items: center; font-size: 13px; color: #666 }
    .debug-overlay-toolbar { display: flex; gap: 8px; align-items: center; padding: 6px 10px; background: #1a1a1a; border-bottom: 1px solid #333; font-size: 12px; flex-shrink: 0 }
    .debug-overlay-toolbar button { background: #333; border: none; color: #eee; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px }
    .debug-overlay-toolbar button:hover { background: #444 }
    .debug-channel-label { font-weight: 600; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap }
    .debug-mosaic-grid { flex: 1; overflow-y: auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 4px; padding: 4px }
    .debug-mosaic-cell { position: relative; background: #000; border-radius: 4px; overflow: hidden; cursor: pointer; aspect-ratio: 16/9 }
    .debug-mosaic-cell:hover { outline: 1px solid var(--accent, #5f6fff) }
    .debug-mosaic-cell rtd-debug-surface { width: 100%; height: 100%; display: block }
    .debug-cell-label { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,.65); font-size: 10px; padding: 2px 4px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap }
    .debug-text-cell { aspect-ratio: unset; min-height: 80px; display: flex; flex-direction: column; justify-content: flex-end }
    .debug-tag-content { padding: 4px; flex: 1; display: flex; flex-wrap: wrap; gap: 3px; align-content: flex-start }
    .debug-tag-chip { background: #2a7a3a; color: #fff; border-radius: 3px; padding: 2px 6px; font-size: 10px }
    .debug-tag-empty { color: #555; font-size: 10px; padding: 4px }
    .debug-preview-wrapper { flex: 1; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #000 }
    .debug-preview-full { width: 100%; height: 100% }
    .debug-preview-text { flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-wrap: wrap; gap: 4px; align-content: flex-start }
  `

  render() {
    const { debugChannelNames: names, debugMosaicMode, debugPreviewChannel } = this._stream.value
    if (!names.length) {
      return html`<div class="debug-overlay debug-overlay-empty"><span>Debug streams: waiting for frames…</span></div>`
    }
    if (!debugMosaicMode && debugPreviewChannel) {
      const channelData = getDebugChannelData()
      const val = channelData.get(debugPreviewChannel) ?? ''
      const isText = this._isTextChannel(debugPreviewChannel)
      const surface = this._surfaceFor(debugPreviewChannel, val)
      return html`<div class="debug-overlay">
        <div class="debug-overlay-toolbar">
          <button @click=${() => $stream.setKey('debugMosaicMode', true)}>← Mosaic</button>
          <span class="debug-channel-label">${debugPreviewChannel}</span>
        </div>
        ${isText
          ? html`<div class="debug-preview-text">
              ${(val || '').split(',').map((t) => t.trim()).filter(Boolean).map((tag) => html`<span class="debug-tag-chip">${tag}</span>`)}
            </div>`
          : html`<div class="debug-preview-wrapper">${surface ? html`<rtd-debug-surface class="debug-preview-full" .surface=${surface}></rtd-debug-surface>` : html`<span class="debug-tag-empty">stream pending...</span>`}</div>`}
      </div>`
    }
    return html`<div class="debug-overlay">
      <div class="debug-overlay-toolbar">
        <span>${names.length} channel${names.length === 1 ? '' : 's'}</span>
        <button @click=${() => {
          $stream.setKey('debugStreamsEnabled', false)
          $stream.setKey('debugChannelNames', [])
        }}>Close</button>
      </div>
      <div class="debug-mosaic-grid">
        ${names.map((name) => this._renderCell(name))}
      </div>
    </div>`
  }

  private _renderCell(name: string) {
    const channelData = getDebugChannelData()
    const val = channelData.get(name) ?? ''
    const isText = this._isTextChannel(name)
    if (isText) {
      const tags = val ? val.split(',').map((t) => t.trim()).filter(Boolean) : []
      return html`<div class="debug-mosaic-cell debug-text-cell" title=${name}>
        <div class="debug-tag-content">
          ${tags.length ? tags.map((tag) => html`<span class="debug-tag-chip">${tag}</span>`) : html`<span class="debug-tag-empty">waiting…</span>`}
        </div>
        <span class="debug-cell-label">${name}</span>
      </div>`
    }
    return html`<div class="debug-mosaic-cell" title=${name}
      @click=${() => { $stream.setKey('debugPreviewChannel', name); $stream.setKey('debugMosaicMode', false) }}>
      ${this._renderSurface(name, val)}
      <span class="debug-cell-label">${name}</span>
    </div>`
  }

  private _renderSurface(name: string, value: string) {
    const surface = this._surfaceFor(name, value)
    return surface ? html`<rtd-debug-surface .surface=${surface}></rtd-debug-surface>` : html`<span class="debug-tag-empty">stream pending...</span>`
  }

  private _surfaceFor(name: string, value: string): DebugSurface | undefined {
    const surfaces = getDebugSurfaces()
    if (value.startsWith('stream:')) return surfaces.get(value.slice('stream:'.length)) ?? surfaces.get(name)
    return surfaces.get(name)
  }

  private _isTextChannel(name: string) {
    return name.startsWith('tagger:') || name.startsWith('meta:')
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-debug-overlay': RtdDebugOverlay }
}
