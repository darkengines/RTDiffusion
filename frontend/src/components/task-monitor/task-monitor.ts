import { LitElement, css, html } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { StoreController } from '@nanostores/lit'
import { $layer } from '../../stores/layer.store'
import { backendUrl } from '../../services/api'
import type { GenerationTask, MotionTaskProgress } from '../../types'

@customElement('rtd-task-monitor')
export class RtdTaskMonitor extends LitElement {
  private _layer = new StoreController(this, $layer)

  @state() private open = false

  static styles = css`
    :host { display: contents }
    .footer-task-host { position: relative; display: flex; align-items: center; gap: 6px }
    .footer-task-button { display: flex; gap: 8px; align-items: center; background: none; border: none; cursor: pointer; padding: 4px 8px; font-size: 12px; border-radius: 4px }
    .footer-task-button:hover { background: var(--hover-bg, rgba(0,0,0,.06)) }
    .footer-task-bar { flex: 1; height: 3px; background: var(--accent-dim, #b0b4e8); border-radius: 2px; overflow: hidden; min-width: 60px }
    .footer-task-bar span { display: block; height: 100%; background: var(--accent, #5f6fff); transition: width 0.2s }
    .task-popover { position: absolute; bottom: calc(100% + 6px); left: 0; width: 320px; max-height: 480px; overflow-y: auto; background: var(--surface, #fff); border: 1px solid var(--border, #ccc); border-radius: 6px; box-shadow: 0 4px 16px rgba(0,0,0,.2); z-index: 200 }
    .task-popover-head { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-bottom: 1px solid var(--border, #ccc); font-size: 13px; font-weight: 600 }
    .task-popover-head button { background: none; border: none; cursor: pointer; font-size: 16px; line-height: 1 }
    .task-popover-list { padding: 8px; display: flex; flex-direction: column; gap: 6px }
    .task-empty { padding: 12px; font-size: 12px; color: var(--muted, #888); text-align: center }
    .task-item { padding: 8px; border-radius: 4px; background: var(--surface-2, #f5f5f5) }
    .task-item.error-task { background: var(--error-bg, #fff0f0) }
    .task-row { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px }
    .task-row strong { font-weight: 600 }
    .task-bar { height: 4px; background: var(--accent-dim, #dde); border-radius: 2px; overflow: hidden; margin: 4px 0 }
    .task-bar span { display: block; height: 100%; background: var(--accent, #5f6fff); transition: width 0.3s }
    .task-detail { color: var(--muted, #888) }
    .task-item p { margin: 4px 0 0; font-size: 11px; word-break: break-all; color: var(--error, #c33) }
    .motion-frame-strip { display: flex; gap: 4px; margin-top: 6px; overflow-x: auto }
    .motion-frame-strip img { height: 48px; width: auto; border-radius: 2px; object-fit: cover }
  `

  render() {
    const { generationTasks } = this._layer.value
    const pending = generationTasks.filter((t) => t.status === 'queued' || t.status === 'running')
    const queued = pending.filter((t) => t.status === 'queued' || t.message === 'Waiting for GPU worker')
    const running = pending.length - queued.length
    const progress = pending.length
      ? pending.reduce((sum, t) => sum + (t.progress || 0), 0) / pending.length
      : generationTasks[0]?.progress ?? 0
    const percent = Math.round(progress * 100)

    return html`<div class="footer-task-host">
      <button class="footer-task-button" @click=${() => (this.open = !this.open)}>
        <span>Tasks ${pending.length}</span>
        <span>Queued ${queued.length}</span>
        <span>Running ${running}</span>
        <span>${percent}%</span>
      </button>
      <div class="footer-task-bar"><span style="width:${percent}%"></span></div>
      ${this.open ? html`<div class="task-popover">
        <div class="task-popover-head">
          <strong>Tasks</strong>
          <button @click=${() => (this.open = false)}>×</button>
        </div>
        ${generationTasks.length
          ? html`<div class="task-popover-list">${generationTasks.map((t) => this._renderTask(t))}</div>`
          : html`<div class="task-empty">No tasks yet</div>`}
      </div>` : ''}
    </div>`
  }

  private _renderTask(task: GenerationTask) {
    const percent = Math.round((task.progress || 0) * 100)
    return html`<div class=${task.status === 'error' ? 'task-item error-task' : 'task-item'}>
      <div class="task-row"><strong>${task.label}</strong><span>${task.status}</span></div>
      <div class="task-bar"><span style="width:${percent}%"></span></div>
      <div class="task-row task-detail"><span>${task.phase}</span><span>${percent}%</span></div>
      ${task.error ? html`<p>${task.error}</p>` : task.message ? html`<p style="color:inherit">${task.message}</p>` : ''}
    </div>`
  }

  renderMotionTask(task: MotionTaskProgress) {
    const percent = Math.round((task.progress || 0) * 100)
    const frames = task.preview_frames || task.result?.frames || []
    return html`<div class=${task.status === 'error' ? 'task-item error-task' : 'task-item'}>
      <div class="task-row"><strong>Motion</strong><span>${task.status}</span></div>
      <div class="task-bar"><span style="width:${percent}%"></span></div>
      <div class="task-row task-detail"><span>${task.phase}</span><span>${percent}%</span></div>
      ${task.error ? html`<p>${task.error}</p>` : ''}
      ${frames.length ? html`<div class="motion-frame-strip">
        ${frames.slice(-6).map((f) => html`<img src=${backendUrl(f.url, { t: String(Date.now()) })} alt="frame ${f.index + 1}" />`)}
      </div>` : ''}
    </div>`
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-task-monitor': RtdTaskMonitor }
}
