import { LitElement, css, html } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export interface ComboOption { label: string; value: string; meta?: string }

/**
 * DearImGui-style combo box (searchable dropdown select).
 *
 * Events:
 *   rtd-change: CustomEvent<{ value: string }>
 *
 * Usage:
 *   <rtd-combo .value=${v} .options=${opts} searchable
 *     @rtd-change=${(e) => v = e.detail.value}></rtd-combo>
 */
@customElement('rtd-combo')
export class RtdCombo extends LitElement {
  @property({ type: String }) value = ''
  @property({ type: Array }) options: ComboOption[] = []
  @property({ type: Boolean }) searchable = false
  @property({ type: String }) placeholder = 'Select…'
  @property({ type: Boolean }) disabled = false

  @state() private _open = false
  @state() private _search = ''

  static styles = css`
    :host { display: block; position: relative; }
    .trigger {
      display: flex; align-items: center; justify-content: space-between;
      height: var(--ctrl-h, 22px);
      padding: 0 var(--pad-x, 6px);
      background: var(--bg-input, #14141a);
      border: 1px solid var(--border, #3a3a4c);
      border-radius: var(--radius, 2px);
      cursor: pointer; width: 100%; box-sizing: border-box;
      color: var(--fg, #d4d4d8);
      font-size: var(--font-size, 12px);
      text-align: left;
    }
    .trigger:hover { border-color: var(--border-hi, #5a5a7a); }
    .trigger .label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1 }
    .trigger .arrow { flex-shrink: 0; color: var(--fg-dim, #7a7a8a); font-size: 10px; margin-left: 4px }
    .dropdown {
      position: fixed; z-index: 9999;
      background: var(--bg-popup, #1c1c24);
      border: 1px solid var(--border-hi, #5a5a7a);
      border-radius: var(--radius, 2px);
      box-shadow: 0 4px 16px rgba(0,0,0,0.5);
      min-width: 160px; max-height: 280px;
      display: flex; flex-direction: column;
    }
    .search-input {
      padding: 4px 6px; border: none;
      border-bottom: 1px solid var(--border, #3a3a4c);
      background: var(--bg-input, #14141a);
      color: var(--fg, #d4d4d8);
      font-size: var(--font-size, 12px);
      outline: none; width: 100%; box-sizing: border-box;
    }
    .list { overflow-y: auto; flex: 1; }
    .option {
      display: flex; align-items: center; justify-content: space-between;
      padding: 3px 8px; cursor: pointer;
      font-size: var(--font-size, 12px);
      color: var(--fg, #d4d4d8);
      white-space: nowrap;
    }
    .option:hover, .option.hi { background: var(--bg-sel, #1a3060); }
    .option.sel { color: var(--fg-accent, #7aadff); }
    .option .meta { color: var(--fg-dim, #7a7a8a); font-size: 10px; margin-left: 8px }
    :host([disabled]) .trigger { opacity: 0.4; pointer-events: none; }
  `

  private _rect?: DOMRect

  render() {
    const selected = this.options.find(o => o.value === this.value)
    const label = selected?.label ?? this.placeholder
    const filtered = this._search
      ? this.options.filter(o => o.label.toLowerCase().includes(this._search.toLowerCase()) || o.value.toLowerCase().includes(this._search.toLowerCase()))
      : this.options
    return html`
      <button class="trigger" @click=${this._toggle} type="button">
        <span class="label">${label}</span>
        <span class="arrow">${this._open ? '▲' : '▼'}</span>
      </button>
      ${this._open ? html`
        <div class="dropdown" style=${this._dropdownStyle()}>
          ${this.searchable ? html`<input class="search-input" type="text" placeholder="Search…"
            .value=${this._search}
            @input=${(e: Event) => { this._search = (e.target as HTMLInputElement).value }}
            @keydown=${(e: KeyboardEvent) => e.stopPropagation()}
            @click=${(e: Event) => e.stopPropagation()} />` : ''}
          <div class="list">
            ${filtered.map(o => html`
              <div class=${`option ${o.value === this.value ? 'sel' : ''}`}
                @click=${() => this._select(o.value)}>
                <span>${o.label}</span>
                ${o.meta ? html`<span class="meta">${o.meta}</span>` : ''}
              </div>`)}
          </div>
        </div>
        <div style="position:fixed;inset:0;z-index:9998" @click=${this._close}></div>
      ` : ''}
    `
  }

  private _dropdownStyle() {
    const r = this._rect
    if (!r) return ''
    const spaceBelow = window.innerHeight - r.bottom - 4
    const spaceAbove = r.top - 4
    const h = Math.min(280, Math.max(spaceBelow, spaceAbove))
    const top = spaceBelow >= spaceAbove ? r.bottom + 2 : r.top - Math.min(280, spaceAbove) - 2
    return `top:${top}px;left:${r.left}px;width:${Math.max(r.width, 200)}px;max-height:${h}px`
  }

  private _toggle(e: MouseEvent) {
    e.stopPropagation()
    this._rect = (this.renderRoot.querySelector('.trigger') as HTMLElement)?.getBoundingClientRect()
    this._open = !this._open
    this._search = ''
  }

  private _close() { this._open = false }

  private _select(v: string) {
    this._open = false
    this.dispatchEvent(new CustomEvent('rtd-change', { detail: { value: v }, bubbles: true, composed: true }))
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-combo': RtdCombo }
}
