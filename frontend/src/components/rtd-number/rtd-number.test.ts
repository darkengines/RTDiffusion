import { describe, it, expect, vi, beforeEach } from 'vitest'
import './rtd-number'
import type { RtdNumber } from './rtd-number'

async function fixtureWithProps<T extends HTMLElement>(
  tagName: string,
  props: Record<string, unknown>,
): Promise<T> {
  const el = document.createElement(tagName) as T
  for (const [key, value] of Object.entries(props)) {
    ;(el as Record<string, unknown>)[key] = value
  }
  document.body.appendChild(el)
  if ('updateComplete' in el) {
    await (el as unknown as { updateComplete: Promise<boolean> }).updateComplete
  }
  return el
}

describe('RtdNumber', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
  })

  it('renders label and value', async () => {
    const el = await fixtureWithProps<RtdNumber>('rtd-number', {
      label: 'CFG',
      value: 1.5,
      decimals: 1,
    })
    const input = el.shadowRoot!.querySelector('input')!
    expect(input.value).toBe('1.5')
    const label = el.shadowRoot!.querySelector('.label')!
    expect(label.textContent).toBe('CFG')
  })

  it('emits rtd-change on dec button click', async () => {
    const el = await fixtureWithProps<RtdNumber>('rtd-number', {
      value: 5,
      min: 0,
      max: 10,
      step: 1,
    })
    const handler = vi.fn()
    el.addEventListener('rtd-change', handler)
    const dec = el.shadowRoot!.querySelector('.btn') as HTMLButtonElement
    dec.click()
    expect(handler).toHaveBeenCalledOnce()
    expect((handler.mock.calls[0][0] as CustomEvent).detail.value).toBe(4)
  })

  it('emits rtd-change on inc button click', async () => {
    const el = await fixtureWithProps<RtdNumber>('rtd-number', {
      value: 5,
      min: 0,
      max: 10,
      step: 1,
    })
    const handler = vi.fn()
    el.addEventListener('rtd-change', handler)
    const buttons = el.shadowRoot!.querySelectorAll('.btn')
    ;(buttons[1] as HTMLButtonElement).click()
    expect(handler).toHaveBeenCalledOnce()
    expect((handler.mock.calls[0][0] as CustomEvent).detail.value).toBe(6)
  })

  it('emits rtd-change on text input before blur', async () => {
    const el = await fixtureWithProps<RtdNumber>('rtd-number', {
      value: 1,
      min: 0,
      max: 10,
      step: 0.1,
      decimals: 1,
    })
    const handler = vi.fn()
    el.addEventListener('rtd-change', handler)
    const input = el.shadowRoot!.querySelector('input') as HTMLInputElement

    input.value = '2.5'
    input.dispatchEvent(new Event('input', { bubbles: true, composed: true }))

    expect(handler).toHaveBeenCalledOnce()
    expect((handler.mock.calls[0][0] as CustomEvent).detail.value).toBe(2.5)
  })

  it('clamps value to min/max', async () => {
    const el = await fixtureWithProps<RtdNumber>('rtd-number', {
      value: 0,
      min: 0,
      max: 10,
      step: 1,
    })
    const handler = vi.fn()
    el.addEventListener('rtd-change', handler)
    const dec = el.shadowRoot!.querySelector('.btn') as HTMLButtonElement
    dec.click()
    expect((handler.mock.calls[0][0] as CustomEvent).detail.value).toBe(0) // clamped
  })
})
