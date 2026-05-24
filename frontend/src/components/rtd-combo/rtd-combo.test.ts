import { describe, it, expect, vi, beforeEach } from 'vitest'
import './rtd-combo'
import type { RtdCombo } from './rtd-combo'
import type { ComboOption } from './rtd-combo'

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

const opts: ComboOption[] = [
  { label: 'Euler', value: 'euler' },
  { label: 'DPM++', value: 'dpmpp' },
]

describe('RtdCombo', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
  })

  it('shows selected label', async () => {
    const el = await fixtureWithProps<RtdCombo>('rtd-combo', {
      value: 'euler',
      options: opts,
    })
    const trigger = el.shadowRoot!.querySelector('.trigger')!
    expect(trigger.textContent).toContain('Euler')
  })

  it('opens dropdown on click', async () => {
    const el = await fixtureWithProps<RtdCombo>('rtd-combo', {
      value: 'euler',
      options: opts,
    })
    const trigger = el.shadowRoot!.querySelector('.trigger') as HTMLButtonElement
    trigger.click()
    await (el as unknown as { updateComplete: Promise<boolean> }).updateComplete
    expect(el.shadowRoot!.querySelector('.dropdown')).not.toBeNull()
  })

  it('emits rtd-change on option click', async () => {
    const el = await fixtureWithProps<RtdCombo>('rtd-combo', {
      value: 'euler',
      options: opts,
    })
    const handler = vi.fn()
    el.addEventListener('rtd-change', handler)
    const trigger = el.shadowRoot!.querySelector('.trigger') as HTMLButtonElement
    trigger.click()
    await (el as unknown as { updateComplete: Promise<boolean> }).updateComplete
    const options = el.shadowRoot!.querySelectorAll('.option')
    ;(options[1] as HTMLElement).click()
    expect(handler).toHaveBeenCalledOnce()
    expect((handler.mock.calls[0][0] as CustomEvent).detail.value).toBe('dpmpp')
  })
})
