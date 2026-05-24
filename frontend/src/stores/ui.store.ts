import { map } from 'nanostores'
import type { GenerationTask, PickerDialog, PickerOption } from '../types'

export interface ErrorEntry {
  id: string
  message: string
  time: number
}

export interface UiState {
  leftPanelWidth: number
  rightPanelWidth: number
  taskPopoverOpen: boolean
  errorPopoverOpen: boolean
  errorLog: ErrorEntry[]
  generationTasks: GenerationTask[]
  projectStatus: string
  loraSearch: string
  pickerDialog: PickerDialog | undefined
  pickerDialogPosition: { left: number; top: number; width: number; maxHeight: number }
}

export const $ui = map<UiState>({
  leftPanelWidth: 220,
  rightPanelWidth: 320,
  taskPopoverOpen: false,
  errorPopoverOpen: false,
  errorLog: [],
  generationTasks: [],
  projectStatus: 'Local project ready',
  loraSearch: '',
  pickerDialog: undefined,
  pickerDialogPosition: { left: 0, top: 0, width: 420, maxHeight: 420 },
})

export function reportError(message: string) {
  const text = message.trim()
  if (!text) return
  const entry: ErrorEntry = { id: crypto.randomUUID(), message: text, time: Date.now() }
  $ui.setKey('errorLog', [$ui.get().errorLog[0] ? entry : entry, ...$ui.get().errorLog].slice(0, 12))
}

export function clearErrors() {
  $ui.setKey('errorLog', [])
  $ui.setKey('errorPopoverOpen', false)
}

export function upsertGenerationTask(task: GenerationTask) {
  const tasks = $ui.get().generationTasks
  const existing = tasks.find((t) => t.task_id === task.task_id)
  $ui.setKey(
    'generationTasks',
    existing
      ? tasks.map((t) => (t.task_id === task.task_id ? { ...t, ...task } : t))
      : [task, ...tasks].slice(0, 12),
  )
}

export function removeStaleSystemTasks(activeIds: Set<string>) {
  $ui.setKey(
    'generationTasks',
    $ui.get().generationTasks.filter((t) => !t.task_id.startsWith('sys_') || activeIds.has(t.task_id)),
  )
}

export function openOptionPicker(
  event: MouseEvent,
  label: string,
  value: string,
  options: PickerOption[],
  onSelect: (v: string) => void,
) {
  const anchor = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const maxHeight = window.innerHeight - anchor.bottom - 8
  $ui.setKey('pickerDialogPosition', {
    left: anchor.left,
    top: anchor.bottom + 4,
    width: Math.max(anchor.width, 240),
    maxHeight: Math.max(maxHeight, 120),
  })
  $ui.setKey('pickerDialog', { kind: 'option', label, value, options, onSelect })
}

export function openLoraPicker(
  event: MouseEvent,
  selectedPaths: string[],
  onToggle: (v: string) => void,
) {
  const anchor = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const maxHeight = window.innerHeight - anchor.bottom - 8
  $ui.setKey('pickerDialogPosition', {
    left: anchor.left,
    top: anchor.bottom + 4,
    width: Math.max(anchor.width, 240),
    maxHeight: Math.max(maxHeight, 120),
  })
  $ui.setKey('pickerDialog', { kind: 'lora', label: 'LoRAs', selectedPaths, onToggle })
}

export function closePickerDialog() {
  $ui.setKey('pickerDialog', undefined)
}
