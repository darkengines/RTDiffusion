import { $source } from '../stores/source.store'
import { reportError } from '../stores/ui.store'
import { fetchSources, pickSourceRoot as apiPickRoot } from './api'

export async function loadSourceImages(): Promise<void> {
  const root = $source.get().root.trim()
  if (!root) return
  $source.setKey('isLoading', true)
  try {
    const catalog = await fetchSources(root)
    $source.setKey('root', catalog.root)
    $source.setKey('images', catalog.images)
    $source.setKey('selectedRel', catalog.images[0]?.rel ?? '')
    window.localStorage.setItem('rtdiffusion.sourceRoot', catalog.root)
  } catch (err) {
    reportError(err instanceof Error ? err.message : String(err))
    $source.setKey('images', [])
    $source.setKey('selectedRel', '')
  } finally {
    $source.setKey('isLoading', false)
  }
}

export function setSourceRoot(root: string): void {
  $source.setKey('root', root.trim())
  $source.setKey('selectedRel', '')
  $source.setKey('images', [])
  if (root.trim()) window.localStorage.setItem('rtdiffusion.sourceRoot', root.trim())
  else window.localStorage.removeItem('rtdiffusion.sourceRoot')
}

export async function pickSourceRoot(): Promise<void> {
  $source.setKey('isLoading', true)
  try {
    const result = await apiPickRoot()
    if (!result.root) return
    setSourceRoot(result.root)
    await loadSourceImages()
  } catch (err) {
    reportError(err instanceof Error ? err.message : String(err))
  } finally {
    $source.setKey('isLoading', false)
  }
}
