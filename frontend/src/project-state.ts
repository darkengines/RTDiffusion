export const CANVAS_CUSTOM_PROPERTIES = [
  '__uid',
  '__preset',
  '__paintChild',
  '__parentLayerId',
  'globalCompositeOperation',
  'name',
  'selectable',
  'evented',
] as const

export const ARCHIVE_ROOT_URL = 'archive:///'
export const PRESERVED_PRESET_STORAGE_KEYS = ['rtdiffusion.samplerPresets.v1'] as const

export type SerializableCanvasObject = {
  type?: string
  name?: string
  __paintChild?: boolean
  __parentLayerId?: string
  objects?: SerializableCanvasObject[]
  [key: string]: unknown
}

export function isPaintChildObject(object: { __paintChild?: boolean; __parentLayerId?: string }) {
  return object.__paintChild === true || !!object.__parentLayerId
}

export function normalizeCanvasPaintChildren(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => normalizeCanvasPaintChildren(item))
  if (!value || typeof value !== 'object') return value
  const object = value as SerializableCanvasObject
  if (object.__parentLayerId) object.__paintChild = true
  if (object.objects) object.objects = object.objects.map((item) => normalizeCanvasPaintChildren(item) as SerializableCanvasObject)
  return object
}

export function topLevelLayerObjects<T extends { __paintChild?: boolean; __parentLayerId?: string }>(objects: T[]) {
  return objects.filter((object) => !isPaintChildObject(object))
}

export type RectTransform = { x: number; y: number; width: number; height: number }

export function strokeParentLayerId(pendingLayerId: string | undefined, selectedLayerId: string, strokeObjectId?: string) {
  const target = pendingLayerId || selectedLayerId
  return target && target !== strokeObjectId ? target : ''
}

export function centeredResizeRect(rect: RectTransform, patch: Partial<RectTransform> & { centerX?: number; centerY?: number }) {
  const width = Math.max(1, patch.width ?? rect.width)
  const height = Math.max(1, patch.height ?? rect.height)
  const centerX = patch.centerX ?? (patch.x !== undefined ? patch.x + width / 2 : rect.x + rect.width / 2)
  const centerY = patch.centerY ?? (patch.y !== undefined ? patch.y + height / 2 : rect.y + rect.height / 2)
  return { x: centerX - width / 2, y: centerY - height / 2, width, height }
}

export function extensionForMediaType(mediaType: string) {
  if (mediaType.includes('jpeg')) return 'jpg'
  if (mediaType.includes('png')) return 'png'
  if (mediaType.includes('webp')) return 'webp'
  if (mediaType.includes('gif')) return 'gif'
  if (mediaType.includes('mp4')) return 'mp4'
  if (mediaType.includes('webm')) return 'webm'
  return 'bin'
}

export function archivePathForResource(hash: string, mediaType: string) {
  return `resources/${hash}.${extensionForMediaType(mediaType)}`
}

export function archiveUrlForResource(hash: string, mediaType: string) {
  return `${ARCHIVE_ROOT_URL}${archivePathForResource(hash, mediaType)}`
}

export function resourceHashFromArchiveUrl(url: string) {
  if (!url.startsWith(ARCHIVE_ROOT_URL)) return undefined
  return /resources\/([a-f0-9]{64})\./.exec(url.slice(ARCHIVE_ROOT_URL.length))?.[1]
}

export function isFlushableProjectStorageKey(key: string) {
  return key.startsWith('rtdiffusion.') && !PRESERVED_PRESET_STORAGE_KEYS.includes(key as (typeof PRESERVED_PRESET_STORAGE_KEYS)[number])
}