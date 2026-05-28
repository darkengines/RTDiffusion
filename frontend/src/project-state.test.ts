import { describe, expect, it } from 'vitest'
import {
  ARCHIVE_ROOT_URL,
  archivePathForResource,
  archiveUrlForResource,
  centeredResizeRect,
  isFlushableProjectStorageKey,
  isPaintChildObject,
  normalizeCanvasPaintChildren,
  resourceHashFromArchiveUrl,
  strokeParentLayerId,
  topLevelLayerObjects,
} from './project-state'

const HASH = 'f3f90a26aaf73d9ba1cbd04047447e3322233a97669b044d83223e9e4160d926'

describe('paint child layer ownership', () => {
  it('does not count brush strokes as layers after reload', () => {
    const objects = [
      { type: 'rect', __uid: 'layer-1', name: 'Layer 1' },
      { type: 'path', __parentLayerId: 'layer-1', name: 'Paint stroke' },
      { type: 'path', __parentLayerId: 'layer-1', name: 'Paint stroke' },
      { type: 'path', __parentLayerId: 'layer-1', name: 'Paint stroke' },
    ]

    const normalized = normalizeCanvasPaintChildren({ objects }) as { objects: { __paintChild?: boolean; __parentLayerId?: string }[] }

    expect(normalized.objects.slice(1).every((object) => object.__paintChild)).toBe(true)
    expect(topLevelLayerObjects(normalized.objects)).toHaveLength(1)
  })

  it('treats parent-owned objects as paint children even if the boolean flag is missing', () => {
    expect(isPaintChildObject({ __parentLayerId: 'active-layer' })).toBe(true)
    expect(isPaintChildObject({ __paintChild: true })).toBe(true)
    expect(isPaintChildObject({})).toBe(false)
  })

  it('keeps eraser strokes attached to the layer where the stroke started', () => {
    expect(strokeParentLayerId('layer-1', 'new-path-id', 'new-path-id')).toBe('layer-1')
  })

  it('does not attach a stroke to itself when Fabric briefly selects the new path', () => {
    expect(strokeParentLayerId('', 'new-path-id', 'new-path-id')).toBe('')
  })
})

describe('archive resource urls', () => {
  it('uses content hashes as stable archive paths', () => {
    expect(archivePathForResource(HASH, 'image/png')).toBe(`resources/${HASH}.png`)
    expect(archiveUrlForResource(HASH, 'image/png')).toBe(`${ARCHIVE_ROOT_URL}resources/${HASH}.png`)
  })

  it('extracts hashes from resource urls so masks can be resolved from storage', () => {
    const url = archiveUrlForResource(HASH, 'image/png')

    expect(resourceHashFromArchiveUrl(url)).toBe(HASH)
    expect(resourceHashFromArchiveUrl('blob:abc')).toBeUndefined()
  })

  it('keeps future video resources in the same archive namespace', () => {
    expect(archivePathForResource(HASH, 'video/webm')).toBe(`resources/${HASH}.webm`)
  })
})

describe('project memory flush', () => {
  it('flushes project state and prompt presets while preserving scheduler presets', () => {
    expect(isFlushableProjectStorageKey('rtdiffusion.projectState.v1')).toBe(true)
    expect(isFlushableProjectStorageKey(`rtdiffusion.resource.v1.${HASH}`)).toBe(true)
    expect(isFlushableProjectStorageKey('rtdiffusion.options.v1')).toBe(true)
    expect(isFlushableProjectStorageKey('rtdiffusion.sourceRoot')).toBe(true)

    expect(isFlushableProjectStorageKey('rtdiffusion.scenePresets.v1')).toBe(true)
    expect(isFlushableProjectStorageKey('rtdiffusion.samplerPresets.v1')).toBe(false)
    expect(isFlushableProjectStorageKey('unrelated.key')).toBe(false)
  })
})

describe('center anchored transforms', () => {
  it('keeps the same center when resizing a rectangle', () => {
    expect(centeredResizeRect({ x: 10, y: 20, width: 100, height: 80 }, { width: 50, height: 40 })).toEqual({
      x: 35,
      y: 40,
      width: 50,
      height: 40,
    })
  })

  it('moves a rectangle by center without changing its size', () => {
    expect(centeredResizeRect({ x: 10, y: 20, width: 100, height: 80 }, { centerX: 200, centerY: 300 })).toEqual({
      x: 150,
      y: 260,
      width: 100,
      height: 80,
    })
  })
})