/**
 * app-shell — root orchestrator using nanostores.
 *
 * Responsibility: layout, lifecycle (init / persist), and wiring events from
 * child components to service calls. All reactive state lives in the stores;
 * this component is thin by design.
 *
 * This is the active runtime shell. Keep orchestration here and place feature
 * logic in focused child components or services.
 */

import { LitElement, css, html } from 'lit'
import { customElement, query } from 'lit/decorators.js'
import { StoreController } from '@nanostores/lit'
import type { RtdCanvasEditor } from '../canvas-editor/canvas-editor'

import { $scene } from '../../stores/scene.store'
import { $stream } from '../../stores/stream.store'
import { $canvas } from '../../stores/canvas.store'
import { $layer } from '../../stores/layer.store'
import { $assets } from '../../stores/assets.store'
import { $source } from '../../stores/source.store'
import { $motion } from '../../stores/motion.store'
import { $ui } from '../../stores/ui.store'
import type { StreamState } from '../../stores/stream.store'

import { loadAllAssets, loadGpuDevices, loadRendererCapabilities } from '../../services/assets.service'
import { loadSourceImages } from '../../services/source.service'
import { startSystemTaskPolling, stopSystemTaskPolling } from '../../services/layer.service'
import { requestRealtimeFrameUpdate, startWebRtc, stopStream, wireFullFrameExporter, wireLayerConditionsMetadataExporter, wireSceneSettingsExporter } from '../../services/stream.service'
import { OPTIONS_KEY, OPTIONS_VERSION } from '../../constants'
import type { LayerPreset, MaskChannel, RegionItem, RegionTarget, ScenePanelTab, StoredOptions } from '../../types'

// v2 layer aggregation: augment the outgoing wire payload with the new
// rgba_b64 / cfg_map_b64 / denoise_map_b64 / prompts[] fields decoded by
// backend ``app.render_plan.plan_from_wire``. The legacy ``layer_conditions``
// stay alongside for ControlNet / tagger plumbing.
import { aggregate } from '../../model/aggregator'
import { aggregatedToWire, createCanvasPngEncoder } from '../../model/encoder'
import { buildSceneFromLayerConditions, createDomImageDecoder, type LegacyLayerCondition, type MaskChannelKind } from '../../model/from-layer-conditions'

import '../toolbar/toolbar'
import '../layer-list/layer-list'
import '../task-monitor/task-monitor'
import '../debug-overlay/debug-overlay'
import '../color-picker/color-picker'
import '../canvas-editor/canvas-editor'
import '../scene-panel/scene-panel'
import '../layer-panel/layer-panel'

// ── Persistence helpers ───────────────────────────────────────────────────────

function _saveOptions() {
  const s = $scene.get()
  const c = $canvas.get()
  const st = $stream.get()
  const l = $layer.get()
  const ui = $ui.get()
  const opts: StoredOptions = {
    version: OPTIONS_VERSION,
    prompt: s.prompt,
    negativePrompt: s.negativePrompt,
    strength: s.strength,
    cfg: s.cfg,
    steps: s.steps,
    stageWidth: s.stageWidth,
    stageHeight: s.stageHeight,
    brushSize: c.brushSize,
    brushHardness: c.brushHardness,
    fillEdgeStrategy: c.fillEdgeStrategy,
    fillTolerance: c.fillTolerance,
    shapeDrawMode: c.shapeDrawMode,
    textFontSize: c.textFontSize,
    brushColor: c.brushColor,
    secondaryBrushColor: c.secondaryBrushColor,
    activeColorSlot: c.activeColorSlot,
    colorPlannerMode: c.colorPlannerMode,
    toolMode: c.toolMode,
    inputZoom: c.inputZoom,
    outputZoom: c.outputZoom,
    inputPaintVisible: c.inputPaintVisible,
    inputMaskVisible: c.inputMaskVisible,
    inputActiveMaskOnly: c.inputActiveMaskOnly,
    editorDest: c.editorDest,
    selectedModel: s.selectedModel,
    selectedLoras: [...s.selectedLoras],
    layerVariationCount: l.variationCount,
    generationWidth: l.generationWidth,
    generationHeight: l.generationHeight,
    layerGenerationSteps: l.generationSteps,
    layerGenerationCfg: l.generationCfg,
    layerGenerationSampler: l.generationSampler,
    layerGenerationScheduler: l.generationScheduler,
    layerGenerationSeed: l.generationSeed,
    layerTransparentBackground: l.transparentBackground,
    layerTransparentMethod: l.transparentMethod,
    layerTransparentMode: l.transparentMode,
    layerTransparentColor: l.transparentColor,
    layerTransparentTolerance: l.transparentTolerance,
    layerTransparentAlphaBlur: l.transparentAlphaBlur,
    layerTransparentAlphaThreshold: l.transparentAlphaThreshold,
    realtimeDevice: st.realtimeDevice,
    sessionDirectory: st.sessionDirectory,
    layerDevice: l.device,
    sceneSampler: s.sampler,
    sceneScheduler: s.scheduler,
    sceneSeed: s.seed,
    sceneSeedMode: s.seedMode,
    sceneSeedRotationMode: s.seedRotationMode,
    sceneSeedRotationIntervalMs: s.seedRotationIntervalMs,
    sceneReuseLatent: s.reuseLatent,
    sceneReuseLatentDenoise: s.reuseLatentDenoise,
    sceneReuseLatentNoise: s.reuseLatentNoise,
    sceneReuseLatentNoiseMode: s.reuseLatentNoiseMode,
    sceneTransparentBackground: s.transparentBackground,
    sceneTransparentMode: s.transparentMode,
    sceneTransparentColor: s.transparentColor,
    sceneTransparentTolerance: s.transparentTolerance,
    sceneTransparentAlphaBlur: s.transparentAlphaBlur,
    sceneTransparentAlphaThreshold: s.transparentAlphaThreshold,
    streamTimestepIndices: st.timestepIndices,
    streamFrameBufferSize: st.frameBufferSize,
    streamMaxPasses: st.maxPasses,
    streamCfgType: st.cfgType,
    streamRuntimePreset: st.runtimePreset,
    streamMotionMode: st.motionMode,
    streamMotionIntensity: st.motionIntensity,
    streamMotionSpeed: st.motionSpeed,
    streamTritonCompile: st.tritonCompile,
    streamVaeMode: st.vaeMode,
    streamOutputTransport: st.outputTransport,
    scenePanelTab: st.scenePanelTab,
    selectedEntity: l.selectedEntity,
    sceneMaskNegated: l.maskNegated,
    sceneMaskAbsolute: l.maskAbsolute,
    regionShape: l.regionShape,
    generatorActive: l.generatorActive,
    leftPanelWidth: ui.leftPanelWidth,
    rightPanelWidth: ui.rightPanelWidth,
  }
  window.localStorage.setItem(OPTIONS_KEY, JSON.stringify(opts))
}

function _loadOptions() {
  try {
    const raw = window.localStorage.getItem(OPTIONS_KEY)
    if (!raw) return
    const opts = JSON.parse(raw) as StoredOptions
    if (!opts.version || opts.version < 2) return
    if (opts.prompt !== undefined) $scene.setKey('prompt', opts.prompt)
    if (opts.negativePrompt !== undefined) $scene.setKey('negativePrompt', opts.negativePrompt)
    if (opts.strength !== undefined) $scene.setKey('strength', opts.strength)
    if (opts.cfg !== undefined) $scene.setKey('cfg', opts.cfg)
    if (opts.steps !== undefined) $scene.setKey('steps', opts.steps)
    if (opts.stageWidth !== undefined) $scene.setKey('stageWidth', opts.stageWidth)
    if (opts.stageHeight !== undefined) $scene.setKey('stageHeight', opts.stageHeight)
    if (opts.selectedModel !== undefined) $scene.setKey('selectedModel', opts.selectedModel)
    if (opts.selectedLoras !== undefined) $scene.setKey('selectedLoras', opts.selectedLoras)
    if (opts.sceneSampler !== undefined) $scene.setKey('sampler', opts.sceneSampler)
    if (opts.sceneScheduler !== undefined) $scene.setKey('scheduler', opts.sceneScheduler)
    if (opts.sceneSeed !== undefined) $scene.setKey('seed', opts.sceneSeed)
    if (opts.sceneSeedMode !== undefined) $scene.setKey('seedMode', opts.sceneSeedMode)
    if (opts.sceneSeedRotationMode !== undefined) $scene.setKey('seedRotationMode', opts.sceneSeedRotationMode)
    if (opts.sceneSeedRotationIntervalMs !== undefined) $scene.setKey('seedRotationIntervalMs', opts.sceneSeedRotationIntervalMs)
    if (opts.sceneReuseLatent !== undefined) $scene.setKey('reuseLatent', opts.sceneReuseLatent)
    if (opts.sceneReuseLatentDenoise !== undefined) $scene.setKey('reuseLatentDenoise', opts.sceneReuseLatentDenoise)
    if (opts.sceneReuseLatentNoise !== undefined) $scene.setKey('reuseLatentNoise', opts.sceneReuseLatentNoise)
    if (opts.sceneReuseLatentNoiseMode !== undefined) $scene.setKey('reuseLatentNoiseMode', opts.sceneReuseLatentNoiseMode)
    if (opts.sceneTransparentBackground !== undefined) $scene.setKey('transparentBackground', opts.sceneTransparentBackground)
    if (opts.sceneTransparentMode !== undefined) $scene.setKey('transparentMode', opts.sceneTransparentMode)
    if (opts.sceneTransparentColor !== undefined) $scene.setKey('transparentColor', opts.sceneTransparentColor)
    if (opts.sceneTransparentTolerance !== undefined) $scene.setKey('transparentTolerance', opts.sceneTransparentTolerance)

    if (opts.brushSize !== undefined) $canvas.setKey('brushSize', opts.brushSize)
    if (opts.brushHardness !== undefined) $canvas.setKey('brushHardness', opts.brushHardness)
    if (opts.fillEdgeStrategy !== undefined) $canvas.setKey('fillEdgeStrategy', opts.fillEdgeStrategy)
    if (opts.fillTolerance !== undefined) $canvas.setKey('fillTolerance', opts.fillTolerance)
    if (opts.shapeDrawMode !== undefined) $canvas.setKey('shapeDrawMode', opts.shapeDrawMode)
    if (opts.textFontSize !== undefined) $canvas.setKey('textFontSize', opts.textFontSize)
    if (opts.brushColor !== undefined) $canvas.setKey('brushColor', opts.brushColor)
    if (opts.secondaryBrushColor !== undefined) $canvas.setKey('secondaryBrushColor', opts.secondaryBrushColor)
    if (opts.activeColorSlot !== undefined) $canvas.setKey('activeColorSlot', opts.activeColorSlot)
    if (opts.colorPlannerMode !== undefined) $canvas.setKey('colorPlannerMode', opts.colorPlannerMode)
    if (opts.toolMode !== undefined && opts.toolMode !== 'mask') $canvas.setKey('toolMode', opts.toolMode)
    if (opts.inputZoom !== undefined) $canvas.setKey('inputZoom', opts.inputZoom)
    if (opts.outputZoom !== undefined) $canvas.setKey('outputZoom', opts.outputZoom)
    if (opts.inputPaintVisible !== undefined) $canvas.setKey('inputPaintVisible', opts.inputPaintVisible)
    if (opts.inputMaskVisible !== undefined) $canvas.setKey('inputMaskVisible', opts.inputMaskVisible)
    if (opts.inputActiveMaskOnly !== undefined) $canvas.setKey('inputActiveMaskOnly', opts.inputActiveMaskOnly)
    if (opts.editorDest !== undefined) $canvas.setKey('editorDest', opts.editorDest)

    if (opts.layerVariationCount !== undefined) $layer.setKey('variationCount', opts.layerVariationCount)
    if (opts.generationWidth !== undefined) $layer.setKey('generationWidth', opts.generationWidth)
    if (opts.generationHeight !== undefined) $layer.setKey('generationHeight', opts.generationHeight)
    if (opts.layerGenerationSteps !== undefined) $layer.setKey('generationSteps', opts.layerGenerationSteps)
    if (opts.layerGenerationCfg !== undefined) $layer.setKey('generationCfg', opts.layerGenerationCfg)
    if (opts.layerGenerationSampler !== undefined) $layer.setKey('generationSampler', opts.layerGenerationSampler)
    if (opts.layerGenerationScheduler !== undefined) $layer.setKey('generationScheduler', opts.layerGenerationScheduler)
    if (opts.layerGenerationSeed !== undefined) $layer.setKey('generationSeed', opts.layerGenerationSeed)
    if (opts.layerDevice !== undefined) $layer.setKey('device', opts.layerDevice)
    if (opts.layerTransparentBackground !== undefined) $layer.setKey('transparentBackground', opts.layerTransparentBackground)
    if (opts.layerTransparentMethod !== undefined) $layer.setKey('transparentMethod', opts.layerTransparentMethod)
    if (opts.layerTransparentMode !== undefined) $layer.setKey('transparentMode', opts.layerTransparentMode)
    if (opts.layerTransparentColor !== undefined) $layer.setKey('transparentColor', opts.layerTransparentColor)
    if (opts.layerTransparentTolerance !== undefined) $layer.setKey('transparentTolerance', opts.layerTransparentTolerance)
    if (opts.layerTransparentAlphaBlur !== undefined) $layer.setKey('transparentAlphaBlur', opts.layerTransparentAlphaBlur)
    if (opts.layerTransparentAlphaThreshold !== undefined) $layer.setKey('transparentAlphaThreshold', opts.layerTransparentAlphaThreshold)
    if (opts.selectedEntity !== undefined) $layer.setKey('selectedEntity', opts.selectedEntity as 'scene' | 'layer')
    if (opts.sceneMaskNegated !== undefined) $layer.setKey('maskNegated', opts.sceneMaskNegated)
    if (opts.sceneMaskAbsolute !== undefined) $layer.setKey('maskAbsolute', opts.sceneMaskAbsolute)
    if (opts.regionShape !== undefined) $layer.setKey('regionShape', opts.regionShape)
    if (opts.generatorActive !== undefined) $layer.setKey('generatorActive', opts.generatorActive)

    if (opts.realtimeDevice !== undefined) $stream.setKey('realtimeDevice', opts.realtimeDevice)
    if (opts.sessionDirectory !== undefined) $stream.setKey('sessionDirectory', opts.sessionDirectory)
    if (opts.streamTimestepIndices !== undefined) $stream.setKey('timestepIndices', opts.streamTimestepIndices)
    if (opts.streamFrameBufferSize !== undefined) $stream.setKey('frameBufferSize', opts.streamFrameBufferSize)
    if (opts.streamMaxPasses !== undefined) $stream.setKey('maxPasses', opts.streamMaxPasses)
    if (opts.streamCfgType !== undefined) $stream.setKey('cfgType', opts.streamCfgType)
    if (opts.streamRuntimePreset !== undefined) $stream.setKey('runtimePreset', opts.streamRuntimePreset)
    if (opts.streamMotionMode !== undefined) $stream.setKey('motionMode', opts.streamMotionMode)
    if (opts.streamMotionIntensity !== undefined) $stream.setKey('motionIntensity', opts.streamMotionIntensity)
    if (opts.streamMotionSpeed !== undefined) $stream.setKey('motionSpeed', opts.streamMotionSpeed)
    // Migration: keep Triton compile disabled by default for legacy option blobs.
    // Restore persisted value only for v4+ options explicitly saved by users.
    if ((opts.version ?? 0) >= 4 && opts.streamTritonCompile !== undefined) {
      $stream.setKey('tritonCompile', opts.streamTritonCompile)
    } else {
      $stream.setKey('tritonCompile', false)
    }
    if (opts.streamVaeMode !== undefined) $stream.setKey('vaeMode', opts.streamVaeMode)
    if (opts.streamOutputTransport !== undefined) $stream.setKey('outputTransport', opts.streamOutputTransport)
    if (opts.scenePanelTab !== undefined) $stream.setKey('scenePanelTab', opts.scenePanelTab)

    if (opts.leftPanelWidth !== undefined) $ui.setKey('leftPanelWidth', opts.leftPanelWidth)
    if (opts.rightPanelWidth !== undefined) $ui.setKey('rightPanelWidth', opts.rightPanelWidth)
  } catch { /* corrupt storage — ignore */ }
}

// ── Component ─────────────────────────────────────────────────────────────────

@customElement('rtd-app-shell')
export class RtdAppShell extends LitElement {
  @query('rtd-canvas-editor') private _editor?: RtdCanvasEditor

  private _scene = new StoreController(this, $scene)
  private _stream = new StoreController(this, $stream)
  private _layer = new StoreController(this, $layer)
  private _assets = new StoreController(this, $assets)
  private _source = new StoreController(this, $source)
  private _motion = new StoreController(this, $motion)
  private _ui = new StoreController(this, $ui)

  private _saveTimer?: number
  private _sceneUnsubscribe?: () => void
  private _streamUnsubscribe?: () => void
  private _resizing: { panel: 'left' | 'right'; startX: number; startW: number } | null = null
  private _seedRotationLastAt = 0

  private _effectiveSeedMode() {
    const sc = $scene.get()
    if (sc.seedRotationMode === 'frame') return sc.seedMode
    if (sc.seedRotationMode === 'interval') {
      const now = performance.now()
      const interval = Math.max(1, Math.round(sc.seedRotationIntervalMs || 1000))
      if (this._seedRotationLastAt <= 0 || now - this._seedRotationLastAt >= interval) {
        this._seedRotationLastAt = now
        return sc.seedMode
      }
    } else {
      this._seedRotationLastAt = 0
    }
    return 'fixed'
  }

  private _streamSceneSignature(stream: StreamState) {
    return JSON.stringify({
      scenePanelTab: stream.scenePanelTab,
      realtimeDevice: stream.realtimeDevice,
      sessionDirectory: stream.sessionDirectory,
      runtimePreset: stream.runtimePreset,
      vaeMode: stream.vaeMode,
      timestepIndices: stream.timestepIndices,
      frameBufferSize: stream.frameBufferSize,
      cfgType: stream.cfgType,
      outputTransport: stream.outputTransport,
      tritonCompile: stream.tritonCompile,
      debugStreamsEnabled: stream.debugStreamsEnabled,
    })
  }

  // Lazy reusable canvas for the v2 encoder + decoder so we don't allocate
  // one per frame. Allocated on first use; survives for the lifetime of the
  // shell. Encoder + decoder share the canvas; the decoder's
  // ``willReadFrequently:true`` context creation wins (first-call attrs are
  // sticky), which is fine because the encoder doesn't getImageData.
  private _v2Canvas: HTMLCanvasElement | null = null
  private _v2CanvasFactory = () => {
    if (!this._v2Canvas) this._v2Canvas = document.createElement('canvas')
    return this._v2Canvas
  }
  private _v2Encoder = createCanvasPngEncoder(this._v2CanvasFactory)
  private _v2Decoder = createDomImageDecoder(this._v2CanvasFactory)
  private _v2LoggedOnce = false
  private _v2LastStateKey = ''

  private _countNonZero(buf: Uint8ClampedArray): number {
    let n = 0
    for (let i = 0; i < buf.length; i++) if (buf[i] > 0) n++
    return n
  }
  // Tiny scratch canvas for lifting painted mask canvases to Uint8 buffers.
  // Kept distinct from _v2Canvas so the mask-lift doesn't clobber the
  // shared encode/decode canvas mid-augmentation.
  private _v2MaskCanvas: HTMLCanvasElement | null = null

  /**
   * Encode a HxW Uint8ClampedArray mask as a "data:image/png;base64,..."
   * grayscale PNG (luma=mask, alpha=255) -- the format the backend's
   * legacy ``decode_data_url(condition.prompt_mask).convert('L')`` and
   * the WebRTC splitter's blob-ref extraction both expect. Reuses the
   * shared scratch canvas; each call resets width/height which clears
   * the canvas, so it's safe to interleave with ``_lift8BitMask``.
   */
  private _encodeMaskToDataUrl(buf: Uint8ClampedArray, w: number, h: number): string {
    if (!this._v2MaskCanvas) this._v2MaskCanvas = document.createElement('canvas')
    const c = this._v2MaskCanvas
    c.width = w
    c.height = h
    const ctx = c.getContext('2d', { alpha: true, willReadFrequently: true })
    if (!ctx) return ''
    const rgba = new Uint8ClampedArray(w * h * 4)
    for (let i = 0, j = 0; i < buf.length; i++, j += 4) {
      const v = buf[i]
      rgba[j] = v
      rgba[j + 1] = v
      rgba[j + 2] = v
      rgba[j + 3] = 255
    }
    ctx.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0)
    return c.toDataURL('image/png')
  }

  /**
   * For each layer with a painted prompt mask, encode it once and set it
   * as ``condition.prompt_mask`` on all of that layer's region entries.
   * The legacy diffusers engine's ``_regional_prompt_mask`` reads this
   * field directly; without it, the regional CLIP attention silently fell
   * back to ``rgba.getchannel('A')`` -- which is why painted prompts only
   * worked where the user had also painted RGBA. canvas-editor sets
   * ``prompt_mask: undefined`` in its inline export, so this patch step
   * is the only way to ship the painted mask through the legacy field.
   */
  private _patchLegacyPromptMasksLastSig = ''
  private _patchLegacyPromptMasks(layerConditions: LegacyLayerCondition[], w: number, h: number): void {
    const editor = this._editor
    if (!editor) return
    const cache = new Map<string, { url: string; nz: number }>()
    let patched = 0
    let skippedHadDataUrl = 0
    let skippedNoCanvas = 0
    let skippedEmptyLift = 0
    for (const cond of layerConditions) {
      const layerId = String((cond.layer_id ?? '') as string)
      if (!layerId) continue
      // Skip only if a real PAINTED-MASK data URL is already in place.
      // canvas-editor's WebRTC export path pre-fills ``prompt_mask`` with a
      // blob-id STRING (truthy but not a data URL); leaving that intact
      // would defeat the backend's data-URL decoder and skip the painted
      // mask entirely. Same logic for cfg/denoise -- only honour an
      // existing field if it's already a data URL.
      const existing = typeof cond.prompt_mask === 'string' ? cond.prompt_mask : ''
      if (existing.startsWith('data:')) {
        skippedHadDataUrl++
        ;(cond as Record<string, unknown>)._v2_dbg = 'fe:had-data-url'
        continue
      }
      if (!cache.has(layerId)) {
        const canvas = editor.getLayerChannelMaskCanvas(layerId, 'prompt')
        if (!canvas) { cache.set(layerId, { url: '', nz: -1 }); }
        else {
          const mask = this._lift8BitMask(canvas, w, h)
          if (!mask) { cache.set(layerId, { url: '', nz: 0 }); }
          else {
            let nz = 0
            for (let i = 0; i < mask.length; i++) if (mask[i] > 0) nz++
            cache.set(layerId, { url: this._encodeMaskToDataUrl(mask, w, h), nz })
          }
        }
      }
      const entry = cache.get(layerId)!
      if (!entry.url) {
        if (entry.nz === -1) {
          skippedNoCanvas++
          ;(cond as Record<string, unknown>)._v2_dbg = 'fe:no-canvas'
        } else {
          skippedEmptyLift++
          ;(cond as Record<string, unknown>)._v2_dbg = 'fe:empty-lift'
        }
        continue
      }
      cond.prompt_mask = entry.url
      ;(cond as Record<string, unknown>)._v2_dbg = `fe:patched(nz=${entry.nz})`
      // The WebRTC splitter extracts ``prompt_mask`` into a blob ref and
      // then DELETES the inline field. If a stale ``prompt_mask_ref_name``
      // was set in a previous frame (different mask shape), the splitter
      // would happily keep using the stale ref. Clearing it forces a
      // fresh extraction.
      delete (cond as Record<string, unknown>).prompt_mask_ref_name
      patched++
    }
    // Log on state change so the user can see in devtools whether the
    // patch did anything this frame. Most useful field: ``patched`` -- if
    // 0 the painted prompt mask is NOT reaching the backend's legacy
    // ``condition.prompt_mask`` field and the SDXL regional CLIP attention
    // will skip with reason ``"no painted prompt_mask"``.
    const sig = `p=${patched}|sExisting=${skippedHadDataUrl}|sNoCv=${skippedNoCanvas}|sEmpty=${skippedEmptyLift}|nzs=${[...cache.values()].map(e => e.nz).join(',')}`
    if (sig !== this._patchLegacyPromptMasksLastSig) {
      this._patchLegacyPromptMasksLastSig = sig
      console.info('[rtd-v2 patch]', {
        patched, skippedHadDataUrl, skippedNoCanvas, skippedEmptyLift,
        layerNzCounts: Object.fromEntries([...cache.entries()].map(([k, v]) => [k, v.nz])),
      })
    }
  }

  private _lift8BitMask(source: HTMLCanvasElement, w: number, h: number): Uint8ClampedArray | null {
    try {
      if (source.width === 0 || source.height === 0) return null
      if (!this._v2MaskCanvas) this._v2MaskCanvas = document.createElement('canvas')
      const c = this._v2MaskCanvas
      c.width = w
      c.height = h
      const ctx = c.getContext('2d', { alpha: true, willReadFrequently: true })
      if (!ctx) return null
      ctx.clearRect(0, 0, w, h)
      // drawImage scales the source mask canvas (which is at stage size
      // by default) to the target wxh -- bilinear by default; OK for masks.
      ctx.drawImage(source, 0, 0, w, h)
      const id = ctx.getImageData(0, 0, w, h)
      const out = new Uint8ClampedArray(w * h)
      // canvas-editor uses two distinct mask polarities depending on the
      // code path:
      //   - The live brush target is RGBA-alpha based (transparent bg,
      //     painted strokes have alpha=255 in the brush colour).
      //   - The canonical "rendered" mask
      //     (``renderMaskDataCanvas``) is luma-based: painted=white,
      //     unpainted=black, alpha=255 throughout.
      // Reading only one channel breaks the OTHER form silently. Taking
      // the per-pixel max of alpha + r + g + b handles both.
      for (let i = 0, j = 0; i < out.length; i++, j += 4) {
        let v = id.data[j + 3]
        if (id.data[j] > v) v = id.data[j]
        if (id.data[j + 1] > v) v = id.data[j + 1]
        if (id.data[j + 2] > v) v = id.data[j + 2]
        out[i] = v
      }
      // All-zero mask -> treat as "no mask" so the bind falls back to rgba.
      // Also reject all-255: canvas-editor sometimes inits a mask canvas to
      // fully opaque before any paint, which would otherwise read as
      // "painted everywhere" and the prompt would become unconstrained.
      let nonZero = 0
      let allMax = true
      for (let i = 0; i < out.length; i++) {
        if (out[i] > 0) nonZero++
        if (out[i] < 255) allMax = false
      }
      if (nonZero === 0) return null
      if (allMax) return null  // uniform-painted = no signal
      return out
    } catch {
      return null
    }
  }

  private async _augmentWithV2(
    settings: Record<string, unknown>,
    layerConditions: LegacyLayerCondition[],
    sceneSettings: ReturnType<RtdAppShell['_rtcSceneSettingsPayload']>,
  ): Promise<void> {
    const w = Number(sceneSettings.width) || 0
    const h = Number(sceneSettings.height) || 0
    if (!w || !h) return
    // No augmentation-level cache: the painted mask canvases live outside
    // of ``layerConditions`` (they're lifted via ``maskOverride`` below),
    // so a stringify-of-layer-conditions key cannot detect "user painted
    // another stroke into the prompt mask". The decoder's URL-keyed cache
    // still skips re-decoding the RGBA when a layer image is unchanged.
    const editor = this._editor
    const maskOverride = editor
      ? (layerId: string, channel: MaskChannelKind): Uint8ClampedArray | null => {
          const canvas = editor.getLayerChannelMaskCanvas(layerId, channel)
          return canvas ? this._lift8BitMask(canvas, w, h) : null
        }
      : undefined
    const scene = await buildSceneFromLayerConditions(layerConditions, {
      width: w,
      height: h,
      basePrompt: String(sceneSettings.prompt ?? ''),
      baseNegativePrompt: String(sceneSettings.negative_prompt ?? ''),
      baseCfg: Number(sceneSettings.cfg ?? 1.5),
      baseDenoise: Number(sceneSettings.strength ?? 1.0),
      maskOverride,
    }, this._v2Decoder)
    if (scene.layers.length === 0) return
    const wire = aggregatedToWire(aggregate(scene), this._v2Encoder)
    const fields: Record<string, unknown> = {
      rgba_b64: wire.rgba_b64,
      cfg_map_b64: wire.cfg_map_b64,
      denoise_map_b64: wire.denoise_map_b64,
      prompts: wire.prompts,
      base_prompt: wire.base_prompt,
      base_negative_prompt: wire.base_negative_prompt,
      base_cfg: wire.base_cfg,
      base_denoise: wire.base_denoise,
    }
    // Fire a console.info whenever the v2 shape changes: number of layers,
    // number of prompts, or which prompts carry a painted attention mask.
    // Lets the user verify in devtools that painted prompt masks are
    // actually reaching the wire (and how many non-zero pixels they have)
    // -- the most common "ignored prompt mask" symptom is the mask being
    // read as empty due to a polarity / canvas-init mismatch.
    const promptSigs = scene.layers.flatMap(l =>
      l.prompts.map(p => `${l.id}/"${p.text.slice(0, 12)}":${p.mask ? this._countNonZero(p.mask) : '-'}`)
    )
    const stateKey = `L${scene.layers.length}|P${wire.prompts.length}|${promptSigs.join(',')}`
    if (stateKey !== this._v2LastStateKey) {
      this._v2LastStateKey = stateKey
      console.info('[rtd-v2]', {
        layers: scene.layers.length,
        prompts: wire.prompts.length,
        promptMasks: promptSigs,
        rgbaBytes: wire.rgba_b64?.length ?? 0,
      })
    }
    if (!this._v2LoggedOnce) this._v2LoggedOnce = true
    Object.assign(settings, fields)
  }

  private _rtcSceneSettingsPayload() {
    const sc = $scene.get()
    const st = $stream.get()
    return {
      prompt: sc.prompt,
      negative_prompt: sc.negativePrompt,
      model_path: sc.selectedModel || undefined,
      lora_paths: sc.selectedLoras,
      width: sc.stageWidth,
      height: sc.stageHeight,
      steps: sc.steps,
      strength: sc.strength,
      cfg: sc.cfg,
      sampler: sc.sampler || undefined,
      scheduler: sc.scheduler || undefined,
      device: st.realtimeDevice || undefined,
      session_directory: st.sessionDirectory.trim(),
      seed: sc.seed ? Number(sc.seed) : undefined,
      seed_mode: this._effectiveSeedMode(),
      reuse_previous_latent: sc.reuseLatent,
      latent_reuse_denoise: sc.reuseLatentDenoise,
      latent_reuse_noise: sc.reuseLatentNoise,
      latent_reuse_noise_mode: sc.reuseLatentNoiseMode,
      transparent_background: sc.transparentBackground,
      transparent_background_mode: sc.transparentMode,
      transparent_background_color: sc.transparentColor,
      transparent_background_tolerance: sc.transparentTolerance,
      transparent_alpha_blur: sc.transparentAlphaBlur,
      transparent_alpha_threshold: sc.transparentAlphaThreshold,
      debug_streams: st.debugStreamsEnabled,
      render_strategy: sc.renderStrategy,
      tile_divisions: sc.tileDivisions,
      tile_overlap: sc.tileOverlap,
      layer_bbox_padding: sc.layerBboxPadding,
      stream_diffusion: st.scenePanelTab === 'streamdiffusion',
      stream_timestep_indices: st.timestepIndices.split(',').map(Number).filter((n: number) => !isNaN(n)),
      stream_frame_buffer_size: st.frameBufferSize,
      stream_max_passes: st.maxPasses,
      stream_cfg_type: st.cfgType,
      stream_triton_compile: st.tritonCompile,
      stream_vae_mode: st.vaeMode,
      output_transport: st.outputTransport,
    }
  }

  private _startResize(panel: 'left' | 'right', e: PointerEvent) {
    e.preventDefault()
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    const w = panel === 'left' ? this._ui.value.leftPanelWidth : this._ui.value.rightPanelWidth
    this._resizing = { panel, startX: e.clientX, startW: w }
  }

  private _doResize(e: PointerEvent) {
    if (!this._resizing) return
    const dx = e.clientX - this._resizing.startX
    const delta = this._resizing.panel === 'left' ? dx : -dx
    const newW = Math.max(160, Math.min(600, this._resizing.startW + delta))
    if (this._resizing.panel === 'left') $ui.setKey('leftPanelWidth', newW)
    else $ui.setKey('rightPanelWidth', newW)
  }

  private _endResize(e: PointerEvent) {
    if (!this._resizing) return
    ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
    this._resizing = null
  }

  static styles = css`
    :host {
      display: block;
      height: 100svh;
      color: var(--fg, #d4d4d8);
      background: var(--bg, #1a1a1e);
      overflow: hidden;
    }

    .shell {
      height: 100svh;
      padding: 4px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) minmax(28px, auto);
      gap: 4px;
      overflow: hidden;
      box-sizing: border-box;
    }

    .topbar {
      display: grid;
      grid-template-rows: minmax(28px, auto) auto;
      gap: 4px;
      background: var(--bg-panel, #24242c);
      border-radius: var(--radius, 4px);
      padding: 4px 8px;
      border: 1px solid var(--border, #3a3a4c);
      overflow: hidden;
    }

    .topbar-main {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      overflow: hidden;
    }

    .topbar rtd-toolbar {
      min-width: 0;
      border-top: 1px solid rgba(90,90,122,.35);
      padding-top: 4px;
    }

    .renderer-select {
      min-width: 150px;
      max-width: 210px;
      height: 26px;
      background: var(--bg-input, #14141a);
      color: var(--fg, #d4d4d8);
      border: 1px solid var(--border, #3a3a4c);
      border-radius: 3px;
      font-size: 12px;
    }

    .workspace {
      display: grid;
      grid-template-columns: var(--left-w, 220px) 5px minmax(0, 1fr) 5px var(--right-w, 320px);
      overflow: hidden;
    }

    .resizer {
      cursor: col-resize;
      background: var(--border, #3a3a4c);
      z-index: 10;
      touch-action: none;
      transition: background 0.15s;
    }
    .resizer:hover { background: var(--accent, #4d87c4); }

    .panel {
      background: var(--bg-panel, #24242c);
      border-radius: var(--radius, 4px);
      border: 1px solid var(--border, #3a3a4c);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .left-panel,
    .right-panel {
      min-height: 0;
      overflow: hidden;
    }

    .left-panel rtd-scene-panel {
      flex: 1 1 auto;
      height: auto;
      min-height: 0;
    }

    .right-panel rtd-layer-list {
      flex: 0 1 36%;
      height: auto;
      min-height: 120px;
      border-bottom: 1px solid var(--border, #3a3a4c);
    }

    .right-panel rtd-layer-panel {
      flex: 1 1 auto;
      height: auto;
      min-height: 0;
    }

    .canvas-panel {
      padding: 0;
      overflow: hidden;
    }

    rtd-canvas-editor {
      flex: 1;
      min-height: 0;
    }

    .footer {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--bg-panel, #24242c);
      border-radius: var(--radius, 4px);
      border: 1px solid var(--border, #3a3a4c);
      padding: 0 8px;
      font-size: 11px;
      color: var(--fg-dim, #7a7a8a);
    }

    .status-chip {
      background: var(--bg-input, #14141a);
      border-radius: 3px;
      padding: 1px 7px;
      font-size: 10px;
      color: var(--fg-dim, #7a7a8a);
      border: 1px solid var(--border, #3a3a4c);
    }

    .stream-btn {
      padding: 3px 12px;
      border-radius: 3px;
      border: none;
      cursor: pointer;
      font-size: 11px;
      font-weight: 600;
      background: var(--fg-accent, #7aadff);
      color: var(--bg, #1a1a1e);
    }
    .stream-btn:disabled { opacity: 0.4 }
    .stream-btn.stop { background: #e04455; color: #fff; }
  `

  connectedCallback() {
    super.connectedCallback()
    _loadOptions()
    const savedRoot = window.localStorage.getItem('rtdiffusion.sourceRoot')
    if (savedRoot) { $source.setKey('root', savedRoot); void loadSourceImages() }
    void loadAllAssets()
    void loadGpuDevices()
    void loadRendererCapabilities()
    startSystemTaskPolling()
    this._saveTimer = window.setInterval(_saveOptions, 5000)
    this._sceneUnsubscribe = $scene.subscribe(() => {
      const stream = $stream.get()
      if (stream.isStreaming) requestRealtimeFrameUpdate({ refreshLayerConditions: true })
    })
    let streamSceneSignature = this._streamSceneSignature($stream.get())
    this._streamUnsubscribe = $stream.subscribe((stream) => {
      const nextSignature = this._streamSceneSignature(stream)
      if (nextSignature === streamSceneSignature) return
      streamSceneSignature = nextSignature
      if (stream.isStreaming) requestRealtimeFrameUpdate({ refreshLayerConditions: true })
    })
  }

  firstUpdated() {
    const editor = this._editor
    if (editor) {
      // Full scene payload: WebSocket sends it as one JSON frame; WebRTC splits
      // it into a normalized scene structure plus named resource blobs.
      wireFullFrameExporter(async (pcId) => {
        const image = await editor.exportFrame()
        if (!image) return null
        const mask = editor.exportMaskForStream()
        const layerConditions = pcId
          ? await editor.exportLayerConditionsForRtcResources()
          : editor.exportLayerConditions()
        const sceneSettings = this._rtcSceneSettingsPayload()
        // Patch the legacy ``layer_conditions[*].prompt_mask`` BEFORE the
        // WebRTC splitter sees the payload -- canvas-editor's inline
        // exporter sets that field to undefined, which made the engine's
        // regional CLIP attention fall back to rgba alpha (the
        // user-reported "regional clip only works on rgba area" bug). The
        // splitter handles the data-URL form correctly: extracts it as a
        // blob ref and removes the inline copy.
        try {
          const w = Number(sceneSettings.width) || 0
          const h = Number(sceneSettings.height) || 0
          if (w && h) this._patchLegacyPromptMasks(layerConditions as LegacyLayerCondition[], w, h)
        } catch (err) {
          console.warn('[rtd] legacy prompt_mask patch failed', err)
        }
        const settings: Record<string, unknown> = {
          image,
          mask,
          ...sceneSettings,
          layer_conditions: layerConditions,
          pipeline_nodes: editor.exportPipelineNodes(),
        }
        // v2 aggregation: derive rgba_b64/cfg_map_b64/denoise_map_b64/prompts[]
        // from the same layer_conditions list and attach to the settings.
        // Backend rtc/session.py activates plan_from_wire when rgba_b64 is
        // present; legacy fields remain available alongside for CN plumbing.
        try {
          await this._augmentWithV2(settings, layerConditions as LegacyLayerCondition[], sceneSettings)
        } catch (err) {
          console.warn('[rtd] v2 augmentation failed; falling back to legacy fields only', err)
        }
        if (pcId) return settings
        return JSON.stringify({
          ...settings,
        })
      })
      wireSceneSettingsExporter(async () => this._rtcSceneSettingsPayload())
      wireLayerConditionsMetadataExporter(async () => editor.exportLayerConditionMetadata())
    }
    this.addEventListener('rtd-persist-options', () => _saveOptions())
    this.addEventListener('rtd-persist-project', () => _saveOptions())
    this.addEventListener('rtd-error', (e) => {
      const msg = (e as CustomEvent<{ message: string }>).detail?.message
      if (msg) console.error('[rtd]', msg)
    })
    // "Generate scene" buttons in scene-panel dispatch rtd:toggle-stream
    this.addEventListener('rtd:toggle-stream', () => {
      if ($stream.get().isStreaming) stopStream()
      else this._startStream()
    })
    this.addEventListener('rtd-send-frame', (event) => {
      const detail = (event as CustomEvent<{ refreshResources?: boolean; refreshLayerConditions?: boolean; liveInputChanged?: boolean }>).detail
      const refreshResources = !!detail?.refreshResources
      const refreshLayerConditions = !!detail?.refreshLayerConditions
      const liveInputChanged = !!detail?.liveInputChanged
      requestRealtimeFrameUpdate({ refreshResources, refreshLayerConditions, liveInputChanged })
    })
    this.addEventListener('rtd-mask-select', (e) => this._editor?.selectMaskTool((e as CustomEvent<{ layerId: string; regionId?: string; channel?: MaskChannel; color?: string }>).detail))
    this.addEventListener('rtd-layer-paint-mode', (e) => this._editor?.enterPaintMode((e as CustomEvent<{ layerId?: string }>).detail?.layerId))
    this.addEventListener('rtd-region-use-color', (e) => this._editor?.useRegionColor((e as CustomEvent<{ color: string }>).detail.color))
    this.addEventListener('rtd-layer-use-region-color', (e) => this._editor?.useLayerRegionColor((e as CustomEvent<{ id: string }>).detail.id))
    this.addEventListener('rtd-layer-hide-inherited-mask', (e) => this._editor?.hideInheritedMask((e as CustomEvent<{ id: string }>).detail.id))
    this.addEventListener('rtd-layer-restore-inherited-mask', (e) => this._editor?.restoreInheritedMask((e as CustomEvent<{ id: string }>).detail.id))
    this.addEventListener('rtd-layer-preset', (e) => {
      const detail = (e as CustomEvent<{ id: string; patch: Partial<LayerPreset> }>).detail
      this._editor?.updateLayerPreset(detail.id, detail.patch)
    })
    this.addEventListener('rtd-layer-channel-fit-alpha', (e) => {
      const detail = (e as CustomEvent<{ id: string; channel: Extract<MaskChannel, 'color' | 'cfg' | 'denoise'>; regionId?: string; value?: number }>).detail
      this._editor?.fitLayerChannelMaskToAlpha(detail.id, detail.channel, detail.regionId, detail.value)
    })
    this.addEventListener('rtd-layer-center', (e) => {
      const detail = (e as CustomEvent<{ id: string; centerX: number; centerY: number }>).detail
      this._editor?.setLayerCenter(detail.id, detail.centerX, detail.centerY)
    })
    this.addEventListener('rtd-layer-size', (e) => {
      const detail = (e as CustomEvent<{ id: string; width: number; height: number }>).detail
      this._editor?.setLayerSize(detail.id, detail.width, detail.height)
    })
    this.addEventListener('rtd-layer-scale', (e) => {
      const detail = (e as CustomEvent<{ id: string; scale: number }>).detail
      this._editor?.setLayerScale(detail.id, detail.scale)
    })
    this.addEventListener('rtd-layer-angle', (e) => {
      const detail = (e as CustomEvent<{ id: string; angle: number }>).detail
      this._editor?.setLayerAngle(detail.id, detail.angle)
    })
    this.addEventListener('rtd-layer-center-viewport', (e) => {
      const detail = (e as CustomEvent<{ id: string }>).detail
      this._editor?.centerLayer(detail.id)
    })
    this.addEventListener('rtd-layer-fit-viewport', (e) => {
      const detail = (e as CustomEvent<{ id: string }>).detail
      this._editor?.fitLayerToViewport(detail.id)
    })
    this.addEventListener('rtd-mask-fill-viewport', (e) => {
      const detail = (e as CustomEvent<{ layerId: string; channel: Extract<MaskChannel, 'color' | 'cfg' | 'denoise'>; regionId?: string; value?: number }>).detail
      this._editor?.fillMaskViewport(detail.layerId, detail.channel, detail.regionId, detail.value)
    })
    this.addEventListener('rtd-mask-clear', (e) => {
      const detail = (e as CustomEvent<{ layerId: string; channel: Extract<MaskChannel, 'color' | 'cfg' | 'denoise'>; regionId?: string }>).detail
      this._editor?.clearLayerMask(detail.layerId, detail.channel, detail.regionId)
    })
    this.addEventListener('rtd-rgba-mask-load', (e) => { void this._editor?.loadRgbaMaskImage((e as CustomEvent<{ layerId?: string }>).detail?.layerId) })
    this.addEventListener('rtd-rgba-mask-empty', () => this._editor?.addEmptyRgbaMask())
    this.addEventListener('rtd-region-add', (e) => {
      const detail = (e as CustomEvent<{ target: RegionTarget; layerId?: string }>).detail
      this._editor?.addRegionFromSelection(detail.target, detail.layerId)
    })
    this.addEventListener('rtd-region-color', (e) => {
      const detail = (e as CustomEvent<{ id: string; color: string }>).detail
      this._editor?.setRegionColor(detail.id, detail.color)
    })
    this.addEventListener('rtd-region-update', (e) => {
      const detail = (e as CustomEvent<{ id: string; patch: Partial<RegionItem> }>).detail
      this._editor?.updateRegion(detail.id, detail.patch)
    })
    this.addEventListener('rtd-region-delete', (e) => this._editor?.deleteRegion((e as CustomEvent<{ id: string }>).detail.id))
    this.addEventListener('rtd-region-restore', (e) => this._editor?.restoreRegionInheritance((e as CustomEvent<{ id: string }>).detail.id))
    this.addEventListener('rtd-selection-clear', () => this._editor?.clearSelection())
    this.addEventListener('rtd-selection-invert', () => this._editor?.invertSelection())
    this.addEventListener('rtd-selection-delete', () => void this._editor?.deleteSelection())
    this.addEventListener('rtd-selection-crop', () => void this._editor?.cropToSelection())
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    stopStream()
    stopSystemTaskPolling()
    if (this._saveTimer) window.clearInterval(this._saveTimer)
    this._sceneUnsubscribe?.()
    this._sceneUnsubscribe = undefined
    this._streamUnsubscribe?.()
    this._streamUnsubscribe = undefined
    _saveOptions()
  }

  render() {
    const st = this._stream.value
    const ui = this._ui.value
    const scene = this._scene.value
    const layer = this._layer.value
    const assets = this._assets.value
    const source = this._source.value
    const motion = this._motion.value
    const leftW = ui.leftPanelWidth
    const rightW = ui.rightPanelWidth
    const renderers: { value: ScenePanelTab; label: string }[] = [
      { value: 'sdxl', label: 'SDXL' },
      { value: 'z-image', label: 'Z-Image' },
      { value: 'streamdiffusion', label: 'StreamDiffusion' },
    ]

    return html`<div class="shell" style="--left-w:${leftW}px; --right-w:${rightW}px">
      <header class="topbar">
        <div class="topbar-main">
          <select class="renderer-select"
            @change=${(e: Event) => $stream.setKey('scenePanelTab', (e.target as HTMLSelectElement).value as ScenePanelTab)}>
            ${renderers.map((renderer) => html`<option value=${renderer.value} ?selected=${renderer.value === st.scenePanelTab}>${renderer.label}</option>`)}
          </select>
          ${assets.isLoading ? html`<span class="status-chip">Loading…</span>` : ''}
          ${motion.isGenerating ? html`<span class="status-chip">Video…</span>` : ''}
          ${st.isStreaming
            ? html`<button class="stream-btn stop" @click=${() => stopStream()}>Stop</button>`
            : html`<button class="stream-btn"
                ?disabled=${st.status === 'connecting'}
                @click=${() => this._startStream()}>
                ${st.status === 'connecting' ? 'Connecting…' : 'Stream'}
              </button>`}
          <span class="status-chip">${st.isStreaming ? `${st.scenePanelTab} · ${st.status}` : st.status}</span>
          ${st.backendMode && st.backendMode !== 'mock' ? html`<span class="status-chip" title=${st.backendMode}>${st.backendMode.split(':')[0].split('/').slice(-1)[0]}</span>` : ''}
          ${st.fps > 0 ? html`<span class="status-chip">${st.fps.toFixed(1)} FPS · ${st.latency} ms</span>` : ''}
        </div>
        <rtd-toolbar></rtd-toolbar>
      </header>

      <div class="workspace">
        <aside class="panel left-panel">
          <rtd-scene-panel></rtd-scene-panel>
        </aside>

        <div class="resizer"
          @pointerdown=${(e: PointerEvent) => this._startResize('left', e)}
          @pointermove=${(e: PointerEvent) => this._doResize(e)}
          @pointerup=${(e: PointerEvent) => this._endResize(e)}
          @pointercancel=${(e: PointerEvent) => this._endResize(e)}></div>

        <main class="panel canvas-panel">
          <rtd-canvas-editor style="flex:1;min-height:0;display:flex;flex-direction:column;"></rtd-canvas-editor>
          ${st.debugStreamsEnabled ? html`<rtd-debug-overlay></rtd-debug-overlay>` : ''}
        </main>

        <div class="resizer"
          @pointerdown=${(e: PointerEvent) => this._startResize('right', e)}
          @pointermove=${(e: PointerEvent) => this._doResize(e)}
          @pointerup=${(e: PointerEvent) => this._endResize(e)}
          @pointercancel=${(e: PointerEvent) => this._endResize(e)}></div>

        <aside class="panel right-panel">
          <rtd-layer-list
            @layer-add=${() => this._editor?.addBlankLayer()}
            @layer-delete=${() => this._editor?.deleteActiveLayer()}
            @layer-toggle=${(e: CustomEvent<{ id: string }>) => this._editor?.toggleLayer(e.detail.id)}
            @layer-move-up=${() => this._editor?.moveLayerUp()}
            @layer-move-down=${() => this._editor?.moveLayerDown()}
          ></rtd-layer-list>
          <rtd-layer-panel></rtd-layer-panel>
        </aside>
      </div>

      <footer class="footer">
        <rtd-task-monitor></rtd-task-monitor>
        <div class="spacer"></div>
        ${source.root ? html`<span class="status-chip">${source.root.split(/[/\\]/).at(-1)}</span>` : ''}
        ${layer.isGeneratingLayer ? html`<span class="status-chip">Generating…</span>` : ''}
        <span class="status-chip">${scene.selectedModel ? scene.selectedModel.split(/[/\\]/).at(-1) : st.backendMode}</span>
      </footer>
    </div>`
  }

  private _startStream() {
    void startWebRtc()
  }
}

declare global {
  interface HTMLElementTagNameMap { 'rtd-app-shell': RtdAppShell }
}
