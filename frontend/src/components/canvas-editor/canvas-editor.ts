import { Canvas, Ellipse, FabricImage, FabricObject as FabricObjectClass, Path, PencilBrush, Point, Rect, Shadow, Textbox, type FabricObject } from 'fabric'
import { LitElement, css, html } from 'lit'
import { keyed } from 'lit/directives/keyed.js'
import { customElement, query, state } from 'lit/decorators.js'
import {
  CANVAS_CUSTOM_PROPERTIES,
  centeredResizeRect,
  isPaintChildObject,
  normalizeCanvasPaintChildren,
  strokeParentLayerId,
  topLevelLayerObjects,
} from '../../project-state'
import type {
  ActiveEntityTarget,
  AssetCatalog,
  ColorPlannerMode,
  ColorSlot,
  ConditionAggregateOperator,
  ControlNetConfig,
  EditorDest,
  EditorSnapshot,
  FillEdgeStrategy,
  GeneratedLayer,
  GenerationTask,
  LayerItem,
  LayerPreset,
  MaskChannel,
  MaskScope,
  RegionItem,
  RegionPoint,
  RegionSchedule,
  RegionShape,
  RegionTarget,
  RgbaColor,
  SelectionRect,
  ShapeDrawMode,
  ToolMode,
  VideoFpsMode,
  ViewportTarget,
  StreamTimingMap,
} from '../../types'
import {
  COLOR_PLANNER_MODES,
  HISTORY_LIMIT,
  PALETTE,
  QUICK_PALETTE,
  TOOL_DEFS,
  TOOL_SHORTCUTS,
} from '../../constants'
import { $layer } from '../../stores/layer.store'
import { $canvas } from '../../stores/canvas.store'
import { $scene } from '../../stores/scene.store'
import { $stream } from '../../stores/stream.store'
import { serializeChannelRefs } from '../../services/mask-blob.service'
import { getDebugSurfaceStream, getDebugSurfaces, getSceneResourceDebugData, getWebRtcOutputMediaStream, registerDebugCanvasSurface, registerWebRtcResourceMediaTrack, requestDebugSurfaceFrame, unregisterDebugSurface, type DebugSurface } from '../../services/stream.service'

const CFG_MASK_F32_MIME = 'application/x-rtd-mask-f32'
const CFG_MASK_F32_MAGIC = 'RTF1'
const CFG_MASK_MAX = 30
const CFG_FACTOR_DEFAULT = 1.0

type SvgTransformHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'
type SvgTransformMode = 'move' | 'rotate' | SvgTransformHandle
type TransformOverlayHandle = { key: SvgTransformHandle; x: number; y: number; cursor: string }
type TransformOverlayState = {
  points: string
  handles: TransformOverlayHandle[]
  rotate: { x: number; y: number; anchorX: number; anchorY: number }
  frame: SvgTransformFrame
}
type SvgTransformFrame = { center: Point; angle: number; width: number; height: number; corners: Point[] }
type SvgTransformObjectState = {
  object: FabricObject
  center: Point
  angle: number
  scaleX: number
  scaleY: number
}
type SvgTransformDrag = {
  pointerId: number
  mode: SvgTransformMode
  layerId: string
  objects: SvgTransformObjectState[]
  startPointer: Point
  center: Point
  angle: number
  scaleX: number
  scaleY: number
  width: number
  height: number
  fixedScene?: Point
  fixedLocal?: Point
  handleSign?: { x: -1 | 0 | 1; y: -1 | 0 | 1 }
  startPointerAngle?: number
}
type SelectionDragState = { pointerId: number; startPoint: RegionPoint; startSelection: SelectionRect }
type LayerMediaSurface = {
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  stream?: MediaStream
  track?: MediaStreamTrack
  registered: boolean
  revision: number
}
type LayerConditionExportPass = {
  layerCanvases: Map<string, HTMLCanvasElement>
}

FabricObjectClass.customProperties = [...new Set([...(FabricObjectClass.customProperties ?? []), ...CANVAS_CUSTOM_PROPERTIES])]

@customElement('rtd-canvas-editor')
export class RtdCanvasEditor extends LitElement {
  @query('#edit-canvas') private editCanvasElement!: HTMLCanvasElement
  @query('#mask-canvas') private maskCanvasElement!: HTMLCanvasElement
  @query('#mask-preview-canvas') private maskPreviewCanvasElement!: HTMLCanvasElement
  @query('#input-capture-canvas') private inputCaptureCanvasElement?: HTMLCanvasElement
  @query('#editor-overlay') private editorOverlayCanvas?: HTMLCanvasElement
  @query('#signal-output-video') private signalOutputVideoElement?: HTMLVideoElement
  @query('.stage') private stageElement!: HTMLElement
  @query('.canvas-pane.work-pane') private inputPaneElement!: HTMLElement

  @state() private prompt = ''
  @state() private negativePrompt = ''
  @state() private strength = 1.0
  @state() private stageWidth = 832
  @state() private stageHeight = 1216
  @state() public generationWidth = 832
  @state() public generationHeight = 1216
  @state() private videoFpsMode: VideoFpsMode = 'fps'
  @state() private brushSize = 34
  @state() private brushHardness = 100
  @state() private fillEdgeStrategy: FillEdgeStrategy = 'transparent_blend'
  @state() private fillTolerance = 32
  @state() private shapeDrawMode: ShapeDrawMode = 'fill'
  @state() private textFontSize = 48
  @state() private brushColor = '#ffffff'
  @state() private secondaryBrushColor = '#000000'
  @state() private activeColorSlot: ColorSlot = 'primary'
  @state() private colorPlannerMode: ColorPlannerMode = 'complementary'
  @state() private colorPopupOpen = false
  @state() private colorPopupPosition = { x: 92, y: 92 }
  @state() private toolMode: ToolMode = 'brush'
  @state() private outputImage = ''
  @state() private outputImageNonce = 0
  @state() private assets: AssetCatalog = { models: [], loras: [] }
  @state() private selectedModel = ''
  @state() private selectedLoras = new Set<string>()
  @state() private loraSearch = ''
  @state() private layers: LayerItem[] = []
  private cursor = { x: 0, y: 0, visible: false }
  @state() private inputZoom = 1
  @state() private outputZoom = 1
  @state() private inputPaintVisible = true
  @state() private inputMaskVisible = true
  @state() private inputChannelVisible = true
  @state() private inputActiveMaskOnly = false
  @state() private activeMaskChannel: MaskChannel = 'color'
  @state() private maskBrushValue = 255
  @state() private editorDest: EditorDest = 'paint'
  @state() private inputOverlayVisible = true
  @state() private outputOverlayVisible = true
  @state() private outputTimings: StreamTimingMap = {}
  @state() private outputTimingsExpanded = false
  @state() private selectedOutputSignal = 'output'
  @state() private signalPickerOpen = false
  @state() private resourceDebugVersion = 0
  @state() private selectionRect?: SelectionRect
  @state() private layerGenerationSteps = 24
  @state() private layerGenerationCfg = 5
  @state() private layerGenerationSampler = 'euler_ancestral'
  @state() private layerGenerationScheduler = 'simple'
  @state() private generatedLayers: GeneratedLayer[] = []
  @state() private generationTasks: GenerationTask[] = []
  @state() private selectedEntity: ActiveEntityTarget = 'scene'
  @state() private regions: RegionItem[] = []
  @state() private regionName = ''
  @state() private sceneMaskNegated = false
  @state() private sceneMaskAbsolute = false
  @state() private noDefaultMaskLayers: string[] = []
  @state() private layerControlNetConfigs = new Map<string, ControlNetConfig>()
  @state() private videoUrl = ''
  @state() private videoLoopMode: 'loop' | 'ping-pong' | 'once' = 'loop'
  @state() private videoStart = 0
  @state() private videoEnd = 0
  @state() private videoFps = 30
  @state() private videoDuration = 0
  @state() private awaitingFrame = false
  @state() private realtimeVideoActive = false
  @state() private activeMotionFrame = ''
  @state() private activeMotionVideo = ''
  @state() private activeMotionIsStream = false
  @state() private motionStreamHasSegments = false
  @state() private transformOverlayPoints = ''
  @state() private transformOverlay?: TransformOverlayState

  private fabricCanvas?: Canvas
  private maskContext?: CanvasRenderingContext2D
  private maskPreviewContext?: CanvasRenderingContext2D
  private drawing = false
  private lastPointer?: { x: number; y: number }
  private panningTarget?: HTMLElement
  private panStart = { x: 0, y: 0, left: 0, top: 0 }
  private overlayTimers: Partial<Record<ViewportTarget, number>> = {}
  private selectedLayerId = ''
  private undoStack: EditorSnapshot[] = []
  private redoStack: EditorSnapshot[] = []
  private isRestoringHistory = false
  private isSyncingLayers = false
  private pendingBrushButton = 0
  private pendingPaintLayerId = ''
  private rightPaintStroke?: { pointerId: number; points: { x: number; y: number }[] }
  private _shapeDrawState: { startX: number; startY: number } | null = null
  private _selectionDragState: SelectionDragState | null = null
  private upperCanvasElement?: HTMLCanvasElement
  private maskStrokeMode: 'paint' | 'erase' = 'paint'
  private maskStrokePoints: RegionPoint[] = []
  private activeStrokeClipCanvas?: HTMLCanvasElement
  private maskStreamUpdateScheduled = false
  private maskPreviewTimer?: number
  private lastMaskPreviewAt = 0
  private selectionTransformLayerId = ''
  private activeMaskScope?: MaskScope
  private activeMaskRegionId = ''
  private maskCanvases = new Map<MaskScope, HTMLCanvasElement>()
  private channelMaskCanvases = new Map<string, HTMLCanvasElement>()
  private channelMaskRevisions = new Map<string, number>()
  private parameterMaskDataUrlCache = new Map<string, { revision: number; value: string | undefined }>()
  private parameterMaskBlobCache = new Map<string, { revision: number; value: Blob | null | undefined }>()
  private layerConditionRevisions = new Map<string, number>()
  private layerRegionConditionImageCache = new Map<string, { signature: string; value: string }>()
  private layerRegionConditionImageBlobCache = new Map<string, { signature: string; value: Blob | null }>()
  private layerRegionMaskDataUrlCache = new Map<string, { signature: string; value: string }>()
  private layerRegionMaskBlobCache = new Map<string, { signature: string; value: Blob | null }>()
  private layerMaskNegated = new Map<string, boolean>()
  private colorPopupDrag?: { pointerId: number; startX: number; startY: number; x: number; y: number; target: HTMLElement }
  private videoObjectUrl = ''
  private videoReversing = false
  private _videoOffscreenCanvas: HTMLCanvasElement | null = null
  private _videoRafId: number | null = null
  private liveStreamUpdateRaf?: number
  private maskStreamUpdateRaf?: number
  private _syncingToStore = false
  private liveInputRevision = 0
  private queuedLiveInputRevision = 0
  private liveStreamUpdateScheduled = false
  private liveStreamResourceRefreshScheduled = false
  private liveStreamLayerConditionRefreshScheduled = false
  private lastLayerConditionTopologySignature = ''
  private layerMediaSurfaces = new Map<string, LayerMediaSurface>()
  private layerMediaRefreshRaf?: number
  private drawSignalRefreshRaf?: number
  private upperCanvasLiveInputRaf?: number
  private upperCanvasLiveInputPending = false
  private layerMediaDirtyIds = new Set<string>()
  private layerPreviewCache = new Map<string, string>()
  private inputCanvasMediaStream?: MediaStream
  private inputCanvasMediaTrack?: MediaStreamTrack
  private inputCanvasMediaSource?: HTMLCanvasElement
  private inputCanvasCaptureCanvas?: HTMLCanvasElement
  private inputCanvasCaptureRaf?: number
  private readonly inputCanvasCaptureFps = 0
  private inputCanvasMediaCaptureFps = 0
  private inputCanvasMediaRegistered = false
  private streamOutputBackingCanvas?: HTMLCanvasElement
  private streamOutputVideo?: HTMLVideoElement
  private streamOutputVideoRaf?: number
  private streamOutputFrameSeq = 0
  private signalInspectVideo?: HTMLVideoElement
  private signalInspectRaf?: number
  private signalInspectCanvasStreams = new Map<string, { canvas: HTMLCanvasElement; stream: MediaStream; track?: MediaStreamTrack }>()
  private pendingRgbaPlacementId = ''
  private pendingRgbaPlacementLayerId = ''
  private transformOverlayRaf?: number
  private svgTransformDrag?: SvgTransformDrag

  private _onRtcFrame = (e: Event) => {
    const src = (e as CustomEvent<string>).detail
    const seq = ++this.streamOutputFrameSeq
    const backing = this.ensureStreamOutputBackingCanvas()
    registerDebugCanvasSurface('output', 'output', backing)
    const image = new Image()
    image.onload = () => {
      if (seq !== this.streamOutputFrameSeq) return
      this.drawStreamOutputFrame(image)
    }
    image.src = src
  }

  private _onRtcOutputStream = (event: Event) => {
    const stream = (event as CustomEvent<{ stream?: MediaStream }>).detail?.stream
    if (!stream) return
    this.attachStreamOutputVideo(stream)
  }

  private attachStreamOutputVideo(stream?: MediaStream) {
    if (!stream) return
    const video = this.streamOutputVideoElement ?? this.streamOutputVideo ?? document.createElement('video')
    this.streamOutputVideo = video
    video.muted = true
    video.autoplay = true
    video.playsInline = true
    if (video.srcObject !== stream) video.srcObject = stream
    void video.play().catch(() => undefined)
    this.scheduleStreamOutputVideoDraw()
  }

  private scheduleStreamOutputVideoDraw() {
    if (this.streamOutputVideoRaf !== undefined) return
    this.streamOutputVideoRaf = window.requestAnimationFrame(() => {
      this.streamOutputVideoRaf = undefined
      const video = this.streamOutputVideo
      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        if (video?.srcObject) this.scheduleStreamOutputVideoDraw()
        return
      }
      this.drawStreamOutputFrame(video)
      if (video.srcObject) this.scheduleStreamOutputVideoDraw()
    })
  }

  private _onSceneResourcesUpdate = () => {
    this.resourceDebugVersion++
    this.ensureSelectedOutputSignal()
    this.scheduleSignalInspectionDraw()
  }

  private _onDebugSurfacesUpdate = () => {
    this.resourceDebugVersion++
    this.ensureSelectedOutputSignal()
    this.scheduleSignalInspectionDraw()
  }

  private _onRtcMediaPlaneReady = (event?: Event) => {
    const detail = (event as CustomEvent<{ force?: boolean }> | undefined)?.detail
    if (detail?.force) this.inputCanvasMediaRegistered = false
    this.ensureInputCanvasMediaTrack()
    this.syncLayerMediaSurfaces()
    this.scheduleLayerMediaSurfaceRefresh()
  }

  private _onRtcInputFrameRequest = () => {
    this.ensureInputCanvasMediaTrack()
    this.scheduleInputCanvasCaptureRefresh(true)
  }

  connectedCallback() {
    super.connectedCallback()
    this.addEventListener('paste', this.handlePaste as EventListener)
    this.addEventListener('keydown', this.handleKeyDown as EventListener)
    this.setAttribute('tabindex', '0')
    window.addEventListener('rtd:rtc-frame', this._onRtcFrame)
    window.addEventListener('rtd:rtc-output-stream', this._onRtcOutputStream as EventListener)
    window.addEventListener('rtd:scene-resources-update', this._onSceneResourcesUpdate)
    window.addEventListener('rtd:debug-surfaces-update', this._onDebugSurfacesUpdate)
    window.addEventListener('rtd:rtc-media-plane-ready', this._onRtcMediaPlaneReady)
    window.addEventListener('rtd:rtc-input-frame-request', this._onRtcInputFrameRequest)
  }

  firstUpdated() {
    this.setupCanvas()
    this.setupMaskCanvas()
    this.registerInputDebugSurfaces()
    this.seedCanvas()
    $layer.subscribe((state) => {
      if (this._syncingToStore) return
      if (state.selectedLayerId && state.selectedLayerId !== this.selectedLayerId) {
        this.selectLayer(state.selectedLayerId)
      }
    })
    // Sync canvas tool/brush/viewport state from store (toolbar writes here)
    $canvas.subscribe((s) => {
      if (this._syncingToStore) return
      if (s.toolMode !== this.toolMode) this.setTool(s.toolMode)
      if (s.brushSize !== this.brushSize) { this.brushSize = s.brushSize; this.configureFabricBrush(0) }
      if (s.brushHardness !== this.brushHardness) { this.brushHardness = s.brushHardness; this.configureFabricBrush(0) }
      if (s.fillEdgeStrategy !== this.fillEdgeStrategy) this.fillEdgeStrategy = s.fillEdgeStrategy
      if (s.fillTolerance !== this.fillTolerance) this.fillTolerance = s.fillTolerance
      if (s.shapeDrawMode !== this.shapeDrawMode) this.shapeDrawMode = s.shapeDrawMode
      if (s.textFontSize !== this.textFontSize) this.textFontSize = s.textFontSize
      if (s.brushColor !== this.brushColor) { this.brushColor = s.brushColor; this.configureFabricBrush(0) }
      if (s.secondaryBrushColor !== this.secondaryBrushColor) this.secondaryBrushColor = s.secondaryBrushColor
      if (s.activeColorSlot !== this.activeColorSlot) this.activeColorSlot = s.activeColorSlot
      if (s.editorDest !== this.editorDest) this.setEditorDest(s.editorDest)
      if (s.inputZoom !== this.inputZoom) this.inputZoom = s.inputZoom
      if (s.outputZoom !== this.outputZoom) this.outputZoom = s.outputZoom
      if (s.inputPaintVisible !== this.inputPaintVisible) this.inputPaintVisible = s.inputPaintVisible
      if (s.inputMaskVisible !== this.inputMaskVisible) this.inputMaskVisible = s.inputMaskVisible
      if (s.inputChannelVisible !== this.inputChannelVisible) { this.inputChannelVisible = s.inputChannelVisible; this.scheduleMaskPreviewRefresh(true) }
      if (s.inputActiveMaskOnly !== this.inputActiveMaskOnly) { this.inputActiveMaskOnly = s.inputActiveMaskOnly; this.scheduleMaskPreviewRefresh(true) }
      if (s.activeMaskChannel !== this.activeMaskChannel) this.setActiveMaskChannel(s.activeMaskChannel)
      if (s.maskBrushValue !== this.maskBrushValue) this.maskBrushValue = s.maskBrushValue
      if (s.colorPlannerMode !== this.colorPlannerMode) this.colorPlannerMode = s.colorPlannerMode
      if (s.noDefaultMaskLayers !== this.noDefaultMaskLayers) this.noDefaultMaskLayers = s.noDefaultMaskLayers
    })
    // Sync stage dimensions from scene store
    $scene.subscribe((s) => {
      if (this._syncingToStore) return
      if (s.stageWidth !== this.stageWidth || s.stageHeight !== this.stageHeight) {
        this.resizeStage(s.stageWidth, s.stageHeight)
      }
    })
    // Sync output frames and RTC video state from stream store
    $stream.subscribe((s) => {
      if (s.outputImage !== this.outputImage) this.outputImage = s.outputImage
      if (s.outputImageNonce !== this.outputImageNonce) this.outputImageNonce = s.outputImageNonce
      if (s.realtimeVideoActive !== this.realtimeVideoActive) this.realtimeVideoActive = s.realtimeVideoActive
      if (s.outputTimings !== this.outputTimings) this.outputTimings = s.outputTimings
      if (s.outputTimingsExpanded !== this.outputTimingsExpanded) this.outputTimingsExpanded = s.outputTimingsExpanded
      if (s.isStreaming) {
        this.ensureInputCanvasMediaTrack()
        this.syncLayerMediaSurfaces()
        this.scheduleLayerMediaSurfaceRefresh()
      } else {
        this.stopInputCanvasMediaTrack()
        this.stopLayerMediaTracks()
      }
    })
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.detachRightPaintHandlers()
    this.fabricCanvas?.dispose()
    if (this.liveStreamUpdateRaf !== undefined) window.cancelAnimationFrame(this.liveStreamUpdateRaf)
    if (this.maskStreamUpdateRaf !== undefined) window.cancelAnimationFrame(this.maskStreamUpdateRaf)
    if (this.transformOverlayRaf !== undefined) window.cancelAnimationFrame(this.transformOverlayRaf)
    if (this.layerMediaRefreshRaf !== undefined) window.cancelAnimationFrame(this.layerMediaRefreshRaf)
    if (this.drawSignalRefreshRaf !== undefined) window.cancelAnimationFrame(this.drawSignalRefreshRaf)
    if (this.streamOutputVideoRaf !== undefined) window.cancelAnimationFrame(this.streamOutputVideoRaf)
    if (this.signalInspectRaf !== undefined) window.cancelAnimationFrame(this.signalInspectRaf)
    this.liveStreamUpdateScheduled = false
    this.maskStreamUpdateScheduled = false
    this.liveStreamUpdateRaf = undefined
    this.maskStreamUpdateRaf = undefined
    this.transformOverlayRaf = undefined
    this.layerMediaRefreshRaf = undefined
    this.drawSignalRefreshRaf = undefined
    this.streamOutputVideoRaf = undefined
    this.signalInspectRaf = undefined
    if (this.streamOutputVideo) this.streamOutputVideo.srcObject = null
    if (this.signalInspectVideo) this.signalInspectVideo.srcObject = null
    this.disposeSignalInspectCanvasStreams()
    this.disposeLayerMediaSurfaces()
    this.stopInputCanvasMediaTrack()
    unregisterDebugSurface('input/frame/canvas')
    unregisterDebugSurface('input/mask/active')
    unregisterDebugSurface('input/mask/preview')
    unregisterDebugSurface('output')
    this.removeEventListener('paste', this.handlePaste as EventListener)
    this.removeEventListener('keydown', this.handleKeyDown as EventListener)
    window.removeEventListener('rtd:rtc-frame', this._onRtcFrame)
    window.removeEventListener('rtd:rtc-output-stream', this._onRtcOutputStream as EventListener)
    window.removeEventListener('rtd:scene-resources-update', this._onSceneResourcesUpdate)
    window.removeEventListener('rtd:debug-surfaces-update', this._onDebugSurfacesUpdate)
    window.removeEventListener('rtd:rtc-media-plane-ready', this._onRtcMediaPlaneReady)
    window.removeEventListener('rtd:rtc-input-frame-request', this._onRtcInputFrameRequest)
  }

  protected updated(changedProperties: Map<PropertyKey, unknown>) {
    if (changedProperties.has('stageWidth') || changedProperties.has('stageHeight') || changedProperties.has('inputZoom')) {
      this.syncCanvasGeometry()
      this.scheduleTransformOverlayUpdate()
    }
    if (changedProperties.has('regions')) this.syncRegionsToStore()
    if (changedProperties.has('selectionRect') || changedProperties.has('selectedEntity')) {
      this.syncSelectionToStore()
      this.scheduleTransformOverlayUpdate()
    }
    if (changedProperties.has('selectedEntity') || changedProperties.has('layerPanelTab') || changedProperties.has('regions') || changedProperties.has('activeMaskChannel') || changedProperties.has('inputMaskVisible') || changedProperties.has('inputChannelVisible') || changedProperties.has('sceneMaskNegated') || changedProperties.has('sceneMaskAbsolute') || changedProperties.has('stageWidth') || changedProperties.has('stageHeight')) {
      this.applyToolMode()
      this.syncMaskScope()
      this.scheduleMaskPreviewRefresh()
    }
    if (changedProperties.has('realtimeVideoActive') || changedProperties.has('stageWidth') || changedProperties.has('stageHeight') || changedProperties.has('outputZoom')) {
      this.syncVisibleStreamOutputCanvas()
      this.attachStreamOutputVideo(getWebRtcOutputMediaStream())
    }
    if (changedProperties.has('selectedOutputSignal') || changedProperties.has('resourceDebugVersion') || changedProperties.has('stageWidth') || changedProperties.has('stageHeight')) {
      this.syncSignalInspectionSurface()
    }
    if (changedProperties.has('signalPickerOpen') || changedProperties.has('selectedOutputSignal') || changedProperties.has('resourceDebugVersion')) {
      this.syncSignalPickerPreviewStreams()
    }
  }

  public applyOutput() {
    if (!this.outputImage || !this.fabricCanvas) return
    this.pushHistory()
    FabricImage.fromURL(this.outputImage).then((image) => {
      this.fitImage(image)
      image.set({ name: 'Applied output' })
      this.fabricCanvas?.add(image)
      this.fabricCanvas?.setActiveObject(image)
      this.fabricCanvas?.requestRenderAll()
      this.clearMask()
      this.syncLayers()
    })
  }

  public async exportFrame(): Promise<string> {
    return this.exportCanvasImage(true, 'image/png') ?? ''
  }

  public exportMaskForStream(): string {
    return this.exportMask()
  }

  public getCanvasData() {
    return {
      layers: this.layers,
      regions: this.regions,
      selectionRect: this.selectionRect,
      stageWidth: this.stageWidth,
      stageHeight: this.stageHeight,
      selectedLayerId: this.selectedLayerId,
    }
  }

  render() {
    return html`
      <div class="canvas-editor">
        <div class="viewer"
          @dragover=${this.allowDrop}
          @drop=${this.dropImages}
        >
          <div class="canvas-pane work-pane"
            @pointermove=${(event: PointerEvent) => this.showViewportOverlay(event, 'input')}
            @pointerleave=${(event: PointerEvent) => this.leaveViewport(event, 'input')}
          >
            <div class=${this.inputOverlayVisible || this.editorDest === 'mask' ? 'viewport-overlay pane-head visible' : 'viewport-overlay pane-head'}>
              <strong>Input</strong>
              <span>${this.stageWidth} x ${this.stageHeight}</span>
              <div class="viewport-toggles">
                <button title="Load RGBA mask image" @click=${this.loadRgbaMaskImage}>Load RGBA</button>
                <button title="Add empty RGBA mask" @click=${this.addEmptyRgbaMask}>Empty RGBA</button>
                <button class=${this.inputPaintVisible ? 'active' : ''} @click=${() => { this.inputPaintVisible = !this.inputPaintVisible; $canvas.setKey('inputPaintVisible', this.inputPaintVisible) }}>Color</button>
                <button class=${this.inputMaskVisible ? 'active' : ''} @click=${() => { this.inputMaskVisible = !this.inputMaskVisible; $canvas.setKey('inputMaskVisible', this.inputMaskVisible); this.scheduleMaskPreviewRefresh(true) }}>Masks</button>
                <button class=${this.inputChannelVisible ? 'active' : ''} @click=${() => { this.inputChannelVisible = !this.inputChannelVisible; $canvas.setKey('inputChannelVisible', this.inputChannelVisible); this.scheduleMaskPreviewRefresh(true) }}>Channels</button>
                <button class=${this.editorDest === 'paint' ? 'active' : ''} @click=${() => this.setEditorDest('paint')}>To Paint</button>
                <button class=${this.editorDest === 'mask' ? 'active' : ''} @click=${() => this.setEditorDest('mask')}>To Masks</button>
                <button class=${this.inputActiveMaskOnly ? 'active' : ''} @click=${() => { this.inputActiveMaskOnly = !this.inputActiveMaskOnly; $canvas.setKey('inputActiveMaskOnly', this.inputActiveMaskOnly); this.scheduleMaskPreviewRefresh(true) }}>Active</button>
              </div>
            </div>
            <div class="work-surface"
              @wheel=${(event: WheelEvent) => this.zoomViewport(event, 'input')}
              @scroll=${() => this.scheduleTransformOverlayUpdate()}
              @pointerdown=${this.startViewportPan}
              @pointermove=${(event: PointerEvent) => this.moveViewport(event, 'input')}
              @pointerup=${this.endViewportPan}
              @contextmenu=${(event: Event) => event.preventDefault()}
            >
              <div class="stage-wrap">
                <div class=${this.inputPaintVisible ? 'stage' : 'stage paint-hidden'}
                  style="--stage-width: ${this.stageWidth}; --stage-height: ${this.stageHeight}; --zoom: ${this.inputZoom};"
                  @pointermove=${this.trackCursor}
                  @pointerenter=${this.showCursor}
                  @pointerleave=${this.hideCursor}
                >
                  <canvas id="edit-canvas"></canvas>
                  <canvas id="input-capture-canvas" class="input-capture-canvas" aria-hidden="true"></canvas>
                  <canvas id="mask-preview-canvas" class=${this.maskPreviewVisible() ? 'mask-preview visible' : 'mask-preview'}></canvas>
                  <canvas id="mask-canvas"
                    class=${['mask', this.maskCanvasActive() && 'active', (this.toolMode === 'brush' || this.toolMode === 'eraser') && 'brush-mode'].filter(Boolean).join(' ')}
                    @pointerdown=${this.startOverlayStroke}
                    @pointermove=${this.moveOverlayStroke}
                    @pointerup=${this.endOverlayStroke}
                    @pointerleave=${this.endOverlayStroke}
                  ></canvas>
                  <canvas id="editor-overlay"></canvas>
                  ${this.renderSelectionOverlay()}
                  <div class=${this.cursor.visible && (this.toolMode === 'brush' || this.toolMode === 'eraser') ? 'cursor visible' : 'cursor'}></div>
                </div>
              </div>
            </div>
            ${this.renderTransformOverlay()}
            <div class=${this.inputOverlayVisible || this.editorDest === 'mask' ? 'viewport-overlay pane-foot visible' : 'viewport-overlay pane-foot'}>
              <span>${this.toolMode}</span>
            </div>
          </div>
          <div class="output-pane work-pane"
            @pointermove=${(event: PointerEvent) => this.showViewportOverlay(event, 'output')}
            @pointerleave=${(event: PointerEvent) => this.leaveViewport(event, 'output')}
          >
            <div class=${this.outputOverlayVisible ? 'viewport-overlay pane-head visible' : 'viewport-overlay pane-head'}>
              <strong>Output</strong>
              <span>${this.stageWidth} x ${this.stageHeight}</span>
              <span>${Math.round(this.outputZoom * 100)}%</span>
              <div class="output-overlay-tools">
                ${this.renderOutputTimingToggle()}
                <button
                  class=${this.signalPickerOpen ? 'signal-grid-toggle active' : 'signal-grid-toggle'}
                  title="Open signal grid"
                  @click=${this.toggleSignalPicker}
                >
                  ${this.selectedOutputSignal === 'output' ? 'Signals' : this.selectedOutputSignal}
                </button>
              </div>
            </div>
            ${this.renderOutputTimingPanel()}
            ${this.signalPickerOpen ? this.renderSignalPickerGrid() : null}
            <div class="work-surface"
              @wheel=${(event: WheelEvent) => this.zoomViewport(event, 'output')}
              @pointerdown=${this.startViewportPan}
              @pointermove=${(event: PointerEvent) => this.moveViewport(event, 'output')}
              @pointerup=${this.endViewportPan}
              @contextmenu=${(event: Event) => event.preventDefault()}
            >
              <div class="stage-wrap">
                <div class=${this.outputPreviewClass()} style=${this.outputPreviewStyle()}>
                    ${this.realtimeVideoActive
                      ? html`<canvas id="stream-output-canvas" class="stream-output-video" aria-label="realtime output"></canvas><video id="stream-output-video" class="stream-output-source" muted autoplay playsinline></video>`
                      : this.activeMotionVideo
                      ? html`<video class=${this.activeMotionIsStream && !this.motionStreamHasSegments ? 'stream-pending' : ''} src=${this.activeMotionVideo} autoplay muted playsinline controls ?loop=${!this.activeMotionIsStream}></video>`
                      : this.activeMotionFrame || this.outputImage
                          ? keyed(this.outputImageNonce, html`<img src=${this.activeMotionFrame || this.outputImage} alt="latest generated frame" />`)
                        : html`<div class="empty">Waiting for diffusion</div>`}
                    ${this.selectedOutputSignal !== 'output'
                      ? html`<video id="signal-output-video" class="signal-output-video signal-output-overlay" muted autoplay playsinline></video>`
                      : null}
                </div>
              </div>
            </div>
            <div class=${this.outputOverlayVisible ? 'viewport-overlay pane-foot pane-action-overlay visible' : 'viewport-overlay pane-foot pane-action-overlay'}>
              <button @click=${this.applyOutput}>Apply output as layer</button>
            </div>
          </div>
        </div>
        ${this.renderColorPopup()}
        <video id="video-input" style="display:none" muted playsinline preload="auto" @loadedmetadata=${this.onVideoMetadata} @ended=${this.onVideoEnded}></video>
      </div>
    `
  }

  private renderTransformOverlay() {
    const overlay = this.transformOverlay
    if (!overlay) return null
    return html`<svg class="transform-svg-overlay" aria-hidden="true" @wheel=${(event: WheelEvent) => this.zoomViewport(event, 'input')}>
      <line class="transform-rotate-line" x1=${overlay.rotate.anchorX} y1=${overlay.rotate.anchorY} x2=${overlay.rotate.x} y2=${overlay.rotate.y} vector-effect="non-scaling-stroke"></line>
      <polygon class="transform-box" points=${overlay.points} vector-effect="non-scaling-stroke" @pointerdown=${(event: PointerEvent) => this.startSvgTransform(event, 'move')}></polygon>
      <circle class="transform-rotate-handle" cx=${overlay.rotate.x} cy=${overlay.rotate.y} r="6" @pointerdown=${(event: PointerEvent) => this.startSvgTransform(event, 'rotate')}></circle>
    </svg>
    ${overlay.handles.map((handle) => html`<button
      type="button"
      class="transform-dom-rotate-band"
      data-handle=${handle.key}
      aria-label=${`Rotate near ${handle.key}`}
      style=${`left:${handle.x}px;top:${handle.y}px`}
      @wheel=${(event: WheelEvent) => this.zoomViewport(event, 'input')}
      @pointerdown=${(event: PointerEvent) => this.startSvgTransform(event, 'rotate')}></button>`)}
    ${overlay.handles.map((handle) => html`<button
      type="button"
      class="transform-dom-handle"
      data-handle=${handle.key}
      aria-label=${`Scale ${handle.key}`}
      style=${`left:${handle.x}px;top:${handle.y}px;cursor:${handle.cursor}`}
      @wheel=${(event: WheelEvent) => this.zoomViewport(event, 'input')}
      @pointerdown=${(event: PointerEvent) => this.startSvgTransform(event, handle.key)}></button>`)}`
  }

  private outputTimingSummaryMs() {
    const preferred = [
      'server_end_to_end_ms',
      'server_total_ms',
      'latency_ms',
      'inference_ms',
      'model_inference_ms',
    ]
    for (const key of preferred) {
      const value = this.outputTimings[key]
      if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value
    }
    return 0
  }

  private formatTimingLabel(key: string) {
    const labels: Record<string, string> = {
      queue_wait_ms: 'Queue wait',
      canvas_decode_ms: 'Canvas decode',
      primary_input_select_ms: 'Primary input select',
      input_compose_ms: 'Input compose',
      mask_decode_ms: 'Mask decode',
      layer_frames_decode_ms: 'Layer frames decode',
      layer_conditions_prepare_ms: 'Layer conditions',
      graph_build_ms: 'Graph build',
      graph_run_ms: 'Graph run',
      conditioning_resolve_ms: 'Condition resolve',
      composition_plan_ms: 'Composition plan',
      request_build_ms: 'Request build',
      inference_ms: 'Inference wrapper',
      model_inference_ms: 'Backend inference',
      output_encode_ms: 'Output encode',
      output_persist_ms: 'Output persist',
      persist_output_ms: 'Output persist',
      debug_persist_ms: 'Debug persist',
      server_total_ms: 'Server total',
      server_end_to_end_ms: 'End to end',
    }
    return labels[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (match) => match.toUpperCase())
  }

  private outputTimingEntries() {
    const preferred = [
      'queue_wait_ms',
      'canvas_decode_ms',
      'primary_input_select_ms',
      'input_compose_ms',
      'mask_decode_ms',
      'layer_frames_decode_ms',
      'layer_conditions_prepare_ms',
      'graph_build_ms',
      'graph_run_ms',
      'conditioning_resolve_ms',
      'composition_plan_ms',
      'request_build_ms',
      'inference_ms',
      'model_inference_ms',
      'output_encode_ms',
      'output_persist_ms',
      'debug_persist_ms',
      'server_total_ms',
      'server_end_to_end_ms',
    ]
    const keys = [...preferred, ...Object.keys(this.outputTimings).filter((key) => !preferred.includes(key))]
    return keys
      .map((key) => ({ key, value: this.outputTimings[key] }))
      .filter((entry): entry is { key: string; value: number } => typeof entry.value === 'number' && Number.isFinite(entry.value) && entry.value >= 0)
  }

  private toggleOutputTimings = (event?: Event) => {
    event?.preventDefault()
    event?.stopPropagation()
    const next = !this.outputTimingsExpanded
    this.outputTimingsExpanded = next
    $stream.setKey('outputTimingsExpanded', next)
  }

  private renderOutputTimingToggle() {
    const summaryMs = this.outputTimingSummaryMs()
    if (summaryMs <= 0) return null
    return html`<button
      class=${this.outputTimingsExpanded ? 'output-timing-chip active' : 'output-timing-chip'}
      title="Show backend timing breakdown"
      @click=${this.toggleOutputTimings}
    >${summaryMs >= 100 ? Math.round(summaryMs) : summaryMs.toFixed(1)} ms</button>`
  }

  private renderOutputTimingPanel() {
    const items = this.outputTimingEntries()
    if (!this.outputTimingsExpanded || !items.length) return null
    const queuedAt = typeof this.outputTimings.queued_at === 'string' ? this.outputTimings.queued_at : ''
    const renderStartedAt = typeof this.outputTimings.render_started_at === 'string' ? this.outputTimings.render_started_at : ''
    return html`<div class="output-timing-panel">
      <div class="output-timing-head">
        <strong>Backend timings</strong>
        <button @click=${this.toggleOutputTimings}>Hide</button>
      </div>
      <div class="output-timing-grid">
        ${items.map((entry) => html`<div class="output-timing-row"><span>${this.formatTimingLabel(entry.key)}</span><strong>${entry.value >= 100 ? entry.value.toFixed(0) : entry.value.toFixed(1)} ms</strong></div>`)}
      </div>
      ${queuedAt || renderStartedAt ? html`<div class="output-timing-meta">
        ${queuedAt ? html`<span>Queued ${queuedAt}</span>` : null}
        ${renderStartedAt ? html`<span>Started ${renderStartedAt}</span>` : null}
      </div>` : null}
    </div>`
  }

  private clearTransformOverlay() {
    if (this.transformOverlayPoints) this.transformOverlayPoints = ''
    if (this.transformOverlay) this.transformOverlay = undefined
  }

  private scheduleTransformOverlayUpdate() {
    if (this.transformOverlayRaf !== undefined) return
    this.transformOverlayRaf = window.requestAnimationFrame(() => {
      this.transformOverlayRaf = undefined
      this.updateTransformOverlay()
    })
  }

  private updateTransformOverlay() {
    if (this.toolMode !== 'select' || this.editorDest !== 'paint') {
      this.clearTransformOverlay()
      return
    }
    const targetLayerId = this.transformTargetLayerId()
    const frame = this.selectionRect ? this.selectionTransformFrame() : targetLayerId ? this.layerTransformFrame(targetLayerId) : undefined
    if (!frame || !this.stageElement || !this.inputPaneElement) {
      this.clearTransformOverlay()
      return
    }
    const stageRect = this.stageElement.getBoundingClientRect()
    const paneRect = this.inputPaneElement.getBoundingClientRect()
    const panePoints = frame.corners.map((point) => {
      const x = stageRect.left - paneRect.left + point.x * this.inputZoom
      const y = stageRect.top - paneRect.top + point.y * this.inputZoom
      return { x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10 }
    })
    const [topLeft, topRight, bottomRight, bottomLeft] = panePoints
    if (!topLeft || !topRight || !bottomRight || !bottomLeft) {
      this.clearTransformOverlay()
      return
    }
    const midpoint = (a: { x: number; y: number }, b: { x: number; y: number }) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })
    const center = { x: (topLeft.x + bottomRight.x) / 2, y: (topLeft.y + bottomRight.y) / 2 }
    const top = midpoint(topLeft, topRight)
    const right = midpoint(topRight, bottomRight)
    const bottom = midpoint(bottomRight, bottomLeft)
    const left = midpoint(bottomLeft, topLeft)
    const normalLength = Math.hypot(top.x - center.x, top.y - center.y) || 1
    const rotate = {
      x: top.x + ((top.x - center.x) / normalLength) * 28,
      y: top.y + ((top.y - center.y) / normalLength) * 28,
      anchorX: top.x,
      anchorY: top.y,
    }
    const handles: TransformOverlayHandle[] = [
      { key: 'nw', ...topLeft, cursor: 'nwse-resize' },
      { key: 'n', ...top, cursor: 'ns-resize' },
      { key: 'ne', ...topRight, cursor: 'nesw-resize' },
      { key: 'e', ...right, cursor: 'ew-resize' },
      { key: 'se', ...bottomRight, cursor: 'nwse-resize' },
      { key: 's', ...bottom, cursor: 'ns-resize' },
      { key: 'sw', ...bottomLeft, cursor: 'nesw-resize' },
      { key: 'w', ...left, cursor: 'ew-resize' },
    ]
    const points = panePoints.map((point) => `${point.x},${point.y}`).join(' ')
    const next = { points, handles, rotate, frame }
    if (points !== this.transformOverlayPoints) this.transformOverlayPoints = points
    this.transformOverlay = next
  }

  private layerTransformFrame(layerId: string): SvgTransformFrame | undefined {
    const object = this.findLayer(layerId)
    if (!object) return undefined
    object.setCoords()
    const angle = object.angle ?? 0
    const origin = object.getCenterPoint()
    const objects = [object, ...this.childrenForLayer(layerId)]
    const localPoints = objects.flatMap((item) => {
      item.setCoords()
      return item.getCoords().map((point) => this.rotatePoint(this.subtractPoints(point, origin), -angle))
    })
    if (!localPoints.length) return undefined
    const xs = localPoints.map((point) => point.x)
    const ys = localPoints.map((point) => point.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const width = Math.max(1, maxX - minX)
    const height = Math.max(1, maxY - minY)
    const centerLocal = new Point((minX + maxX) / 2, (minY + maxY) / 2)
    const center = this.addPoints(origin, this.rotatePoint(centerLocal, angle))
    const localCorners = [
      new Point(minX, minY),
      new Point(maxX, minY),
      new Point(maxX, maxY),
      new Point(minX, maxY),
    ]
    const corners = localCorners.map((point) => this.addPoints(origin, this.rotatePoint(point, angle)))
    return { center, angle, width, height, corners }
  }

  private selectionTransformFrame(): SvgTransformFrame | undefined {
    if (!this.selectionRect) return undefined
    const selection = this.normalizedSelection()
    const center = new Point(selection.x + selection.width / 2, selection.y + selection.height / 2)
    const corners = [
      new Point(selection.x, selection.y),
      new Point(selection.x + selection.width, selection.y),
      new Point(selection.x + selection.width, selection.y + selection.height),
      new Point(selection.x, selection.y + selection.height),
    ]
    return { center, angle: 0, width: selection.width, height: selection.height, corners }
  }

  private transformTargetLayerId() {
    if (this.selectionTransformLayerId && this.findLayer(this.selectionTransformLayerId)) return this.selectionTransformLayerId
    return this.selectedEntity === 'layer' && this.selectedLayerId ? this.selectedLayerId : ''
  }

  private renderLayerBlockCanvas(layerId: string, object = this.findLayer(layerId)) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    if (!this.fabricCanvas || !object) return canvas
    const source = this.fabricCanvas.getElement()
    if (!source) return canvas
    const blockSet = new Set<FabricObject>([object, ...this.childrenForLayer(layerId)])
    const allObjects = this.fabricCanvas.getObjects()
    const visibility = new Map<FabricObject, boolean | undefined>()
    const opacity = new Map<FabricObject, number | undefined>()
    const active = this.fabricCanvas.getActiveObject()
    const wasSyncingLayers = this.isSyncingLayers
    try {
      this.isSyncingLayers = true
      for (const candidate of allObjects) {
        visibility.set(candidate, candidate.visible)
        opacity.set(candidate, candidate.opacity)
        candidate.set('visible', blockSet.has(candidate))
        if (blockSet.has(candidate)) candidate.set('opacity', 1)
      }
      this.fabricCanvas.discardActiveObject()
      this.fabricCanvas.renderAll()
      context.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
    } finally {
      for (const candidate of allObjects) {
        candidate.set('visible', visibility.get(candidate))
        candidate.set('opacity', opacity.get(candidate))
      }
      if (active) this.fabricCanvas.setActiveObject(active)
      this.fabricCanvas.renderAll()
      this.isSyncingLayers = wasSyncingLayers
    }
    return canvas
  }

  private prepareSelectionPixelTransform() {
    if (!this.fabricCanvas || !this.selectionRect) return false
    const existingFloating = this.selectionTransformLayerId ? this.findLayer(this.selectionTransformLayerId) as (FabricObject & { __parentLayerId?: string }) | undefined : undefined
    if (existingFloating) {
      this.selectedEntity = 'layer'
      this.selectedLayerId = existingFloating.__parentLayerId || this.selectedLayerId
      this.fabricCanvas.setActiveObject(existingFloating)
      return true
    }
    if (this.selectionRect.inverted) return false
    const layerId = this.selectedLayerId
    const object = layerId ? this.findLayer(layerId) : undefined
    if (!layerId || !object || isPaintChildObject(object as FabricObject & { __paintChild?: boolean; __parentLayerId?: string })) return false
    const blockObjects = [object, ...this.childrenForLayer(layerId)]
    const blockSet = new Set<FabricObject>(blockObjects)
    this.pushHistory()
    const source = this.fabricCanvas.getElement()
    if (!source) return false
    const layerCanvas = document.createElement('canvas')
    layerCanvas.width = this.stageWidth
    layerCanvas.height = this.stageHeight
    const layerContext = layerCanvas.getContext('2d')!
    const allObjects = this.fabricCanvas.getObjects()
    const visibility = new Map<FabricObject, boolean | undefined>()
    const opacity = new Map<FabricObject, number | undefined>()
    const active = this.fabricCanvas.getActiveObject()
    const layerName = object.get('name') || 'Selection edit'
    const layerOpacity = object.opacity ?? 1
    const layerVisible = object.visible ?? true
    const preset = (object as FabricObject & { __preset?: LayerPreset }).__preset
    const isRgbaMaskLayer = !!(object as FabricObject & { __isRgbaMaskLayer?: boolean }).__isRgbaMaskLayer
    try {
      this.isSyncingLayers = true
      for (const candidate of allObjects) {
        visibility.set(candidate, candidate.visible)
        opacity.set(candidate, candidate.opacity)
        candidate.set('visible', blockSet.has(candidate))
        if (blockSet.has(candidate)) candidate.set('opacity', 1)
      }
      this.fabricCanvas.discardActiveObject()
      this.fabricCanvas.renderAll()
      layerContext.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
    } finally {
      for (const candidate of allObjects) {
        candidate.set('visible', visibility.get(candidate))
        candidate.set('opacity', opacity.get(candidate))
      }
      if (active) this.fabricCanvas.setActiveObject(active)
      this.fabricCanvas.renderAll()
      this.isSyncingLayers = false
    }
    const selection = this.normalizedSelection()
    const floatingWidth = Math.max(1, Math.ceil(selection.width))
    const floatingHeight = Math.max(1, Math.ceil(selection.height))
    const baseCanvas = document.createElement('canvas')
    const floatingCanvas = document.createElement('canvas')
    baseCanvas.width = this.stageWidth
    baseCanvas.height = this.stageHeight
    floatingCanvas.width = floatingWidth
    floatingCanvas.height = floatingHeight
    const baseContext = baseCanvas.getContext('2d')!
    const floatingContext = floatingCanvas.getContext('2d')!
    baseContext.drawImage(layerCanvas, 0, 0, this.stageWidth, this.stageHeight)
    floatingContext.save()
    floatingContext.translate(-selection.x, -selection.y)
    this.clipContextToSelection(floatingContext, this.selectionRect)
    floatingContext.drawImage(layerCanvas, 0, 0, this.stageWidth, this.stageHeight)
    floatingContext.restore()
    baseContext.save()
    baseContext.globalCompositeOperation = 'destination-out'
    this.clipContextToSelection(baseContext, this.selectionRect)
    baseContext.fillStyle = '#000'
    baseContext.fillRect(0, 0, this.stageWidth, this.stageHeight)
    baseContext.restore()

    const base = new FabricImage(baseCanvas, { left: 0, top: 0, originX: 'left', originY: 'top', name: layerName, opacity: layerOpacity, visible: layerVisible }) as FabricImage & { __uid?: string; __preset?: LayerPreset; __isRgbaMaskLayer?: boolean }
    const floating = new FabricImage(floatingCanvas, { left: selection.x, top: selection.y, originX: 'left', originY: 'top', name: 'Selected pixels' }) as FabricImage & { __uid?: string; __paintChild?: boolean; __parentLayerId?: string }
    base.__uid = layerId
    if (preset) base.__preset = this.normalizeLayerPreset(preset)
    else this.attachPreset(base as unknown as FabricObject)
    if (isRgbaMaskLayer) base.__isRgbaMaskLayer = true
    this.attachPreset(floating as unknown as FabricObject)
    const floatingId = this.assignLayerId(floating as unknown as FabricObject)
    floating.__paintChild = true
    floating.__parentLayerId = layerId
    this.styleTransformControls(base as unknown as FabricObject)
    this.styleTransformControls(floating as unknown as FabricObject)
    this.replaceLayerObject(layerId, object, base as unknown as FabricObject)
    this.fabricCanvas.add(floating as unknown as FabricObject)
    this.selectionTransformLayerId = floatingId
    this.selectedEntity = 'layer'
    this.selectedLayerId = layerId
    this.selectionRect = undefined
    this.fabricCanvas.setActiveObject(floating as unknown as FabricObject)
    this.fabricCanvas.requestRenderAll()
    this.syncLayers()
    this.scheduleProjectPersist()
    this.queueLiveInputChanged(true, true)
    return true
  }

  private clipContextToSelection(context: CanvasRenderingContext2D, selectionRect: SelectionRect) {
    const selection = this.normalizedRegionRect(selectionRect)
    const points = selectionRect.points
    context.beginPath()
    if (points?.length) {
      context.moveTo(points[0].x, points[0].y)
      for (const point of points.slice(1)) context.lineTo(point.x, point.y)
      context.closePath()
    } else if (selectionRect.shape === 'ellipse') {
      context.ellipse(selection.x + selection.width / 2, selection.y + selection.height / 2, selection.width / 2, selection.height / 2, 0, 0, Math.PI * 2)
    } else {
      context.rect(selection.x, selection.y, selection.width, selection.height)
    }
    context.clip()
  }

  private startSvgTransform(event: PointerEvent, mode: SvgTransformMode) {
    if (!this.fabricCanvas || this.toolMode !== 'select' || this.editorDest !== 'paint') return
    if (this.selectionRect && !this.prepareSelectionPixelTransform()) return
    const targetLayerId = this.transformTargetLayerId()
    if (!targetLayerId) return
    const object = this.findLayer(targetLayerId)
    const frame = this.layerTransformFrame(targetLayerId)
    if (!object || !frame) return
    const pointer = this.scenePointFromPanePointer(event)
    const center = frame.center
    const angle = frame.angle
    const width = frame.width
    const height = frame.height
    const scaleX = 1
    const scaleY = 1
    const transformObjects = targetLayerId === this.selectionTransformLayerId ? [object] : [object, ...this.childrenForLayer(targetLayerId)]
    const objects = transformObjects.map((item) => ({
      object: item,
      center: item.getCenterPoint(),
      angle: item.angle ?? 0,
      scaleX: item.scaleX ?? 1,
      scaleY: item.scaleY ?? 1,
    }))
    const drag: SvgTransformDrag = {
      pointerId: event.pointerId,
      mode,
      layerId: targetLayerId,
      objects,
      startPointer: pointer,
      center,
      angle,
      scaleX,
      scaleY,
      width,
      height,
    }
    if (mode === 'rotate') {
      drag.startPointerAngle = Math.atan2(pointer.y - center.y, pointer.x - center.x)
    } else if (mode !== 'move') {
      const sign = this.svgTransformHandleSign(mode)
      const halfWidth = width * scaleX / 2
      const halfHeight = height * scaleY / 2
      const fixedLocal = new Point(sign.x === 0 ? 0 : -sign.x * halfWidth, sign.y === 0 ? 0 : -sign.y * halfHeight)
      drag.handleSign = sign
      drag.fixedLocal = fixedLocal
      drag.fixedScene = this.addPoints(center, this.rotatePoint(fixedLocal, angle))
    }
    this.svgTransformDrag = drag
    this.pushHistory()
    event.preventDefault()
    event.stopPropagation()
    ;(event.currentTarget as Element).setPointerCapture?.(event.pointerId)
    window.addEventListener('pointermove', this.moveSvgTransform, true)
    window.addEventListener('pointerup', this.endSvgTransform, true)
    window.addEventListener('pointercancel', this.endSvgTransform, true)
  }

  private moveSvgTransform = (event: PointerEvent) => {
    const drag = this.svgTransformDrag
    if (!drag || event.pointerId !== drag.pointerId || !this.fabricCanvas) return
    const object = this.findLayer(drag.layerId)
    if (!object) return
    const pointer = this.scenePointFromPanePointer(event)
    if (drag.mode === 'move') {
      const delta = this.subtractPoints(pointer, drag.startPointer)
      for (const state of drag.objects) {
        state.object.setPositionByOrigin(this.addPoints(state.center, delta), 'center', 'center')
        state.object.setCoords()
      }
    } else if (drag.mode === 'rotate') {
      const startAngle = drag.startPointerAngle ?? 0
      const currentAngle = Math.atan2(pointer.y - drag.center.y, pointer.x - drag.center.x)
      const angleDelta = ((currentAngle - startAngle) * 180) / Math.PI
      for (const state of drag.objects) {
        const relative = this.subtractPoints(state.center, drag.center)
        const center = this.addPoints(drag.center, this.rotatePoint(relative, angleDelta))
        state.object.rotate(state.angle + angleDelta)
        state.object.setPositionByOrigin(center, 'center', 'center')
        state.object.setCoords()
      }
    } else {
      this.applySvgScaleTransform(drag, pointer, event.shiftKey)
    }
    object.setCoords()
    this.fabricCanvas.setActiveObject(object)
    this.fabricCanvas.requestRenderAll()
    this.scheduleTransformOverlayUpdate()
    event.preventDefault()
    event.stopPropagation()
  }

  private endSvgTransform = (event: PointerEvent) => {
    const drag = this.svgTransformDrag
    if (!drag || event.pointerId !== drag.pointerId) return
    window.removeEventListener('pointermove', this.moveSvgTransform, true)
    window.removeEventListener('pointerup', this.endSvgTransform, true)
    window.removeEventListener('pointercancel', this.endSvgTransform, true)
    this.svgTransformDrag = undefined
    const object = this.findLayer(drag.layerId)
    if (object) this.afterLayerTransform(object)
    event.preventDefault()
    event.stopPropagation()
  }

  private applySvgScaleTransform(drag: SvgTransformDrag, pointer: Point, keepAspect: boolean) {
    if (!drag.fixedScene || !drag.fixedLocal || !drag.handleSign) return
    const sign = drag.handleSign
    const deltaLocal = this.rotatePoint(this.subtractPoints(pointer, drag.fixedScene), -drag.angle)
    let scaledWidth = Math.max(1, drag.width * drag.scaleX)
    let scaledHeight = Math.max(1, drag.height * drag.scaleY)
    if (sign.x !== 0) scaledWidth = Math.max(1, Math.abs(deltaLocal.x))
    if (sign.y !== 0) scaledHeight = Math.max(1, Math.abs(deltaLocal.y))
    if (keepAspect && sign.x !== 0 && sign.y !== 0) {
      const ratio = Math.max(scaledWidth / Math.max(1, drag.width * drag.scaleX), scaledHeight / Math.max(1, drag.height * drag.scaleY))
      scaledWidth = Math.max(1, drag.width * drag.scaleX * ratio)
      scaledHeight = Math.max(1, drag.height * drag.scaleY * ratio)
    }
    const scaleRatioX = scaledWidth / Math.max(1, drag.width * drag.scaleX)
    const scaleRatioY = scaledHeight / Math.max(1, drag.height * drag.scaleY)
    for (const state of drag.objects) {
      const relative = this.subtractPoints(state.center, drag.fixedScene)
      const local = this.rotatePoint(relative, -drag.angle)
      const scaledLocal = new Point(local.x * scaleRatioX, local.y * scaleRatioY)
      const center = this.addPoints(drag.fixedScene, this.rotatePoint(scaledLocal, drag.angle))
      state.object.set({ scaleX: state.scaleX * scaleRatioX, scaleY: state.scaleY * scaleRatioY })
      state.object.rotate(state.angle)
      state.object.setPositionByOrigin(center, 'center', 'center')
      state.object.setCoords()
    }
  }

  private svgTransformHandleSign(handle: SvgTransformHandle): { x: -1 | 0 | 1; y: -1 | 0 | 1 } {
    const x = handle.includes('w') ? -1 : handle.includes('e') ? 1 : 0
    const y = handle.includes('n') ? -1 : handle.includes('s') ? 1 : 0
    return { x, y }
  }

  private scenePointFromPanePointer(event: PointerEvent) {
    const stageRect = this.stageElement.getBoundingClientRect()
    return new Point((event.clientX - stageRect.left) / this.inputZoom, (event.clientY - stageRect.top) / this.inputZoom)
  }

  private rotatePoint(point: Point, angleDegrees: number) {
    const radians = (angleDegrees * Math.PI) / 180
    const cos = Math.cos(radians)
    const sin = Math.sin(radians)
    return new Point(point.x * cos - point.y * sin, point.x * sin + point.y * cos)
  }

  private addPoints(a: Point, b: Point) {
    return new Point(a.x + b.x, a.y + b.y)
  }

  private subtractPoints(a: Point, b: Point) {
    return new Point(a.x - b.x, a.y - b.y)
  }

  private onVideoMetadata = () => {
    const vid = this.shadowRoot?.querySelector('#video-input') as HTMLVideoElement | null
    if (!vid) return
    this.videoDuration = vid.duration
    vid.currentTime = this.videoStart
  }

  private onVideoEnded = () => {
    const vid = this.shadowRoot?.querySelector('#video-input') as HTMLVideoElement | null
    if (!vid) return
    if (this.videoLoopMode === 'loop') {
      vid.currentTime = this.videoStart
      void vid.play()
    } else if (this.videoLoopMode === 'ping-pong') {
      this.videoReversing = !this.videoReversing
      vid.currentTime = this.videoReversing ? (this.videoEnd > 0 ? this.videoEnd : this.videoDuration) : this.videoStart
      void vid.play()
    }
  }

  private get videoInputElement(): HTMLVideoElement | undefined {
    return this.shadowRoot?.querySelector('#video-input') as HTMLVideoElement | undefined
  }

  private get streamOutputCanvasElement(): HTMLCanvasElement | undefined {
    return this.shadowRoot?.querySelector('#stream-output-canvas') as HTMLCanvasElement | undefined
  }

  private get streamOutputVideoElement(): HTMLVideoElement | undefined {
    return this.shadowRoot?.querySelector('#stream-output-video') as HTMLVideoElement | undefined
  }

  private ensureStreamOutputBackingCanvas() {
    if (!this.streamOutputBackingCanvas) this.streamOutputBackingCanvas = document.createElement('canvas')
    return this.streamOutputBackingCanvas
  }

  private outputSignalOptions() {
    const names = new Set<string>(['output'])
    void this.resourceDebugVersion
    for (const surface of getDebugSurfaces().values()) {
      if (!surface.id) continue
      if (surface.id.startsWith('final/')) names.add(surface.id)
    }
    for (const name of getSceneResourceDebugData().keys()) {
      if (!name || names.has(name)) continue
      if (name.startsWith('final/')) names.add(name)
    }
    return [...names].sort((a, b) => {
      if (a === 'output') return -1
      if (b === 'output') return 1
      return a.localeCompare(b)
    })
  }

  private toggleSignalPicker = (event: Event) => {
    event.preventDefault()
    event.stopPropagation()
    this.signalPickerOpen = !this.signalPickerOpen
  }

  private renderSignalPickerGrid() {
    const signals = this.outputSignalOptions()
    return html`
      <div class="signal-picker-panel">
        <div class="signal-picker-grid">
          ${signals.map((name) => html`
            <button
              class=${this.selectedOutputSignal === name ? 'signal-picker-tile active' : 'signal-picker-tile'}
              title=${name}
              @click=${() => this.selectOutputSignal(name)}
            >
              <div class="signal-picker-thumb">
                <video class="signal-picker-video" data-signal-name=${name} muted autoplay playsinline></video>
              </div>
              <span class="signal-picker-name">${name}</span>
            </button>
          `)}
        </div>
      </div>
    `
  }

  private selectOutputSignal(name: string) {
    // Cancel any pending animation frame
    if (this.signalInspectRaf !== undefined) {
      window.cancelAnimationFrame(this.signalInspectRaf)
      this.signalInspectRaf = undefined
    }

    this.selectedOutputSignal = name || 'output'
    this.signalPickerOpen = false
    this.syncSignalInspectionSurface()
  }

  private ensureSelectedOutputSignal() {
    if (this.selectedOutputSignal === 'output') return
    if (!this.outputSignalOptions().includes(this.selectedOutputSignal)) this.selectedOutputSignal = 'output'
  }

  private selectedSignalSurface(signalName = this.selectedOutputSignal): DebugSurface | undefined {
    const surfaces = getDebugSurfaces()
    if (signalName === 'output') return surfaces.get('output')
    const localSurface = surfaces.get(signalName) ?? [...surfaces.values()].find((surface) => surface.name === signalName)
    if (localSurface) return localSurface
    const signalData = getSceneResourceDebugData()
    const value = signalData.get(signalName) || ''
    if (value.startsWith('stream:')) {
      const id = value.slice('stream:'.length)
      return surfaces.get(id)
        ?? [...surfaces.values()].find((surface) => surface.name === id || surface.name === signalName)
    }
    return surfaces.get(signalName)
      ?? [...surfaces.values()].find((surface) => surface.name === signalName)
  }

  private scheduleSignalInspectionDraw() {
    if (this.selectedOutputSignal === 'output') return
    this.syncSignalInspectionSurface()
    if (this.signalInspectRaf !== undefined) return
    this.signalInspectRaf = window.requestAnimationFrame(this.drawSignalInspectionFrame)
  }

  private drawSignalInspectionFrame = () => {
    this.signalInspectRaf = undefined
    if (this.selectedOutputSignal === 'output') return
    const stream = this.selectedSignalStream()
    this.attachSignalInspectVideo(stream)
    if (!stream) {
      const selectedSignal = this.selectedOutputSignal
      window.setTimeout(() => {
        if (this.selectedOutputSignal === selectedSignal) this.scheduleSignalInspectionDraw()
      }, 100)
    }
  }

  private ensureSignalInspectVideo(stream: MediaStream) {
    const video = this.signalOutputVideoElement ?? this.signalInspectVideo ?? document.createElement('video')
    if (!this.signalOutputVideoElement) this.signalInspectVideo = video
    video.muted = true
    video.playsInline = true
    video.autoplay = true
    if (video.srcObject !== stream) {
      video.srcObject = stream
      void video.play().catch(() => undefined)
    }
    return video
  }

  private attachSignalInspectVideo(stream?: MediaStream) {
    if (!stream) {
      if (this.signalOutputVideoElement) this.signalOutputVideoElement.srcObject = null
      if (this.signalInspectVideo) this.signalInspectVideo.srcObject = null
      return
    }
    this.ensureSignalInspectVideo(stream)
  }

  private syncSignalInspectionSurface() {
    if (this.selectedOutputSignal === 'output') {
      this.attachSignalInspectVideo(undefined)
      return
    }
    const stream = this.selectedSignalStream()
    this.attachSignalInspectVideo(stream)
    if (!stream) this.scheduleSignalInspectionDraw()
  }

  private selectedSignalStream(signalName = this.selectedOutputSignal) {
    const surface = this.selectedSignalSurface(signalName)
    if (!surface) return undefined
    if (surface.kind === 'stream') return surface.stream
    const sharedStream = getDebugSurfaceStream(surface.id)
    if (sharedStream) return sharedStream
    const existing = this.signalInspectCanvasStreams.get(surface.id)
    if (existing && existing.canvas === surface.canvas && existing.track?.readyState !== 'ended') return existing.stream
    existing?.track?.stop()
    if (typeof surface.canvas.captureStream !== 'function') return undefined
    const stream = surface.canvas.captureStream(30)
    const track = stream.getVideoTracks()[0]
    this.signalInspectCanvasStreams.set(surface.id, { canvas: surface.canvas, stream, track })
    return stream
  }

  private disposeSignalInspectCanvasStreams() {
    for (const item of this.signalInspectCanvasStreams.values()) item.track?.stop()
    this.signalInspectCanvasStreams.clear()
  }

  private syncSignalPickerPreviewStreams() {
    if (!this.signalPickerOpen) return
    const videos = this.shadowRoot?.querySelectorAll<HTMLVideoElement>('video.signal-picker-video[data-signal-name]')
    if (!videos?.length) return
    for (const video of videos) {
      const signalName = video.getAttribute('data-signal-name') || ''
      const stream = this.selectedSignalStream(signalName)
      if (video.srcObject !== stream) video.srcObject = stream ?? null
      if (stream) void video.play().catch(() => undefined)
    }
  }

  private drawStreamOutputFrame(image: CanvasImageSource) {
    const backing = this.ensureStreamOutputBackingCanvas()
    backing.width = this.stageWidth
    backing.height = this.stageHeight
    const context = backing.getContext('2d', { alpha: false })
    context?.clearRect(0, 0, backing.width, backing.height)
    context?.drawImage(image, 0, 0, backing.width, backing.height)
    this.syncVisibleStreamOutputCanvas()
  }

  private syncVisibleStreamOutputCanvas() {
    const backing = this.streamOutputBackingCanvas
    const canvas = this.streamOutputCanvasElement
    if (!backing || !canvas || backing.width <= 0 || backing.height <= 0) return
    if (canvas.width !== backing.width) canvas.width = backing.width
    if (canvas.height !== backing.height) canvas.height = backing.height
    const context = canvas.getContext('2d', { alpha: false })
    context?.clearRect(0, 0, canvas.width, canvas.height)
    context?.drawImage(backing, 0, 0, canvas.width, canvas.height)
  }

  private registerInputDebugSurfaces() {
    const capture = this.ensureInputCanvasCaptureCanvas()
    registerDebugCanvasSurface('input/frame/canvas', 'input/frame/canvas', this.editCanvasElement)
    registerDebugCanvasSurface('input/frame/capture', 'input/frame/capture', capture)
    registerDebugCanvasSurface('input/mask/active', 'input/mask/active', this.maskCanvasElement)
    registerDebugCanvasSurface('input/mask/preview', 'input/mask/preview', this.maskPreviewCanvasElement)
  }

  private ensureInputCanvasCaptureCanvas() {
    const domCanvas = this.inputCaptureCanvasElement
    if (domCanvas && this.inputCanvasCaptureCanvas !== domCanvas) {
      this.inputCanvasCaptureCanvas = domCanvas
    }
    if (!this.inputCanvasCaptureCanvas) this.inputCanvasCaptureCanvas = document.createElement('canvas')
    if (this.inputCanvasCaptureCanvas.width !== this.stageWidth) this.inputCanvasCaptureCanvas.width = this.stageWidth
    if (this.inputCanvasCaptureCanvas.height !== this.stageHeight) this.inputCanvasCaptureCanvas.height = this.stageHeight
    return this.inputCanvasCaptureCanvas
  }

  private refreshInputCanvasCaptureSurface() {
    const lowerCanvas = this.fabricCanvas?.getElement()
    const capture = this.ensureInputCanvasCaptureCanvas()
    const context = capture.getContext('2d', { alpha: true })
    if (!lowerCanvas || !context) return
    context.clearRect(0, 0, this.stageWidth, this.stageHeight)
    context.drawImage(lowerCanvas, 0, 0, this.stageWidth, this.stageHeight)
    const upperCanvas = (this.fabricCanvas as (Canvas & { upperCanvasEl?: HTMLCanvasElement }) | undefined)?.upperCanvasEl
    const isErasePreview = this.isEraseStroke(this.pendingBrushButton)
    if (upperCanvas && this.pendingPaintLayerId && !isErasePreview) {
      context.save()
      context.globalCompositeOperation = 'source-over'
      context.drawImage(upperCanvas, 0, 0, this.stageWidth, this.stageHeight)
      context.restore()
    }
  }

  private scheduleUpperCanvasLiveInput = (event?: PointerEvent | MouseEvent) => {
    if (event?.type.includes('move') && 'buttons' in event && event.buttons === 0 && !this.rightPaintStroke) return
    if (!this.fabricCanvas?.isDrawingMode && !this.rightPaintStroke) return
    if (!this.pendingPaintLayerId && !this.selectedLayerId) return
    this.upperCanvasLiveInputPending = true
    if (this.upperCanvasLiveInputRaf !== undefined) return
    this.upperCanvasLiveInputRaf = window.requestAnimationFrame(() => {
      this.upperCanvasLiveInputRaf = undefined
      if (!this.upperCanvasLiveInputPending) return
      this.upperCanvasLiveInputPending = false
      this.scheduleActiveDrawSignalRefresh()
      this.scheduleInputCanvasCaptureRefresh()
      this.queueLiveInputChanged()
    })
  }

  private hasActiveDrawStroke() {
    return !!this.rightPaintStroke || !!this.pendingPaintLayerId || (this.drawing && this.maskCanvasEditable())
  }

  private scheduleActiveDrawSignalRefresh() {
    if (this.drawSignalRefreshRaf !== undefined) return
    this.drawSignalRefreshRaf = window.requestAnimationFrame(() => {
      this.drawSignalRefreshRaf = undefined
      if (!this.hasActiveDrawStroke()) return
      const layerId = this.pendingPaintLayerId || this.selectedLayerId
      if (layerId) this.scheduleLayerMediaSurfaceRefresh(layerId)
      else this.scheduleLayerMediaSurfaceRefresh()
      this.refreshMaskPreview()
      requestDebugSurfaceFrame('input/mask/active')
      requestDebugSurfaceFrame('input/mask/preview')
      this.scheduleInputCanvasCaptureRefresh(true)
      if (this.hasActiveDrawStroke()) this.scheduleActiveDrawSignalRefresh()
    })
  }

  private scheduleInputCanvasCaptureRefresh(force = false) {
    if (force) {
      if (this.inputCanvasCaptureRaf !== undefined) window.cancelAnimationFrame(this.inputCanvasCaptureRaf)
      this.inputCanvasCaptureRaf = undefined
      this.refreshInputCanvasCaptureSurface()
      ;(this.inputCanvasMediaTrack as (MediaStreamTrack & { requestFrame?: () => void }) | undefined)?.requestFrame?.()
      return
    }
    if (this.inputCanvasCaptureRaf !== undefined) return
    this.inputCanvasCaptureRaf = window.requestAnimationFrame(() => {
      this.inputCanvasCaptureRaf = undefined
      this.refreshInputCanvasCaptureSurface()
      ;(this.inputCanvasMediaTrack as (MediaStreamTrack & { requestFrame?: () => void }) | undefined)?.requestFrame?.()
    })
  }

  private ensureInputCanvasMediaTrack() {
    const capture = this.ensureInputCanvasCaptureCanvas()
    if (!$stream.get().isStreaming || typeof capture.captureStream !== 'function') {
      this.stopInputCanvasMediaTrack()
      return
    }
    if (this.inputCanvasMediaStream && this.inputCanvasMediaSource !== capture) {
      this.stopInputCanvasMediaTrack()
    }
    if (this.inputCanvasMediaStream && this.inputCanvasMediaCaptureFps !== this.inputCanvasCaptureFps) {
      this.stopInputCanvasMediaTrack()
    }
    if (this.inputCanvasMediaTrack?.readyState === 'ended') {
      this.inputCanvasMediaStream = undefined
      this.inputCanvasMediaTrack = undefined
      this.inputCanvasMediaCaptureFps = 0
      this.inputCanvasMediaRegistered = false
    }
    if (!this.inputCanvasMediaStream) {
      this.refreshInputCanvasCaptureSurface()
      this.inputCanvasMediaStream = capture.captureStream(this.inputCanvasCaptureFps)
      this.inputCanvasMediaSource = capture
      this.inputCanvasMediaCaptureFps = this.inputCanvasCaptureFps
      this.inputCanvasMediaTrack = this.inputCanvasMediaStream.getVideoTracks()[0]
      this.inputCanvasMediaTrack.onended = () => {
        this.inputCanvasMediaStream = undefined
        this.inputCanvasMediaTrack = undefined
        this.inputCanvasMediaSource = undefined
        this.inputCanvasMediaCaptureFps = 0
        this.inputCanvasMediaRegistered = false
      }
    }
    if (!this.inputCanvasMediaTrack || this.inputCanvasMediaRegistered) return
    this.inputCanvasMediaRegistered = true
    void registerWebRtcResourceMediaTrack(this.inputCanvasMediaTrack, {
      id: 'input:canvas',
      stream_id: 'input:canvas',
      channel: 'canvas',
      label: 'input/frame/canvas',
      kind: 'input_surface',
    }).then((ok) => {
      if (!ok) this.inputCanvasMediaRegistered = false
      else (this.inputCanvasMediaTrack as (MediaStreamTrack & { requestFrame?: () => void }) | undefined)?.requestFrame?.()
    })
  }

  private stopInputCanvasMediaTrack() {
    this.inputCanvasMediaTrack?.stop()
    if (this.inputCanvasCaptureRaf !== undefined) window.cancelAnimationFrame(this.inputCanvasCaptureRaf)
    this.inputCanvasCaptureRaf = undefined
    this.inputCanvasMediaStream = undefined
    this.inputCanvasMediaTrack = undefined
    this.inputCanvasMediaSource = undefined
    this.inputCanvasMediaCaptureFps = 0
    this.inputCanvasMediaRegistered = false
  }

  public reportError(message: string) {
    this.dispatchEvent(new CustomEvent('rtd-error', { bubbles: true, composed: true, detail: { message } }))
  }
  public persistOptions() {
    this.dispatchEvent(new CustomEvent('rtd-persist-options', { bubbles: true, composed: true }))
  }
  private scheduleProjectPersist() {
    this.dispatchEvent(new CustomEvent('rtd-persist-project', { bubbles: true, composed: true }))
  }
  private markLiveInputChanged() {
    this.liveInputRevision++
  }
  private queueLiveStreamFrame(force = false, refreshResources = false, refreshLayerConditions = false) {
    this.liveStreamResourceRefreshScheduled ||= refreshResources
    this.liveStreamLayerConditionRefreshScheduled ||= refreshLayerConditions
    if (force) {
      if (this.liveStreamUpdateRaf !== undefined) window.cancelAnimationFrame(this.liveStreamUpdateRaf)
      this.liveStreamUpdateRaf = undefined
      this.liveStreamUpdateScheduled = false
      const shouldRefreshResources = this.liveStreamResourceRefreshScheduled
      const shouldRefreshLayerConditions = this.liveStreamLayerConditionRefreshScheduled
      this.liveStreamResourceRefreshScheduled = false
      this.liveStreamLayerConditionRefreshScheduled = false
      this.dispatchEvent(new CustomEvent('rtd-send-frame', { bubbles: true, composed: true, detail: { refreshResources: shouldRefreshResources, refreshLayerConditions: shouldRefreshLayerConditions, liveInputChanged: true } }))
      return
    }
    if (this.liveStreamUpdateScheduled) return
    this.liveStreamUpdateScheduled = true
    this.liveStreamUpdateRaf = window.requestAnimationFrame(() => {
      this.liveStreamUpdateRaf = undefined
      this.liveStreamUpdateScheduled = false
      const shouldRefreshResources = this.liveStreamResourceRefreshScheduled
      const shouldRefreshLayerConditions = this.liveStreamLayerConditionRefreshScheduled
      this.liveStreamResourceRefreshScheduled = false
      this.liveStreamLayerConditionRefreshScheduled = false
      this.dispatchEvent(new CustomEvent('rtd-send-frame', { bubbles: true, composed: true, detail: { refreshResources: shouldRefreshResources, refreshLayerConditions: shouldRefreshLayerConditions, liveInputChanged: true } }))
    })
  }
  private queueLiveStreamFrameIfInputChanged(force = false, refreshResources = false, refreshLayerConditions = false) {
    if (this.liveInputRevision === this.queuedLiveInputRevision) return
    this.queuedLiveInputRevision = this.liveInputRevision
    this.queueLiveStreamFrame(force, refreshResources, refreshLayerConditions)
  }
  private queueLiveInputChanged(force = false, refreshResources = false, refreshLayerConditions = false) {
    if ($stream.get().isStreaming) {
      if (!this.inputCanvasMediaTrack || this.inputCanvasMediaTrack.readyState === 'ended' || !this.inputCanvasMediaRegistered) {
        this.ensureInputCanvasMediaTrack()
      }
      this.refreshInputCanvasCaptureSurface()
      this.scheduleInputCanvasCaptureRefresh(force)
    } else if (this.inputCanvasMediaTrack) {
      this.stopInputCanvasMediaTrack()
    }
    this.markLiveInputChanged()
    this.queueLiveStreamFrameIfInputChanged(force, refreshResources, refreshLayerConditions)
  }

  private shouldRefreshLiveResourcesOnDraw() {
    return $stream.get().isStreaming
  }

  private layerConditionTopologySignature() {
    return JSON.stringify({
      noDefaultMaskLayers: [...this.noDefaultMaskLayers].sort(),
      layers: this.activeSamplingLayersBottomUp().map((layer) => ({
        id: layer.id,
        z: this.layerStackIndex(layer.id),
        transform: layer.transform,
        isVideo: layer.isVideo,
        regions: this.layerRegionSpecs(layer).map((region) => ({
          id: region.id,
          target: region.target,
          layerId: region.layerId,
          color: region.color,
          rect: region.rect,
          shape: region.shape,
          maskStrength: region.maskStrength,
          innerBlur: region.innerBlur,
          outerBlur: region.outerBlur,
          negate: region.negate,
          cfgMaskInherited: region.cfgMaskInherited,
          denoiseMaskInherited: region.denoiseMaskInherited,
          blendingEnabled: region.blendingEnabled,
          blendingRadius: region.blendingRadius,
          blendingStrength: region.blendingStrength,
        })),
      })),
    })
  }

  private queueLayerConditionAwareLiveInput() {
    const signature = this.layerConditionTopologySignature()
    const refreshResources = signature !== this.lastLayerConditionTopologySignature
    this.lastLayerConditionTopologySignature = signature
    this.queueLiveInputChanged(false, refreshResources, !refreshResources)
  }
  private currentStreamMotionTransform(): [number, number, number, number, number, number] | undefined {
    return undefined
  }

  static styles = css`
    :host { display: block; width: 100%; height: 100%; }
    .canvas-editor { display: flex; flex-direction: column; width: 100%; height: 100%; background: var(--bg-header, #1e1e28); }
    .viewer { display: flex; flex: 1; gap: 4px; min-height: 0; }
    .work-pane { flex: 1; display: flex; flex-direction: column; position: relative; overflow: hidden; }
    .work-surface { flex: 1; overflow: auto; background: var(--bg-header, #1e1e28); }
    .stage-wrap { width: max-content; height: max-content; min-width: 100%; min-height: 100%; display: flex; align-items: center; justify-content: center; }
    .stage { position: relative; width: calc(var(--stage-width) * 1px * var(--zoom, 1)); height: calc(var(--stage-height) * 1px * var(--zoom, 1)); background: repeating-conic-gradient(#666 0% 25%, #999 0% 50%) 0 0 / 16px 16px; overflow: visible; }
    .stage .canvas-container { overflow: visible !important; }
    .stage canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
    .stage canvas.input-capture-canvas { width: 1px; height: 1px; opacity: 0; pointer-events: none; z-index: -1; }
    .stream-output-source { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
    .mask { cursor: crosshair; }
    /* Non-active mask and all permanent overlays must never capture pointer events */
    .mask:not(.active) { pointer-events: none; opacity: 0; }
    .mask.active { pointer-events: all; }
    .mask.active.brush-mode { cursor: none; }
    .mask-preview { pointer-events: none; opacity: 0; transition: opacity 0.2s; }
    .mask-preview.visible { opacity: 0.6; }
    #editor-overlay { pointer-events: none; }
    .cursor { position: absolute; pointer-events: none; border-radius: 50%; border: 1.5px solid rgba(255,255,255,0.8); box-shadow: 0 0 0 1px rgba(0,0,0,0.5); transform: translate(-50%, -50%); opacity: 0; left: var(--cursor-x, 50%); top: var(--cursor-y, 50%); width: var(--cursor-w, 20px); height: var(--cursor-h, 20px); transition: width 0.08s, height 0.08s; }
    .cursor.visible { opacity: 1; }
    .viewport-overlay { position: absolute; left: 0; right: 0; display: flex; align-items: center; gap: 4px; padding: 2px 6px; background: rgba(20,20,26,0.8); z-index: 10; opacity: 0; transition: opacity 0.2s; pointer-events: none; font-size: 11px; color: var(--fg, #d4d4d8); }
    .viewport-overlay.visible { opacity: 1; pointer-events: all; }
    .pane-head { top: 0; }
    .pane-foot { bottom: 0; }
    .pane-action-overlay { justify-content: flex-end; }
    .viewport-toggles { display: flex; gap: 2px; margin-left: auto; }
    .viewport-toggles button { font-size: 10px; padding: 1px 5px; background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c); border-radius: var(--radius, 2px); color: var(--fg, #d4d4d8); cursor: pointer; }
    .viewport-toggles button.active { background: var(--bg-sel, #1a3060); border-color: var(--border-focus, #4d87c4); }
    .output-overlay-tools { margin-left: auto; display: flex; align-items: center; gap: 6px; }
    .signal-grid-toggle { min-width: 132px; max-width: 240px; height: 20px; font-size: 10px; padding: 1px 8px; background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c); border-radius: 2px; color: var(--fg, #d4d4d8); text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .signal-grid-toggle.active { border-color: var(--border-focus, #4d87c4); background: var(--bg-sel, #1a3060); }
    .output-timing-chip { height: 20px; font-size: 10px; padding: 1px 8px; background: rgba(8, 28, 24, 0.92); border: 1px solid rgba(87, 198, 162, 0.55); border-radius: 999px; color: #bbf7df; cursor: pointer; white-space: nowrap; }
    .output-timing-chip.active { background: rgba(18, 68, 56, 0.96); border-color: rgba(129, 236, 199, 0.85); color: #e9fff7; }
    .output-timing-panel { position: absolute; top: 26px; right: 6px; width: min(320px, calc(100% - 12px)); z-index: 24; display: flex; flex-direction: column; gap: 8px; padding: 8px 9px; background: rgba(10, 14, 18, 0.96); border: 1px solid rgba(87, 198, 162, 0.35); border-radius: 6px; box-shadow: 0 10px 30px rgba(0,0,0,0.34); color: var(--fg, #d4d4d8); }
    .output-timing-head { display: flex; align-items: center; gap: 8px; }
    .output-timing-head button { margin-left: auto; font-size: 10px; padding: 1px 6px; background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c); border-radius: 2px; color: var(--fg, #d4d4d8); cursor: pointer; }
    .output-timing-grid { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 10px; font-size: 11px; }
    .output-timing-row { display: contents; }
    .output-timing-row span { color: var(--fg-dim, #a5a7b3); }
    .output-timing-row strong { text-align: right; color: #effff8; font-variant-numeric: tabular-nums; }
    .output-timing-meta { display: flex; flex-direction: column; gap: 2px; padding-top: 2px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 10px; color: var(--fg-dim, #8f92a0); }
    .signal-picker-panel { position: absolute; top: 22px; left: 6px; right: 6px; max-height: min(46%, 360px); z-index: 22; background: rgba(12,12,16,0.95); border: 1px solid var(--border, #3a3a4c); border-radius: 4px; overflow: hidden; box-shadow: 0 10px 28px rgba(0,0,0,0.38); display: flex; flex-direction: column; }
    .signal-picker-grid { overflow-y: auto; flex: 1; min-height: 0; padding: 6px; display: grid; grid-template-columns: repeat(auto-fill, minmax(136px, 1fr)); gap: 6px; }
    .signal-picker-tile { display: flex; flex-direction: column; gap: 4px; margin: 0; padding: 4px; background: rgba(8,8,12,0.7); border: 1px solid rgba(255,255,255,0.12); border-radius: 4px; color: var(--fg, #d4d4d8); text-align: left; }
    .signal-picker-tile.active { border-color: #7aadff; box-shadow: 0 0 0 1px rgba(122,173,255,0.4) inset; }
    .signal-picker-thumb { width: 100%; aspect-ratio: 4 / 3; background: repeating-conic-gradient(#2b2d36 0% 25%, #6f7280 0% 50%) 0 0 / 16px 16px; border-radius: 3px; overflow: hidden; }
    .signal-picker-video { width: 100%; height: 100%; object-fit: contain; display: block; }
    .signal-picker-name { font-size: 10px; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .resource-debug-panel { position: absolute; top: 26px; left: 6px; right: 6px; max-height: min(46%, 360px); z-index: 30; display: flex; flex-direction: column; background: rgba(12,12,16,0.94); border: 1px solid var(--border, #3a3a4c); border-radius: 4px; overflow: hidden; box-shadow: 0 10px 28px rgba(0,0,0,0.38); }
    .resource-debug-head { display: flex; align-items: center; gap: 8px; padding: 5px 7px; border-bottom: 1px solid var(--border, #3a3a4c); font-size: 11px; color: var(--fg, #d4d4d8); }
    .resource-debug-head span { color: var(--fg-dim, #7a7a8a); }
    .resource-debug-head button { margin-left: auto; font-size: 10px; padding: 1px 6px; background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c); border-radius: 2px; color: var(--fg, #d4d4d8); cursor: pointer; }
    .resource-debug-grid { overflow: auto; padding: 5px; display: grid; grid-template-columns: repeat(auto-fill, minmax(132px, 1fr)); gap: 5px; }
    .resource-debug-cell { margin: 0; position: relative; min-height: 92px; aspect-ratio: 4 / 3; background: #050507; border: 1px solid rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden; }
    .resource-debug-cell rtd-debug-surface { width: 100%; height: 100%; display: block; }
    .text-resource-debug-cell { aspect-ratio: unset; min-height: 92px; }
    .text-resource-debug-cell pre { margin: 0; height: 100%; padding: 6px 6px 22px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; color: #d7e3ff; font-size: 10px; line-height: 1.35; }
    .resource-debug-cell figcaption { position: absolute; left: 0; right: 0; bottom: 0; padding: 2px 4px; background: rgba(0,0,0,0.72); color: #e6e6ec; font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .resource-debug-empty { padding: 18px; text-align: center; font-size: 11px; color: var(--fg-dim, #7a7a8a); }
    .paint-hidden canvas:first-child { opacity: 0; }
    .preview { position: relative; overflow: hidden; }
    .preview img, .preview video { width: 100%; height: 100%; object-fit: contain; display: block; }
    .preview.signal-inspection { background: repeating-conic-gradient(#2b2d36 0% 25%, #6f7280 0% 50%) 0 0 / 16px 16px; }
    .signal-output-canvas { width: 100%; height: 100%; object-fit: contain; display: block; }
    .signal-output-overlay { position: absolute; inset: 0; z-index: 4; pointer-events: none; }
    .empty { color: var(--fg-dim, #7a7a8a); font-size: 12px; text-align: center; padding: 16px; }
    .selection-box { position: absolute; border: 2px dashed rgba(0, 200, 255, 0.8); pointer-events: none; box-sizing: border-box; }
    .selection-box.inverted { border-style: solid; background: rgba(0, 200, 255, 0.08); }
    .selection-path { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
    .selection-path polygon, .selection-path ellipse { fill: rgba(0, 200, 255, 0.08); stroke: rgba(0, 200, 255, 0.8); stroke-width: 0.5; stroke-dasharray: 2; }
    .transform-svg-overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 9; overflow: visible; }
    .transform-box { fill: rgba(0, 229, 255, 0.05); stroke: rgba(0, 229, 255, 0.95); stroke-width: 1.5; stroke-dasharray: 7 4; pointer-events: all; cursor: move; }
    .transform-dom-rotate-band { position: absolute; z-index: 10; width: 46px; height: 46px; min-width: 46px; min-height: 46px; padding: 0; transform: translate(-50%, -50%); border-radius: 50%; border: 0; background: rgba(255,255,255,0.001); pointer-events: all; cursor: grab; }
    .transform-dom-handle { position: absolute; z-index: 11; width: 18px; height: 18px; min-width: 18px; min-height: 18px; padding: 0; transform: translate(-50%, -50%); border-radius: 50%; border: 2px solid #00e5ff; background: #f3f1ec; box-shadow: 0 1px 3px rgba(0,0,0,0.7); pointer-events: all; }
    .transform-dom-handle::after { content: ''; position: absolute; inset: -8px; border-radius: 50%; }
    .transform-rotate-line { stroke: rgba(0, 229, 255, 0.75); stroke-width: 1.25; pointer-events: none; }
    .transform-rotate-handle { fill: #7aadff; stroke: #050507; stroke-width: 1.5; pointer-events: all; cursor: grab; }
    .color-popup { position: fixed; z-index: 200; background: var(--bg-panel, #24242c); border: 1px solid var(--border, #3a3a4c); border-radius: 4px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); min-width: 280px; }
    .color-popup-head { display: flex; align-items: center; justify-content: space-between; padding: 6px 8px; border-bottom: 1px solid var(--border, #3a3a4c); cursor: move; user-select: none; font-size: 12px; font-weight: 600; color: var(--fg, #d4d4d8); }
    .color-popup-body { padding: 8px; display: flex; flex-direction: column; gap: 8px; }
    .color-preview-large { width: 100%; height: 28px; border-radius: 2px; background: var(--chip, #fff); border: 1px solid var(--border, #3a3a4c); }
    .color-native { display: flex; gap: 4px; align-items: center; }
    .color-native input[type="color"] { width: 32px; height: 24px; padding: 0; border: none; cursor: pointer; }
    .hex-input { flex: 1; font-size: 11px; padding: 2px 4px; background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c); color: var(--fg, #d4d4d8); border-radius: 2px; }
    .channel-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
    .channel-input { display: flex; flex-direction: column; gap: 1px; font-size: 10px; color: var(--fg-label, #9898a8); }
    .channel-input input { width: 100%; font-size: 10px; padding: 1px 3px; background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c); color: var(--fg, #d4d4d8); border-radius: 2px; text-align: center; }
    .hsl-sliders { display: flex; flex-direction: column; gap: 3px; }
    .color-slider { display: grid; grid-template-columns: 12px 1fr 28px; align-items: center; gap: 4px; font-size: 10px; color: var(--fg-label, #9898a8); }
    .color-slider input { width: 100%; }
    .color-slider output { text-align: right; }
    .palette { display: flex; flex-wrap: wrap; gap: 3px; }
    .swatch { width: 20px; height: 20px; border-radius: 2px; border: 1px solid rgba(255,255,255,0.15); cursor: pointer; background: var(--swatch, #fff); }
    .swatch.active { border-color: #7aadff; box-shadow: 0 0 0 1px #7aadff; }
    .planner { display: flex; flex-direction: column; gap: 4px; }
    .planner-mode { font-size: 11px; }
    .planner-row { display: flex; gap: 3px; }
    .popup-planner .planner-mode select { font-size: 10px; background: var(--bg-input, #14141a); border: 1px solid var(--border, #3a3a4c); color: var(--fg, #d4d4d8); }
    .stream-output-video { width: 100%; height: 100%; object-fit: contain; display: block; }
    .stream-pending { opacity: 0.5; }
    .stream-preview-fallback { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
    .native { width: calc(var(--stage-width) * 1px * var(--zoom, 1)); height: calc(var(--stage-height) * 1px * var(--zoom, 1)); }
  `


  private syncCanvasGeometry() {
    // Scale the Fabric.js canvas-container via CSS transform instead of overriding its
    // inline width/height. This leaves Fabric.js's own pixel dimensions untouched so
    // getPointer() can compute the correct cssScale (buffer / boundsWidth), while
    // getBoundingClientRect() automatically accounts for the transform.
    const wrapper = this.editCanvasElement?.parentElement as HTMLElement | null
    if (wrapper) {
      wrapper.style.transformOrigin = '0 0'
      wrapper.style.transform = `scale(${this.inputZoom})`
    }
    window.requestAnimationFrame(() => {
      this.fabricCanvas?.calcOffset()
      this.fabricCanvas?.requestRenderAll()
    })
  }

  public validTransparentMethod(value: unknown): value is string {
    return value === 'auto' || value === 'fast' || value === 'layerdiffuse'
  }

  private handleKeyDown = (event: KeyboardEvent) => {
    if (this.eventStartedInEditable(event)) return
    if (event.key === 'Enter' && this.pendingRgbaPlacementId) {
      event.preventDefault()
      this.commitPendingRgbaPlacement()
      return
    }
    if (event.key === 'Enter' && this.toolMode === 'select') {
      event.preventDefault()
      this.commitActiveLayerTransform()
      return
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault()
      void this.undo()
      return
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
      event.preventDefault()
      void this.redo()
      return
    }
    if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return
    const shortcut = TOOL_SHORTCUTS[event.key.toLowerCase()]
    if (shortcut) {
      event.preventDefault()
      this.setTool(shortcut)
    }
  }

  private pushHistory() {
    if (this.isRestoringHistory || !this.fabricCanvas || !this.maskCanvasElement) return
    const snapshot = this.captureSnapshot()
    if (!snapshot) return
    this.undoStack = [...this.undoStack, snapshot].slice(-HISTORY_LIMIT)
    this.redoStack = []
    this.requestUpdate()
  }

  private captureSnapshot(): EditorSnapshot | undefined {
    if (!this.fabricCanvas || !this.maskCanvasElement) return undefined
    const canvas = normalizeCanvasPaintChildren((this.fabricCanvas as Canvas & { toJSON: (properties?: string[]) => unknown }).toJSON([...CANVAS_CUSTOM_PROPERTIES]))
    return {
      canvas,
      mask: this.maskCanvasElement.toDataURL('image/png'),
      channelMasks: this.serializedChannelMasks(),
      regions: this.regions.map((region) => ({ ...region, rect: { ...region.rect, points: region.rect.points ? [...region.rect.points] : undefined } })),
      stageWidth: this.stageWidth,
      stageHeight: this.stageHeight,
      selectionRect: this.selectionRect ? { ...this.selectionRect } : undefined,
      selectedLayerId: this.selectedLayerId,
    }
  }

  private undo = async () => {
    const snapshot = this.undoStack.at(-1)
    if (!snapshot || !this.fabricCanvas) return
    const current = this.captureSnapshot()
    if (current) this.redoStack = [...this.redoStack, current].slice(-HISTORY_LIMIT)
    this.undoStack = this.undoStack.slice(0, -1)
    this.requestUpdate()
    await this.restoreSnapshot(snapshot)
  }

  private redo = async () => {
    const snapshot = this.redoStack.at(-1)
    if (!snapshot || !this.fabricCanvas) return
    const current = this.captureSnapshot()
    if (current) this.undoStack = [...this.undoStack, current].slice(-HISTORY_LIMIT)
    this.redoStack = this.redoStack.slice(0, -1)
    this.requestUpdate()
    await this.restoreSnapshot(snapshot)
  }

  public restoreMask(dataUrl: string) {
    return new Promise<void>((resolve) => {
      const image = new Image()
      image.onload = () => {
        if (!this.maskContext) return resolve()
        this.maskContext.clearRect(0, 0, this.stageWidth, this.stageHeight)
        this.maskContext.drawImage(image, 0, 0, this.stageWidth, this.stageHeight)
        resolve()
      }
      image.onerror = () => resolve()
      image.src = dataUrl
    })
  }

  private drawDataUrlToCanvas(dataUrl: string, canvas: HTMLCanvasElement) {
    return new Promise<void>((resolve) => {
      const image = new Image()
      image.onload = () => {
        canvas.getContext('2d')?.drawImage(image, 0, 0, this.stageWidth, this.stageHeight)
        resolve()
      }
      image.onerror = () => resolve()
      image.src = dataUrl
    })
  }

  private serializedChannelMasks(): { scope: MaskScope; channel: MaskChannel; image: string }[] {
    this.saveCurrentMaskScope()
    const out: { scope: MaskScope; channel: MaskChannel; image: string }[] = []
    for (const [key, canvas] of this.channelMaskCanvases) {
      if (!this.canvasHasNonZeroLuma(canvas)) continue
      const index = key.lastIndexOf('::')
      if (index <= 0) continue
      const scope = key.slice(0, index) as MaskScope
      const channel = key.slice(index + 2) as MaskChannel
      out.push({ scope, channel, image: canvas.toDataURL('image/png') })
    }
    return out
  }

  public toolButton(mode: ToolMode, _label: string) {
    const tool = TOOL_DEFS[mode]
    return html`<button
      class=${this.toolMode === mode ? 'tool-icon active' : 'tool-icon'}
      title=${`${tool.label} (${tool.shortcut})`}
      aria-label=${`${tool.label}, shortcut ${tool.shortcut}`}
      @click=${() => this.setTool(mode)}
    >${tool.icon}</button>`
  }

  public renderEditorDestination() {
    return html`<div class="dest-switch" role="group" aria-label="editor destination">
      <button class=${this.editorDest === 'paint' ? 'active' : ''} @click=${() => this.setEditorDest('paint')}>Paint</button>
      <button class=${this.editorDest === 'mask' ? 'active' : ''} @click=${() => this.setEditorDest('mask')}>Masks</button>
    </div>`
  }

  private outputPreviewClass() {
    const classes = ['preview', 'native']
    if (this.awaitingFrame) classes.push('sampling')
    if (this.selectedOutputSignal !== 'output') classes.push('signal-inspection')
    return classes.join(' ')
  }

  private outputPreviewStyle() {
    return [
      `--stage-width: ${this.stageWidth}`,
      `--stage-height: ${this.stageHeight}`,
      `--zoom: ${this.outputZoom}`,
    ].join('; ')
  }

  public renderColorPanel() {
    const rgba = this.activeRgba()
    return html`<div class="color-panel">
      <div class="color-stack" aria-label="primary and secondary colors">
        <button
          class=${this.activeColorSlot === 'primary' ? 'color-chip primary active' : 'color-chip primary'}
          style="--chip: ${this.brushColor}"
          title="Primary color"
          @click=${() => (this.activeColorSlot = 'primary')}
        ></button>
        <button
          class=${this.activeColorSlot === 'secondary' ? 'color-chip secondary active' : 'color-chip secondary'}
          style="--chip: ${this.secondaryBrushColor}"
          title="Secondary color / right-click color"
          @click=${() => (this.activeColorSlot = 'secondary')}
        ></button>
        <button class="swap-colors" title="Swap colors" @click=${this.swapColors}>â†”ï¸</button>
      </div>
      <div class="color-editor">
        <label class="color-native" title="Active color">
          <input type="color" .value=${this.colorInputValue()} @input=${this.pickColor} />
          <input class="hex-input" spellcheck="false" .value=${this.colorTextValue()} @input=${this.pickHexColor} />
        </label>
        <div class="channel-grid rgba-grid">
          ${this.channelInput('R', rgba.r, 0, 255, (value) => this.setActiveRgba({ r: value }))}
          ${this.channelInput('G', rgba.g, 0, 255, (value) => this.setActiveRgba({ g: value }))}
          ${this.channelInput('B', rgba.b, 0, 255, (value) => this.setActiveRgba({ b: value }))}
          ${this.channelInput('A', Math.round(rgba.a * 100), 0, 100, (value) => this.setActiveRgba({ a: value / 100 }))}
        </div>
      </div>
      <div class="palette quick-palette" aria-label="quick color palette">
        ${QUICK_PALETTE.slice(0, 15).map((color) => this.colorSwatch(color, 'Quick color'))}
        <button class="swatch palette-more" title="Open full color palette" aria-label="Open full color palette" @click=${this.openColorPopup}>+</button>
      </div>
    </div>`
  }

  private renderColorPopup() {
    if (!this.colorPopupOpen) return ''
    const rgb = this.activeRgb()
    const hsl = this.activeHsl()
    const rgba = this.activeRgba()
    return html`<section
      class="color-popup"
      style="left: ${this.colorPopupPosition.x}px; top: ${this.colorPopupPosition.y}px;"
      @pointermove=${this.dragColorPopup}
      @pointerup=${this.endColorPopupDrag}
      @pointercancel=${this.endColorPopupDrag}
    >
      <header class="color-popup-head" @pointerdown=${this.beginColorPopupDrag}>
        <strong>Color Palette</strong>
        <button title="Close palette" aria-label="Close color palette" @click=${() => (this.colorPopupOpen = false)}>Ã—</button>
      </header>
      <div class="color-popup-body">
        <div class="popup-color-editor">
          <div class="color-preview-large" style="--chip: ${this.activeColor()}"></div>
          <label class="color-native" title="Active color">
            <input type="color" .value=${this.colorInputValue()} @input=${this.pickColor} />
            <input class="hex-input" spellcheck="false" .value=${this.colorTextValue()} @input=${this.pickHexColor} />
          </label>
          <div class="channel-grid rgba-grid popup-rgba">
            ${this.channelInput('R', rgb.r, 0, 255, (value) => this.setActiveRgba({ r: value }))}
            ${this.channelInput('G', rgb.g, 0, 255, (value) => this.setActiveRgba({ g: value }))}
            ${this.channelInput('B', rgb.b, 0, 255, (value) => this.setActiveRgba({ b: value }))}
            ${this.channelInput('A', Math.round(rgba.a * 100), 0, 100, (value) => this.setActiveRgba({ a: value / 100 }))}
          </div>
          <div class="hsl-sliders">
            ${this.colorSlider('H', hsl.h, 0, 360, 1, (value) => this.setActiveHsl({ h: value }))}
            ${this.colorSlider('S', hsl.s, 0, 100, 1, (value) => this.setActiveHsl({ s: value }))}
            ${this.colorSlider('L', hsl.l, 0, 100, 1, (value) => this.setActiveHsl({ l: value }))}
            ${this.colorSlider('A', Math.round(rgba.a * 100), 0, 100, 1, (value) => this.setActiveRgba({ a: value / 100 }))}
          </div>
        </div>
        <div class="palette full-palette" aria-label="full color palette">
          ${PALETTE.map((color) => this.colorSwatch(this.withActiveAlpha(color), 'Palette color'))}
        </div>
        <div class="planner popup-planner">
          <label class="field inline planner-mode">
            <span>Plan</span>
            <select .value=${this.colorPlannerMode} @change=${(event: Event) => (this.colorPlannerMode = (event.target as HTMLSelectElement).value as ColorPlannerMode)}>
              ${COLOR_PLANNER_MODES.map((mode) => html`<option value=${mode.value}>${mode.label}</option>`)}
            </select>
          </label>
          <div class="planner-row" aria-label="planned harmony colors">
            ${this.plannedColors().map((color) => this.colorSwatch(color, 'Planned color'))}
          </div>
          <div class="planner-row" aria-label="primary to secondary blend">
            ${this.blendColors(this.brushColor, this.secondaryBrushColor, 7).map((color) => this.colorSwatch(color, 'Blend color'))}
          </div>
          <div class="planner-row" aria-label="tints and shades">
            ${this.toneRamp(this.activeColor(), 7).map((color) => this.colorSwatch(color, 'Tone color'))}
          </div>
        </div>
      </div>
    </section>`
  }

  private colorSwatch(color: string, label: string) {
    return html`<button
      class=${this.colorIsActive(color) ? 'swatch active' : 'swatch'}
      style="--swatch: ${color}"
      title=${`${label} ${color}. Left click active slot, right click secondary.`}
      @click=${() => this.setActiveColor(color)}
      @contextmenu=${(event: Event) => this.setSecondaryFromPalette(event, color)}
    ></button>`
  }

  private channelInput(label: string, value: number, min: number, max: number, onChange: (value: number) => void) {
    return html`<label class="channel-input">
      <span>${label}</span>
      <input type="number" min=${String(min)} max=${String(max)} .value=${String(value)} @input=${(event: Event) => onChange(Math.round(this.clamp(Number((event.target as HTMLInputElement).value), min, max, value)))} />
    </label>`
  }

  private colorSlider(label: string, value: number, min: number, max: number, step: number, onChange: (value: number) => void) {
    return html`<label class="color-slider">
      <span>${label}</span>
      <input type="range" min=${min} max=${max} step=${step} .valueAsNumber=${value} @input=${(event: Event) => onChange(Number((event.target as HTMLInputElement).value))} />
      <output>${Math.round(value)}</output>
    </label>`
  }

  private openColorPopup = () => {
    this.colorPopupOpen = true
  }

  private beginColorPopupDrag = (event: PointerEvent) => {
    if ((event.target as HTMLElement).closest('button')) return
    event.preventDefault()
    this.colorPopupDrag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      x: this.colorPopupPosition.x,
      y: this.colorPopupPosition.y,
      target: event.currentTarget as HTMLElement,
    }
    this.colorPopupDrag.target.setPointerCapture(event.pointerId)
  }

  private dragColorPopup = (event: PointerEvent) => {
    if (!this.colorPopupDrag || event.pointerId !== this.colorPopupDrag.pointerId) return
    const x = this.colorPopupDrag.x + event.clientX - this.colorPopupDrag.startX
    const y = this.colorPopupDrag.y + event.clientY - this.colorPopupDrag.startY
    this.colorPopupPosition = {
      x: Math.round(this.clamp(x, 8, Math.max(8, window.innerWidth - 420), this.colorPopupPosition.x)),
      y: Math.round(this.clamp(y, 8, Math.max(8, window.innerHeight - 420), this.colorPopupPosition.y)),
    }
  }

  private endColorPopupDrag = (event: PointerEvent) => {
    if (!this.colorPopupDrag || event.pointerId !== this.colorPopupDrag.pointerId) return
    this.colorPopupDrag.target.releasePointerCapture?.(event.pointerId)
    this.colorPopupDrag = undefined
  }

  public addVideoLayer = async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'video/*'
    await new Promise<void>((resolve) => {
      input.onchange = async () => {
        const file = input.files?.[0]
        if (!file) { resolve(); return }
        if (this.videoObjectUrl) URL.revokeObjectURL(this.videoObjectUrl)
        this.videoObjectUrl = URL.createObjectURL(file)
        this.videoUrl = this.videoObjectUrl
        this.videoReversing = false

        const vid = this.videoInputElement
        if (!vid) { resolve(); return }
        vid.src = this.videoObjectUrl
        vid.currentTime = this.videoStart

        // Wait for metadata so we know dimensions
        await new Promise<void>((res) => {
          if (vid.readyState >= 1) { res(); return }
          vid.onloadedmetadata = () => res()
        })
        this.videoDuration = vid.duration

        // Create off-screen canvas sized to the stage
        const offscreen = document.createElement('canvas')
        offscreen.width = this.stageWidth
        offscreen.height = this.stageHeight
        this._videoOffscreenCanvas = offscreen

        // Draw first frame before creating FabricImage
        if (vid.readyState >= 2) {
          const ctx = offscreen.getContext('2d')!
          const vw = vid.videoWidth || this.stageWidth
          const vh = vid.videoHeight || this.stageHeight
          const scale = Math.min(offscreen.width / vw, offscreen.height / vh)
          const sw = vw * scale
          const sh = vh * scale
          ctx.drawImage(vid, (offscreen.width - sw) / 2, (offscreen.height - sh) / 2, sw, sh)
        }

        // Create FabricImage from the off-screen canvas (objectCaching=false so it re-reads on each render)
        const img = new FabricImage(offscreen, {
          objectCaching: false,
          left: 0,
          top: 0,
          scaleX: 1,
          scaleY: 1,
        }) as unknown as FabricObject & { __isVideoLayer?: boolean; id?: string }
        img.__isVideoLayer = true

        this.pushHistory()
        this.fabricCanvas?.add(img as unknown as FabricObject)
        this.fabricCanvas?.setActiveObject(img as unknown as FabricObject)
        this.fabricCanvas?.requestRenderAll()
        this.syncLayers()

        // Record the layer id
        // Start playback and rAF loop
        vid.loop = this.videoLoopMode === 'loop'
        if (this.videoFpsMode === 'fps') void vid.play()
        this._startVideoRaf()

        // Select the video tab in the layer panel
        $layer.setKey('layerPanelTab', 'video')
        resolve()
      }
      input.click()
    })
  }

  public loadRgbaMaskImage = async (layerId?: string | Event) => {
    const targetLayerId = typeof layerId === 'string' ? layerId : this.selectedLayerId
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/png,image/webp,image/*'
    await new Promise<void>((resolve) => {
      input.onchange = async () => {
        const file = input.files?.[0]
        if (!file) { resolve(); return }
        try {
          const image = await this.loadHtmlImage(URL.createObjectURL(file))
          const source = document.createElement('canvas')
          source.width = image.naturalWidth || image.width || 1
          source.height = image.naturalHeight || image.height || 1
          source.getContext('2d')!.drawImage(image, 0, 0)
          const cropped = this.cropCanvasToAlphaBounds(source)
          this.addRgbaMaskPlacement(cropped.canvas, file.name.replace(/\.[^.]+$/, '') || 'RGBA image', targetLayerId)
        } finally {
          resolve()
        }
      }
      input.click()
    })
  }

  public addEmptyRgbaMask = () => {
    this.commitPendingRgbaPlacement()
    const width = Math.max(64, Math.round(Math.min(this.stageWidth, 512) * 0.5))
    const height = Math.max(64, Math.round(Math.min(this.stageHeight, 512) * 0.5))
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    this.addRgbaMaskCanvas(canvas, `RGBA mask ${this.layers.length + 1}`, true)
  }

  private loadHtmlImage(url: string) {
    return new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image()
      image.onload = () => { URL.revokeObjectURL(url); resolve(image) }
      image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not load image')) }
      image.src = url
    })
  }

  private addRgbaMaskCanvas(canvas: HTMLCanvasElement, name: string, fitToStage: boolean) {
    if (!this.fabricCanvas) return
    const scale = fitToStage ? Math.min(1, (this.stageWidth * 0.92) / canvas.width, (this.stageHeight * 0.92) / canvas.height) : 1
    const image = new FabricImage(canvas, {
      left: Math.round((this.stageWidth - canvas.width * scale) / 2),
      top: Math.round((this.stageHeight - canvas.height * scale) / 2),
      scaleX: scale,
      scaleY: scale,
      objectCaching: false,
      name,
    }) as unknown as FabricObject & { __isRgbaMaskLayer?: boolean }
    image.__isRgbaMaskLayer = true
    this.attachPreset(image)
    this.pushHistory()
    this.fabricCanvas.add(image as unknown as FabricObject)
    this.fabricCanvas.setActiveObject(image as unknown as FabricObject)
    this.selectedEntity = 'layer'
    this.selectedLayerId = this.assignLayerId(image as unknown as FabricObject)
    this.setEditorDest('paint')
    this.fabricCanvas.requestRenderAll()
    this.syncLayers()
  }

  private ensureRgbaMaskLayer(layerId = this.selectedLayerId) {
    let object = layerId ? this.findLayer(layerId) as (FabricObject & { __isRgbaMaskLayer?: boolean; __preset?: LayerPreset }) | undefined : undefined
    if (!object) {
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(64, Math.round(Math.min(this.stageWidth, 512) * 0.5))
      canvas.height = Math.max(64, Math.round(Math.min(this.stageHeight, 512) * 0.5))
      this.addRgbaMaskCanvas(canvas, `RGBA mask ${this.layers.length + 1}`, true)
      object = this.selectedLayerId ? this.findLayer(this.selectedLayerId) as (FabricObject & { __isRgbaMaskLayer?: boolean; __preset?: LayerPreset }) | undefined : undefined
    }
    if (!object) return undefined
    object.__isRgbaMaskLayer = true
    return { id: String((object as FabricObject & { __uid?: string }).__uid ?? this.assignLayerId(object as FabricObject)), object }
  }

  private addRgbaMaskPlacement(canvas: HTMLCanvasElement, name: string, layerId = this.selectedLayerId) {
    if (!this.fabricCanvas) return
    this.commitPendingRgbaPlacement()
    const target = this.ensureRgbaMaskLayer(layerId)
    if (!target) return
    const scale = Math.min(1, (this.stageWidth * 0.92) / canvas.width, (this.stageHeight * 0.92) / canvas.height)
    const placement = new FabricImage(canvas, {
      left: Math.round((this.stageWidth - canvas.width * scale) / 2),
      top: Math.round((this.stageHeight - canvas.height * scale) / 2),
      scaleX: scale,
      scaleY: scale,
      objectCaching: false,
      name,
    }) as unknown as FabricObject & { __paintChild?: boolean; __parentLayerId?: string }
    placement.__paintChild = true
    placement.__parentLayerId = target.id
    this.styleTransformControls(placement as unknown as FabricObject)
    this.pushHistory()
    this.fabricCanvas.add(placement as unknown as FabricObject)
    this.fabricCanvas.setActiveObject(placement as unknown as FabricObject)
    this.selectedEntity = 'layer'
    this.selectedLayerId = target.id
    this.pendingRgbaPlacementId = this.assignLayerId(placement as unknown as FabricObject)
    this.pendingRgbaPlacementLayerId = target.id
    this.setEditorDest('paint', false)
    this.setTool('select', false)
    this.fabricCanvas.requestRenderAll()
    this.syncLayers()
  }

  private commitPendingRgbaPlacement() {
    if (!this.pendingRgbaPlacementId || !this.pendingRgbaPlacementLayerId) return
    const layerId = this.pendingRgbaPlacementLayerId
    this.pendingRgbaPlacementId = ''
    this.pendingRgbaPlacementLayerId = ''
    this.rasterizeRgbaMaskLayer(layerId)
    this.queueLiveInputChanged(true, true)
  }

  private cropCanvasToAlphaBounds(source: HTMLCanvasElement, fallback?: { x: number; y: number; width: number; height: number }) {
    const context = source.getContext('2d')!
    const imageData = context.getImageData(0, 0, source.width, source.height)
    let minX = source.width, minY = source.height, maxX = -1, maxY = -1
    for (let y = 0; y < source.height; y++) {
      for (let x = 0; x < source.width; x++) {
        const alpha = imageData.data[(y * source.width + x) * 4 + 3]
        if (alpha <= 0) continue
        if (x < minX) minX = x
        if (y < minY) minY = y
        if (x > maxX) maxX = x
        if (y > maxY) maxY = y
      }
    }
    if (maxX < minX || maxY < minY) {
      const fb = fallback ?? { x: 0, y: 0, width: Math.min(256, source.width || 256), height: Math.min(256, source.height || 256) }
      minX = Math.max(0, Math.floor(fb.x)); minY = Math.max(0, Math.floor(fb.y))
      maxX = Math.min(source.width - 1, Math.ceil(fb.x + Math.max(1, fb.width)) - 1)
      maxY = Math.min(source.height - 1, Math.ceil(fb.y + Math.max(1, fb.height)) - 1)
    }
    const width = Math.max(1, maxX - minX + 1)
    const height = Math.max(1, maxY - minY + 1)
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    canvas.getContext('2d')!.drawImage(source, minX, minY, width, height, 0, 0, width, height)
    return { canvas, x: minX, y: minY, width, height }
  }

  private rasterizeRgbaMaskLayer(layerId: string) {
    const object = this.findLayer(layerId) as (FabricObject & { __isRgbaMaskLayer?: boolean; __preset?: LayerPreset }) | undefined
    if (!object?.__isRgbaMaskLayer || !this.fabricCanvas) return
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    this.drawLayerCondition(context, object, layerId)
    const bounds = object.getBoundingRect()
    const cropped = this.cropCanvasToAlphaBounds(canvas, { x: bounds.left, y: bounds.top, width: bounds.width, height: bounds.height })
    const replacement = new FabricImage(cropped.canvas, {
      left: cropped.x,
      top: cropped.y,
      scaleX: 1,
      scaleY: 1,
      objectCaching: false,
      name: object.get('name') || 'RGBA mask',
      opacity: object.opacity ?? 1,
      visible: object.visible ?? true,
    }) as unknown as FabricObject & { __isRgbaMaskLayer?: boolean; __preset?: LayerPreset }
    replacement.__isRgbaMaskLayer = true
    replacement.__preset = object.__preset
    this.attachPreset(replacement as unknown as FabricObject, object.__preset)
    this.replaceLayerObject(layerId, object as unknown as FabricObject, replacement as unknown as FabricObject)
  }

  private _isVideoLayer(layerId: string): boolean {
    const obj = this.findLayer(layerId)
    return !!(obj && (obj as FabricObject & { __isVideoLayer?: boolean }).__isVideoLayer)
  }

  private _startVideoRaf() {
    if (this._videoRafId != null) return
    const step = () => {
      const vid = this.videoInputElement
      const canvas = this._videoOffscreenCanvas
      if (vid && canvas && vid.readyState >= 2) {
        const ctx = canvas.getContext('2d')!
        const vw = vid.videoWidth || canvas.width
        const vh = vid.videoHeight || canvas.height
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        const scale = Math.min(canvas.width / vw, canvas.height / vh)
        const sw = vw * scale
        const sh = vh * scale
        ctx.drawImage(vid, (canvas.width - sw) / 2, (canvas.height - sh) / 2, sw, sh)
        this.fabricCanvas?.requestRenderAll()
      }
      this._videoRafId = requestAnimationFrame(step)
    }
    this._videoRafId = requestAnimationFrame(step)
  }

  public advanceVideoSequential() {
    const vid = this.videoInputElement
    if (!vid || !this.videoUrl) return
    const end = this.videoEnd > 0 ? this.videoEnd : this.videoDuration
    const step = 1 / this.videoFps
    if (!this.videoReversing) {
      vid.currentTime = Math.min(vid.currentTime + step, end)
      if (vid.currentTime >= end - step * 0.5) {
        if (this.videoLoopMode === 'loop') {
          vid.currentTime = this.videoStart
        } else if (this.videoLoopMode === 'ping-pong') {
          this.videoReversing = true
        }
      }
    } else {
      vid.currentTime = Math.max(vid.currentTime - step, this.videoStart)
      if (vid.currentTime <= this.videoStart + step * 0.5) {
        this.videoReversing = false
      }
    }
  }

  public renderRegionTimeline(currentLayer?: LayerItem) {
    const activeLayers = this.layers.filter((layer) => layer.visible && (!currentLayer || layer.preset.samplingEnabled || layer.id === currentLayer.id))
    const layerSchedules = this.autoLayerSchedules(activeLayers)
    const rows = activeLayers.flatMap((layer, index) => {
        const timing = layerSchedules.get(layer.id) ?? { start: 0, end: 1, schedule: 'linear' as RegionSchedule, active: true }
        if (!timing.active) return [{ id: layer.id, label: `${layer.name} Â· occluded`, color: this.layerRegionColor(layer.id), timing }]
        return this.layerRegionSpecs(layer).map((region) => ({
          id: region.id,
          label: currentLayer?.id === layer.id ? (region.name || layer.name) : `${layer.name}${layer.preset.samplingEnabled ? '' : ' Â· off'}`,
          color: region.color || this.hslToHex((index * 41 + 212) % 360, 70, 54),
          timing: this.resolvedRegionTiming(region, timing),
        }))
      })
    return html`<section class="schedule-timeline">
      <div class="timeline-head"><strong>Mask schedule</strong><span>0</span><span>0.5</span><span>1</span></div>
      ${rows.length
        ? rows.map((row) => {
            const start = Math.round(row.timing.start * 100)
            const width = Math.max(2, Math.round((row.timing.end - row.timing.start) * 100))
            return html`<div class=${row.timing.active ? 'timeline-row' : 'timeline-row muted'}>
              <span>${row.label}</span>
              <div class="timeline-track">
                <i style=${`left: ${start}%; width: ${width}%; background: ${row.color};`}></i>
              </div>
            </div>`
          })
        : html`<div class="timeline-empty">No scheduled masks</div>`}
    </section>`
  }

  private layerRegionSpecs(layer: LayerItem) {
    const explicit = this.regions.filter((region) => region.target === 'layer' && region.layerId === layer.id)
    if (this.noDefaultMaskLayers.includes(layer.id)) return explicit
    return [this.defaultLayerRegion(layer), ...explicit]
  }

  private defaultLayerRegion(layer: LayerItem): RegionItem {
    const transform = layer.transform
    const scene = $scene.get()
    const denoise = layer.preset.strength
    const cfg = CFG_FACTOR_DEFAULT
    return {
      id: `inherited:${layer.id}`,
      target: 'layer',
      layerId: layer.id,
      color: this.layerRegionColor(layer.id),
      name: `${layer.name} alpha`,
      shape: 'box',
      rect: {
        x: transform ? transform.centerX - transform.width / 2 : 0,
        y: transform ? transform.centerY - transform.height / 2 : 0,
        width: transform?.width ?? this.stageWidth,
        height: transform?.height ?? this.stageHeight,
        inverted: false,
        shape: 'box',
      },
      maskStrength: layer.preset.conditionWeight,
      innerBlur: layer.preset.conditionInnerBlur,
      outerBlur: layer.preset.conditionOuterBlur,
      negate: layer.preset.conditionNegate,
      inherited: false,
      prompt: layer.preset.rgbaPromptInherited ? scene.prompt : layer.preset.prompt,
      negativePrompt: layer.preset.rgbaNegativePromptInherited ? scene.negativePrompt : layer.preset.negativePrompt,
      denoise,
      mode: layer.preset.conditionMode,
      maskOperator: layer.preset.maskOperator,
      denoiseOperator: layer.preset.denoiseOperator,
      schedule: layer.preset.schedule,
      start: layer.preset.scheduleStart,
      end: layer.preset.scheduleEnd,
      cfgMaskInherited: false,
      denoiseMaskInherited: false,
      regionCfg: cfg,
      regionSteps: layer.preset.layerSteps,
      blendingEnabled: true,
      blendingRadius: 32,
      blendingStrength: 1,
    }
  }

  private resolvedLayerRgbaRegionPatch(layer: LayerItem): Partial<RegionItem> {
    return {
      prompt: layer.preset.prompt,
      negativePrompt: layer.preset.negativePrompt,
      denoise: layer.preset.strength,
      regionCfg: CFG_FACTOR_DEFAULT,
      maskStrength: layer.preset.conditionWeight,
      blendingEnabled: true,
      blendingRadius: 32,
      blendingStrength: 1,
    }
  }

  private activeSamplingLayersBottomUp() {
    return [...this.layers].reverse().filter((layer) => layer.visible && layer.preset.samplingEnabled)
  }

  private layerRegionColor(layerId: string) {
    const index = Math.max(0, this.layers.findIndex((layer) => layer.id === layerId))
    return this.hslToHex((index * 47 + 205) % 360, 78, 54)
  }

  private promptMaskName(layer: LayerItem, region: RegionItem) {
    const typed = region.name?.trim()
    if (typed) return typed
    const prompt = region.prompt?.trim()
    if (prompt) return prompt.slice(0, 16)
    const index = Math.max(0, this.layers.findIndex((item) => item.id === layer.id))
    return `Layer ${index + 1}`
  }

  public hideInheritedMask(layerId: string) {
    if (!this.noDefaultMaskLayers.includes(layerId))
      this.noDefaultMaskLayers = [...this.noDefaultMaskLayers, layerId]
    $canvas.setKey('noDefaultMaskLayers', this.noDefaultMaskLayers)
    this.scheduleMaskPreviewRefresh(true)
    this.queueLiveInputChanged()
    this.scheduleProjectPersist()
  }

  public restoreInheritedMask(layerId: string) {
    this.noDefaultMaskLayers = this.noDefaultMaskLayers.filter((id) => id !== layerId)
    $canvas.setKey('noDefaultMaskLayers', this.noDefaultMaskLayers)
    this.scheduleMaskPreviewRefresh(true)
    this.queueLiveInputChanged()
    this.scheduleProjectPersist()
  }

  public formatBytes(bytes: number) {
    if (!Number.isFinite(bytes) || bytes <= 0) return ''
    const units = ['B', 'KB', 'MB', 'GB', 'TB']
    let value = bytes
    let unit = 0
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024
      unit += 1
    }
    return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`
  }

  private renderSelectionOverlay() {
    if (!this.selectionRect) return ''
    if (this.selectionRect.shape === 'ellipse') {
      const { x, y, width, height, inverted } = this.normalizedSelection()
      return html`<svg class=${inverted ? 'selection-path inverted' : 'selection-path'} viewBox="0 0 100 100" preserveAspectRatio="none">
        <ellipse cx=${((x + width / 2) / this.stageWidth) * 100} cy=${((y + height / 2) / this.stageHeight) * 100} rx=${(width / this.stageWidth) * 50} ry=${(height / this.stageHeight) * 50} vector-effect="non-scaling-stroke"></ellipse>
      </svg>`
    }
    if (this.selectionRect.points?.length) {
      const points = this.selectionRect.points.map((point) => `${(point.x / this.stageWidth) * 100},${(point.y / this.stageHeight) * 100}`).join(' ')
      return html`<svg class="selection-path" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polygon points=${points} vector-effect="non-scaling-stroke"></polygon>
      </svg>`
    }
    const { x, y, width, height, inverted } = this.normalizedSelection()
    return html`<div
      class=${inverted ? 'selection-box inverted' : 'selection-box'}
      style="left: ${(x / this.stageWidth) * 100}%; top: ${(y / this.stageHeight) * 100}%; width: ${(width / this.stageWidth) * 100}%; height: ${(height / this.stageHeight) * 100}%;"
    ></div>`
  }

  public numberControl(
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    decimals: number,
    onChange: (value: number) => void,
    _compact = false,
    _defaultValue?: number,
  ) {
    return html`<rtd-number
      label=${label}
      .value=${value}
      min=${min}
      max=${max}
      step=${step}
      decimals=${decimals}
      @rtd-change=${(e: CustomEvent<{ value: number }>) => onChange(e.detail.value)}
    ></rtd-number>`
  }

  private setupCanvas() {
    this.fabricCanvas = new Canvas(this.editCanvasElement, {
      width: this.stageWidth,
      height: this.stageHeight,
      selection: true,
      fireRightClick: true,
      stopContextMenu: true,
      selectionColor: 'rgba(47, 155, 255, 0.14)',
      selectionBorderColor: '#00e5ff',
      selectionLineWidth: 2,
      selectionDashArray: [7, 4],
    })
    const syncIfReady = () => {
      if (!this.isRestoringHistory && !this.isSyncingLayers && !this.pendingPaintLayerId) this.syncLayers()
    }
    this.fabricCanvas.on('selection:created', syncIfReady)
    this.fabricCanvas.on('selection:updated', syncIfReady)
    this.fabricCanvas.on('selection:cleared', () => { syncIfReady(); this.clearTransformOverlay() })
    this.fabricCanvas.on('object:added', (event) => { this.invalidateLayerPreview(this.layerIdForObject(event.target as FabricObject | undefined)); syncIfReady() })
    this.fabricCanvas.on('object:removed', (event) => { this.invalidateLayerPreview(this.layerIdForObject(event.target as FabricObject | undefined)); syncIfReady() })
    this.fabricCanvas.on('object:modified', (event) => { const layerId = this.layerIdForObject(event.target as FabricObject | undefined); this.invalidateLayerPreview(layerId); syncIfReady(); this.scheduleTransformOverlayUpdate(); this.scheduleLayerMediaSurfaceRefresh(layerId) })
    this.fabricCanvas.on('object:moving', (event) => { this.scheduleTransformOverlayUpdate(); this.scheduleLayerMediaSurfaceRefresh(this.layerIdForObject(event.target as FabricObject | undefined)) })
    this.fabricCanvas.on('object:scaling', (event) => { this.scheduleTransformOverlayUpdate(); this.scheduleLayerMediaSurfaceRefresh(this.layerIdForObject(event.target as FabricObject | undefined)) })
    this.fabricCanvas.on('object:rotating', (event) => { this.scheduleTransformOverlayUpdate(); this.scheduleLayerMediaSurfaceRefresh(this.layerIdForObject(event.target as FabricObject | undefined)) })
    this.fabricCanvas.on('object:skewing', (event) => { this.scheduleTransformOverlayUpdate(); this.scheduleLayerMediaSurfaceRefresh(this.layerIdForObject(event.target as FabricObject | undefined)) })
    this.fabricCanvas.on('after:render', () => {
      this.scheduleTransformOverlayUpdate()
      this.scheduleInputCanvasCaptureRefresh()
    })
    this.fabricCanvas.on('before:transform', () => this.pushHistory())
    this.fabricCanvas.on('mouse:down', (event) => {
      if (this.toolMode === 'eyedropper') {
        const pt = this.fabricCanvas!.getScenePoint(event.e)
        this.pickCanvasColorAt(pt.x, pt.y)
        return
      }
      if (this._isShapeTool()) {
        if (this.editorDest !== 'paint' || !this.selectedLayerId) return
        const pt = this.fabricCanvas!.getScenePoint(event.e)
        if (this.toolMode === 'fill') {
          this.pushHistory()
          this.executeFillTool(pt.x, pt.y)
        } else {
          this.pushHistory()
          this._shapeDrawState = { startX: pt.x, startY: pt.y }
        }
        return
      }
      if (!this.fabricCanvas?.isDrawingMode) return
      this.pushHistory()
      this.pendingPaintLayerId = this.selectedLayerId
      this.pendingBrushButton = this.pointerButton(event.e)
      this.configureFabricBrush(this.pendingBrushButton)
      this.scheduleActiveDrawSignalRefresh()
    })
    this.fabricCanvas.on('mouse:up', (event) => {
      if (this._isShapeTool() && this._shapeDrawState) {
        const pt = this.fabricCanvas!.getScenePoint(event.e)
        this.clearEditorOverlay()
        this.finalizeShape(this._shapeDrawState.startX, this._shapeDrawState.startY, pt.x, pt.y)
        this._shapeDrawState = null
        return
      }
      this.pendingBrushButton = 0
      if (!this.rightPaintStroke) queueMicrotask(() => (this.pendingPaintLayerId = ''))
      this.configureFabricBrush(0)
      this.queueLiveInputChanged(true)
    })
    this.fabricCanvas.on('mouse:move', (event) => {
      if (this._isShapeTool() && this._shapeDrawState) {
        const pt = this.fabricCanvas!.getScenePoint(event.e)
        this.drawShapePreview(this._shapeDrawState.startX, this._shapeDrawState.startY, pt.x, pt.y)
        return
      }
      if (this.fabricCanvas?.isDrawingMode && this.pendingPaintLayerId) {
        this.scheduleLayerMediaSurfaceRefresh(this.pendingPaintLayerId)
        this.scheduleInputCanvasCaptureRefresh()
        this.queueLiveInputChanged(false, this.shouldRefreshLiveResourcesOnDraw())
      }
    })
    this.fabricCanvas.on('path:created', (event) => {
      this.attachStrokeToLayer(event.path as FabricObject, this.pendingPaintLayerId, this.isEraseStroke(this.pendingBrushButton))
      // Synchronous re-render so the destination-out path is committed to the lower
      // canvas before the next RAF-queued frame export fires.
      this.fabricCanvas?.renderAll()
      this.invalidateLayerPreview(this.pendingPaintLayerId)
      this.scheduleLayerMediaSurfaceRefresh(this.pendingPaintLayerId)
      this.scheduleInputCanvasCaptureRefresh(true)
      this.queueLiveInputChanged(true, this.shouldRefreshLiveResourcesOnDraw())
    })
    const brush = new PencilBrush(this.fabricCanvas)
    brush.color = this.brushColor
    brush.width = this.brushSize
    this.fabricCanvas.freeDrawingBrush = brush
    this.attachRightPaintHandlers()
    this.setTool(this.toolMode)
    // Fabric.js sets explicit inline pixel dimensions on its wrapper; override immediately.
    this.syncCanvasGeometry()
  }

  private attachRightPaintHandlers() {
    const upperCanvas = (this.fabricCanvas as (Canvas & { upperCanvasEl?: HTMLCanvasElement }) | undefined)?.upperCanvasEl
    if (!upperCanvas || this.upperCanvasElement === upperCanvas) return
    this.detachRightPaintHandlers()
    this.upperCanvasElement = upperCanvas
    upperCanvas.addEventListener('pointerdown', this.startRightPaintStroke, true)
    upperCanvas.addEventListener('pointermove', this.moveRightPaintStroke, true)
    upperCanvas.addEventListener('pointerup', this.endRightPaintStroke, true)
    upperCanvas.addEventListener('pointercancel', this.cancelRightPaintStroke, true)
    upperCanvas.addEventListener('pointerdown', this.scheduleUpperCanvasLiveInput, true)
    upperCanvas.addEventListener('pointermove', this.scheduleUpperCanvasLiveInput, true)
    upperCanvas.addEventListener('pointerup', this.scheduleUpperCanvasLiveInput, true)
    upperCanvas.addEventListener('pointercancel', this.scheduleUpperCanvasLiveInput, true)
    upperCanvas.addEventListener('contextmenu', this.preventPaintContextMenu, true)
  }

  private detachRightPaintHandlers() {
    if (!this.upperCanvasElement) return
    this.upperCanvasElement.removeEventListener('pointerdown', this.startRightPaintStroke, true)
    this.upperCanvasElement.removeEventListener('pointermove', this.moveRightPaintStroke, true)
    this.upperCanvasElement.removeEventListener('pointerup', this.endRightPaintStroke, true)
    this.upperCanvasElement.removeEventListener('pointercancel', this.cancelRightPaintStroke, true)
    this.upperCanvasElement.removeEventListener('pointerdown', this.scheduleUpperCanvasLiveInput, true)
    this.upperCanvasElement.removeEventListener('pointermove', this.scheduleUpperCanvasLiveInput, true)
    this.upperCanvasElement.removeEventListener('pointerup', this.scheduleUpperCanvasLiveInput, true)
    this.upperCanvasElement.removeEventListener('pointercancel', this.scheduleUpperCanvasLiveInput, true)
    this.upperCanvasElement.removeEventListener('contextmenu', this.preventPaintContextMenu, true)
    this.upperCanvasElement = undefined
  }

  private setupMaskCanvas() {
    this.maskCanvasElement.width = this.stageWidth
    this.maskCanvasElement.height = this.stageHeight
    this.maskContext = this.maskCanvasElement.getContext('2d') ?? undefined
    this.maskPreviewCanvasElement.width = this.stageWidth
    this.maskPreviewCanvasElement.height = this.stageHeight
    this.maskPreviewContext = this.maskPreviewCanvasElement.getContext('2d') ?? undefined
  }

  private seedCanvas() {
    this.fabricCanvas?.clear()
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private setTool(mode: ToolMode, commitPending = true) {
    if (commitPending && this.pendingRgbaPlacementId && mode !== this.toolMode) this.commitPendingRgbaPlacement()
    if (commitPending && this.toolMode === 'select' && mode !== 'select') this.commitActiveLayerTransform()
    this.toolMode = mode
    if (!this._syncingToStore) { this._syncingToStore = true; $canvas.setKey('toolMode', mode); this._syncingToStore = false }
    if (mode === 'text') this.addTextLayer()
    this.applyToolMode()
    this.clearEditorOverlay()
    this.scheduleTransformOverlayUpdate()
  }

  private setEditorDest(dest: EditorDest, commitPending = true) {
    if (commitPending && this.pendingRgbaPlacementId && dest !== this.editorDest) this.commitPendingRgbaPlacement()
    if (commitPending && this.toolMode === 'select' && dest !== this.editorDest) this.commitActiveLayerTransform()
    this.editorDest = dest
    if (!this._syncingToStore) { this._syncingToStore = true; $canvas.setKey('editorDest', dest); this._syncingToStore = false }
    if (dest === 'paint') {
      this.activeMaskRegionId = ''
      this.setActiveMaskChannel('color')
    }
    this.applyToolMode()
    this.syncMaskScope()
    this.scheduleMaskPreviewRefresh(true)
  }

  public enterPaintMode(layerId?: string) {
    if (layerId) this.selectLayer(layerId)
    this.activeMaskRegionId = ''
    this.setEditorDest('paint')
    this.setActiveMaskChannel('color')
    this.setTool('brush')
  }

  private setActiveMaskChannel(channel: MaskChannel) {
    if (channel === this.activeMaskChannel) return
    this.saveCurrentMaskScope()
    this.activeMaskChannel = channel
    if (channel === 'color' && this.activeMaskRegionId.startsWith('special:')) this.activeMaskRegionId = ''
    if ((channel === 'cfg' || channel === 'denoise') && this.selectedLayerId) this.activeMaskRegionId = `special:${channel}:${this.selectedLayerId}`
    if (!this._syncingToStore) { this._syncingToStore = true; $canvas.setKey('activeMaskChannel', channel); this._syncingToStore = false }
    this.loadMaskScope(this.activeMaskScope)
    this.scheduleMaskPreviewRefresh(true)
  }

  public selectMaskTool(detail: { layerId: string; regionId?: string; channel?: MaskChannel; color?: string }) {
    const channel = detail.channel ?? 'color'
    this.selectLayer(detail.layerId)
    this.activeMaskRegionId = detail.regionId ?? ''
    this.setEditorDest('mask')
    this.setActiveMaskChannel(channel)
    if (detail.color && channel === 'color') {
      this.activeColorSlot = 'primary'
      this.setBrushColor(detail.color)
      $canvas.setKey('brushColor', this.brushColor)
    }
    if (channel !== 'color') {
      this.maskBrushValue = 255
      $canvas.setKey('maskBrushValue', 255)
    }
    this.setTool('brush')
  }

  private applyToolMode() {
    if (!this.fabricCanvas) return
    const selectedLayer = this.selectedLayerId ? this.findLayer(this.selectedLayerId) : undefined
    const activeObject = this.fabricCanvas.getActiveObject()
    const activeIsPaintChild = activeObject ? isPaintChildObject(activeObject as FabricObject & { __paintChild?: boolean; __parentLayerId?: string }) : false
    if (this.editorDest === 'mask' && activeObject) this.fabricCanvas.discardActiveObject()
    if (this.editorDest === 'paint' && selectedLayer && activeObject !== selectedLayer && !activeIsPaintChild) {
      this.fabricCanvas.setActiveObject(selectedLayer)
    }
    this.fabricCanvas.isDrawingMode =
      this.editorDest === 'paint' && (this.toolMode === 'brush' || this.toolMode === 'eraser') && this.selectedEntity === 'layer' && !!this.selectedLayerId
    this.fabricCanvas.selection = this.toolMode === 'select'
    this.configureFabricBrush(0)
  }

  private maskPreviewVisible() {
    return (this.inputMaskVisible || this.inputChannelVisible) && this.currentMaskScope() !== undefined
  }

  private maskCanvasEditable() {
    if (!this.hasEditableMaskLayer()) return false
    if (this.activeMaskChannel === 'color' && this.activeMaskRegionId.startsWith('inherited:')) return false
    return this.editorDest === 'mask' && this.currentMaskScope() !== undefined && (this.toolMode === 'brush' || this.toolMode === 'eraser')
  }

  private maskCanvasActive() {
    return this._isSelectionTool() || ((this.inputMaskVisible || this.inputChannelVisible) && this.editorDest === 'mask' && this.maskCanvasEditable())
  }

  private hasEditableMaskLayer() {
    return this.selectedEntity === 'layer' && !!this.selectedLayerId && this.layers.some((layer) => layer.id === this.selectedLayerId)
  }

  private currentMaskScope(): MaskScope | undefined {
    if (this.hasEditableMaskLayer()) return `layer:${this.selectedLayerId}`
    return undefined
  }

  private pointerButton(event: Event) {
    const pointer = event as PointerEvent | MouseEvent
    return pointer.button === 2 ? 2 : 0
  }

  private isEraseStroke(button = this.pendingBrushButton) {
    return this.toolMode === 'eraser' || button === 2
  }

  private configureFabricBrush(button = 0) {
    if (!this.fabricCanvas?.freeDrawingBrush) return
    const color = button === 2 ? this.secondaryBrushColor : this.brushColor
    const fillColor = this.isEraseStroke(button) ? '#000000' : color
    this.fabricCanvas.freeDrawingBrush.color = fillColor
    this.fabricCanvas.freeDrawingBrush.width = this.brushSize
    // Soft brush: shadow with same color creates a blurred halo that softens edges
    if (this.brushHardness < 100) {
      const blur = ((100 - this.brushHardness) / 100) * this.brushSize * 0.8
      this.fabricCanvas.freeDrawingBrush.shadow = new Shadow({ color: fillColor, blur, offsetX: 0, offsetY: 0 })
    } else {
      this.fabricCanvas.freeDrawingBrush.shadow = null
    }
  }

  private _isShapeTool() {
    return this.toolMode === 'line' || this.toolMode === 'rect' || this.toolMode === 'ellipse' || this.toolMode === 'fill'
  }

  private _isSelectionTool() {
    return this.toolMode === 'region' || this.toolMode === 'lassoSelect' || this.toolMode === 'ellipseSelect' || this.toolMode === 'magicWand' || this.toolMode === 'moveSelection'
  }

  private drawShapePreview(x1: number, y1: number, x2: number, y2: number) {
    this.drawEditorOverlay((ctx) => {
      ctx.strokeStyle = this.brushColor
      ctx.fillStyle = this.secondaryBrushColor
      ctx.lineWidth = this.brushSize
      ctx.lineCap = 'round'
      const drawFill = this.shapeDrawMode === 'fill' || this.shapeDrawMode === 'both'
      const drawStroke = this.shapeDrawMode === 'stroke' || this.shapeDrawMode === 'both'
      if (this.toolMode === 'line') {
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
      } else if (this.toolMode === 'rect' || this.toolMode === 'ellipse') {
        const rx = Math.min(x1, x2), ry = Math.min(y1, y2)
        const rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1)
        if (this.toolMode === 'ellipse') {
          ctx.beginPath()
          ctx.ellipse(rx + rw / 2, ry + rh / 2, rw / 2, rh / 2, 0, 0, Math.PI * 2)
          if (drawFill) ctx.fill()
          if (drawStroke) ctx.stroke()
        } else {
          if (drawFill) ctx.fillRect(rx, ry, rw, rh)
          if (drawStroke) ctx.strokeRect(rx, ry, rw, rh)
        }
      }
    })
  }

  private finalizeShape(x1: number, y1: number, x2: number, y2: number) {
    if (!this.fabricCanvas || !this.selectedLayerId) return
    let obj: FabricObject
    const strokeColor = this.brushColor
    const fillColor = this.secondaryBrushColor
    const w = this.brushSize
    if (this.toolMode === 'line') {
      obj = new Path(`M ${x1} ${y1} L ${x2} ${y2}`, {
        stroke: strokeColor, fill: '', strokeWidth: w, strokeLineCap: 'round',
        selectable: false, evented: false, objectCaching: false,
      }) as unknown as FabricObject
    } else if (this.toolMode === 'rect') {
      const rx = Math.min(x1, x2), ry = Math.min(y1, y2)
      const rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1)
      obj = new Rect({
        left: rx, top: ry, width: rw, height: rh,
        fill: this.shapeDrawMode === 'stroke' ? '' : fillColor,
        stroke: this.shapeDrawMode === 'fill' ? '' : strokeColor,
        strokeWidth: this.shapeDrawMode === 'fill' ? 0 : w,
        selectable: false, evented: false, objectCaching: false,
      }) as unknown as FabricObject
    } else {
      const rx = Math.min(x1, x2), ry = Math.min(y1, y2)
      const rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1)
      obj = new Ellipse({
        left: rx, top: ry, rx: rw / 2, ry: rh / 2,
        fill: this.shapeDrawMode === 'stroke' ? '' : fillColor,
        stroke: this.shapeDrawMode === 'fill' ? '' : strokeColor,
        strokeWidth: this.shapeDrawMode === 'fill' ? 0 : w,
        selectable: false, evented: false, objectCaching: false,
      }) as unknown as FabricObject
    }
    this.fabricCanvas.add(obj)
    this.attachStrokeToLayer(obj, this.selectedLayerId, false)
    this.fabricCanvas.renderAll()
    this.queueLiveInputChanged(true, true)
  }

  private executeFillTool(clickX: number, clickY: number) {
    if (!this.fabricCanvas || !this.selectedLayerId) return
    const layerObject = this.findLayer(this.selectedLayerId)
    if (!layerObject || isPaintChildObject(layerObject as FabricObject & { __paintChild?: boolean; __parentLayerId?: string })) return
    const sourceCanvas = this.renderLayerBlockCanvas(this.selectedLayerId, layerObject)
    const ctx = sourceCanvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    const cw = sourceCanvas.width, ch = sourceCanvas.height
    const sx = cw / this.stageWidth, sy = ch / this.stageHeight
    const px = Math.max(0, Math.min(cw - 1, Math.round(clickX * sx)))
    const py = Math.max(0, Math.min(ch - 1, Math.round(clickY * sy)))
    const imageData = ctx.getImageData(0, 0, cw, ch)
    const d = imageData.data
    const tidx = (py * cw + px) * 4
    const tr = d[tidx], tg = d[tidx + 1], tb = d[tidx + 2], ta = d[tidx + 3]
    const fill = this.activeRgba()
    if (!fill) return
    const fillAlpha = Math.round(this.clamp(fill.a, 0, 1, 1) * 255)
    const tol = this.clamp(this.fillTolerance, 0, 255, 32)
    const same = (i: number) =>
      Math.abs(d[i] - tr) <= tol && Math.abs(d[i + 1] - tg) <= tol &&
      Math.abs(d[i + 2] - tb) <= tol && Math.abs(d[i + 3] - ta) <= tol
    const out = new Uint8ClampedArray(cw * ch * 4)
    const vis = new Uint8Array(cw * ch)
    const stack = [py * cw + px]
    vis[py * cw + px] = 1
    while (stack.length) {
      const pos = stack.pop()!
      const x = pos % cw, y = (pos / cw) | 0
      const i = pos * 4
      out[i] = fill.r; out[i + 1] = fill.g; out[i + 2] = fill.b; out[i + 3] = fillAlpha
      for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]] as [number, number][]) {
        const nx = x + dx, ny = y + dy
        if (nx < 0 || nx >= cw || ny < 0 || ny >= ch) continue
        const ni = ny * cw + nx
        if (vis[ni] || !same(ni * 4)) continue
        vis[ni] = 1; stack.push(ni)
      }
    }
    this.applyFillEdgeStrategy(vis, out, d, cw, ch, fill, fillAlpha, ta)
    const tmp = document.createElement('canvas')
    tmp.width = cw; tmp.height = ch
    tmp.getContext('2d')!.putImageData(new ImageData(out, cw, ch), 0, 0)
    const img = new FabricImage(tmp, {
      left: 0, top: 0, scaleX: this.stageWidth / cw, scaleY: this.stageHeight / ch,
      selectable: false, evented: false, objectCaching: false,
    }) as unknown as FabricObject
    this.fabricCanvas.add(img)
    this.attachStrokeToLayer(img, this.selectedLayerId, false)
    this.fabricCanvas.renderAll()
    this.queueLiveInputChanged(true, true)
  }

  private pickCanvasColorAt(x: number, y: number) {
    if (!this.fabricCanvas) return
    const fc = this.fabricCanvas as Canvas & { lowerCanvasEl: HTMLCanvasElement }
    const ctx = fc.lowerCanvasEl.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    const cw = fc.lowerCanvasEl.width, ch = fc.lowerCanvasEl.height
    const px = Math.max(0, Math.min(cw - 1, Math.round(x * (cw / this.stageWidth))))
    const py = Math.max(0, Math.min(ch - 1, Math.round(y * (ch / this.stageHeight))))
    const data = ctx.getImageData(px, py, 1, 1).data
    const color = this.rgbaToCss({ r: data[0], g: data[1], b: data[2], a: data[3] / 255 })
    this.setActiveColor(color)
    $canvas.setKey(this.activeColorSlot === 'primary' ? 'brushColor' : 'secondaryBrushColor', color)
  }

  private applyFillEdgeStrategy(
    fillMask: Uint8Array,
    out: Uint8ClampedArray,
    source: Uint8ClampedArray,
    width: number,
    height: number,
    fill: RgbaColor,
    fillAlpha: number,
    targetAlpha: number,
  ) {
    if (this.fillEdgeStrategy === 'none') return
    if (this.fillEdgeStrategy === 'transparent_blend') {
      this.blendFillIntoTransparentPixels(fillMask, out, source, width, height, fill, fillAlpha)
      return
    }
    this.expandFillIntoAntialiasFringe(fillMask, out, source, width, height, fill, fillAlpha, targetAlpha)
  }

  private expandFillIntoAntialiasFringe(
    fillMask: Uint8Array,
    out: Uint8ClampedArray,
    source: Uint8ClampedArray,
    width: number,
    height: number,
    fill: RgbaColor,
    fillAlpha: number,
    targetAlpha: number,
  ) {
    const fringeMask = new Uint8Array(fillMask)
    const maxFringeAlpha = targetAlpha < 96 ? 245 : Math.min(245, targetAlpha + 64)
    for (let pass = 0; pass < 2; pass++) {
      const next = new Uint8Array(fringeMask)
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const pos = y * width + x
          if (!fringeMask[pos]) continue
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue
              const nx = x + dx, ny = y + dy
              if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue
              const neighbor = ny * width + nx
              if (next[neighbor]) continue
              const index = neighbor * 4
              const alpha = source[index + 3]
              if (alpha > maxFringeAlpha) continue
              next[neighbor] = 1
              out[index] = fill.r
              out[index + 1] = fill.g
              out[index + 2] = fill.b
              out[index + 3] = fillAlpha
            }
          }
        }
      }
      fringeMask.set(next)
    }
  }

  private blendFillIntoTransparentPixels(
    fillMask: Uint8Array,
    out: Uint8ClampedArray,
    source: Uint8ClampedArray,
    width: number,
    height: number,
    fill: RgbaColor,
    fillAlpha: number,
  ) {
    const edgeMask = new Uint8Array(fillMask)
    for (let pass = 0; pass < 2; pass++) {
      const next = new Uint8Array(edgeMask)
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const pos = y * width + x
          if (!edgeMask[pos]) continue
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue
              const nx = x + dx, ny = y + dy
              if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue
              const neighbor = ny * width + nx
              if (next[neighbor]) continue
              const index = neighbor * 4
              const sourceAlpha = source[index + 3]
              if (sourceAlpha >= 255) continue
              const coverage = 1 - sourceAlpha / 255
              next[neighbor] = 1
              out[index] = fill.r
              out[index + 1] = fill.g
              out[index + 2] = fill.b
              out[index + 3] = Math.round(fillAlpha * coverage)
            }
          }
        }
      }
      edgeMask.set(next)
    }
  }

  private canRightPaint() {
    return this.editorDest === 'paint' && !!this.fabricCanvas?.isDrawingMode && (this.toolMode === 'brush' || this.toolMode === 'eraser') && this.selectedEntity === 'layer' && !!this.selectedLayerId
  }

  private preventPaintContextMenu = (event: MouseEvent) => {
    if (!this.canRightPaint()) return
    event.preventDefault()
    event.stopImmediatePropagation()
  }

  private startRightPaintStroke = (event: PointerEvent) => {
    if (event.button !== 2 || !this.canRightPaint() || !this.upperCanvasElement) return
    event.preventDefault()
    event.stopImmediatePropagation()
    this.pushHistory()
    this.pendingPaintLayerId = this.selectedLayerId
    this.pendingBrushButton = 2
    this.configureFabricBrush(2)
    const point = this.canvasPointFromPointer(event)
    this.rightPaintStroke = { pointerId: event.pointerId, points: [point] }
    this.upperCanvasElement.setPointerCapture(event.pointerId)
    this.renderRightPaintPreview()
    this.scheduleLayerMediaSurfaceRefresh(this.pendingPaintLayerId)
    this.scheduleActiveDrawSignalRefresh()
    this.scheduleInputCanvasCaptureRefresh()
    this.queueLiveInputChanged(false, this.shouldRefreshLiveResourcesOnDraw())
  }

  private moveRightPaintStroke = (event: PointerEvent) => {
    if (!this.rightPaintStroke || this.rightPaintStroke.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopImmediatePropagation()
    const point = this.canvasPointFromPointer(event)
    const last = this.rightPaintStroke.points.at(-1)
    if (!last || Math.hypot(point.x - last.x, point.y - last.y) >= 1.5) {
      this.rightPaintStroke = { ...this.rightPaintStroke, points: [...this.rightPaintStroke.points, point] }
      this.renderRightPaintPreview()
      this.scheduleLayerMediaSurfaceRefresh(this.pendingPaintLayerId)
      this.scheduleActiveDrawSignalRefresh()
      this.scheduleInputCanvasCaptureRefresh()
      this.queueLiveInputChanged(false, this.shouldRefreshLiveResourcesOnDraw())
    }
  }

  private endRightPaintStroke = (event: PointerEvent) => {
    if (!this.rightPaintStroke || this.rightPaintStroke.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopImmediatePropagation()
    this.finishRightPaintStroke()
  }

  private cancelRightPaintStroke = (event: PointerEvent) => {
    if (!this.rightPaintStroke || this.rightPaintStroke.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopImmediatePropagation()
    this.clearRightPaintPreview()
    this.rightPaintStroke = undefined
    this.pendingPaintLayerId = ''
    this.pendingBrushButton = 0
    this.configureFabricBrush(0)
    this.queueLiveInputChanged(true, this.shouldRefreshLiveResourcesOnDraw())
  }

  private finishRightPaintStroke() {
    if (!this.fabricCanvas || !this.rightPaintStroke) return
    const points = this.rightPaintStroke.points
    this.clearRightPaintPreview()
    if (this.upperCanvasElement?.hasPointerCapture(this.rightPaintStroke.pointerId)) {
      this.upperCanvasElement.releasePointerCapture(this.rightPaintStroke.pointerId)
    }
    const strokePath = this.pathFromPoints(points)
    const path = new Path(strokePath, {
      fill: '',
      stroke: '#000000',
      strokeWidth: this.brushSize,
      strokeLineCap: 'round',
      strokeLineJoin: 'round',
      selectable: false,
      evented: false,
      name: 'Paint stroke',
    })
    path.set({ globalCompositeOperation: 'destination-out' } as Partial<FabricObject>)
    const parentLayerId = this.pendingPaintLayerId
    this.fabricCanvas.add(path)
    this.attachStrokeToLayer(path, parentLayerId, true)
    this.fabricCanvas.requestRenderAll()
    this.rightPaintStroke = undefined
    this.pendingPaintLayerId = ''
    this.pendingBrushButton = 0
    this.configureFabricBrush(0)
    this.scheduleInputCanvasCaptureRefresh()
    this.queueLiveInputChanged(true, this.shouldRefreshLiveResourcesOnDraw())
  }

  private canvasPointFromPointer(event: PointerEvent) {
    const rect = this.editCanvasElement.getBoundingClientRect()
    return {
      x: this.clamp(((event.clientX - rect.left) / rect.width) * this.stageWidth, 0, this.stageWidth, 0),
      y: this.clamp(((event.clientY - rect.top) / rect.height) * this.stageHeight, 0, this.stageHeight, 0),
    }
  }

  private pathFromPoints(points: { x: number; y: number }[]) {
    const first = points[0] ?? { x: 0, y: 0 }
    if (points.length < 2) return `M ${first.x} ${first.y} L ${first.x + 0.01} ${first.y}`
    return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ')
  }

  private renderRightPaintPreview() {
    if (!this.fabricCanvas || !this.rightPaintStroke) return
    const canvas = this.fabricCanvas as Canvas & { contextTop: CanvasRenderingContext2D; clearContext: (context: CanvasRenderingContext2D) => void }
    const context = canvas.contextTop
    canvas.clearContext(context)
    const points = this.rightPaintStroke.points
    const first = points[0]
    if (!first) return
    context.save()
    context.strokeStyle = '#000000'
    context.lineWidth = this.brushSize
    context.lineCap = 'round'
    context.lineJoin = 'round'
    context.beginPath()
    context.moveTo(first.x, first.y)
    for (const point of points.slice(1)) context.lineTo(point.x, point.y)
    if (points.length === 1) context.lineTo(first.x + 0.01, first.y)
    context.stroke()
    context.restore()
  }

  private clearRightPaintPreview() {
    const canvas = this.fabricCanvas as (Canvas & { contextTop?: CanvasRenderingContext2D; clearContext?: (context: CanvasRenderingContext2D) => void }) | undefined
    if (canvas?.contextTop && canvas.clearContext) canvas.clearContext(canvas.contextTop)
  }

  private setBrushColor(color: string) {
    this.brushColor = this.normalizeColor(color) ?? this.brushColor
    this.configureFabricBrush(this.pendingBrushButton)
    if (this.inputActiveMaskOnly) this.scheduleMaskPreviewRefresh(true)
  }

  private setSecondaryBrushColor(color: string) {
    this.secondaryBrushColor = this.normalizeColor(color) ?? this.secondaryBrushColor
    this.configureFabricBrush(this.pendingBrushButton)
  }

  private activeColor() {
    return this.activeColorSlot === 'primary' ? this.brushColor : this.secondaryBrushColor
  }

  private colorInputValue() {
    return this.rgbToHex(this.activeRgba())
  }

  private colorTextValue() {
    const color = this.activeRgba()
    return color.a < 1 ? this.rgbaToHex8(color) : this.rgbToHex(color)
  }

  private colorIsActive(color: string) {
    const active = this.activeRgba()
    const candidate = this.parseColor(color)
    return !!candidate && active.r === candidate.r && active.g === candidate.g && active.b === candidate.b && Math.round(active.a * 100) === Math.round(candidate.a * 100)
  }

  private setActiveColor(color: string) {
    const nextColor = this.normalizeColor(color)
    if (!nextColor) return
    if (this.activeColorSlot === 'primary') this.setBrushColor(nextColor)
    else this.setSecondaryBrushColor(nextColor)
  }

  public useLayerRegionColor(layerId: string) {
    this.activeColorSlot = 'primary'
    this.setBrushColor(this.layerRegionColor(layerId))
    this.setSecondaryBrushColor('rgba(0, 0, 0, 0)')
    this.setTool('brush')
  }

  public useRegionColor(color: string) {
    this.activeColorSlot = 'primary'
    this.setBrushColor(color)
    this.setSecondaryBrushColor('rgba(0, 0, 0, 0)')
    this.setTool('brush')
  }

  private setSecondaryFromPalette(event: Event, color: string) {
    event.preventDefault()
    this.setSecondaryBrushColor(color)
  }

  private swapColors = () => {
    const primary = this.brushColor
    this.brushColor = this.secondaryBrushColor
    this.secondaryBrushColor = primary
    this.configureFabricBrush(this.pendingBrushButton)
    if (this.inputActiveMaskOnly) this.scheduleMaskPreviewRefresh(true)
  }

  private pickHexColor = (event: Event) => {
    const color = this.normalizeColor((event.target as HTMLInputElement).value)
    if (color) this.setActiveColor(color)
  }

  private activeRgb() {
    const { r, g, b } = this.activeRgba()
    return { r, g, b }
  }

  private activeRgba() {
    return this.parseColor(this.activeColor()) ?? { r: 255, g: 255, b: 255, a: 1 }
  }

  private activeHsl() {
    return this.rgbToHsl(this.activeRgb())
  }

  private setActiveRgba(patch: Partial<RgbaColor>) {
    const next = { ...this.activeRgba(), ...patch }
    this.setActiveColor(this.rgbaToCss(next))
  }

  private setActiveHsl(patch: Partial<{ h: number; s: number; l: number }>) {
    const hsl = { ...this.activeHsl(), ...patch }
    this.setActiveColor(this.withActiveAlpha(this.hslToHex(hsl.h, hsl.s, hsl.l)))
  }

  private withActiveAlpha(color: string) {
    const parsed = this.parseColor(color)
    if (!parsed) return color
    return this.rgbaToCss({ ...parsed, a: this.activeRgba().a })
  }

  private plannedColors() {
    const hsl = this.activeHsl()
    const wheel = (offset: number, saturation = hsl.s, lightness = hsl.l) => this.withActiveAlpha(this.hslToHex(hsl.h + offset, saturation, lightness))
    if (this.colorPlannerMode === 'analogous') return [-36, -18, 0, 18, 36, 54].map((offset) => wheel(offset))
    if (this.colorPlannerMode === 'triad') return [0, 120, 240, 120, 240, 0].map((offset, index) => wheel(offset, hsl.s, index > 2 ? Math.min(92, hsl.l + 18) : hsl.l))
    if (this.colorPlannerMode === 'tetrad') return [0, 60, 180, 240, 30, 210].map((offset) => wheel(offset))
    if (this.colorPlannerMode === 'split') return [0, 150, 210, 30, 180, 330].map((offset) => wheel(offset))
    if (this.colorPlannerMode === 'monochrome') return [18, 30, 42, 54, 66, 78].map((lightness) => wheel(0, hsl.s, lightness))
    return [0, 180, 150, 210, 0, 180].map((offset, index) => wheel(offset, hsl.s, index > 3 ? Math.min(90, hsl.l + 18) : hsl.l))
  }

  private blendColors(startColor: string, endColor: string, count: number) {
    const start = this.parseColor(startColor) ?? { r: 255, g: 255, b: 255, a: 1 }
    const end = this.parseColor(endColor) ?? { r: 0, g: 0, b: 0, a: 1 }
    return Array.from({ length: count }, (_, index) => {
      const mix = count === 1 ? 0 : index / (count - 1)
      return this.rgbaToCss({
        r: Math.round(start.r + (end.r - start.r) * mix),
        g: Math.round(start.g + (end.g - start.g) * mix),
        b: Math.round(start.b + (end.b - start.b) * mix),
        a: start.a + (end.a - start.a) * mix,
      })
    })
  }

  private toneRamp(color: string, count: number) {
    const parsed = this.parseColor(color) ?? { r: 255, g: 255, b: 255, a: 1 }
    const hsl = this.rgbToHsl(parsed)
    return Array.from({ length: count }, (_, index) => this.rgbaToCss({ ...this.parseColor(this.hslToHex(hsl.h, hsl.s, 12 + index * (76 / Math.max(1, count - 1))))!, a: parsed.a }))
  }

  private normalizeColor(color: string) {
    const parsed = this.parseColor(color)
    return parsed ? this.rgbaToCss(parsed) : undefined
  }

  private parseColor(color: string): RgbaColor | undefined {
    const trimmed = color.trim()
    const short = /^#?([0-9a-f]{3})$/i.exec(trimmed)
    if (short) {
      const expanded = short[1].split('').map((value) => value + value).join('')
      return this.hexChannels(expanded, 'ff')
    }
    const shortAlpha = /^#?([0-9a-f]{4})$/i.exec(trimmed)
    if (shortAlpha) {
      const expanded = shortAlpha[1].split('').map((value) => value + value).join('')
      return this.hexChannels(expanded.slice(0, 6), expanded.slice(6, 8))
    }
    const full = /^#?([0-9a-f]{6})$/i.exec(trimmed)
    if (full) return this.hexChannels(full[1], 'ff')
    const fullAlpha = /^#?([0-9a-f]{8})$/i.exec(trimmed)
    if (fullAlpha) return this.hexChannels(fullAlpha[1].slice(0, 6), fullAlpha[1].slice(6, 8))
    const rgba = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)$/i.exec(trimmed)
    if (!rgba) return undefined
    return {
      r: Math.round(this.clamp(Number(rgba[1]), 0, 255, 0)),
      g: Math.round(this.clamp(Number(rgba[2]), 0, 255, 0)),
      b: Math.round(this.clamp(Number(rgba[3]), 0, 255, 0)),
      a: this.clamp(rgba[4] === undefined ? 1 : Number(rgba[4]), 0, 1, 1),
    }
  }

  private hexChannels(rgb: string, alpha: string): RgbaColor {
    return {
      r: Number.parseInt(rgb.slice(0, 2), 16),
      g: Number.parseInt(rgb.slice(2, 4), 16),
      b: Number.parseInt(rgb.slice(4, 6), 16),
      a: Number.parseInt(alpha, 16) / 255,
    }
  }

  private rgbToHex(rgb: { r: number; g: number; b: number }) {
    const channel = (value: number) => Math.round(this.clamp(value, 0, 255, 0)).toString(16).padStart(2, '0')
    return `#${channel(rgb.r)}${channel(rgb.g)}${channel(rgb.b)}`
  }

  private rgbaToHex8(color: RgbaColor) {
    const alpha = Math.round(this.clamp(color.a, 0, 1, 1) * 255).toString(16).padStart(2, '0')
    return `${this.rgbToHex(color)}${alpha}`
  }

  private rgbaToCss(color: RgbaColor) {
    const normalized = {
      r: Math.round(this.clamp(color.r, 0, 255, 0)),
      g: Math.round(this.clamp(color.g, 0, 255, 0)),
      b: Math.round(this.clamp(color.b, 0, 255, 0)),
      a: this.clamp(color.a, 0, 1, 1),
    }
    return normalized.a >= 0.995 ? this.rgbToHex(normalized) : `rgba(${normalized.r}, ${normalized.g}, ${normalized.b}, ${Number(normalized.a.toFixed(3))})`
  }

  private sameRgb(first: string, second: string) {
    const firstColor = this.parseColor(first)
    const secondColor = this.parseColor(second)
    return !!firstColor && !!secondColor && firstColor.r === secondColor.r && firstColor.g === secondColor.g && firstColor.b === secondColor.b
  }

  private rgbToHsl(rgb: { r: number; g: number; b: number }) {
    const red = rgb.r / 255
    const green = rgb.g / 255
    const blue = rgb.b / 255
    const max = Math.max(red, green, blue)
    const min = Math.min(red, green, blue)
    const lightness = (max + min) / 2
    if (max === min) return { h: 0, s: 0, l: Math.round(lightness * 100) }
    const delta = max - min
    const saturation = lightness > 0.5 ? delta / (2 - max - min) : delta / (max + min)
    let hue = 0
    if (max === red) hue = (green - blue) / delta + (green < blue ? 6 : 0)
    if (max === green) hue = (blue - red) / delta + 2
    if (max === blue) hue = (red - green) / delta + 4
    return { h: Math.round(hue * 60), s: Math.round(saturation * 100), l: Math.round(lightness * 100) }
  }

  private hslToHex(hue: number, saturation: number, lightness: number) {
    const normalizedHue = ((hue % 360) + 360) % 360
    const sat = this.clamp(saturation, 0, 100, 0) / 100
    const light = this.clamp(lightness, 0, 100, 50) / 100
    const chroma = (1 - Math.abs(2 * light - 1)) * sat
    const x = chroma * (1 - Math.abs(((normalizedHue / 60) % 2) - 1))
    const match = light - chroma / 2
    const [red, green, blue] = normalizedHue < 60
      ? [chroma, x, 0]
      : normalizedHue < 120
        ? [x, chroma, 0]
        : normalizedHue < 180
          ? [0, chroma, x]
          : normalizedHue < 240
            ? [0, x, chroma]
            : normalizedHue < 300
              ? [x, 0, chroma]
              : [chroma, 0, x]
    return this.rgbToHex({ r: (red + match) * 255, g: (green + match) * 255, b: (blue + match) * 255 })
  }

  public setBrushSize(value: number) {
    this.brushSize = Math.round(this.clamp(value, 1, 4096, this.brushSize))
    this.configureFabricBrush(this.pendingBrushButton)
  }

  public setBrushHardness(value: number) {
    this.brushHardness = Math.round(this.clamp(value, 0, 100, this.brushHardness))
    this.configureFabricBrush(this.pendingBrushButton)
  }

  private clamp(value: number, min: number, max: number, fallback: number) {
    if (!Number.isFinite(value)) return fallback
    return Math.max(min, Math.min(max, value))
  }

  private pickColor = (event: Event) => {
    this.setActiveColor(this.withActiveAlpha((event.target as HTMLInputElement).value))
  }

  private currentPreset(): LayerPreset {
    return {
      prompt: this.prompt,
      negativePrompt: this.negativePrompt,
      modelPath: this.selectedModel,
      loraPaths: [...this.selectedLoras],
      samplingEnabled: true,
      conditionMode: 'mask',
      conditionWeight: 1,
      maskOperator: 'add',
      denoiseOperator: 'replace',
      conditionInnerBlur: 0,
      conditionOuterBlur: 0,
      conditionNegate: false,
      strength: this.strength,
      cfg: this.layerGenerationCfg,
      steps: this.layerGenerationSteps,
      sampler: this.layerGenerationSampler,
      scheduler: this.layerGenerationScheduler,
      schedule: 'linear',
      scheduleStart: 0,
      scheduleEnd: 1,
      autoTag: false,
      autoTagThreshold: 0.35,
      autoTagRefreshFrames: 5,
      layerCfg: null,
      layerSteps: null,
      layerSampler: null,
      layerScheduler: null,
      rgbaPromptInherited: true,
      rgbaNegativePromptInherited: true,
      rgbaDenoiseInherited: true,
      rgbaCfgInherited: true,
      rgbaWeightInherited: true,
      rgbaBlendingInherited: true,
      rgbaMaskBlendingEnabled: false,
      rgbaMaskBlendingRadius: 32,
      rgbaMaskBlendingStrength: 1,
      cfgMaskBlendingEnabled: false,
      cfgMaskBlendingRadius: 32,
      cfgMaskBlendingStrength: 1,
      denoiseMaskBlendingEnabled: false,
      denoiseMaskBlendingRadius: 32,
      denoiseMaskBlendingStrength: 1,
    }
  }

  private normalizeLayerPreset(preset: Partial<LayerPreset> = {}): LayerPreset {
    return {
      prompt: preset.prompt ?? this.prompt,
      negativePrompt: preset.negativePrompt ?? this.negativePrompt,
      modelPath: preset.modelPath ?? this.selectedModel,
      loraPaths: [...(preset.loraPaths ?? this.selectedLoras)],
      samplingEnabled: preset.samplingEnabled ?? true,
      conditionMode: preset.conditionMode ?? 'mask',
      conditionWeight: this.clamp(preset.conditionWeight ?? 1, 0, 4, 1),
      maskOperator: this.normalizeAggregateOperator(preset.maskOperator, 'add'),
      denoiseOperator: this.normalizeAggregateOperator(preset.denoiseOperator, 'replace'),
      conditionInnerBlur: Math.round(this.clamp(preset.conditionInnerBlur ?? 0, 0, 160, 0)),
      conditionOuterBlur: Math.round(this.clamp(preset.conditionOuterBlur ?? 0, 0, 160, 0)),
      conditionNegate: preset.conditionNegate ?? false,
      strength: this.clamp(preset.strength ?? this.strength, 0, 0.999, this.strength),
      cfg: this.clamp(preset.cfg ?? this.layerGenerationCfg, 0, 30, this.layerGenerationCfg),
      steps: Math.round(this.clamp(preset.steps ?? this.layerGenerationSteps, 1, 40, this.layerGenerationSteps)),
      sampler: preset.sampler ?? this.layerGenerationSampler,
      scheduler: preset.scheduler ?? this.layerGenerationScheduler,
      schedule: ['linear', 'ease_in', 'ease_out', 'ease_in_out'].includes(preset.schedule ?? '') ? (preset.schedule as RegionSchedule) : 'linear',
      scheduleStart: this.clamp(preset.scheduleStart ?? 0, 0, 1, 0),
      scheduleEnd: this.clamp(preset.scheduleEnd ?? 1, 0, 1, 1),
      autoTag: preset.autoTag ?? false,
      autoTagThreshold: this.clamp(preset.autoTagThreshold ?? 0.35, 0, 1, 0.35),
      autoTagRefreshFrames: Math.round(this.clamp(preset.autoTagRefreshFrames ?? 5, 1, 60, 5)),
      layerCfg: preset.layerCfg !== undefined ? preset.layerCfg : null,
      layerSteps: preset.layerSteps !== undefined ? preset.layerSteps : null,
      layerSampler: preset.layerSampler !== undefined ? preset.layerSampler : null,
      layerScheduler: preset.layerScheduler !== undefined ? preset.layerScheduler : null,
      rgbaPromptInherited: preset.rgbaPromptInherited ?? true,
      rgbaNegativePromptInherited: preset.rgbaNegativePromptInherited ?? true,
      rgbaDenoiseInherited: preset.rgbaDenoiseInherited ?? true,
      rgbaCfgInherited: preset.rgbaCfgInherited ?? true,
      rgbaWeightInherited: preset.rgbaWeightInherited ?? true,
      rgbaBlendingInherited: preset.rgbaBlendingInherited ?? true,
      rgbaMaskBlendingEnabled: preset.rgbaMaskBlendingEnabled ?? false,
      rgbaMaskBlendingRadius: Math.round(this.clamp(preset.rgbaMaskBlendingRadius ?? 32, -160, 160, 32)),
      rgbaMaskBlendingStrength: this.clamp(preset.rgbaMaskBlendingStrength ?? 1, 0, 4, 1),
      cfgMaskBlendingEnabled: preset.cfgMaskBlendingEnabled ?? false,
      cfgMaskBlendingRadius: Math.round(this.clamp(preset.cfgMaskBlendingRadius ?? 32, -160, 160, 32)),
      cfgMaskBlendingStrength: this.clamp(preset.cfgMaskBlendingStrength ?? 1, 0, 4, 1),
      denoiseMaskBlendingEnabled: preset.denoiseMaskBlendingEnabled ?? false,
      denoiseMaskBlendingRadius: Math.round(this.clamp(preset.denoiseMaskBlendingRadius ?? 32, -160, 160, 32)),
      denoiseMaskBlendingStrength: this.clamp(preset.denoiseMaskBlendingStrength ?? 1, 0, 4, 1),
    }
  }

  private attachPreset(object: FabricObject, preset = this.presetForNewLayer()) {
    ;(object as FabricObject & { __preset?: LayerPreset }).__preset = this.normalizeLayerPreset(preset)
    this.styleTransformControls(object)
  }

  private styleTransformControls(object: FabricObject) {
    object.set({
      hasBorders: false,
      hasControls: false,
    } as Partial<FabricObject>)
  }

  private attachStrokeToLayer(object: FabricObject, layerId = this.pendingPaintLayerId || this.selectedLayerId, erase = this.isEraseStroke()) {
    const withMeta = object as FabricObject & { __uid?: string; __paintChild?: boolean; __parentLayerId?: string }
    const parentLayerId = strokeParentLayerId(layerId, this.selectedLayerId, withMeta.__uid)
    const parentLayer = parentLayerId ? this.findLayer(parentLayerId) as (FabricObject & { __paintChild?: boolean; __parentLayerId?: string }) | undefined : undefined
    const validParent = !!parentLayer && !isPaintChildObject(parentLayer)
    if (!parentLayerId || !validParent) {
      // A stroke with no valid parent layer must never remain on canvas,
      // otherwise it becomes a global top-level object and can alter
      // content outside the selected layer.
      this.fabricCanvas?.remove(object)
      this.pendingPaintLayerId = ''
      this.syncLayers()
      return
    }
    withMeta.__paintChild = true
    withMeta.__parentLayerId = parentLayerId
    if (erase) {
      object.clipPath = this.layerEraseClipPath(parentLayerId)
      object.set({ globalCompositeOperation: 'destination-out' } as Partial<FabricObject>)
    }
    object.set({
      name: 'Paint stroke',
      selectable: false,
      evented: false,
      hasControls: false,
      hasBorders: false,
      lockMovementX: true,
      lockMovementY: true,
      lockRotation: true,
      lockScalingX: true,
      lockScalingY: true,
    } as Partial<FabricObject>)
    this.selectedEntity = 'layer'
    this.selectedLayerId = parentLayerId
    const parent = parentLayer
    if (parent) this.fabricCanvas?.setActiveObject(parent)
    this.pendingPaintLayerId = ''
    if (erase) {
      // Keep destination-out strokes layer-local by flattening immediately.
      // Leaving erase paths as live Fabric children punches through lower layers.
      if ((parent as (FabricObject & { __isRgbaMaskLayer?: boolean }) | undefined)?.__isRgbaMaskLayer) {
        this.rasterizeRgbaMaskLayer(parentLayerId)
      } else {
        this.rasterizePaintLayer(parentLayerId)
      }
      return
    }
    this.syncLayers()
  }

  private rasterizePaintLayer(layerId: string) {
    const object = this.findLayer(layerId) as (FabricObject & { __preset?: LayerPreset; __isRgbaMaskLayer?: boolean }) | undefined
    if (!object || !this.fabricCanvas) return
    const canvas = this.renderLayerBlockCanvas(layerId, object)
    const replacement = new FabricImage(canvas, {
      left: 0,
      top: 0,
      originX: 'left',
      originY: 'top',
      name: object.get('name') || 'Layer',
      opacity: object.opacity ?? 1,
      visible: object.visible ?? true,
    }) as FabricImage & { __uid?: string; __preset?: LayerPreset; __isRgbaMaskLayer?: boolean }
    replacement.__uid = layerId
    if (object.__preset) replacement.__preset = this.normalizeLayerPreset(object.__preset)
    if (object.__isRgbaMaskLayer) replacement.__isRgbaMaskLayer = true
    replacement.set({ selectable: true, evented: true })
    this.styleTransformControls(replacement as unknown as FabricObject)
    this.replaceLayerObject(layerId, object, replacement as unknown as FabricObject)
  }

  private layerEraseClipPath(layerId: string) {
    const layer = this.findLayer(layerId)
    const objects = layer ? [layer, ...this.childrenForLayer(layerId)] : []
    let left = Number.POSITIVE_INFINITY
    let top = Number.POSITIVE_INFINITY
    let right = Number.NEGATIVE_INFINITY
    let bottom = Number.NEGATIVE_INFINITY
    for (const object of objects) {
      const bounds = object.getBoundingRect()
      left = Math.min(left, bounds.left)
      top = Math.min(top, bounds.top)
      right = Math.max(right, bounds.left + bounds.width)
      bottom = Math.max(bottom, bounds.top + bounds.height)
    }
    if (!Number.isFinite(left) || !Number.isFinite(top) || !Number.isFinite(right) || !Number.isFinite(bottom) || right <= left || bottom <= top) {
      return new Rect({
        left: 0,
        top: 0,
        width: this.stageWidth,
        height: this.stageHeight,
        absolutePositioned: true,
      })
    }
    return new Rect({
      left: this.clamp(left, 0, this.stageWidth, 0),
      top: this.clamp(top, 0, this.stageHeight, 0),
      width: this.clamp(right, 0, this.stageWidth, this.stageWidth) - this.clamp(left, 0, this.stageWidth, 0),
      height: this.clamp(bottom, 0, this.stageHeight, this.stageHeight) - this.clamp(top, 0, this.stageHeight, 0),
      absolutePositioned: true,
    })
  }

  private resizeStage(width: number, height: number) {
    this.stageWidth = Math.max(64, width || this.stageWidth)
    this.stageHeight = Math.max(64, height || this.stageHeight)
    if (!this._syncingToStore) {
      this._syncingToStore = true
      $scene.setKey('stageWidth', this.stageWidth)
      $scene.setKey('stageHeight', this.stageHeight)
      this._syncingToStore = false
    }
    this.generationWidth = this.alignGenerationSize(this.stageWidth)
    this.generationHeight = this.alignGenerationSize(this.stageHeight)
    this.fabricCanvas?.setDimensions({ width: this.stageWidth, height: this.stageHeight })
    this.maskCanvasElement.width = this.stageWidth
    this.maskCanvasElement.height = this.stageHeight
    this.maskPreviewCanvasElement.width = this.stageWidth
    this.maskPreviewCanvasElement.height = this.stageHeight
    if (this.editorOverlayCanvas) {
      this.editorOverlayCanvas.width = this.stageWidth
      this.editorOverlayCanvas.height = this.stageHeight
    }
    this.ensureInputCanvasCaptureCanvas()
    this.scheduleInputCanvasCaptureRefresh(true)
    this.clearMask()
    this.resizeStoredMasks()
    this.resizeLayerMediaSurfaces()
    this.parameterMaskDataUrlCache.clear()
    this.fabricCanvas?.requestRenderAll()
  }

  private startOverlayStroke(event: PointerEvent) {
    if (event.button !== 0 && event.button !== 2) return
    event.preventDefault()
    event.stopPropagation()
    if (this._isSelectionTool()) {
      if (event.button !== 0) return
      const point = this.maskPoint(event)
      if (this.toolMode === 'magicWand') {
        this.pushHistory()
        this.createMagicWandSelection(point.x, point.y)
        return
      }
      if (this.toolMode === 'moveSelection') {
        if (!this.selectionRect || !this.selectionContainsPoint(point)) return
        this.pushHistory()
        this._selectionDragState = { pointerId: event.pointerId, startPoint: point, startSelection: { ...this.selectionRect, points: this.selectionRect.points ? [...this.selectionRect.points] : undefined } }
        this.maskCanvasElement.setPointerCapture(event.pointerId)
        this.drawing = true
        return
      }
      this.pushHistory()
      this.maskCanvasElement.setPointerCapture(event.pointerId)
      this.drawing = true
      this.lastPointer = point
      this.selectionTransformLayerId = ''
      this.selectedEntity = 'scene'
      this.selectionRect = this.toolMode === 'lassoSelect'
        ? this.selectionFromPoints([point], 'rope')
        : { x: point.x, y: point.y, width: 0, height: 0, inverted: false, shape: this.toolMode === 'ellipseSelect' ? 'ellipse' : 'box' }
      return
    }
    if (!this.maskCanvasEditable() || !this.maskContext) return
    this.syncMaskScope()
    this.pushHistory()
    this.maskStrokeMode = this.toolMode === 'eraser' || event.button === 2 ? 'erase' : 'paint'
    this.activeStrokeClipCanvas = this.activeMaskStrokeClipCanvas()
    this.maskCanvasElement.setPointerCapture(event.pointerId)
    this.drawing = true
    this.lastPointer = this.maskPoint(event)
    this.maskStrokePoints = [this.lastPointer]
    this.drawMaskPoint(this.lastPointer)
    this.scheduleActiveDrawSignalRefresh()
    this.scheduleMaskPreviewRefresh()
    this.queueLiveMaskStreamUpdate()
  }

  private moveOverlayStroke(event: PointerEvent) {
    event.stopPropagation()
    this.trackCursor(event)
    if (!this.drawing && (event.buttons & 3) !== 0 && this.maskCanvasEditable() && this.maskContext) {
      try {
        this.maskCanvasElement.setPointerCapture(event.pointerId)
      } catch {
        // Pointer capture is best-effort; drawing still works without it.
      }
      this.pushHistory()
      this.maskStrokeMode = this.toolMode === 'eraser' || (event.buttons & 2) !== 0 ? 'erase' : 'paint'
      this.activeStrokeClipCanvas = this.activeMaskStrokeClipCanvas()
      this.drawing = true
      this.lastPointer = this.maskPoint(event)
      this.maskStrokePoints = [this.lastPointer]
      this.drawMaskPoint(this.lastPointer)
      this.scheduleActiveDrawSignalRefresh()
      this.scheduleMaskPreviewRefresh()
      this.queueLiveMaskStreamUpdate()
      return
    }
    if (this._selectionDragState && this.drawing) {
      const point = this.maskPoint(event)
      this.moveSelectionBy(point.x - this._selectionDragState.startPoint.x, point.y - this._selectionDragState.startPoint.y)
      return
    }
    if (this._isSelectionTool() && this.drawing && this.lastPointer) {
      const point = this.maskPoint(event)
      if (this.toolMode === 'lassoSelect') {
        const points = [...(this.selectionRect?.points ?? [])]
        const lastPoint = points.at(-1)
        if (!lastPoint || Math.hypot(point.x - lastPoint.x, point.y - lastPoint.y) > 4) {
          points.push(point)
          this.selectionRect = this.selectionFromPoints(points, 'rope')
        }
        return
      }
      this.selectionRect = {
        x: this.lastPointer.x,
        y: this.lastPointer.y,
        width: point.x - this.lastPointer.x,
        height: point.y - this.lastPointer.y,
        inverted: this.selectionRect?.inverted ?? false,
        shape: this.toolMode === 'ellipseSelect' ? 'ellipse' : 'box',
      }
      return
    }
    if (!this.drawing || !this.maskContext || !this.lastPointer) return
    const point = this.maskPoint(event)
    if (this.eraseActiveColorMaskAt((context) => {
      context.beginPath()
      context.moveTo(this.lastPointer!.x, this.lastPointer!.y)
      context.lineTo(point.x, point.y)
      context.stroke()
    }, this.maskStrokeDirtyRect(this.lastPointer, point))) {
      this.lastPointer = point
      const previousPoint = this.maskStrokePoints.at(-1)
      if (!previousPoint || Math.hypot(point.x - previousPoint.x, point.y - previousPoint.y) > 2) this.maskStrokePoints = [...this.maskStrokePoints, point]
      this.scheduleMaskPreviewRefresh()
      this.queueLiveMaskStreamUpdate()
      return
    }
    const context = this.maskContext
    context.save()
    context.globalCompositeOperation = this.maskStrokeMode === 'erase' ? 'destination-out' : 'source-over'
    context.strokeStyle = this.maskStrokeStyle()
    context.lineCap = 'round'
    context.lineJoin = 'round'
    context.lineWidth = this.brushSize
    context.beginPath()
    context.moveTo(this.lastPointer.x, this.lastPointer.y)
    context.lineTo(point.x, point.y)
    context.stroke()
    context.restore()
    this.clipActiveMaskStroke()
    this.lastPointer = point
    const previousPoint = this.maskStrokePoints.at(-1)
    if (!previousPoint || Math.hypot(point.x - previousPoint.x, point.y - previousPoint.y) > 2) this.maskStrokePoints = [...this.maskStrokePoints, point]
    this.scheduleActiveDrawSignalRefresh()
    this.scheduleMaskPreviewRefresh()
    this.queueLiveMaskStreamUpdate()
  }

  private endOverlayStroke(event: PointerEvent) {
    event.stopPropagation()
    if (this.drawing && this.maskCanvasElement.hasPointerCapture(event.pointerId)) this.maskCanvasElement.releasePointerCapture(event.pointerId)
    if (this._isSelectionTool()) {
      this.drawing = false
      this._selectionDragState = null
      this.lastPointer = undefined
      this.scheduleProjectPersist()
      return
    }
    if (this.drawing && !this._isSelectionTool()) this.createRegionFromMaskStroke()
    this.drawing = false
    this._selectionDragState = null
    this.lastPointer = undefined
    this.maskStrokePoints = []
    this.activeStrokeClipCanvas = undefined
    this.saveCurrentMaskScope()
    this.scheduleMaskPreviewRefresh()
    this.scheduleProjectPersist()
    this.queueLiveMaskStreamUpdate(true, true)
  }

  private normalizedSelection() {
    const selection = this.selectionRect ?? { x: 0, y: 0, width: this.stageWidth, height: this.stageHeight, inverted: false }
    const x = Math.max(0, Math.min(this.stageWidth, selection.width < 0 ? selection.x + selection.width : selection.x))
    const y = Math.max(0, Math.min(this.stageHeight, selection.height < 0 ? selection.y + selection.height : selection.y))
    const right = Math.max(0, Math.min(this.stageWidth, selection.width < 0 ? selection.x : selection.x + selection.width))
    const bottom = Math.max(0, Math.min(this.stageHeight, selection.height < 0 ? selection.y : selection.y + selection.height))
    return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y), inverted: selection.inverted }
  }

  public setSelectionTransform(patch: Partial<SelectionRect> & { centerX?: number; centerY?: number }) {
    if (!this.selectionRect) return
    this.pushHistory()
    const previous = this.normalizedSelection()
    const requested = centeredResizeRect(previous, patch)
    const width = Math.min(this.stageWidth, Math.max(1, requested.width))
    const height = Math.min(this.stageHeight, Math.max(1, requested.height))
    const x = this.clamp(requested.x, 0, this.stageWidth - width, previous.x)
    const y = this.clamp(requested.y, 0, this.stageHeight - height, previous.y)
    const next: SelectionRect = { x, y, width, height, inverted: this.selectionRect.inverted, shape: this.selectionRect.shape }
    const points = this.selectionRect.points
    if (points?.length) {
      const scaleX = width / Math.max(1, previous.width)
      const scaleY = height / Math.max(1, previous.height)
      next.points = points.map((point) => ({
        x: x + (point.x - previous.x) * scaleX,
        y: y + (point.y - previous.y) * scaleY,
      }))
    }
    this.selectionRect = next
  }

  private selectionFromPoints(points: RegionPoint[], shape: RegionShape): SelectionRect {
    const xs = points.map((point) => point.x)
    const ys = points.map((point) => point.y)
    const x = Math.max(0, Math.min(...xs))
    const y = Math.max(0, Math.min(...ys))
    const right = Math.min(this.stageWidth, Math.max(...xs))
    const bottom = Math.min(this.stageHeight, Math.max(...ys))
    return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y), inverted: false, shape, points }
  }

  private selectionContainsPoint(point: RegionPoint) {
    if (!this.selectionRect) return false
    const selection = this.normalizedSelection()
    if (this.selectionRect.shape === 'ellipse') {
      const rx = selection.width / 2
      const ry = selection.height / 2
      const cx = selection.x + rx
      const cy = selection.y + ry
      return ((point.x - cx) ** 2) / Math.max(1, rx ** 2) + ((point.y - cy) ** 2) / Math.max(1, ry ** 2) <= 1
    }
    const points = this.selectionRect.points
    if (points?.length) return this.pointInPolygon(point, points)
    return point.x >= selection.x && point.x <= selection.x + selection.width && point.y >= selection.y && point.y <= selection.y + selection.height
  }

  private pointInPolygon(point: RegionPoint, polygon: RegionPoint[]) {
    let inside = false
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const a = polygon[i]
      const b = polygon[j]
      if (((a.y > point.y) !== (b.y > point.y)) && point.x < ((b.x - a.x) * (point.y - a.y)) / Math.max(0.0001, b.y - a.y) + a.x) inside = !inside
    }
    return inside
  }

  private moveSelectionBy(dx: number, dy: number) {
    if (!this._selectionDragState) return
    const start = this._selectionDragState.startSelection
    const width = Math.min(this.stageWidth, Math.max(1, Math.abs(start.width)))
    const height = Math.min(this.stageHeight, Math.max(1, Math.abs(start.height)))
    const x = this.clamp(start.x + dx, 0, this.stageWidth - width, start.x)
    const y = this.clamp(start.y + dy, 0, this.stageHeight - height, start.y)
    const moved: SelectionRect = { ...start, x, y, width, height }
    if (start.points?.length) moved.points = start.points.map((point) => ({ x: point.x + (x - start.x), y: point.y + (y - start.y) }))
    this.selectionRect = moved
  }

  private createMagicWandSelection(clickX: number, clickY: number) {
    if (!this.fabricCanvas) return
    const fc = this.fabricCanvas as Canvas & { lowerCanvasEl: HTMLCanvasElement }
    const layerObject = this.selectedLayerId ? this.findLayer(this.selectedLayerId) : undefined
    const sourceCanvas = layerObject && !isPaintChildObject(layerObject as FabricObject & { __paintChild?: boolean; __parentLayerId?: string })
      ? this.renderLayerBlockCanvas(this.selectedLayerId, layerObject)
      : fc.lowerCanvasEl
    const ctx = sourceCanvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    const cw = sourceCanvas.width, ch = sourceCanvas.height
    const px = Math.max(0, Math.min(cw - 1, Math.round(clickX * (cw / this.stageWidth))))
    const py = Math.max(0, Math.min(ch - 1, Math.round(clickY * (ch / this.stageHeight))))
    const imageData = ctx.getImageData(0, 0, cw, ch)
    const d = imageData.data
    const tidx = (py * cw + px) * 4
    const tr = d[tidx], tg = d[tidx + 1], tb = d[tidx + 2], ta = d[tidx + 3]
    const tol = this.clamp(this.fillTolerance, 0, 255, 32)
    const same = (i: number) => Math.abs(d[i] - tr) <= tol && Math.abs(d[i + 1] - tg) <= tol && Math.abs(d[i + 2] - tb) <= tol && Math.abs(d[i + 3] - ta) <= tol
    const visited = new Uint8Array(cw * ch)
    const stack = [py * cw + px]
    visited[py * cw + px] = 1
    let minX = px, maxX = px, minY = py, maxY = py
    while (stack.length) {
      const pos = stack.pop()!
      const x = pos % cw, y = (pos / cw) | 0
      minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y)
      for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]] as [number, number][]) {
        const nx = x + dx, ny = y + dy
        if (nx < 0 || nx >= cw || ny < 0 || ny >= ch) continue
        const ni = ny * cw + nx
        if (visited[ni] || !same(ni * 4)) continue
        visited[ni] = 1
        stack.push(ni)
      }
    }
    const sx = this.stageWidth / cw
    const sy = this.stageHeight / ch
    this.selectedEntity = layerObject ? 'layer' : 'scene'
    this.selectionTransformLayerId = ''
    this.selectionRect = {
      x: minX * sx,
      y: minY * sy,
      width: Math.max(1, (maxX - minX + 1) * sx),
      height: Math.max(1, (maxY - minY + 1) * sy),
      inverted: false,
      shape: 'box',
    }
  }

  public clearSelection = () => {
    this.pushHistory()
    this.selectionTransformLayerId = ''
    this.selectionRect = undefined
  }

  public invertSelection = () => {
    if (!this.selectionRect) return
    this.pushHistory()
    this.selectionRect = { ...this.selectionRect, inverted: !this.selectionRect.inverted }
  }

  public deleteSelection = async () => {
    if (!this.selectionRect || !this.fabricCanvas) return
    this.pushHistory()
    const selection = this.normalizedSelection()
    const image = await this.canvasImage()
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    context.drawImage(image, 0, 0)
    context.globalCompositeOperation = 'destination-out'
    if (selection.inverted) {
      context.fillRect(0, 0, this.stageWidth, selection.y)
      context.fillRect(0, selection.y + selection.height, this.stageWidth, this.stageHeight - selection.y - selection.height)
      context.fillRect(0, selection.y, selection.x, selection.height)
      context.fillRect(selection.x + selection.width, selection.y, this.stageWidth - selection.x - selection.width, selection.height)
    } else {
      context.fillRect(selection.x, selection.y, selection.width, selection.height)
    }
    await this.replaceCanvasWithDataUrl(canvas.toDataURL('image/png'), 'Selection edit')
  }

  public cropToSelection = async () => {
    if (!this.selectionRect || !this.fabricCanvas) return
    const selection = this.normalizedSelection()
    if (selection.inverted) return
    this.pushHistory()
    const image = await this.canvasImage()
    const canvas = document.createElement('canvas')
    canvas.width = selection.width
    canvas.height = selection.height
    canvas.getContext('2d')!.drawImage(image, selection.x, selection.y, selection.width, selection.height, 0, 0, selection.width, selection.height)
    this.resizeStage(selection.width, selection.height)
    await this.replaceCanvasWithDataUrl(canvas.toDataURL('image/png'), 'Cropped selection')
    this.clearSelection()
  }

  private canvasImage() {
    return new Promise<HTMLImageElement>((resolve) => {
      const image = new Image()
      image.onload = () => resolve(image)
      image.src = this.exportCanvasImage(true)
    })
  }

  private replaceCanvasWithDataUrl(dataUrl: string, name: string) {
    return FabricImage.fromURL(dataUrl).then((image) => {
      this.fabricCanvas?.clear()
      image.set({ left: 0, top: 0, originX: 'left', originY: 'top', name })
      this.attachPreset(image)
      this.fabricCanvas?.add(image)
      this.fabricCanvas?.setActiveObject(image)
      this.fabricCanvas?.requestRenderAll()
      this.syncLayers()
    })
  }

  private drawMaskPoint(point: { x: number; y: number }) {
    if (!this.maskContext) return
    if (this.eraseActiveColorMaskAt((context) => {
      context.beginPath()
      context.arc(point.x, point.y, this.brushSize / 2, 0, Math.PI * 2)
      context.fill()
    }, this.maskStrokeDirtyRect(point, point))) return
    this.maskContext.save()
    this.maskContext.globalCompositeOperation = this.maskStrokeMode === 'erase' ? 'destination-out' : 'source-over'
    this.maskContext.fillStyle = this.maskStrokeStyle()
    this.maskContext.beginPath()
    this.maskContext.arc(point.x, point.y, this.brushSize / 2, 0, Math.PI * 2)
    this.maskContext.fill()
    this.maskContext.restore()
    this.clipActiveMaskStroke()
  }

  private activeMaskStrokeClipCanvas(): HTMLCanvasElement | undefined {
    const scope = this.currentMaskScope()
    if (!scope?.startsWith('layer:')) return undefined
    const layerId = scope.replace('layer:', '')
    const explicit = this.regions.find((region) => region.id === this.activeMaskRegionId && region.layerId === layerId)
    const layer = this.layers.find((item) => item.id === layerId)
    if (this.activeMaskChannel === 'color') {
      return undefined
    }
    if (explicit) return this.layerRegionConditionMaskCanvas(layerId, explicit)
    return layer ? undefined : undefined
  }

  private activeColorEraseRegion() {
    const scope = this.currentMaskScope()
    if (this.activeMaskChannel !== 'color' || this.maskStrokeMode !== 'erase' || !scope?.startsWith('layer:')) return undefined
    const layerId = scope.replace('layer:', '')
    return this.regions.find((region) => region.id === this.activeMaskRegionId && region.layerId === layerId)
  }

  private maskStrokeDirtyRect(first: { x: number; y: number }, second: { x: number; y: number }) {
    const pad = Math.max(2, this.brushSize / 2 + 2)
    const x = Math.floor(this.clamp(Math.min(first.x, second.x) - pad, 0, this.stageWidth, 0))
    const y = Math.floor(this.clamp(Math.min(first.y, second.y) - pad, 0, this.stageHeight, 0))
    const right = Math.ceil(this.clamp(Math.max(first.x, second.x) + pad, 0, this.stageWidth, this.stageWidth))
    const bottom = Math.ceil(this.clamp(Math.max(first.y, second.y) + pad, 0, this.stageHeight, this.stageHeight))
    return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y) }
  }

  private eraseActiveColorMaskAt(drawEraseShape: (context: CanvasRenderingContext2D) => void, bounds: { x: number; y: number; width: number; height: number }) {
    if (!this.maskContext) return false
    const region = this.activeColorEraseRegion()
    const target = region ? this.parseColor(region.color) : undefined
    if (!target) return false
    const erase = document.createElement('canvas')
    erase.width = bounds.width
    erase.height = bounds.height
    const eraseContext = erase.getContext('2d')!
    eraseContext.fillStyle = '#fff'
    eraseContext.strokeStyle = '#fff'
    eraseContext.lineCap = 'round'
    eraseContext.lineJoin = 'round'
    eraseContext.lineWidth = this.brushSize
    eraseContext.translate(-bounds.x, -bounds.y)
    drawEraseShape(eraseContext)
    const imageData = this.maskContext.getImageData(bounds.x, bounds.y, bounds.width, bounds.height)
    const eraseData = eraseContext.getImageData(0, 0, bounds.width, bounds.height).data
    for (let index = 0; index < imageData.data.length; index += 4) {
      const eraseAlpha = eraseData[index + 3]
      if (eraseAlpha <= 0 || imageData.data[index + 3] <= 0) continue
      const matches = Math.abs(imageData.data[index] - target.r) <= 3 && Math.abs(imageData.data[index + 1] - target.g) <= 3 && Math.abs(imageData.data[index + 2] - target.b) <= 3
      if (!matches) continue
      const nextAlpha = Math.round(imageData.data[index + 3] * (1 - eraseAlpha / 255))
      if (nextAlpha <= 0) {
        imageData.data[index] = 0
        imageData.data[index + 1] = 0
        imageData.data[index + 2] = 0
        imageData.data[index + 3] = 0
      } else {
        imageData.data[index + 3] = nextAlpha
      }
    }
    this.maskContext.putImageData(imageData, bounds.x, bounds.y)
    return true
  }

  private clipActiveMaskStroke() {
    if (!this.maskContext) return
    const clip = this.activeStrokeClipCanvas
    if (!clip) return
    this.maskContext.save()
    this.maskContext.globalCompositeOperation = 'destination-in'
    this.maskContext.drawImage(clip, 0, 0, this.stageWidth, this.stageHeight)
    this.maskContext.restore()
  }

  private queueLiveMaskStreamUpdate(force = false, refreshResources = false) {
    refreshResources ||= this.shouldRefreshLiveResourcesOnDraw()
    if (force) {
      if (this.maskStreamUpdateRaf !== undefined) window.cancelAnimationFrame(this.maskStreamUpdateRaf)
      this.maskStreamUpdateRaf = undefined
      this.maskStreamUpdateScheduled = false
      this.flushLiveMaskStreamUpdate(true, refreshResources)
      return
    }
    if (this.maskStreamUpdateScheduled) return
    this.maskStreamUpdateScheduled = true
    this.maskStreamUpdateRaf = window.requestAnimationFrame(() => {
      this.maskStreamUpdateRaf = undefined
      this.maskStreamUpdateScheduled = false
      this.flushLiveMaskStreamUpdate(false)
    })
  }

  private flushLiveMaskStreamUpdate(force = true, refreshResources = false) {
    this.saveCurrentMaskScope()
    if (this.activeMaskScope) this.bumpChannelMaskRevision(this.activeMaskScope, this.activeMaskChannel)
    this.queueLiveInputChanged(force, refreshResources, true)
  }

  private createRegionFromMaskStroke() {
    const scope = this.currentMaskScope()
    if (this.activeMaskChannel !== 'color' || !scope?.startsWith('layer:') || this.maskStrokeMode !== 'paint' || this.maskStrokePoints.length === 0) return
    const color = this.normalizeColor(this.brushColor)
    const parsed = color ? this.parseColor(color) : undefined
    if (!color || !parsed || parsed.a <= 0.001) return
    const layerId = scope.replace('layer:', '')
    const layer = this.layers.find((item) => item.id === layerId)
    const xs = this.maskStrokePoints.map((point) => point.x)
    const ys = this.maskStrokePoints.map((point) => point.y)
    const pad = Math.max(1, this.brushSize / 2)
    const x = this.clamp(Math.min(...xs) - pad, 0, this.stageWidth, 0)
    const y = this.clamp(Math.min(...ys) - pad, 0, this.stageHeight, 0)
    const right = this.clamp(Math.max(...xs) + pad, 0, this.stageWidth, this.stageWidth)
    const bottom = this.clamp(Math.max(...ys) + pad, 0, this.stageHeight, this.stageHeight)
    const rect = { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y), inverted: false, shape: 'box' as RegionShape }
    const selectedRegion = this.regions.find((region) => region.id === this.activeMaskRegionId && region.target === 'layer' && region.layerId === layerId)
    const existing = selectedRegion && this.sameRgb(selectedRegion.color, color)
      ? selectedRegion
      : this.regions.find((region) => region.target === 'layer' && region.layerId === layerId && this.sameRgb(region.color, color))
    if (existing) {
      const existingRect = this.normalizedRegionRect(existing.rect)
      const unionX = Math.min(existingRect.x, rect.x)
      const unionY = Math.min(existingRect.y, rect.y)
      const unionRight = Math.max(existingRect.x + existingRect.width, rect.x + rect.width)
      const unionBottom = Math.max(existingRect.y + existingRect.height, rect.y + rect.height)
      this.updateRegion(existing.id, {
        rect: { x: unionX, y: unionY, width: unionRight - unionX, height: unionBottom - unionY, inverted: false, shape: 'box' },
        maskStrength: Math.max(existing.maskStrength, parsed.a),
      })
      this.activeMaskRegionId = existing.id
      return
    }
    const region: RegionItem = {
      id: crypto.randomUUID(),
      target: 'layer',
      layerId,
      color: this.rgbToHex(parsed),
      name: `${layer?.name ?? 'Layer'} ${this.rgbToHex(parsed)}`,
      shape: 'box',
      rect,
      maskStrength: 1,
      innerBlur: 0,
      outerBlur: 0,
      negate: false,
      inherited: false,
      prompt: layer ? this.resolvedLayerRgbaRegionPatch(layer).prompt ?? '' : '',
      negativePrompt: layer ? this.resolvedLayerRgbaRegionPatch(layer).negativePrompt ?? '' : '',
      denoise: layer ? this.resolvedLayerRgbaRegionPatch(layer).denoise ?? this.strength : this.strength,
      mode: layer?.preset.conditionMode ?? 'mask',
      maskOperator: layer?.preset.maskOperator ?? 'add',
      denoiseOperator: layer?.preset.denoiseOperator ?? 'replace',
      schedule: 'linear',
      start: 0,
      end: 1,
      cfgMaskInherited: true,
      denoiseMaskInherited: true,
      regionCfg: layer ? this.resolvedLayerRgbaRegionPatch(layer).regionCfg ?? null : null,
      regionSteps: null,
      blendingEnabled: layer ? this.resolvedLayerRgbaRegionPatch(layer).blendingEnabled ?? true : true,
      blendingRadius: layer ? this.resolvedLayerRgbaRegionPatch(layer).blendingRadius ?? 32 : 32,
      blendingStrength: layer ? this.resolvedLayerRgbaRegionPatch(layer).blendingStrength ?? 1 : 1,
    }
    this.regions = [region, ...this.regions]
    this.activeMaskRegionId = region.id
    this.syncRegionsToStore()
    $layer.setKey('layerPanelTab', 'mask')
  }

  private syncRegionsToStore() {
    if (this._syncingToStore) return
    this._syncingToStore = true
    $layer.setKey('regions', this.regions)
    this._syncingToStore = false
    this.queueLayerConditionAwareLiveInput()
  }

  private syncSelectionToStore() {
    if (this._syncingToStore) return
    this._syncingToStore = true
    $layer.setKey('selectionRect', this.selectionRect ? { ...this.selectionRect, points: this.selectionRect.points ? [...this.selectionRect.points] : undefined } : undefined)
    $layer.setKey('selectedEntity', this.selectionRect ? 'scene' : this.selectedEntity)
    this._syncingToStore = false
  }

  private maskPoint(event: PointerEvent) {
    const rect = this.stageElement.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * this.stageWidth
    const y = ((event.clientY - rect.top) / rect.height) * this.stageHeight
    return {
      x: this.clamp(x, 0, this.stageWidth, 0),
      y: this.clamp(y, 0, this.stageHeight, 0),
    }
  }

  private showCursor = () => {
    this.cursor.visible = true
    const isBrushMode = this.toolMode === 'brush' || this.toolMode === 'eraser'
    const cursorEl = this.shadowRoot?.querySelector('.cursor') as HTMLElement | null
    if (cursorEl && isBrushMode) cursorEl.classList.add('visible')
  }

  private hideCursor = () => {
    this.cursor.visible = false
    const cursorEl = this.shadowRoot?.querySelector('.cursor') as HTMLElement | null
    cursorEl?.classList.remove('visible')
  }

  private trackCursor = (event: PointerEvent) => {
    const rect = this.stageElement.getBoundingClientRect()
    const x = this.clamp(((event.clientX - rect.left) / rect.width) * this.stageWidth, 0, this.stageWidth, this.cursor.x)
    const y = this.clamp(((event.clientY - rect.top) / rect.height) * this.stageHeight, 0, this.stageHeight, this.cursor.y)
    this.cursor.x = x
    this.cursor.y = y
    this.cursor.visible = true
    const stageEl = this.stageElement
    if (stageEl) {
      stageEl.style.setProperty('--cursor-x', `${(x / this.stageWidth) * 100}%`)
      stageEl.style.setProperty('--cursor-y', `${(y / this.stageHeight) * 100}%`)
      stageEl.style.setProperty('--cursor-w', `${(this.brushSize / this.stageWidth) * 100}%`)
      stageEl.style.setProperty('--cursor-h', `${(this.brushSize / this.stageHeight) * 100}%`)
    }
    const cursorEl = this.shadowRoot?.querySelector('.cursor') as HTMLElement | null
    if (cursorEl && (this.toolMode === 'brush' || this.toolMode === 'eraser')) cursorEl.classList.add('visible')
  }

  private syncMaskScope() {
    const nextScope = this.currentMaskScope()
    if (nextScope === this.activeMaskScope) return
    this.saveCurrentMaskScope()
    this.activeMaskScope = nextScope
    this.loadMaskScope(nextScope)
  }

  private saveCurrentMaskScope() {
    if (!this.activeMaskScope || !this.maskCanvasElement) return
    const canvas = this.maskCanvasForScope(this.activeMaskScope)
    const context = canvas.getContext('2d')!
    context.clearRect(0, 0, this.stageWidth, this.stageHeight)
    context.drawImage(this.maskCanvasElement, 0, 0, this.stageWidth, this.stageHeight)
    if (this.activeMaskChannel !== 'color') this.bumpChannelMaskRevision(this.activeMaskScope, this.activeMaskChannel)
  }

  private loadMaskScope(scope?: MaskScope) {
    if (!this.maskContext) return
    this.maskContext.clearRect(0, 0, this.stageWidth, this.stageHeight)
    if (!scope) return
    this.maskContext.drawImage(this.maskCanvasForScope(scope), 0, 0, this.stageWidth, this.stageHeight)
  }

  private maskCanvasForScope(scope: MaskScope) {
    return this.maskCanvasForScopeChannel(scope, this.activeMaskChannel)
  }

  private maskCanvasForScopeChannel(scope: MaskScope, channel: MaskChannel = 'color') {
    if (channel === 'color') {
      const existing = this.maskCanvases.get(scope)
      if (existing) return existing
      const canvas = document.createElement('canvas')
      canvas.width = this.stageWidth
      canvas.height = this.stageHeight
      this.maskCanvases.set(scope, canvas)
      return canvas
    }
    const key = this.maskChannelKey(scope, channel)
    const existing = this.channelMaskCanvases.get(key)
    if (existing) return existing
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    this.channelMaskCanvases.set(key, canvas)
    return canvas
  }

  private existingMaskCanvasForScopeChannel(scope: MaskScope, channel: MaskChannel = 'color') {
    if (this.activeMaskScope === scope && this.activeMaskChannel === channel) return this.maskCanvasElement
    if (channel === 'color') return this.maskCanvases.get(scope)
    return this.channelMaskCanvases.get(this.maskChannelKey(scope, channel))
  }

  /**
   * Public accessor for the v2 layer aggregator -- returns the painted mask
   * HTMLCanvasElement for a given layer + channel, or null if none has been
   * painted yet. This bypasses the encoded data-URL round-trip the legacy
   * ``exportLayerConditions`` does, which dropped ``prompt_mask`` entirely
   * (the inline export always set it to ``undefined``) and serialised
   * ``cfg_mask`` in a custom RTF1 float32 format unreadable by ``Image``.
   *
   * Returning the live canvas lets the augmentation lift pixels directly
   * via getImageData. Honoured channels: 'denoise' | 'prompt' | 'cfg'.
   */
  public getLayerChannelMaskCanvas(layerId: string, channel: MaskChannel): HTMLCanvasElement | null {
    if (!layerId) return null
    // Flush the live drawing canvas (``maskCanvasElement``) into the
    // persistent ``channelMaskCanvases`` map BEFORE the lookup -- without
    // this, any paint strokes made while the active scope+channel didn't
    // match the requested (layerId, channel) live only in the shared
    // element and our lookup misses them. The user-reported "painted
    // prompt mask has no effect" was exactly this: the user paints, then
    // shifts focus (away from the layer or to another tool), the active
    // scope changes, and the next frame export queries the prompt-mask
    // store before the live canvas has been copied across.
    //
    // ``regionChannelMaskBlob`` (the v1 export helper that actually
    // worked) does the same save first, see line ~6306. Mirroring it
    // here keeps the v2 patch in sync with v1 semantics.
    try {
      this.saveCurrentMaskScope()
    } catch {
      // Defensive: if the save throws (no active scope, no live canvas
      // yet), fall through to the lookup -- worst case we return null.
    }
    return this.existingMaskCanvasForScopeChannel(`layer:${layerId}` as MaskScope, channel) ?? null
  }

  /**
   * Diagnostic: returns a snapshot of where mask data currently lives so the
   * v2 augmentation can show in the dashboard which scope+channel keys are
   * actually populated. Used when ``getLayerChannelMaskCanvas`` returns null
   * for a layer that should have a painted mask -- the snapshot tells us if
   * the mask is stored under a key we don't look up.
   */
  public maskStateSnapshot(): {
    activeScope: string
    activeChannel: string
    channelKeys: string[]
    sceneScopes: string[]
    liveCanvasReady: boolean
  } {
    return {
      activeScope: String(this.activeMaskScope ?? ''),
      activeChannel: String(this.activeMaskChannel ?? ''),
      channelKeys: [...this.channelMaskCanvases.keys()],
      sceneScopes: [...this.maskCanvases.keys()],
      liveCanvasReady: !!this.maskCanvasElement && this.maskCanvasElement.width > 0,
    }
  }

  private maskChannelKey(scope: MaskScope, channel: MaskChannel) {
    return `${scope}::${channel}`
  }

  private bumpChannelMaskRevision(scope: MaskScope, channel: MaskChannel) {
    const key = this.maskChannelKey(scope, channel)
    this.channelMaskRevisions.set(key, (this.channelMaskRevisions.get(key) ?? 0) + 1)
    for (const cacheKey of [...this.parameterMaskDataUrlCache.keys()]) {
      if (cacheKey.startsWith(`${key}::`)) this.parameterMaskDataUrlCache.delete(cacheKey)
    }
    for (const cacheKey of [...this.parameterMaskBlobCache.keys()]) {
      if (cacheKey.startsWith(`${key}::`)) this.parameterMaskBlobCache.delete(cacheKey)
    }
  }

  private bumpLayerConditionRevision(layerId: string) {
    if (!layerId) return
    this.layerConditionRevisions.set(layerId, (this.layerConditionRevisions.get(layerId) ?? 0) + 1)
  }

  private layerConditionRevision(layerId: string) {
    return this.layerConditionRevisions.get(layerId) ?? 0
  }

  private layerConditionImageSignature(layer: LayerItem, region: RegionItem) {
    return JSON.stringify({
      stage: [this.stageWidth, this.stageHeight],
      layer: layer.id,
      layerRevision: this.layerConditionRevision(layer.id),
      stack: this.layerStackIndex(layer.id),
      inherited: region.id.startsWith('inherited:'),
      region: region.id.startsWith('inherited:') ? this.layerRegionCacheShape(region) : region.id,
    })
  }

  private layerRegionCacheShape(region: RegionItem) {
    return {
      id: region.id,
      color: region.color,
      rect: region.rect,
      maskStrength: region.maskStrength,
      innerBlur: region.innerBlur,
      outerBlur: region.outerBlur,
      negate: region.negate,
      inherited: region.inherited,
      blendingEnabled: region.blendingEnabled,
      blendingRadius: region.blendingRadius,
      blendingStrength: region.blendingStrength,
    }
  }

  private maskStrokeStyle() {
    if (this.maskStrokeMode === 'erase') return 'rgba(0, 0, 0, 1)'
    if (this.activeMaskChannel === 'color') return this.brushColor
    const value = this.clamp(this.maskBrushValue, 0, 255, 255)
    return `rgba(${value}, ${value}, ${value}, 1)`
  }

  private maskValueFromPixel(data: Uint8ClampedArray, index: number) {
    if (data[index + 3] <= 0) return 0
    return Math.round((data[index] + data[index + 1] + data[index + 2]) / 3)
  }

  private resizeStoredMasks() {
    for (const [scope, source] of this.maskCanvases) {
      const resized = document.createElement('canvas')
      resized.width = this.stageWidth
      resized.height = this.stageHeight
      resized.getContext('2d')!.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
      this.maskCanvases.set(scope, resized)
    }
    for (const [key, source] of this.channelMaskCanvases) {
      const resized = document.createElement('canvas')
      resized.width = this.stageWidth
      resized.height = this.stageHeight
      resized.getContext('2d')!.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
      this.channelMaskCanvases.set(key, resized)
    }
    this.activeMaskScope = undefined
    this.syncMaskScope()
    this.scheduleMaskPreviewRefresh()
  }

  private clearMaskScope(scope: MaskScope) {
    this.syncMaskScope()
    this.pushHistory()
    const context = this.maskCanvasForScope(scope).getContext('2d')!
    context.clearRect(0, 0, this.stageWidth, this.stageHeight)
    for (const channel of ['color', 'cfg', 'denoise', 'prompt'] as const) this.bumpChannelMaskRevision(scope, channel)
    if (this.activeMaskScope === scope) this.loadMaskScope(scope)
    this.refreshMaskPreview()
    this.scheduleProjectPersist()
  }

  private refreshMaskPreview() {
    if (!this.maskPreviewContext) return
    this.lastMaskPreviewAt = performance.now()
    this.maskPreviewContext.clearRect(0, 0, this.stageWidth, this.stageHeight)
    const scope = this.currentMaskScope()
    if (!scope) return
    this.maskPreviewContext.drawImage(this.composeMaskPreviewCanvas(scope), 0, 0)
  }

  private scheduleMaskPreviewRefresh(force = false) {
    if (force) {
      if (this.maskPreviewTimer) {
        window.clearTimeout(this.maskPreviewTimer)
        this.maskPreviewTimer = undefined
      }
      this.refreshMaskPreview()
      return
    }
    if (this.maskPreviewTimer) return
    const delay = Math.max(0, 100 - (performance.now() - this.lastMaskPreviewAt))
    this.maskPreviewTimer = window.setTimeout(() => {
      this.maskPreviewTimer = undefined
      this.refreshMaskPreview()
    }, delay)
  }

  private composeMaskPreviewCanvas(scope: MaskScope) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    if (this.inputMaskVisible) this.drawPaintedMaskPreview(context, scope, 'color')
    if (scope === 'scene') {
      if (this.inputMaskVisible) this.drawRegionMaskPreview(context, 'scene')
      if (!this.sceneMaskAbsolute) {
        for (const layer of this.layers) {
          if (this.inputMaskVisible) this.drawPaintedMaskPreview(context, `layer:${layer.id}`, 'color')
          if (this.inputMaskVisible) this.drawRegionMaskPreview(context, 'layer', layer.id)
          if (this.inputChannelVisible) this.drawChannelMaskPreview(context, `layer:${layer.id}`)
        }
      }
    } else {
      const layerId = scope.replace('layer:', '')
      if (this.inputMaskVisible) this.drawRegionMaskPreview(context, 'layer', layerId)
      if (this.inputChannelVisible) this.drawChannelMaskPreview(context, scope)
    }
    return canvas
  }

  private drawPaintedMaskPreview(context: CanvasRenderingContext2D, scope: MaskScope, channel: MaskChannel = 'color') {
    const painted = document.createElement('canvas')
    painted.width = this.stageWidth
    painted.height = this.stageHeight
    const source = this.activeMaskScope === scope && this.activeMaskChannel === channel ? this.maskCanvasElement : this.maskCanvasForScopeChannel(scope, channel)
    painted.getContext('2d')!.drawImage(source, 0, 0)
    if (this.filterRegularMasksToActiveColor()) this.keepOnlyActiveMaskColor(painted)
    context.drawImage(painted, 0, 0)
  }

  private filterRegularMasksToActiveColor() {
    return this.inputActiveMaskOnly && this.activeMaskChannel === 'color'
  }

  private drawChannelMaskPreview(context: CanvasRenderingContext2D, scope: MaskScope) {
    const channels: MaskChannel[] = this.inputActiveMaskOnly && this.activeMaskChannel !== 'color'
      ? [this.activeMaskChannel]
      : ['cfg', 'denoise', 'prompt']
    for (const channel of channels) {
      const source = this.activeMaskScope === scope && this.activeMaskChannel === channel ? this.maskCanvasElement : this.channelMaskCanvases.get(this.maskChannelKey(scope, channel))
      if (!source) continue
      const tinted = document.createElement('canvas')
      tinted.width = this.stageWidth
      tinted.height = this.stageHeight
      const tctx = tinted.getContext('2d')!
      tctx.drawImage(source, 0, 0)
      const tint = this.channelPreviewColor(channel)
      tctx.save()
      tctx.globalCompositeOperation = 'source-in'
      tctx.fillStyle = tint
      tctx.fillRect(0, 0, this.stageWidth, this.stageHeight)
      tctx.restore()
      context.drawImage(tinted, 0, 0)
    }
  }

  private channelPreviewColor(channel: MaskChannel) {
    if (channel === 'cfg') return 'rgba(122, 173, 255, 0.56)'
    if (channel === 'denoise') return 'rgba(255, 180, 95, 0.56)'
    if (channel === 'prompt') return 'rgba(178, 128, 255, 0.52)'
    return 'rgba(255, 255, 255, 0.5)'
  }

  private keepOnlyActiveMaskColor(canvas: HTMLCanvasElement) {
    const active = this.parseColor(this.brushColor)
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const matches = active && alpha > 0 && Math.abs(imageData.data[index] - active.r) <= 3 && Math.abs(imageData.data[index + 1] - active.g) <= 3 && Math.abs(imageData.data[index + 2] - active.b) <= 3
      if (!matches) imageData.data[index + 3] = 0
    }
    context.putImageData(imageData, 0, 0)
  }

  private drawRegionMaskPreview(context: CanvasRenderingContext2D, target: RegionTarget, layerId?: string) {
    for (const region of this.regions.filter((item) => item.target === target && (target === 'scene' || item.layerId === layerId))) {
      if (this.filterRegularMasksToActiveColor() && !this.sameRgb(region.color, this.brushColor)) continue
      if (target === 'layer' && layerId && this.hasLayerRegionMaskPixels(layerId, region)) continue
      context.save()
      context.globalCompositeOperation = 'source-over'
      context.filter = region.outerBlur ? `blur(${region.outerBlur}px)` : 'none'
      context.fillStyle = this.hexToRgba(region.color, region.maskStrength * 0.58)
      this.fillRegionShape(context, region, this.normalizedRegionRect(region.rect))
      if (region.innerBlur) {
        context.filter = `blur(${region.innerBlur}px)`
        context.fillStyle = this.hexToRgba(region.color, region.maskStrength * 0.38)
        this.fillRegionShape(context, region, this.normalizedRegionRect(region.rect))
      }
      context.restore()
    }
  }

  private hexToRgba(hex: string, alpha: number) {
    const color = this.hexToRgb(hex)
    return `rgba(${color.r},${color.g},${color.b},${this.clamp(alpha, 0, 1, 0.5)})`
  }

  private hexToRgb(hex: string) {
    const value = hex.replace('#', '')
    return {
      r: parseInt(value.slice(0, 2), 16) || 255,
      g: parseInt(value.slice(2, 4), 16) || 78,
      b: parseInt(value.slice(4, 6), 16) || 99,
    }
  }

  private _maskCanvasHasContent(scope: MaskScope): boolean {
    const canvas = this.activeMaskScope === scope && this.activeMaskChannel === 'color' ? this.maskCanvasElement : this.maskCanvases.get(scope)
    if (!canvas) return false
    const data = canvas.getContext('2d')?.getImageData(0, 0, canvas.width, canvas.height).data
    if (!data) return false
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true
    return false
  }

  private composeSceneMaskCanvas() {
    this.saveCurrentMaskScope()

    // Auto-selective: if the user explicitly defined where to generate (painted mask or named
    // region), switch to black background so only those areas are denoised.
    // With no explicit masks, keep white background = full img2img stylization of the whole canvas.
    // NOTE: visible layers alone do NOT trigger black mode â€” they send their conditioning via
    // layer_conditions to the backend, which applies them on top of the full-canvas denoising.
    // This ensures the background keeps diffusing even when a new layer with a painted dot is added.
    const hasSceneRegions = this.regions.some((r) => r.target === 'scene')
    const hasLayerRegions = !this.sceneMaskAbsolute && this.regions.some((r) => r.target === 'layer')
    const hasScenePaint = this._maskCanvasHasContent('scene')
    const hasLayerPaint = !this.sceneMaskAbsolute && this.layers.some((l) => this._maskCanvasHasContent(`layer:${l.id}`))
    const autoSelective = hasSceneRegions || hasLayerRegions || hasScenePaint || hasLayerPaint
    const useBlack = this.sceneMaskNegated || autoSelective

    const canvas = this.blankMaskCanvas(useBlack ? '#000' : '#fff')
    const context = canvas.getContext('2d')!
    this.drawPaintedMask(context, 'scene')
    this.drawRegionMasks(context, 'scene')
    if (!this.sceneMaskAbsolute) {
      for (const layer of this.layers) {
        this.drawPaintedMask(context, `layer:${layer.id}`, this.layerMaskNegated.get(layer.id) ?? false)
        this.drawRegionMasks(context, 'layer', layer.id)
      }
    }
    return canvas
  }

  private blankMaskCanvas(fillStyle: string) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    context.fillStyle = fillStyle
    context.fillRect(0, 0, this.stageWidth, this.stageHeight)
    return canvas
  }

  private clearMask = () => {
    this.maskContext?.clearRect(0, 0, this.stageWidth, this.stageHeight)
  }

  public clearCurrentMask = () => {
    this.clearMaskScope(this.currentMaskScope() ?? 'scene')
  }

  public loadImage = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/*'
    input.multiple = true
    input.onchange = () => Array.from(input.files ?? []).forEach((file) => this.addImageFile(file))
    input.click()
  }

  public fileToDataUrl(file: File) {
    return this.blobToDataUrl(file, file.name)
  }

  private blobToDataUrl(blob: Blob, name: string) {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result ?? ''))
      reader.onerror = () => reject(reader.error ?? new Error(`Could not read ${name}`))
      reader.readAsDataURL(blob)
    })
  }

  private handlePaste = (event: ClipboardEvent) => {
    const items = Array.from(event.clipboardData?.items ?? [])
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (file) {
          event.preventDefault()
          void this.addImageFile(file)
        }
      }
    }
  }

  private fitImage(image: FabricImage) {
    const scale = Math.min(this.stageWidth / image.width!, this.stageHeight / image.height!, 1)
    image.scale(scale)
    image.set({ left: this.stageWidth / 2, top: this.stageHeight / 2, originX: 'center', originY: 'center' })
  }

  public addBlankLayer = () => {
    this.pushHistory()
    const rect = new Rect({
      left: 0,
      top: 0,
      width: this.stageWidth,
      height: this.stageHeight,
      fill: 'rgba(255,255,255,0)',
      name: `Layer ${this.layers.length + 1}`,
    })
    this.attachPreset(rect)
    this.fabricCanvas?.add(rect)
    this.fabricCanvas?.setActiveObject(rect)
    this.syncLayers()
  }

  private addTextLayer = () => {
    this.pushHistory()
    const text = new Textbox('Text', {
      left: this.stageWidth * 0.2,
      top: this.stageHeight * 0.2,
      width: 260,
      fill: this.brushColor,
      fontSize: this.textFontSize,
      name: `Text ${this.layers.length + 1}`,
    })
    this.attachPreset(text)
    this.fabricCanvas?.add(text)
    this.fabricCanvas?.setActiveObject(text)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private syncLayers() {
    if (this.isSyncingLayers) return
    this.isSyncingLayers = true
    try {
    const active = this.fabricCanvas?.getActiveObject()
    if (active && !isPaintChildObject(active as FabricObject & { __paintChild?: boolean; __parentLayerId?: string })) {
      this.selectedEntity = 'layer'
      this.selectedLayerId = String((active as FabricObject & { __uid?: string }).__uid ?? this.assignLayerId(active))
    }
    const layerObjects = topLevelLayerObjects([...(this.fabricCanvas?.getObjects() ?? [])] as (FabricObject & { __paintChild?: boolean; __parentLayerId?: string })[])
    if (this.selectedEntity === 'layer' && this.selectedLayerId && !layerObjects.some((object) => (object as FabricObject & { __uid?: string }).__uid === this.selectedLayerId)) {
      this.selectedLayerId = ''
    }
    const layerIds = new Set(layerObjects.map((object) => String((object as FabricObject & { __uid?: string }).__uid ?? this.assignLayerId(object))))
    for (const id of this.layerPreviewCache.keys()) {
      if (!layerIds.has(id)) this.layerPreviewCache.delete(id)
    }
    this.layers = topLevelLayerObjects([...(this.fabricCanvas?.getObjects() ?? [])] as (FabricObject & { __paintChild?: boolean; __parentLayerId?: string })[])
      .reverse()
      .map((object, index) => {
        this.styleTransformControls(object)
        const id = String((object as FabricObject & { __uid?: string }).__uid ?? this.assignLayerId(object))
        return {
          id,
          name: object.get('name') || `Layer ${index + 1}`,
          preview: this.layerPreview(object, id),
          visible: object.visible ?? true,
          opacity: object.opacity ?? 1,
          selected: id === this.selectedLayerId,
          preset: this.normalizeLayerPreset((object as FabricObject & { __preset?: LayerPreset }).__preset ?? this.currentPreset()),
          transform: this.layerTransform(object),
          isVideo: !!(object as FabricObject & { __isVideoLayer?: boolean }).__isVideoLayer,
        }
      })
    this.applyToolMode()
    this.syncMaskScope()
    this.scheduleMaskPreviewRefresh()
    this._syncingToStore = true
    $layer.setKey('layers', this.layers)
    $layer.setKey('selectedLayerId', this.selectedLayerId)
    this._syncingToStore = false
    this.syncLayerMediaSurfaces()
    this.scheduleLayerMediaSurfaceRefresh()
    this.queueLayerConditionAwareLiveInput()
    } finally {
      this.isSyncingLayers = false
    }
  }

  private layerIdForObject(object?: FabricObject) {
    if (!object) return ''
    const meta = object as FabricObject & { __uid?: string; __parentLayerId?: string; __paintChild?: boolean }
    return meta.__paintChild ? (meta.__parentLayerId || '') : (meta.__uid || '')
  }

  private syncLayerMediaSurfaces() {
    const activeIds = new Set(this.layers.map((layer) => layer.id))
    for (const id of activeIds) this.ensureLayerMediaSurface(id)
    for (const [id, surface] of this.layerMediaSurfaces.entries()) {
      if (activeIds.has(id)) continue
      surface.track?.stop()
      unregisterDebugSurface(`layer:${id}:color`)
      this.layerMediaSurfaces.delete(id)
      this.layerMediaDirtyIds.delete(id)
    }
  }

  private ensureLayerMediaSurface(layerId: string) {
    let surface = this.layerMediaSurfaces.get(layerId)
    if (!surface) {
      const canvas = document.createElement('canvas')
      canvas.width = this.stageWidth
      canvas.height = this.stageHeight
      const context = canvas.getContext('2d', { alpha: true })
      if (!context) return undefined
      surface = { canvas, context, registered: false, revision: 0 }
      this.layerMediaSurfaces.set(layerId, surface)
    }
    const layer = this.layers.find((item) => item.id === layerId)
    registerDebugCanvasSurface(`layer:${layerId}:color`, `${layer?.name || layerId}/color`, surface.canvas)
    if (surface.canvas.width !== this.stageWidth || surface.canvas.height !== this.stageHeight) {
      surface.canvas.width = this.stageWidth
      surface.canvas.height = this.stageHeight
    }
    return surface
  }

  private resizeLayerMediaSurfaces() {
    for (const surface of this.layerMediaSurfaces.values()) {
      surface.canvas.width = this.stageWidth
      surface.canvas.height = this.stageHeight
    }
    this.scheduleLayerMediaSurfaceRefresh()
  }

  private disposeLayerMediaSurfaces() {
    for (const [id, surface] of this.layerMediaSurfaces.entries()) {
      surface.track?.stop()
      unregisterDebugSurface(`layer:${id}:color`)
    }
    this.layerMediaSurfaces.clear()
    this.layerMediaDirtyIds.clear()
  }

  private stopLayerMediaTracks() {
    for (const [id, surface] of this.layerMediaSurfaces.entries()) {
      surface.track?.stop()
      surface.stream = undefined
      surface.track = undefined
      surface.registered = false
      unregisterDebugSurface(`layer:${id}:color`)
    }
  }

  private scheduleLayerMediaSurfaceRefresh(layerId = '') {
    if (layerId) {
      this.layerMediaDirtyIds.add(layerId)
      this.bumpLayerConditionRevision(layerId)
    } else {
      for (const layer of this.layers) {
        this.layerMediaDirtyIds.add(layer.id)
        this.bumpLayerConditionRevision(layer.id)
      }
    }
    if (this.layerMediaRefreshRaf !== undefined) return
    this.layerMediaRefreshRaf = window.requestAnimationFrame(() => {
      this.layerMediaRefreshRaf = undefined
      this.refreshLayerMediaSurfaces()
    })
  }

  private refreshLayerMediaSurfaces() {
    if (!this.fabricCanvas || this.layerMediaDirtyIds.size === 0) return
    const objects = this.fabricCanvas.getObjects()
    const upperCanvas = (this.fabricCanvas as (Canvas & { upperCanvasEl?: HTMLCanvasElement }) | undefined)?.upperCanvasEl
    const dirtyIds = [...this.layerMediaDirtyIds]
    this.layerMediaDirtyIds.clear()
    for (const layerId of dirtyIds) {
      const surface = this.ensureLayerMediaSurface(layerId)
      if (!surface) continue
      surface.context.clearRect(0, 0, this.stageWidth, this.stageHeight)
      const object = objects.find((candidate) => (candidate as FabricObject & { __uid?: string }).__uid === layerId)
      if (object && object.visible !== false) this.drawLayerCondition(surface.context, object, layerId)
      if (upperCanvas && this.pendingPaintLayerId === layerId) {
        surface.context.save()
        surface.context.globalCompositeOperation = this.isEraseStroke(this.pendingBrushButton) ? 'destination-out' : 'source-over'
        surface.context.drawImage(upperCanvas, 0, 0, this.stageWidth, this.stageHeight)
        surface.context.restore()
      }
      surface.revision++
      ;(surface.track as (MediaStreamTrack & { requestFrame?: () => void }) | undefined)?.requestFrame?.()
    }
  }

  private invalidateLayerPreview(layerId = '') {
    if (!layerId) return
    this.layerPreviewCache.delete(layerId)
    this.bumpLayerConditionRevision(layerId)
  }

  private layerPreview(object: FabricObject, layerId: string) {
    const cached = this.layerPreviewCache.get(layerId)
    if (cached) return cached
    try {
      const preview = (object as FabricObject & { toDataURL?: (options: { format: string; multiplier: number }) => string }).toDataURL?.({
        format: 'png',
        multiplier: 0.12,
      }) ?? this.transparentPreview()
      this.layerPreviewCache.set(layerId, preview)
      return preview
    } catch {
      const preview = this.transparentPreview()
      this.layerPreviewCache.set(layerId, preview)
      return preview
    }
  }

  private transparentPreview() {
    const canvas = document.createElement('canvas')
    canvas.width = 64
    canvas.height = 64
    return canvas.toDataURL('image/png')
  }

  private assignLayerId(object: FabricObject) {
    const withId = object as FabricObject & { __uid?: string }
    withId.__uid = withId.__uid || crypto.randomUUID()
    return withId.__uid
  }

  private findLayer(id: string) {
    return (this.fabricCanvas?.getObjects() ?? []).find(
      (object) => (object as FabricObject & { __uid?: string }).__uid === id,
    )
  }

  private selectLayer(id: string) {
    const object = this.findLayer(id)
    if (!object) return
    this.selectedEntity = 'layer'
    this.selectedLayerId = id
    this.fabricCanvas?.setActiveObject(object)
    this.applyToolMode()
    this.syncMaskScope()
    this.refreshMaskPreview()
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  public selectSceneEntity = () => {
    this.selectedEntity = 'scene'
    this.selectedLayerId = ''
    this.fabricCanvas?.discardActiveObject()
    this.applyToolMode()
    this.syncMaskScope()
    this.refreshMaskPreview()
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  public toggleLayer(id: string) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    object.set('visible', !object.visible)
    for (const child of this.childrenForLayer(id)) child.set('visible', object.visible)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  public setLayerOpacity(id: string, opacity: number) {
    const object = this.findLayer(id)
    if (!object) return
    opacity = this.clamp(opacity, 0, 1, object.opacity ?? 1)
    this.pushHistory()
    object.set('opacity', opacity)
    for (const child of this.childrenForLayer(id)) child.set('opacity', opacity)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private layerTransform(object: FabricObject) {
    const bounds = object.getBoundingRect()
    const center = object.getCenterPoint()
    return {
      centerX: Math.round(center.x),
      centerY: Math.round(center.y),
      width: Math.max(1, Math.round(bounds.width || 1)),
      height: Math.max(1, Math.round(bounds.height || 1)),
      scale: Math.max(1, Math.round((((object.scaleX ?? 1) + (object.scaleY ?? 1)) / 2) * 100)),
      angle: Math.round(object.angle ?? 0),
    }
  }

  public setLayerCenter(id: string, centerX: number, centerY: number) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    object.setPositionByOrigin(new Point(this.clamp(centerX, 0, this.stageWidth, centerX), this.clamp(centerY, 0, this.stageHeight, centerY)), 'center', 'center')
    this.afterLayerTransform(object)
  }

  public setLayerSize(id: string, width: number, height: number) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    const center = object.getCenterPoint()
    object.scaleToWidth(this.clamp(width, 1, this.stageWidth * 4, width))
    object.scaleToHeight(this.clamp(height, 1, this.stageHeight * 4, height))
    object.setPositionByOrigin(center, 'center', 'center')
    this.afterLayerTransform(object)
  }

  public setLayerScale(id: string, scalePercent: number) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    const center = object.getCenterPoint()
    const scale = this.clamp(scalePercent, 1, 400, scalePercent) / 100
    object.set({ scaleX: scale, scaleY: scale })
    object.setPositionByOrigin(center, 'center', 'center')
    this.afterLayerTransform(object)
  }

  public setLayerAngle(id: string, angle: number) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    const center = object.getCenterPoint()
    object.rotate(this.clamp(angle, -180, 180, angle))
    object.setPositionByOrigin(center, 'center', 'center')
    this.afterLayerTransform(object)
  }

  public centerLayer(id: string) {
    this.setLayerCenter(id, this.stageWidth / 2, this.stageHeight / 2)
  }

  public fitLayerToViewport(id: string) {
    const object = this.findLayer(id)
    if (!object) return
    const bounds = object.getBoundingRect()
    const ratio = Math.min(this.stageWidth / Math.max(1, bounds.width), this.stageHeight / Math.max(1, bounds.height), 1)
    this.setLayerSize(id, Math.max(1, Math.round(bounds.width * ratio)), Math.max(1, Math.round(bounds.height * ratio)))
    this.centerLayer(id)
  }

  private afterLayerTransform(object: FabricObject) {
    object.setCoords()
    this.fabricCanvas?.setActiveObject(object)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
    this.scheduleTransformOverlayUpdate()
  }

  public commitActiveLayerTransform() {
    if (!this.fabricCanvas || this.selectedEntity !== 'layer' || !this.selectedLayerId) return
    const object = this.findLayer(this.selectedLayerId)
    if (!object) return
    const layerId = this.selectedLayerId
    const blockObjects = [object, ...this.childrenForLayer(layerId)]
    const source = this.fabricCanvas.getElement()
    if (!source) return

    this.pushHistory()
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    const allObjects = this.fabricCanvas.getObjects()
    const blockSet = new Set<FabricObject>(blockObjects)
    const visibility = new Map<FabricObject, boolean | undefined>()
    const opacity = new Map<FabricObject, number | undefined>()
    const active = this.fabricCanvas.getActiveObject()
    const layerName = object.get('name') || 'Layer'
    const layerOpacity = object.opacity ?? 1
    const layerVisible = object.visible ?? true
    const preset = (object as FabricObject & { __preset?: LayerPreset }).__preset
    const isRgbaMaskLayer = !!(object as FabricObject & { __isRgbaMaskLayer?: boolean }).__isRgbaMaskLayer
    try {
      this.isSyncingLayers = true
      for (const candidate of allObjects) {
        visibility.set(candidate, candidate.visible)
        opacity.set(candidate, candidate.opacity)
        candidate.set('visible', blockSet.has(candidate))
        if (blockSet.has(candidate)) candidate.set('opacity', 1)
      }
      this.fabricCanvas.discardActiveObject()
      this.fabricCanvas.renderAll()
      context.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
    } finally {
      for (const candidate of allObjects) {
        candidate.set('visible', visibility.get(candidate))
        candidate.set('opacity', opacity.get(candidate))
      }
      if (active) this.fabricCanvas.setActiveObject(active)
      this.fabricCanvas.renderAll()
      this.isSyncingLayers = false
    }

    const replacement = new FabricImage(canvas, {
      left: 0,
      top: 0,
      originX: 'left',
      originY: 'top',
      name: layerName,
      opacity: layerOpacity,
      visible: layerVisible,
    }) as FabricImage & { __uid?: string; __preset?: LayerPreset; __isRgbaMaskLayer?: boolean }
    replacement.__uid = layerId
    if (preset) replacement.__preset = this.normalizeLayerPreset(preset)
    if (isRgbaMaskLayer) replacement.__isRgbaMaskLayer = true
    replacement.set({ selectable: true, evented: true })
    this.styleTransformControls(replacement as unknown as FabricObject)
    this.replaceLayerObject(layerId, object, replacement as unknown as FabricObject)
    if (this.selectionTransformLayerId && blockObjects.some((item) => (item as FabricObject & { __uid?: string }).__uid === this.selectionTransformLayerId)) this.selectionTransformLayerId = ''
    this.clearTransformOverlay()
    this.scheduleProjectPersist()
    this.queueLiveInputChanged(true, true)
  }

  private childrenForLayer(id: string) {
    return (this.fabricCanvas?.getObjects() ?? []).filter(
      (object) => (object as FabricObject & { __parentLayerId?: string }).__parentLayerId === id,
    )
  }

  private layerBlocks() {
    const objects = (this.fabricCanvas?.getObjects() ?? []) as (FabricObject & { __uid?: string; __parentLayerId?: string; __paintChild?: boolean })[]
    const seen = new Set<FabricObject>()
    return topLevelLayerObjects(objects).map((object) => {
      const id = String(object.__uid ?? this.assignLayerId(object))
      const blockObjects = [object, ...objects.filter((candidate) => candidate.__parentLayerId === id)]
        .filter((candidate) => {
          if (seen.has(candidate)) return false
          seen.add(candidate)
          return true
        })
      return {
        id,
        object,
        objects: blockObjects,
      }
    })
  }

  private removeLayerState(id: string) {
    this.regions = this.regions.filter((region) => region.target !== 'layer' || region.layerId !== id)
    this.maskCanvases.delete(`layer:${id}`)
    for (const key of [...this.channelMaskCanvases.keys()]) if (key.startsWith(`layer:${id}::`)) this.channelMaskCanvases.delete(key)
    for (const key of [...this.channelMaskRevisions.keys()]) if (key.startsWith(`layer:${id}::`)) this.channelMaskRevisions.delete(key)
    for (const key of [...this.parameterMaskDataUrlCache.keys()]) if (key.startsWith(`layer:${id}::`)) this.parameterMaskDataUrlCache.delete(key)
    for (const key of [...this.parameterMaskBlobCache.keys()]) if (key.startsWith(`layer:${id}::`)) this.parameterMaskBlobCache.delete(key)
    for (const key of [...this.layerRegionConditionImageCache.keys()]) if (key.startsWith(`${id}:`)) this.layerRegionConditionImageCache.delete(key)
    for (const key of [...this.layerRegionConditionImageBlobCache.keys()]) if (key.startsWith(`${id}:`)) this.layerRegionConditionImageBlobCache.delete(key)
    for (const key of [...this.layerRegionMaskDataUrlCache.keys()]) if (key.startsWith(`${id}:`)) this.layerRegionMaskDataUrlCache.delete(key)
    for (const key of [...this.layerRegionMaskBlobCache.keys()]) if (key.startsWith(`${id}:`)) this.layerRegionMaskBlobCache.delete(key)
    this.layerConditionRevisions.delete(id)
    if (this.activeMaskScope === `layer:${id}`) this.activeMaskScope = undefined
    this.generatedLayers = this.generatedLayers.filter((variation) => variation.layerId !== id)
    this.generationTasks = this.generationTasks.filter((task) => task.layerId !== id)
  }

  private rebuildLayerBlocks(blocks: { id: string; object: FabricObject; objects: FabricObject[] }[], activeId = this.selectedLayerId) {
    if (!this.fabricCanvas) return
    const uniqueObjects = [...new Set(blocks.flatMap((block) => block.objects))]
    this.isRestoringHistory = true
    for (const object of uniqueObjects) this.fabricCanvas.remove(object)
    for (const block of blocks) for (const object of block.objects) this.fabricCanvas.add(object)
    this.isRestoringHistory = false
    const activeBlock = blocks.find((block) => block.id === activeId) ?? blocks.at(-1)
    if (activeBlock) {
      this.selectedEntity = 'layer'
      this.selectedLayerId = activeBlock.id
      this.fabricCanvas.setActiveObject(activeBlock.object)
    } else {
      this.selectedLayerId = ''
      this.fabricCanvas.discardActiveObject()
    }
    this.fabricCanvas.renderAll()
    this.syncLayers()
  }

  public replaceLayerObject(layerId: string, target: FabricObject, replacement: FabricObject) {
    if (!this.fabricCanvas) return
    const blocks = this.layerBlocks()
    const block = blocks.find((item) => item.id === layerId)
    if (!block) return
    const removedObjects = new Set(this.fabricCanvas.getObjects().filter((object) => {
      const meta = object as FabricObject & { __uid?: string; __parentLayerId?: string }
      return object === target || meta.__parentLayerId === layerId
    }))
    for (const item of blocks) item.objects = item.objects.filter((object) => !removedObjects.has(object))
    ;(replacement as FabricObject & { __uid?: string; __paintChild?: boolean; __parentLayerId?: string }).__uid = layerId
    ;(replacement as FabricObject & { __paintChild?: boolean; __parentLayerId?: string }).__paintChild = false
    ;(replacement as FabricObject & { __parentLayerId?: string }).__parentLayerId = undefined
    replacement.clipPath = undefined
    replacement.set({ globalCompositeOperation: 'source-over' } as Partial<FabricObject>)
    block.object = replacement
    block.objects = [replacement]
    this.isRestoringHistory = true
    for (const object of removedObjects) this.fabricCanvas.remove(object)
    this.isRestoringHistory = false
    this.rebuildLayerBlocks(blocks, layerId)
  }

  public selectedLayer() {
    return this.layers.find((layer) => layer.selected)
  }

  public updateLayerPreset(layerId: string, patch: Partial<LayerPreset>) {
    const object = this.findLayer(layerId) as (FabricObject & { __preset?: LayerPreset }) | undefined
    if (!object) return
    const previousPreset = object.__preset ?? this.currentPreset()
    const normalizedPatch = { ...patch }
    if (normalizedPatch.rgbaPromptInherited === false && !('prompt' in normalizedPatch)) {
      normalizedPatch.prompt = previousPreset.prompt || $scene.get().prompt
    }
    if (normalizedPatch.rgbaNegativePromptInherited === false && !('negativePrompt' in normalizedPatch)) {
      normalizedPatch.negativePrompt = previousPreset.negativePrompt || $scene.get().negativePrompt
    }
    const preset = this.normalizeLayerPreset({ ...previousPreset, ...normalizedPatch })
    object.__preset = preset
    this.layers = this.layers.map((layer) => (layer.id === layerId ? { ...layer, preset } : layer))
    const nextLayer = this.layers.find((layer) => layer.id === layerId)
    if (nextLayer && (
      'prompt' in normalizedPatch || 'rgbaPromptInherited' in normalizedPatch ||
      'negativePrompt' in normalizedPatch || 'rgbaNegativePromptInherited' in normalizedPatch ||
      'strength' in normalizedPatch || 'rgbaDenoiseInherited' in normalizedPatch ||
      'cfg' in normalizedPatch || 'layerCfg' in normalizedPatch || 'rgbaCfgInherited' in normalizedPatch ||
      'conditionWeight' in normalizedPatch || 'rgbaWeightInherited' in normalizedPatch ||
      'rgbaMaskBlendingEnabled' in normalizedPatch || 'rgbaMaskBlendingRadius' in normalizedPatch ||
      'rgbaMaskBlendingStrength' in normalizedPatch || 'rgbaBlendingInherited' in normalizedPatch
    )) {
      const rgbaPatch = this.resolvedLayerRgbaRegionPatch(nextLayer)
      this.regions = this.regions.map((region) => (
        region.target === 'layer' && region.layerId === layerId && !region.id.startsWith('inherited:')
          ? { ...region, ...rgbaPatch, inherited: false }
          : region
      ))
    }
    this._syncingToStore = true
    $layer.setKey('layers', this.layers)
    $layer.setKey('regions', this.regions)
    this._syncingToStore = false
    if ('cfgMaskBlendingEnabled' in normalizedPatch || 'cfgMaskBlendingRadius' in normalizedPatch || 'cfgMaskBlendingStrength' in normalizedPatch) {
      this.bumpChannelMaskRevision(`layer:${layerId}`, 'cfg')
    }
    if ('denoiseMaskBlendingEnabled' in normalizedPatch || 'denoiseMaskBlendingRadius' in normalizedPatch || 'denoiseMaskBlendingStrength' in normalizedPatch) {
      this.bumpChannelMaskRevision(`layer:${layerId}`, 'denoise')
    }
    if ('rgbaMaskBlendingEnabled' in normalizedPatch || 'rgbaMaskBlendingRadius' in normalizedPatch || 'rgbaMaskBlendingStrength' in normalizedPatch) {
      this.bumpChannelMaskRevision(`layer:${layerId}`, 'color')
    }
    this.queueLiveInputChanged(false, false, true)
  }

  public fitLayerChannelMaskToAlpha(layerId: string, channel: Extract<MaskChannel, 'color' | 'cfg' | 'denoise'>, regionId?: string, value = 1) {
    const object = this.findLayer(layerId)
    if (!object) return
    const intensity = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 1))
    this.saveCurrentMaskScope()
    this.pushHistory()
    const scope: MaskScope = `layer:${layerId}`
    const canvas = this.maskCanvasForScopeChannel(scope, channel)
    if (channel !== 'color') {
      canvas.width = this.stageWidth
      canvas.height = this.stageHeight
    }
    const source = document.createElement('canvas')
    source.width = this.stageWidth
    source.height = this.stageHeight
    const sourceContext = source.getContext('2d')!
    this.drawLayerCondition(sourceContext, object, layerId)
    const sourceData = sourceContext.getImageData(0, 0, this.stageWidth, this.stageHeight).data
    const context = canvas.getContext('2d')!
    const region = channel === 'color' && regionId ? this.regions.find((item) => item.id === regionId && item.layerId === layerId) : undefined
    const parsed = region ? this.parseColor(region.color) : undefined
    if (channel === 'color' && (!region || !parsed)) return
    if (region) this.clearMaskDrawForRegionOnCanvas(region, canvas)
    const imageData = channel === 'color'
      ? context.getImageData(0, 0, this.stageWidth, this.stageHeight)
      : context.createImageData(this.stageWidth, this.stageHeight)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = sourceData[index + 3]
      if (channel === 'color' && alpha <= 0) continue
      const maskValue = Math.round(alpha * intensity)
      imageData.data[index] = parsed?.r ?? maskValue
      imageData.data[index + 1] = parsed?.g ?? maskValue
      imageData.data[index + 2] = parsed?.b ?? maskValue
      imageData.data[index + 3] = channel === 'color' ? maskValue : maskValue > 0 ? 255 : 0
    }
    if (channel !== 'color') context.clearRect(0, 0, this.stageWidth, this.stageHeight)
    context.putImageData(imageData, 0, 0)
    this.bumpChannelMaskRevision(scope, channel)
    if (this.activeMaskScope === scope && this.activeMaskChannel === channel) this.loadMaskScope(scope)
    this.scheduleMaskPreviewRefresh(true)
    this.queueLiveInputChanged(true, true)
    this.scheduleProjectPersist()
  }

  public fillMaskViewport(layerId: string, channel: Extract<MaskChannel, 'color' | 'cfg' | 'denoise'>, regionId?: string, value = 1) {
    const scope: MaskScope = `layer:${layerId}`
    const region = channel === 'color' && regionId ? this.regions.find((item) => item.id === regionId && item.layerId === layerId) : undefined
    const parsed = region ? this.parseColor(region.color) : undefined
    if (channel === 'color' && (!region || !parsed)) return
    const intensity = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 1))
    this.saveCurrentMaskScope()
    this.pushHistory()
    const canvas = this.maskCanvasForScopeChannel(scope, channel)
    const context = canvas.getContext('2d')!
    if (channel === 'color' && parsed) {
      context.fillStyle = `rgba(${parsed.r}, ${parsed.g}, ${parsed.b}, ${intensity})`
    } else {
      const maskValue = Math.round(intensity * 255)
      context.fillStyle = `rgb(${maskValue}, ${maskValue}, ${maskValue})`
    }
    context.fillRect(0, 0, this.stageWidth, this.stageHeight)
    this.bumpChannelMaskRevision(scope, channel)
    if (this.activeMaskScope === scope && this.activeMaskChannel === channel) this.loadMaskScope(scope)
    this.scheduleMaskPreviewRefresh(true)
    this.queueLiveInputChanged(true, true)
    this.scheduleProjectPersist()
  }

  public clearLayerMask(layerId: string, channel: Extract<MaskChannel, 'color' | 'cfg' | 'denoise'>, regionId?: string) {
    const scope: MaskScope = `layer:${layerId}`
    const region = channel === 'color' && regionId ? this.regions.find((item) => item.id === regionId && item.layerId === layerId) : undefined
    if (channel === 'color' && !region) return
    this.saveCurrentMaskScope()
    this.pushHistory()
    if (region) {
      this.clearMaskDrawForRegionOnCanvas(region, this.maskCanvasForScopeChannel(scope, 'color'))
      this.bumpChannelMaskRevision(scope, 'color')
    } else {
      const canvas = this.maskCanvasForScopeChannel(scope, channel)
      canvas.getContext('2d')!.clearRect(0, 0, this.stageWidth, this.stageHeight)
      this.bumpChannelMaskRevision(scope, channel)
    }
    if (this.activeMaskScope === scope && this.activeMaskChannel === channel) this.loadMaskScope(scope)
    this.scheduleMaskPreviewRefresh(true)
    this.queueLiveInputChanged(true, true)
    this.scheduleProjectPersist()
  }

  public addRegionFromSelection(target: RegionTarget, layerId?: string) {
    if (!this.selectionRect) return
    const normalized = this.normalizedSelection()
    const layer = layerId ? this.layers.find((item) => item.id === layerId) : undefined
    const region: RegionItem = {
      id: crypto.randomUUID(),
      target,
      layerId: target === 'layer' ? layerId : undefined,
      color: layerId ? this.layerRegionColor(layerId) : this.nextRegionColor(),
      name: this.regionName.trim() || `${layer?.name ?? (target === 'scene' ? 'Scene' : 'Layer')} mask ${this.regions.length + 1}`,
      shape: this.selectionRect.shape ?? 'box',
      rect: { ...normalized, shape: this.selectionRect.shape ?? 'box', points: this.selectionRect.points ? [...this.selectionRect.points] : undefined },
      maskStrength: 1,
      innerBlur: 0,
      outerBlur: 24,
      negate: false,
      inherited: true,
      prompt: layer?.preset.prompt ?? '',
      negativePrompt: layer?.preset.negativePrompt ?? '',
      denoise: layer?.preset.strength ?? this.strength,
      mode: layer?.preset.conditionMode ?? 'mask',
      maskOperator: layer?.preset.maskOperator ?? 'add',
      denoiseOperator: layer?.preset.denoiseOperator ?? 'replace',
      schedule: 'linear',
      start: 0,
      end: 1,
      cfgMaskInherited: true,
      denoiseMaskInherited: true,
      regionCfg: null,
      regionSteps: null,
      blendingEnabled: true,
      blendingRadius: 32,
      blendingStrength: 1,
    }
    this.regions = [region, ...this.regions]
    this.regionName = ''
    this.syncRegionsToStore()
    $layer.setKey('layerPanelTab', 'mask')
  }

  public updateRegion(id: string, patch: Partial<RegionItem>) {
    this.regions = this.regions.map((region) => (region.id === id ? { ...region, ...patch } : region))
    this.syncRegionsToStore()
    this.queueLiveInputChanged(false, false, true)
  }

  public setRegionColor(id: string, color: string) {
    const region = this.regions.find((item) => item.id === id)
    const next = this.normalizeColor(color)
    if (!region || !next) return
    this.recolorMaskDrawForRegion(region, next)
    this.updateRegion(id, { color: next, inherited: false })
    this.loadMaskScope(this.activeMaskScope)
    this.scheduleMaskPreviewRefresh(true)
    this.scheduleProjectPersist()
  }

  public restoreRegionInheritance(id: string) {
    const region = this.regions.find((item) => item.id === id)
    const layer = region?.layerId ? this.layers.find((item) => item.id === region.layerId) : undefined
    this.updateRegion(id, {
      inherited: true,
      prompt: layer?.preset.prompt ?? '',
      negativePrompt: layer?.preset.negativePrompt ?? '',
      denoise: this.strength,
      cfgMaskInherited: true,
      denoiseMaskInherited: true,
      maskOperator: layer?.preset.maskOperator ?? 'add',
      denoiseOperator: layer?.preset.denoiseOperator ?? 'replace',
      schedule: 'linear',
      start: 0,
      end: 1,
    })
  }

  private nextRegionColor() {
    const hue = (this.regions.length * 47 + 205) % 360
    return this.hslToHex(hue, 78, 54)
  }

  public deleteRegion(id: string) {
    const region = this.regions.find((item) => item.id === id)
    if (region) this.clearMaskDrawForRegion(region)
    this.regions = this.regions.filter((item) => item.id !== id)
    this.syncRegionsToStore()
    this.loadMaskScope(this.activeMaskScope)
    this.scheduleMaskPreviewRefresh(true)
    this.scheduleProjectPersist()
  }

  private clearMaskDrawForRegion(region: RegionItem) {
    if (region.target !== 'layer' || !region.layerId) return
    this.saveCurrentMaskScope()
    const target = this.parseColor(region.color)
    if (!target) return
    const scope: MaskScope = `layer:${region.layerId}`
    const canvas = this.maskCanvasForScopeChannel(scope, 'color')
    this.clearMaskDrawForRegionOnCanvas(region, canvas)
  }

  private clearMaskDrawForRegionOnCanvas(region: RegionItem, canvas: HTMLCanvasElement) {
    const target = this.parseColor(region.color)
    if (!target) return
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    let changed = false
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const matches = alpha > 0 && Math.abs(imageData.data[index] - target.r) <= 3 && Math.abs(imageData.data[index + 1] - target.g) <= 3 && Math.abs(imageData.data[index + 2] - target.b) <= 3
      if (matches) {
        imageData.data[index + 3] = 0
        changed = true
      }
    }
    if (changed) context.putImageData(imageData, 0, 0)
    if (changed) this.bumpChannelMaskRevision(`layer:${region.layerId}`, 'color')
  }

  private recolorMaskDrawForRegion(region: RegionItem, color: string) {
    if (region.target !== 'layer' || !region.layerId) return
    this.saveCurrentMaskScope()
    const from = this.parseColor(region.color)
    const to = this.parseColor(color)
    if (!from || !to) return
    const canvas = this.maskCanvasForScopeChannel(`layer:${region.layerId}`, 'color')
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    let changed = false
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const matches = alpha > 0 && Math.abs(imageData.data[index] - from.r) <= 3 && Math.abs(imageData.data[index + 1] - from.g) <= 3 && Math.abs(imageData.data[index + 2] - from.b) <= 3
      if (matches) {
        imageData.data[index] = to.r
        imageData.data[index + 1] = to.g
        imageData.data[index + 2] = to.b
        changed = true
      }
    }
    if (changed) context.putImageData(imageData, 0, 0)
  }

  public clearActiveLayer = () => {
    const object = this.selectedLayerId ? this.findLayer(this.selectedLayerId) : this.fabricCanvas?.getActiveObject()
    if (!object) return
    this.pushHistory()
    const id = (object as FabricObject & { __uid?: string }).__uid
    if (id) for (const child of this.childrenForLayer(id)) this.fabricCanvas?.remove(child)
    if (object.type === 'textbox' || object.type === 'rect') object.set({ opacity: 0 })
    else object.set({ opacity: 0 })
    this.fabricCanvas?.discardActiveObject()
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  public deleteActiveLayer = () => {
    if (!this.fabricCanvas) return
    const blocks = this.layerBlocks()
    const activeId = this.selectedLayerId || ((this.fabricCanvas.getActiveObject() as FabricObject & { __uid?: string } | undefined)?.__uid ?? '')
    const index = blocks.findIndex((block) => block.id === activeId)
    if (index < 0) return
    this.pushHistory()
    const [removed] = blocks.splice(index, 1)
    const removedObjects = new Set(this.fabricCanvas.getObjects().filter((object) => {
      const meta = object as FabricObject & { __uid?: string; __parentLayerId?: string }
      return meta.__uid === removed.id || meta.__parentLayerId === removed.id
    }))
    for (const block of blocks) block.objects = block.objects.filter((object) => !removedObjects.has(object))
    this.removeLayerState(removed.id)
    this.isRestoringHistory = true
    for (const object of removedObjects) this.fabricCanvas.remove(object)
    this.isRestoringHistory = false
    const nextActive = blocks[Math.min(index, blocks.length - 1)]?.id ?? blocks.at(-1)?.id ?? ''
    this.rebuildLayerBlocks(blocks, nextActive)
  }

  public moveLayerUp = () => {
    const blocks = this.layerBlocks()
    const index = blocks.findIndex((block) => block.id === this.selectedLayerId)
    if (index < 0 || index >= blocks.length - 1) return
    this.pushHistory()
    const [block] = blocks.splice(index, 1)
    blocks.splice(index + 1, 0, block)
    this.rebuildLayerBlocks(blocks, block.id)
  }

  public moveLayerDown = () => {
    const blocks = this.layerBlocks()
    const index = blocks.findIndex((block) => block.id === this.selectedLayerId)
    if (index <= 0) return
    this.pushHistory()
    const [block] = blocks.splice(index, 1)
    blocks.splice(index - 1, 0, block)
    this.rebuildLayerBlocks(blocks, block.id)
  }

  public layerBounds(object: FabricObject) {
    const bounds = object.getBoundingRect()
    return {
      x: Math.max(0, Math.round(bounds.left || 0)),
      y: Math.max(0, Math.round(bounds.top || 0)),
      width: Math.max(64, Math.round(bounds.width || this.stageWidth)),
      height: Math.max(64, Math.round(bounds.height || this.stageHeight)),
    }
  }

  private alignGenerationSize(value: number) {
    return Math.max(64, Math.round(value / 64) * 64 || 64)
  }

  public visibleLoras(selectedPaths = [...this.selectedLoras]) {
    const query = this.loraSearch.trim().toLowerCase()
    return this.assets.loras.filter(
      (lora) => selectedPaths.includes(lora.path) || !query || lora.name.toLowerCase().includes(query),
    )
  }

  private allowDrop = (event: DragEvent) => {
    event.preventDefault()
  }

  private dropImages = (event: DragEvent) => {
    event.preventDefault()
    Array.from(event.dataTransfer?.files ?? [])
      .filter((file) => file.type.startsWith('image/'))
      .forEach((file) => this.addImageFile(file))
  }

  private async zoomViewport(event: WheelEvent, target: ViewportTarget) {
    event.preventDefault()
    if (event.altKey) {
      const delta = event.deltaY > 0 ? -3 : 3
      this.brushSize = Math.max(1, Math.min(4096, this.brushSize + delta))
      this.configureFabricBrush()
      return
    }
    this.showViewportOverlay(event, target)
    const workSurface = event.currentTarget as HTMLElement
    const currentZoom = target === 'input' ? this.inputZoom : this.outputZoom
    const multiplier = event.deltaY > 0 ? 0.9 : 1.1
    const nextZoom = Math.max(0.25, Math.min(8, Number((currentZoom * multiplier).toFixed(3))))
    if (nextZoom === currentZoom) return

    // Pointer position within the visible scroll viewport
    const rect = workSurface.getBoundingClientRect()
    const pointerX = event.clientX - rect.left
    const pointerY = event.clientY - rect.top

    // Stage size and centering offset before zoom
    // stage-wrap: min-width/height: 100%, flex center → stage is centered when smaller than viewport
    const stageW = this.stageWidth * currentZoom
    const stageH = this.stageHeight * currentZoom
    const stageOffsetX = Math.max(0, (workSurface.clientWidth - stageW) / 2)
    const stageOffsetY = Math.max(0, (workSurface.clientHeight - stageH) / 2)

    // Stage-space coordinate under the pointer
    const stageX = (workSurface.scrollLeft + pointerX - stageOffsetX) / currentZoom
    const stageY = (workSurface.scrollTop + pointerY - stageOffsetY) / currentZoom

    // Apply zoom and wait for DOM to reflect new stage size
    if (target === 'input') {
      this.inputZoom = nextZoom
      $canvas.setKey('inputZoom', nextZoom)
    } else {
      this.outputZoom = nextZoom
      $canvas.setKey('outputZoom', nextZoom)
    }
    await this.updateComplete

    // Scroll so the same stage coordinate stays under the pointer
    const newStageW = this.stageWidth * nextZoom
    const newStageH = this.stageHeight * nextZoom
    const newStageOffsetX = Math.max(0, (workSurface.clientWidth - newStageW) / 2)
    const newStageOffsetY = Math.max(0, (workSurface.clientHeight - newStageH) / 2)
    workSurface.scrollLeft = stageX * nextZoom + newStageOffsetX - pointerX
    workSurface.scrollTop = stageY * nextZoom + newStageOffsetY - pointerY
  }

  private startViewportPan = (event: PointerEvent) => {
    if (event.button !== 1 && !(event.pointerType === 'touch' && this.toolMode === 'select')) return
    event.preventDefault()
    const wrap = event.currentTarget as HTMLElement
    this.panningTarget = wrap
    this.panStart = { x: event.clientX, y: event.clientY, left: wrap.scrollLeft, top: wrap.scrollTop }
    wrap.setPointerCapture(event.pointerId)
  }

  private moveViewport(event: PointerEvent, target: ViewportTarget) {
    this.showViewportOverlay(event, target)
    this.moveViewportPan(event)
  }

  private moveViewportPan(event: PointerEvent) {
    const wrap = event.currentTarget as HTMLElement
    if (this.panningTarget !== wrap) return
    wrap.scrollLeft = this.panStart.left - (event.clientX - this.panStart.x)
    wrap.scrollTop = this.panStart.top - (event.clientY - this.panStart.y)
  }

  private showViewportOverlay(_event: Event, target: ViewportTarget) {
    if (target === 'input') this.inputOverlayVisible = true
    else this.outputOverlayVisible = true
    const existingTimer = this.overlayTimers[target]
    if (existingTimer) window.clearTimeout(existingTimer)
    this.overlayTimers[target] = window.setTimeout(() => {
      if (target === 'input') this.inputOverlayVisible = false
      else this.outputOverlayVisible = false
    }, 2000)
  }

  private leaveViewport(event: PointerEvent, target: ViewportTarget) {
    this.endViewportPan(event)
    const existingTimer = this.overlayTimers[target]
    if (existingTimer) window.clearTimeout(existingTimer)
    this.overlayTimers[target] = undefined
    if (target === 'input') this.inputOverlayVisible = false
    else this.outputOverlayVisible = false
  }

  private endViewportPan = (event: PointerEvent) => {
    const wrap = this.panningTarget
    if (!wrap) return
    if (wrap.hasPointerCapture(event.pointerId)) wrap.releasePointerCapture(event.pointerId)
    this.panningTarget = undefined
  }

  private exportCanvasImage(hideSelection = false, format = 'image/png') {
    if (!this.fabricCanvas) return ''
    const exportImage = () => this.exportCompositedFabricCanvas(format)
    return hideSelection ? this.withSelectionHidden(exportImage) : exportImage()
  }

  private exportCompositedFabricCanvas(format = 'image/png') {
    const lowerCanvas = this.fabricCanvas?.getElement()
    if (!lowerCanvas) return ''
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    // Fill neutral gray so JPEG-flattened transparent regions don't become black,
    // which would otherwise dominate diffusion's image-to-image input.
    if (format === 'image/jpeg') {
      context.fillStyle = '#808080'
      context.fillRect(0, 0, this.stageWidth, this.stageHeight)
    }
    context.drawImage(lowerCanvas, 0, 0, this.stageWidth, this.stageHeight)
    const upperCanvas = (this.fabricCanvas as (Canvas & { upperCanvasEl?: HTMLCanvasElement }) | undefined)?.upperCanvasEl
    const isErasePreview = this.isEraseStroke(this.pendingBrushButton)
    if (upperCanvas && this.pendingPaintLayerId && !isErasePreview) {
      context.save()
      context.globalCompositeOperation = 'source-over'
      context.drawImage(upperCanvas, 0, 0, this.stageWidth, this.stageHeight)
      context.restore()
    }
    return canvas.toDataURL(format, 0.88)
  }

  public exportCompositedFabricCanvasBlob(format = 'image/jpeg', quality = 0.88): Promise<Blob | null> {
    const lowerCanvas = this.fabricCanvas?.getElement()
    if (!lowerCanvas) return Promise.resolve(null)
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    if (format === 'image/jpeg') {
      context.fillStyle = '#808080'
      context.fillRect(0, 0, this.stageWidth, this.stageHeight)
    }
    context.drawImage(lowerCanvas, 0, 0, this.stageWidth, this.stageHeight)
    const upperCanvas = (this.fabricCanvas as (Canvas & { upperCanvasEl?: HTMLCanvasElement }) | undefined)?.upperCanvasEl
    const isErasePreview = this.isEraseStroke(this.pendingBrushButton)
    if (upperCanvas && this.pendingPaintLayerId && !isErasePreview) {
      context.save()
      context.globalCompositeOperation = 'source-over'
      context.drawImage(upperCanvas, 0, 0, this.stageWidth, this.stageHeight)
      context.restore()
    }
    return new Promise((resolve) => canvas.toBlob(resolve, format, quality))
  }

  public exportRealtimeCanvasImage(format = 'image/jpeg') {
    const transform = this.currentStreamMotionTransform()
    // Never call withSelectionHidden during streaming â€” it emits renderAll() twice per frame
    // which disrupts Fabric.js drag state and cancels in-progress layer moves.
    if (!transform) return this.exportCompositedFabricCanvas(format)
    return this.exportTransformedRealtimeSource(transform, format)
  }

  private exportTransformedRealtimeSource(transform: [number, number, number, number, number, number], format = 'image/png') {
    const source = this.outputImage || this.exportCompositedFabricCanvas()
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    const videoCanvas = this.streamOutputBackingCanvas ?? this.streamOutputCanvasElement
    const image = source ? this.loadedImageElement(source) : undefined
    const drawable = videoCanvas && videoCanvas.width > 0 && videoCanvas.height > 0 ? videoCanvas : image
    if (!drawable) return this.exportCompositedFabricCanvas(format)
    if (drawable instanceof HTMLImageElement && (!drawable.complete || drawable.naturalWidth <= 0)) return source || this.exportCompositedFabricCanvas(format)
    context.setTransform(...transform)
    context.drawImage(drawable, 0, 0, this.stageWidth, this.stageHeight)
    context.resetTransform()
    return canvas.toDataURL(format, 0.88)
  }

  public async exportLayerConditionsWithMaskRefs(pcId: string) {
    const refs = new Map<string, Record<string, string>>()
    const activeLayers = this.activeSamplingLayersBottomUp()
    for (const layer of activeLayers) {
      for (const region of this.layerRegionSpecs(layer)) {
        const channels: Partial<Record<'color' | 'denoise' | 'prompt' | 'cfg', Blob>> = {}
        const names: Partial<Record<'color' | 'denoise' | 'prompt' | 'cfg', string>> = {}
        for (const channel of ['color', 'denoise', 'prompt', 'cfg'] as const) {
          const blob = await this.regionChannelMaskBlob(layer.id, region, channel)
          if (blob) {
            channels[channel] = blob
            names[channel] = `${layer.name || layer.id}/${region.name || region.id}/${channel}`
          }
        }
        const channelRefs = await serializeChannelRefs(pcId, channels, names)
        if (Object.keys(channelRefs).length) refs.set(this.regionRefKey(layer.id, region.id), channelRefs)
      }
    }
    return this.exportLayerConditions(refs)
  }

  private regionRefKey(layerId: string, regionId: string) {
    return `${layerId}:${regionId}`
  }

  public exportLayerConditions(channelRefsByRegion = new Map<string, Record<string, string>>()) {
    this.saveCurrentMaskScope()
    const objects = this.fabricCanvas?.getObjects() ?? []
    const conditions = []
    const activeLayers = this.activeSamplingLayersBottomUp()
    const schedules = this.autoLayerSchedules(activeLayers)
    const exportPass: LayerConditionExportPass = { layerCanvases: new Map() }
    for (const layer of activeLayers) {
      const object = objects.find((candidate) => (candidate as FabricObject & { __uid?: string }).__uid === layer.id)
      if (!object) continue
      const timing = schedules.get(layer.id) ?? { start: 0, end: 1, schedule: 'linear' as RegionSchedule, active: true }
      if (!timing.active) continue
      for (const region of this.layerRegionSpecs(layer)) {
        const isDefaultRgbaRegion = region.id.startsWith('inherited:')
        const prompt = region.inherited ? layer.preset.prompt : region.prompt
        const negativePrompt = region.inherited ? layer.preset.negativePrompt : region.negativePrompt
        const regionTiming = this.resolvedRegionTiming(region, timing)
        const channelRefs = channelRefsByRegion.get(this.regionRefKey(layer.id, region.id)) ?? {}
        const cfgValue = region.regionCfg ?? CFG_FACTOR_DEFAULT
        const hasExplicitPrompt = Boolean(channelRefs.prompt || this.layerRegionParameterMaskCanvas(layer.id, region, 'prompt'))
        const explicitCfgMask = this.layerRegionParameterMaskCanvas(layer.id, region, 'cfg')
        const explicitDenoiseMask = this.layerRegionParameterMaskCanvas(layer.id, region, 'denoise')
        const cfgMask = explicitCfgMask ? this.exportCfgMaskDataUrl(explicitCfgMask, cfgValue) : undefined
        const denoiseMask = explicitDenoiseMask ? this.exportMaskCanvas(explicitDenoiseMask) : undefined
        const hasExplicitCfg = Boolean(channelRefs.cfg || cfgMask)
        const hasExplicitDenoise = Boolean(channelRefs.denoise || denoiseMask)
        conditions.push({
          layer_id: layer.id,
          z_index: this.layerStackIndex(layer.id),
          region_id: region.id,
          name: this.promptMaskName(layer, region),
          mask_color: region.color,
          prompt: (isDefaultRgbaRegion && !hasExplicitPrompt) || layer.preset.autoTag ? '' : prompt,
          negative_prompt: isDefaultRgbaRegion && !hasExplicitPrompt ? '' : negativePrompt,
          auto_tag: layer.preset.autoTag,
          auto_tag_threshold: layer.preset.autoTagThreshold,
          auto_tag_refresh_frames: layer.preset.autoTagRefreshFrames,
          image: this.exportLayerRegionConditionImage(object, layer, region, exportPass),
          prompt_mask: undefined,
          cfg_mask: channelRefs.cfg ? undefined : cfgMask,
          denoise_mask: channelRefs.denoise ? undefined : denoiseMask,
          weight: region.inherited ? layer.preset.conditionWeight : region.maskStrength,
          mode: region.inherited ? layer.preset.conditionMode : (region.mode ?? layer.preset.conditionMode),
          mask_operator: 'max',
          denoise_operator: 'replace',
          cfg_operator: 'replace',
          denoise: hasExplicitDenoise ? (region.denoise ?? layer.preset.strength) : undefined,
          cfg: hasExplicitCfg ? cfgValue : undefined,
          steps: undefined,
          sampler: undefined,
          scheduler: undefined,
          schedule: regionTiming.schedule,
          schedule_start: regionTiming.start,
          schedule_end: regionTiming.end,
          primary_input: this._isVideoLayer(layer.id),
          ...channelRefs,
        })
      }
    }
    return conditions
  }

  public exportLayerConditionMetadata() {
    const activeLayers = this.activeSamplingLayersBottomUp()
    const schedules = this.autoLayerSchedules(activeLayers)
    const conditions = []
    for (const layer of activeLayers) {
      const timing = schedules.get(layer.id) ?? { start: 0, end: 1, schedule: 'linear' as RegionSchedule, active: true }
      if (!timing.active) continue
      for (const region of this.layerRegionSpecs(layer)) {
        const isDefaultRgbaRegion = region.id.startsWith('inherited:')
        const regionTiming = this.resolvedRegionTiming(region, timing)
        const prompt = region.inherited ? layer.preset.prompt : region.prompt
        const negativePrompt = region.inherited ? layer.preset.negativePrompt : region.negativePrompt
        const cfgValue = region.regionCfg ?? CFG_FACTOR_DEFAULT
        const hasExplicitPrompt = Boolean(this.layerRegionParameterMaskCanvas(layer.id, region, 'prompt'))
        const hasExplicitCfg = Boolean(this.layerRegionParameterMaskCanvas(layer.id, region, 'cfg'))
        const hasExplicitDenoise = Boolean(this.layerRegionParameterMaskCanvas(layer.id, region, 'denoise'))
        conditions.push({
          layer_id: layer.id,
          z_index: this.layerStackIndex(layer.id),
          region_id: region.id,
          name: this.promptMaskName(layer, region),
          mask_color: region.color,
          prompt: (isDefaultRgbaRegion && !hasExplicitPrompt) || layer.preset.autoTag ? '' : prompt,
          negative_prompt: isDefaultRgbaRegion && !hasExplicitPrompt ? '' : negativePrompt,
          auto_tag: layer.preset.autoTag,
          auto_tag_threshold: layer.preset.autoTagThreshold,
          auto_tag_refresh_frames: layer.preset.autoTagRefreshFrames,
          weight: region.inherited ? layer.preset.conditionWeight : region.maskStrength,
          mode: region.inherited ? layer.preset.conditionMode : (region.mode ?? layer.preset.conditionMode),
          mask_operator: 'max',
          denoise_operator: 'replace',
          cfg_operator: 'replace',
          denoise: hasExplicitDenoise ? (region.denoise ?? layer.preset.strength) : undefined,
          cfg: hasExplicitCfg ? cfgValue : undefined,
          steps: undefined,
          sampler: undefined,
          scheduler: undefined,
          schedule: regionTiming.schedule,
          schedule_start: regionTiming.start,
          schedule_end: regionTiming.end,
          primary_input: this._isVideoLayer(layer.id),
        })
      }
    }
    return conditions
  }

  public async exportLayerConditionsForRtcResources() {
    this.saveCurrentMaskScope()
    const objects = this.fabricCanvas?.getObjects() ?? []
    const conditions = []
    const activeLayers = this.activeSamplingLayersBottomUp()
    const schedules = this.autoLayerSchedules(activeLayers)
    const exportPass: LayerConditionExportPass = { layerCanvases: new Map() }
    for (const layer of activeLayers) {
      const object = objects.find((candidate) => (candidate as FabricObject & { __uid?: string }).__uid === layer.id)
      if (!object) continue
      const timing = schedules.get(layer.id) ?? { start: 0, end: 1, schedule: 'linear' as RegionSchedule, active: true }
      if (!timing.active) continue
      for (const region of this.layerRegionSpecs(layer)) {
        const isDefaultRgbaRegion = region.id.startsWith('inherited:')
        const prompt = region.inherited ? layer.preset.prompt : region.prompt
        const negativePrompt = region.inherited ? layer.preset.negativePrompt : region.negativePrompt
        const regionTiming = this.resolvedRegionTiming(region, timing)
        const cfgValue = region.regionCfg ?? CFG_FACTOR_DEFAULT
        const colorMask = await this.regionChannelMaskBlob(layer.id, region, 'color')
        const promptMask = await this.regionChannelMaskBlob(layer.id, region, 'prompt')
        const cfgMask = await this.regionChannelMaskBlob(layer.id, region, 'cfg')
        const denoiseMask = await this.regionChannelMaskBlob(layer.id, region, 'denoise')
        const hasExplicitPrompt = Boolean(promptMask)
        const hasExplicitCfg = Boolean(cfgMask)
        const hasExplicitDenoise = Boolean(denoiseMask)
        conditions.push({
          layer_id: layer.id,
          z_index: this.layerStackIndex(layer.id),
          region_id: region.id,
          name: this.promptMaskName(layer, region),
          mask_color: region.color,
          prompt: (isDefaultRgbaRegion && !hasExplicitPrompt) || layer.preset.autoTag ? '' : prompt,
          negative_prompt: isDefaultRgbaRegion && !hasExplicitPrompt ? '' : negativePrompt,
          auto_tag: layer.preset.autoTag,
          auto_tag_threshold: layer.preset.autoTagThreshold,
          auto_tag_refresh_frames: layer.preset.autoTagRefreshFrames,
          image: await this.exportLayerRegionConditionImageBlob(object, layer, region, exportPass),
          color_mask: colorMask,
          prompt_mask: isDefaultRgbaRegion ? null : promptMask,
          cfg_mask: cfgMask,
          denoise_mask: denoiseMask,
          weight: region.inherited ? layer.preset.conditionWeight : region.maskStrength,
          mode: region.inherited ? layer.preset.conditionMode : (region.mode ?? layer.preset.conditionMode),
          mask_operator: 'max',
          denoise_operator: 'replace',
          cfg_operator: 'replace',
          denoise: hasExplicitDenoise ? (region.denoise ?? layer.preset.strength) : undefined,
          cfg: hasExplicitCfg ? cfgValue : undefined,
          steps: undefined,
          sampler: undefined,
          scheduler: undefined,
          schedule: regionTiming.schedule,
          schedule_start: regionTiming.start,
          schedule_end: regionTiming.end,
          primary_input: this._isVideoLayer(layer.id),
        })
      }
    }
    return conditions
  }

  public exportPipelineNodes() {
    const nodes: object[] = []
    const activeLayers = this.layers.filter((layer) => layer.visible && layer.preset.samplingEnabled)
    for (const layer of activeLayers) {
      if (layer.preset.autoTag) {
        nodes.push({
          id: `tagger_${layer.id}`,
          type: 'tagger',
          layer_id: layer.id,
          tagger_model: 'wd-eva02-large-v3',
          tagger_threshold: layer.preset.autoTagThreshold,
          tagger_refresh_frames: layer.preset.autoTagRefreshFrames,
        })
      }
    }
    return nodes
  }

  public exportLayerFrames(): Record<string, string> {
    const frames: Record<string, string> = {}
    const activeLayers = this.layers.filter((layer) => layer.visible && layer.preset.samplingEnabled)
    const needsLayerFrames = activeLayers.some((l) =>
      l.preset.autoTag ||
      this._isVideoLayer(l.id) ||
      (this.layerControlNetConfigs.get(l.id)?.enabled && this.layerControlNetConfigs.get(l.id)?.useLayerFrame)
    )
    if (!needsLayerFrames) return frames

    const objects = this.fabricCanvas?.getObjects() ?? []
    for (const layer of activeLayers) {
      if (this._isVideoLayer(layer.id)) {
        // Read directly from the offscreen canvas to avoid Fabric.js async rendering race
        const src = this._videoOffscreenCanvas
        if (!src) continue
        const canvas = document.createElement('canvas')
        canvas.width = this.stageWidth
        canvas.height = this.stageHeight
        const ctx = canvas.getContext('2d')!
        ctx.drawImage(src, 0, 0, canvas.width, canvas.height)
        frames[layer.id] = canvas.toDataURL('image/jpeg', 0.90)
        continue
      }
      const object = objects.find((candidate) => (candidate as FabricObject & { __uid?: string }).__uid === layer.id)
      if (!object) continue
      const canvas = document.createElement('canvas')
      canvas.width = this.stageWidth
      canvas.height = this.stageHeight
      const ctx = canvas.getContext('2d')!
      this.drawLayerCondition(ctx, object, layer.id)
      frames[layer.id] = canvas.toDataURL('image/jpeg', 0.82)
    }
    return frames
  }

  private autoLayerSchedules(layers: LayerItem[]) {
    const schedules = new Map<string, { start: number; end: number; schedule: RegionSchedule; active: boolean }>()
    if (!layers.length) return schedules
    for (const layer of layers) {
      const schedule = layer.preset.schedule
      schedules.set(layer.id, {
        start: layer.preset.scheduleStart,
        end: layer.preset.scheduleEnd,
        schedule,
        active: true,
      })
    }
    return schedules
  }

  private layerStackIndex(layerId: string) {
    const objects = this.fabricCanvas?.getObjects() ?? []
    const index = objects.findIndex((candidate) => (candidate as FabricObject & { __uid?: string }).__uid === layerId)
    return index < 0 ? this.layers.findIndex((layer) => layer.id === layerId) : index
  }

  private resolvedRegionTiming(region: RegionItem, fallback?: { start: number; end: number; schedule: RegionSchedule; active: boolean }) {
    if (region.schedule !== 'auto') return { start: region.start, end: region.end, schedule: region.schedule }
    return { start: fallback?.start ?? 0, end: fallback?.end ?? 1, schedule: fallback?.schedule ?? 'linear' as RegionSchedule, active: fallback?.active ?? true }
  }

  private drawLayerCondition(context: CanvasRenderingContext2D, object: FabricObject, layerId: string) {
    const source = this.fabricCanvas?.getElement()
    if (!source) return
    const isTransforming = !!(this.fabricCanvas as unknown as { _currentTransform?: unknown })._currentTransform
    if (isTransforming) {
      context.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
      return
    }
    const wasSyncingLayers = this.isSyncingLayers
    this.isSyncingLayers = true
    const visibility = new Map<FabricObject, boolean | undefined>()
    const active = this.fabricCanvas?.getActiveObject()
    try {
      for (const candidate of this.fabricCanvas?.getObjects() ?? []) {
        visibility.set(candidate, candidate.visible)
        candidate.set('visible', candidate === object || (candidate as FabricObject & { __parentLayerId?: string }).__parentLayerId === layerId)
      }
      this.fabricCanvas?.discardActiveObject()
      this.fabricCanvas?.renderAll()
      context.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
    } finally {
      for (const [candidate, visible] of visibility) candidate.set('visible', visible)
      if (active) this.fabricCanvas?.setActiveObject(active)
      this.fabricCanvas?.renderAll()
      this.isSyncingLayers = wasSyncingLayers
    }
  }

  private renderedLayerConditionCanvas(object: FabricObject, layerId: string, exportPass: LayerConditionExportPass) {
    const cached = exportPass.layerCanvases.get(layerId)
    if (cached) return cached
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    this.drawLayerCondition(context, object, layerId)
    context.globalCompositeOperation = 'source-over'
    exportPass.layerCanvases.set(layerId, canvas)
    return canvas
  }

  private exportLayerRegionConditionImage(object: FabricObject, layer: LayerItem, region: RegionItem, exportPass: LayerConditionExportPass) {
    const cacheKey = `${layer.id}:${region.id}:image`
    const signature = this.layerConditionImageSignature(layer, region)
    const cached = this.layerRegionConditionImageCache.get(cacheKey)
    if (cached && cached.signature === signature) return cached.value
    const rendered = this.renderedLayerConditionCanvas(object, layer.id, exportPass)
    const value = region.id.startsWith('inherited:')
      ? this.exportInheritedLayerConditionImage(rendered, region)
      : rendered.toDataURL('image/png')
    this.layerRegionConditionImageCache.set(cacheKey, { signature, value })
    return value
  }

  private async exportLayerRegionConditionImageBlob(object: FabricObject, layer: LayerItem, region: RegionItem, exportPass: LayerConditionExportPass) {
    const cacheKey = `${layer.id}:${region.id}:image`
    const signature = this.layerConditionImageSignature(layer, region)
    const cached = this.layerRegionConditionImageBlobCache.get(cacheKey)
    if (cached && cached.signature === signature) return cached.value
    const rendered = this.renderedLayerConditionCanvas(object, layer.id, exportPass)
    const canvas = region.id.startsWith('inherited:') ? this.renderInheritedLayerConditionCanvas(rendered, region) : rendered
    const value = await this.canvasToPngBlob(canvas)
    this.layerRegionConditionImageBlobCache.set(cacheKey, { signature, value })
    return value
  }

  private renderMaskDataCanvas(mask: HTMLCanvasElement) {
    const canvas = document.createElement('canvas')
    canvas.width = mask.width
    canvas.height = mask.height
    const context = canvas.getContext('2d')!
    context.drawImage(mask, 0, 0)
    context.globalCompositeOperation = 'source-in'
    context.fillStyle = '#fff'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.globalCompositeOperation = 'destination-over'
    context.fillStyle = '#000'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.globalCompositeOperation = 'source-over'
    return canvas
  }

  private exportMaskCanvas(mask: HTMLCanvasElement) {
    return this.renderMaskDataCanvas(mask).toDataURL('image/png')
  }

  private exportCfgMaskDataUrl(mask: HTMLCanvasElement, cfgValue: number) {
    const bytes = new Uint8Array(this.cfgMaskFloatBuffer(mask, cfgValue))
    let binary = ''
    const chunkSize = 0x8000
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize))
    }
    return `data:${CFG_MASK_F32_MIME};base64,${btoa(binary)}`
  }

  private cfgMaskFloatBlob(mask: HTMLCanvasElement, cfgValue: number) {
    const payload = this.cfgMaskFloatBuffer(mask, cfgValue)
    return new Blob([payload], { type: CFG_MASK_F32_MIME })
  }

  private cfgMaskFloatBuffer(mask: HTMLCanvasElement, cfgValue: number) {
    const width = mask.width
    const height = mask.height
    const bytes = new Uint8Array(12 + width * height * 4)
    for (let i = 0; i < CFG_MASK_F32_MAGIC.length; i++) bytes[i] = CFG_MASK_F32_MAGIC.charCodeAt(i)
    const view = new DataView(bytes.buffer)
    view.setUint32(4, width, true)
    view.setUint32(8, height, true)
    const data = mask.getContext('2d')!.getImageData(0, 0, width, height).data
    const cfg = this.clamp(Number.isFinite(cfgValue) ? cfgValue : CFG_FACTOR_DEFAULT, 0, CFG_MASK_MAX, CFG_FACTOR_DEFAULT)
    let offset = 12
    for (let index = 0; index < data.length; index += 4) {
      view.setFloat32(offset, (this.maskValueFromPixel(data, index) / 255) * cfg, true)
      offset += 4
    }
    return bytes.buffer
  }

  private renderInheritedLayerConditionCanvas(renderedLayer: HTMLCanvasElement, region: RegionItem) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    const mask = this.renderLayerAlphaMaskCanvas(renderedLayer, region.negate, 1)
    if (!this.canvasHasNonZeroAlpha(mask)) {
      // Keep alpha fully transparent (no spatial influence) while storing
      // neutral RGB in transparent pixels so naive RGBA->RGB decodes do not
      // collapse to black source content.
      context.fillStyle = 'rgba(128,128,128,0)'
      context.fillRect(0, 0, canvas.width, canvas.height)
      return canvas
    }
    // Apply outer blur then inner blur independently.
    // Outer blur: draw blurred version first (extends alpha outward as a halo),
    //   then draw the solid original on top to keep the sprite interior fully opaque.
    // Inner blur: Gaussian-blur the result, then clip with destination-in back to
    //   the original boundary so blur cannot expand outside the sprite edge.
    const w = this.stageWidth, h = this.stageHeight
    const finalMask = document.createElement('canvas')
    finalMask.width = w; finalMask.height = h
    const finalMaskContext = finalMask.getContext('2d')!
    if (region.outerBlur) {
      finalMaskContext.filter = `blur(${region.outerBlur}px)`
      finalMaskContext.drawImage(mask, 0, 0)   // blurred halo extends outward
      finalMaskContext.filter = 'none'
      finalMaskContext.drawImage(mask, 0, 0)   // solid original fills interior back
    } else {
      finalMaskContext.drawImage(mask, 0, 0)
    }
    if (region.innerBlur) {
      // Blur the current state (creates inner fade at edges)
      const innerCanvas = document.createElement('canvas')
      innerCanvas.width = w; innerCanvas.height = h
      const innerCtx = innerCanvas.getContext('2d')!
      innerCtx.filter = `blur(${region.innerBlur}px)`
      innerCtx.drawImage(finalMask, 0, 0)
      // Clip to the original boundary: prevents inner-blur from creating an outer halo
      innerCtx.filter = 'none'
      innerCtx.globalCompositeOperation = 'destination-in'
      innerCtx.drawImage(mask, 0, 0)
      finalMaskContext.clearRect(0, 0, w, h)
      finalMaskContext.drawImage(innerCanvas, 0, 0)
    }
    context.drawImage(renderedLayer, 0, 0)
    context.globalCompositeOperation = 'destination-in'
    context.drawImage(finalMask, 0, 0)
    context.globalCompositeOperation = 'source-over'
    return canvas
  }

  private canvasHasNonZeroAlpha(canvas: HTMLCanvasElement) {
    const data = canvas.getContext('2d')?.getImageData(0, 0, canvas.width, canvas.height).data
    if (!data) return false
    for (let index = 3; index < data.length; index += 4) {
      if (data[index] > 0) return true
    }
    return false
  }

  private exportInheritedLayerConditionImage(renderedLayer: HTMLCanvasElement, region: RegionItem) {
    const canvas = this.renderInheritedLayerConditionCanvas(renderedLayer, region)
    return canvas.toDataURL('image/png')
  }

  private emptyStageCanvas() {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    return canvas
  }

  private renderLayerAlphaMaskCanvas(renderedLayer: HTMLCanvasElement, negate = false, strength = 1) {
    const canvas = this.emptyStageCanvas()
    const context = canvas.getContext('2d')!
    const alpha = this.clamp(strength, 0, 1, 1)
    if (negate) {
      context.fillStyle = `rgba(255,255,255,${alpha})`
      context.fillRect(0, 0, canvas.width, canvas.height)
      context.globalCompositeOperation = 'destination-out'
      context.drawImage(renderedLayer, 0, 0)
    } else {
      context.drawImage(renderedLayer, 0, 0)
      context.globalCompositeOperation = 'source-in'
      context.fillStyle = `rgba(255,255,255,${alpha})`
      context.fillRect(0, 0, canvas.width, canvas.height)
    }
    context.globalCompositeOperation = 'source-over'
    return canvas
  }

  private layerRegionConditionMaskCanvas(layerId: string, region: RegionItem) {
    if (region.id.startsWith('inherited:')) return this.inheritedLayerConditionMaskCanvas(layerId, region)
    const extracted = this.layerRegionColorMaskCanvas(layerId, region, false)
    if (extracted.hasPixels) return extracted.canvas
    const fallback = document.createElement('canvas')
    fallback.width = this.stageWidth
    fallback.height = this.stageHeight
    const context = fallback.getContext('2d')!
    const rect = this.normalizedRegionRect(region.rect)
    const fill = region.negate ? 'rgba(0,0,0,1)' : 'rgba(255,255,255,1)'
    context.fillStyle = fill
    this.fillRegionShape(context, region, rect)  // solid interior / boundary definition
    return fallback
  }

  private inheritedLayerConditionMaskCanvas(layerId: string, region: RegionItem) {
    const object = this.findLayer(layerId)
    if (!object) return this.emptyStageCanvas()
    const rendered = document.createElement('canvas')
    rendered.width = this.stageWidth
    rendered.height = this.stageHeight
    const context = rendered.getContext('2d')!
    this.drawLayerCondition(context, object, layerId)
    const strength = this.clamp(region.maskStrength ?? 1, 0, 4, 1)
    return this.renderLayerAlphaMaskCanvas(rendered, region.negate, strength)
  }

  private hasLayerRegionMaskPixels(layerId: string, region: RegionItem) {
    return this.layerRegionColorMaskCanvas(layerId, region).hasPixels
  }

  private layerRegionColorMaskCanvas(layerId: string, region: RegionItem, applyStrength = true) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    const source = this.activeMaskScope === `layer:${layerId}` && this.activeMaskChannel === 'color' ? this.maskCanvasElement : this.maskCanvasForScopeChannel(`layer:${layerId}`, 'color')
    context.drawImage(source, 0, 0)
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    const target = this.parseColor(region.color)
    let hasPixels = false
    if (!target) {
      context.clearRect(0, 0, this.stageWidth, this.stageHeight)
      return { canvas, hasPixels }
    }
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const matches = alpha > 0 && Math.abs(imageData.data[index] - target.r) <= 3 && Math.abs(imageData.data[index + 1] - target.g) <= 3 && Math.abs(imageData.data[index + 2] - target.b) <= 3
      if (matches) {
        hasPixels = true
        const nextAlpha = Math.round(alpha * (applyStrength ? region.maskStrength : 1))
        imageData.data[index] = 255
        imageData.data[index + 1] = 255
        imageData.data[index + 2] = 255
        imageData.data[index + 3] = nextAlpha
      } else {
        imageData.data[index] = 0
        imageData.data[index + 1] = 0
        imageData.data[index + 2] = 0
        imageData.data[index + 3] = 0
      }
    }
    context.putImageData(imageData, 0, 0)
    return { canvas, hasPixels }
  }

  private async regionChannelMaskBlob(layerId: string, region: RegionItem, channel: MaskChannel): Promise<Blob | null> {
    this.saveCurrentMaskScope()
    const mask = channel === 'color'
      ? this.layerRegionConditionMaskCanvas(layerId, region)
      : channel === 'cfg' || channel === 'denoise'
        ? this.layerRegionParameterMaskCanvas(layerId, region, channel)
        : this.layerRegionParameterMaskCanvas(layerId, region, channel)
    if (!mask) return null
    if (channel === 'color' && !this.canvasHasNonZeroLuma(mask)) return null
    if (channel === 'cfg') {
      const cfgValue = region.regionCfg ?? CFG_FACTOR_DEFAULT
      return this.cfgMaskFloatBlob(mask, cfgValue)
    }
    return this.canvasToPngBlob(mask)
  }

  private layerRegionParameterMaskCanvas(layerId: string, region: RegionItem, channel: Exclude<MaskChannel, 'color'>): HTMLCanvasElement | null {
    const sourceRaw = this.existingMaskCanvasForScopeChannel(`layer:${layerId}`, channel)
    if (!sourceRaw) return null
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    context.drawImage(sourceRaw, 0, 0)
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    if (channel !== 'prompt') {
      if (!region.id.startsWith('inherited:')) return null
      for (let index = 0; index < imageData.data.length; index += 4) {
        const value = Math.round(this.maskValueFromPixel(imageData.data, index))
        imageData.data[index] = value
        imageData.data[index + 1] = value
        imageData.data[index + 2] = value
        imageData.data[index + 3] = 255
      }
      context.putImageData(imageData, 0, 0)
      return canvas
    }
    for (let index = 0; index < imageData.data.length; index += 4) {
      const value = Math.round(this.maskValueFromPixel(imageData.data, index))
      imageData.data[index] = value
      imageData.data[index + 1] = value
      imageData.data[index + 2] = value
      imageData.data[index + 3] = 255
    }
    context.putImageData(imageData, 0, 0)
    return canvas
  }

  private canvasHasNonZeroLuma(canvas: HTMLCanvasElement) {
    const data = canvas.getContext('2d')?.getImageData(0, 0, canvas.width, canvas.height).data
    if (!data) return false
    for (let index = 0; index < data.length; index += 4) {
      if (this.maskValueFromPixel(data, index) > 0) return true
    }
    return false
  }

  private canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/png'))
  }

  private exportMask() {
    const canvas = this.composeSceneMaskCanvas()
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const value = Math.max(imageData.data[index], imageData.data[index + 1], imageData.data[index + 2])
      imageData.data[index] = value
      imageData.data[index + 1] = value
      imageData.data[index + 2] = value
      imageData.data[index + 3] = 255
    }
    context.putImageData(imageData, 0, 0)
    return canvas.toDataURL('image/png')
  }

  public exportRealtimeMask() {
    const transform = this.currentStreamMotionTransform()
    if (!transform) return this.exportMask()
    const canvas = this.composeSceneMaskCanvas()
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    let hasMask = false
    for (let index = 0; index < imageData.data.length; index += 4) {
      const value = Math.max(imageData.data[index], imageData.data[index + 1], imageData.data[index + 2])
      if (value > 8) hasMask = true
      imageData.data[index] = value
      imageData.data[index + 1] = value
      imageData.data[index + 2] = value
      imageData.data[index + 3] = 255
    }
    if (!hasMask) {
      for (let index = 0; index < imageData.data.length; index += 4) {
        imageData.data[index] = 255
        imageData.data[index + 1] = 255
        imageData.data[index + 2] = 255
        imageData.data[index + 3] = 255
      }
    }
    context.putImageData(imageData, 0, 0)
    const transformed = document.createElement('canvas')
    transformed.width = this.stageWidth
    transformed.height = this.stageHeight
    const transformedContext = transformed.getContext('2d')!
    transformedContext.setTransform(...transform)
    transformedContext.drawImage(canvas, 0, 0)
    transformedContext.resetTransform()
    return transformed.toDataURL('image/png')
  }

  private drawPaintedMask(context: CanvasRenderingContext2D, scope: MaskScope, negated = false) {
    const painted = document.createElement('canvas')
    painted.width = this.stageWidth
    painted.height = this.stageHeight
    const paintedContext = painted.getContext('2d')!
    paintedContext.drawImage(this.maskCanvasForScopeChannel(scope, 'color'), 0, 0)
    const imageData = paintedContext.getImageData(0, 0, this.stageWidth, this.stageHeight)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const channel = negated ? 0 : 255
      imageData.data[index] = channel
      imageData.data[index + 1] = channel
      imageData.data[index + 2] = channel
      imageData.data[index + 3] = alpha
    }
    paintedContext.putImageData(imageData, 0, 0)
    context.drawImage(painted, 0, 0)
  }

  private drawRegionMasks(context: CanvasRenderingContext2D, target: RegionTarget, layerId?: string) {
    for (const region of this.regions.filter((item) => item.target === target && (target === 'scene' || item.layerId === layerId))) {
      if (target === 'layer' && layerId && this.hasLayerRegionMaskPixels(layerId, region)) continue
      const rect = this.normalizedRegionRect(region.rect)
      context.save()
      context.globalCompositeOperation = 'source-over'
      context.filter = region.outerBlur ? `blur(${region.outerBlur}px)` : 'none'
      const channel = region.negate ? 0 : 255
      context.fillStyle = `rgba(${channel},${channel},${channel},${region.maskStrength})`
      this.fillRegionShape(context, region, rect)
      if (region.innerBlur) {
        context.filter = `blur(${region.innerBlur}px)`
        context.fillStyle = region.negate ? '#000' : '#fff'
        this.fillRegionShape(context, region, rect)
      }
      context.restore()
    }
  }

  private fillRegionShape(context: CanvasRenderingContext2D, region: RegionItem, rect: SelectionRect) {
    const points = region.rect.points
    if (points?.length) {
      context.beginPath()
      context.moveTo(points[0].x, points[0].y)
      for (const point of points.slice(1)) context.lineTo(point.x, point.y)
      context.closePath()
      context.fill()
      return
    }
    if (region.rect.shape === 'ellipse') {
      context.beginPath()
      context.ellipse(rect.x + rect.width / 2, rect.y + rect.height / 2, rect.width / 2, rect.height / 2, 0, 0, Math.PI * 2)
      context.fill()
      return
    }
    context.fillRect(rect.x, rect.y, rect.width, rect.height)
  }

  private normalizedRegionRect(rect: SelectionRect) {
    const x = Math.max(0, Math.min(this.stageWidth, rect.x))
    const y = Math.max(0, Math.min(this.stageHeight, rect.y))
    return {
      x,
      y,
      width: Math.max(1, Math.min(this.stageWidth - x, rect.width)),
      height: Math.max(1, Math.min(this.stageHeight - y, rect.height)),
      inverted: rect.inverted,
    }
  }

  public downloadOutput = () => {
    const url = this.outputImage || this.exportCanvasImage(true)
    if (!url) return
    const link = document.createElement('a')
    link.href = url
    link.download = 'rtdiffusion-frame.png'
    link.click()
  }

  private eventStartedInEditable(event: Event) {
    return event.composedPath().some((target) => {
      if (!(target instanceof HTMLElement)) return false
      return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || target.isContentEditable
    })
  }

  private restoreSnapshot = async (snapshot: EditorSnapshot) => {
    if (!this.fabricCanvas) return
    this.isRestoringHistory = true
    try {
      this.stageWidth = snapshot.stageWidth
      this.stageHeight = snapshot.stageHeight
      this.generationWidth = this.alignGenerationSize(snapshot.stageWidth)
      this.generationHeight = this.alignGenerationSize(snapshot.stageHeight)
      this.fabricCanvas.setDimensions({ width: snapshot.stageWidth, height: snapshot.stageHeight })
      this.maskCanvasElement.width = snapshot.stageWidth
      this.maskCanvasElement.height = snapshot.stageHeight
      this.maskContext = this.maskCanvasElement.getContext('2d') ?? undefined
      await (this.fabricCanvas as Canvas & { loadFromJSON: (json: unknown) => Promise<Canvas> }).loadFromJSON(normalizeCanvasPaintChildren(snapshot.canvas))
      await this.restoreMask(snapshot.mask)
      this.channelMaskCanvases.clear()
      for (const item of snapshot.channelMasks ?? []) {
        const canvas = this.maskCanvasForScopeChannel(item.scope, item.channel)
        const context = canvas.getContext('2d')!
        context.clearRect(0, 0, this.stageWidth, this.stageHeight)
        await this.drawDataUrlToCanvas(item.image, canvas)
      }
      this.regions = snapshot.regions ?? []
      this.selectionRect = snapshot.selectionRect ? { ...snapshot.selectionRect } : undefined
      const active = snapshot.selectedLayerId ? this.findLayer(snapshot.selectedLayerId) : undefined
      if (active) this.fabricCanvas.setActiveObject(active)
      else this.fabricCanvas.discardActiveObject()
      this.selectedLayerId = snapshot.selectedLayerId
      this.fabricCanvas.requestRenderAll()
    } finally {
      this.isRestoringHistory = false
      this.syncLayers()
      this.syncCanvasGeometry()
    }
  }

  private clearEditorOverlay() {
    const canvas = this.editorOverlayCanvas
    if (!canvas) return
    canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
  }

  private drawEditorOverlay(fn: (ctx: CanvasRenderingContext2D, w: number, h: number) => void) {
    const canvas = this.editorOverlayCanvas
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    fn(ctx, this.stageWidth, this.stageHeight)
  }

  private normalizeAggregateOperator(value: unknown, fallback: ConditionAggregateOperator): ConditionAggregateOperator {
    return ['replace', 'add', 'average', 'multiply', 'max'].includes(String(value)) ? value as ConditionAggregateOperator : fallback
  }

  private presetForNewLayer(): LayerPreset {
    const objects = this.fabricCanvas?.getObjects() ?? []
    const active = this.fabricCanvas?.getActiveObject()
    const activeIndex = active ? objects.indexOf(active) : objects.length
    const source = objects[Math.max(0, activeIndex - 1)] as (FabricObject & { __preset?: LayerPreset }) | undefined
    return this.normalizeLayerPreset(source?.__preset ?? this.currentPreset())
  }

  private async addImageFile(file: File) {
    const url = await this.fileToDataUrl(file)
    this.pushHistory()
    FabricImage.fromURL(url).then((image) => {
      this.fitImage(image)
      image.set({ name: file.name })
      this.attachPreset(image)
      this.fabricCanvas?.add(image)
      this.fabricCanvas?.setActiveObject(image)
      this.fabricCanvas?.requestRenderAll()
      this.syncLayers()
    })
  }

  private withSelectionHidden<T>(callback: () => T): T {
    if (!this.fabricCanvas) return callback()
    const fc = this.fabricCanvas as unknown as { _activeObject: unknown; _currentTransform: unknown }
    const active = fc._activeObject
    if (!active) return callback()
    if (fc._currentTransform) return callback()
    fc._activeObject = undefined
    this.fabricCanvas.renderAll()
    try {
      return callback()
    } finally {
      fc._activeObject = active
      this.fabricCanvas.renderAll()
    }
  }

  private loadedImageElement(source: string) {
    const image = new Image()
    image.decoding = 'sync'
    image.src = source
    return image
  }

}

declare global {
  interface HTMLElementTagNameMap { 'rtd-canvas-editor': RtdCanvasEditor }
}
