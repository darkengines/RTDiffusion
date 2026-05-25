// ── Backend entity types ────────────────────────────────────────────────────
export type AssetItem = { name: string; path: string; size: number; preferred: boolean }
export type AssetCatalog = { models: AssetItem[]; video_models?: AssetItem[]; loras: AssetItem[] }
export type HealthState = { status: string; mode: string }
export type SourceImage = { name: string; rel: string; size: number; modified: number }
export type SourceCatalog = { root: string; images: SourceImage[] }
export type GpuDevice = { id: string; name: string; kind: string; available: boolean; memory_total?: number }
export type GpuCatalog = { default: string; devices: GpuDevice[] }

// ── Layer / region types ────────────────────────────────────────────────────
export type LayerConditionMode = 'mask' | 'add' | 'multiply' | 'override' | 'prompt_mix'
export type ConditionAggregateOperator = 'replace' | 'add' | 'average' | 'multiply' | 'max'
export type RegionSchedule = 'auto' | 'linear' | 'ease_in' | 'ease_out' | 'ease_in_out'
export type RegionShape = 'box' | 'rope' | 'ellipse' | 'polygon'
export type RegionPoint = { x: number; y: number }
export type SelectionRect = { x: number; y: number; width: number; height: number; inverted: boolean; shape?: RegionShape; points?: RegionPoint[] }
export type RegionTarget = 'scene' | 'layer'
export type MaskScope = 'scene' | `layer:${string}`
export type MaskChannel = 'color' | 'denoise' | 'prompt' | 'cfg'

export type LayerPreset = {
  prompt: string
  negativePrompt: string
  modelPath: string
  loraPaths: string[]
  samplingEnabled: boolean
  conditionMode: LayerConditionMode
  conditionWeight: number
  maskOperator: ConditionAggregateOperator
  denoiseOperator: ConditionAggregateOperator
  conditionInnerBlur: number
  conditionOuterBlur: number
  conditionNegate: boolean
  strength: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
  schedule: RegionSchedule
  scheduleStart: number
  scheduleEnd: number
  autoTag: boolean
  autoTagThreshold: number
  autoTagRefreshFrames: number
  layerCfg: number | null
  layerSteps: number | null
  layerSampler: string | null
  layerScheduler: string | null
  rgbaPromptInherited: boolean
  rgbaNegativePromptInherited: boolean
  rgbaDenoiseInherited: boolean
  rgbaCfgInherited: boolean
  rgbaWeightInherited: boolean
  rgbaBlendingInherited: boolean
  rgbaMaskBlendingEnabled: boolean
  rgbaMaskBlendingRadius: number
  rgbaMaskBlendingStrength: number
  cfgMaskBlendingEnabled: boolean
  cfgMaskBlendingRadius: number
  cfgMaskBlendingStrength: number
  denoiseMaskBlendingEnabled: boolean
  denoiseMaskBlendingRadius: number
  denoiseMaskBlendingStrength: number
}

export type LayerTransform = {
  centerX: number
  centerY: number
  width: number
  height: number
  scale: number
  angle: number
}

export type LayerItem = {
  id: string
  name: string
  preview: string
  visible: boolean
  opacity: number
  selected: boolean
  preset: LayerPreset
  transform: LayerTransform | null
  isVideo: boolean
}

export type RegionItem = {
  id: string
  target: RegionTarget
  layerId?: string
  color: string
  name: string
  shape: RegionShape
  rect: SelectionRect
  maskStrength: number
  innerBlur: number
  outerBlur: number
  negate: boolean
  inherited: boolean
  prompt: string
  negativePrompt: string
  denoise: number
  mode: LayerConditionMode
  maskOperator: ConditionAggregateOperator
  denoiseOperator: ConditionAggregateOperator
  schedule: RegionSchedule
  start: number
  end: number
  cfgMaskInherited: boolean
  denoiseMaskInherited: boolean
  regionCfg: number | null
  regionSteps: number | null
  blendingEnabled: boolean
  blendingRadius: number
  blendingStrength: number
}

export type GeneratedLayer = { image: string; seed: number; selected: boolean; layerId: string }

// ── UI state types ──────────────────────────────────────────────────────────
export type LayerPanelTab = 'mask' | 'video'
export type ScenePanelTab = 'sdxl' | 'z-image' | 'streamdiffusion'
export type StreamOutputTransport = 'video' | 'image'
export type ToolMode = 'select' | 'moveSelection' | 'region' | 'lassoSelect' | 'ellipseSelect' | 'magicWand' | 'brush' | 'eraser' | 'text' | 'line' | 'rect' | 'ellipse' | 'fill' | 'eyedropper'
export type FillEdgeStrategy = 'none' | 'fringe' | 'transparent_blend'
export type ShapeDrawMode = 'fill' | 'stroke' | 'both'
export type RenderStrategy = 'single' | 'per_layer' | 'tiled'
export type ViewportTarget = 'input' | 'output'
export type ColorSlot = 'primary' | 'secondary'
export type EditorDest = 'paint' | 'mask'
export type ColorPlannerMode = 'complementary' | 'analogous' | 'triad' | 'tetrad' | 'split' | 'monochrome'
export type RgbaColor = { r: number; g: number; b: number; a: number }
export type EntityTarget = 'scene' | 'sceneMask' | 'layer'
export type ActiveEntityTarget = 'scene' | 'layer'
export type VideoLoopMode = 'loop' | 'ping-pong' | 'once'
export type VideoFpsMode = 'fps' | 'sequential'
export type SeedMode = 'fixed' | 'random' | 'increment' | 'decrement'
export type SeedRotationMode = 'off' | 'interval' | 'frame'
export type LatentReuseNoiseMode = 'none' | 'fixed' | 'random'
export type StreamRuntimePreset = 'diffusers' | 'lcm-lora-sdxl' | 'sdxl-lightning-4step' | 'sdxl-turbo'
export type StreamMotionMode = 'none' | 'sway' | 'orbit' | 'push' | 'zoom'
export type StreamVaeMode = 'auto' | 'tiny' | 'full'
export type SizePreset = { label: string; width: number; height: number }

// ── ControlNet ──────────────────────────────────────────────────────────────
export type ControlNetModel = 'openpose' | 'depth' | 'canny' | 'hed' | 'mlsd' | 'scribble' | 'normal' | 'lineart' | 'seg'
export type ControlNetConfig = {
  enabled: boolean
  model: ControlNetModel
  scale: number
  startAt: number
  endAt: number
  usePreprocessor: boolean
  useLayerFrame: boolean
  modelPath: string
  explicitImageUrl: string
  sourceImageUrl: string
  cnPreviewHeight: number
  preprocessorParams: {
    cannyLow: number
    cannyHigh: number
    openposeHands: boolean
    openposeFace: boolean
    mlsdThrV: number
    mlsdThrD: number
    linartCoarse: boolean
  }
}

// ── Motion / video types ────────────────────────────────────────────────────
export type MotionClipFrame = { index: number; url: string; duration_ms: number }
export type MotionClip = {
  id: string
  name: string
  motion: string
  region: string
  model: string
  prompt: string
  arrival_prompt?: string
  negative_prompt: string
  keyframes?: string[]
  fps: number
  duration_seconds: number
  loop: string
  width: number
  height: number
  frame_count: number
  frames: MotionClipFrame[]
  video_url?: string
  created_at: number
}
export type MotionTaskProgress = {
  task_id: string
  status: 'queued' | 'running' | 'complete' | 'error'
  phase: string
  progress: number
  message: string
  preview_frames?: MotionClipFrame[]
  result?: MotionClip | null
  error?: string | null
}

// ── Task / system types ─────────────────────────────────────────────────────
export type LayerGeneratePayload = { variations: { image: string; seed: number }[]; error?: string | null; mode: string; latency_ms: number }
export type LayerTaskProgress = {
  task_id: string
  status: 'queued' | 'running' | 'complete' | 'error'
  phase: string
  progress: number
  message: string
  result?: LayerGeneratePayload | null
  error?: string | null
}
export type GenerationTask = LayerTaskProgress & { layerId: string; label: string; startedAt: number }
export type SystemTask = {
  task_id: string
  type: 'system' | 'inpaint'
  status: 'queued' | 'running' | 'complete' | 'error'
  phase: string
  progress: number
  message: string
  error?: string | null
}

// ── Renderer capability ─────────────────────────────────────────────────────
export type VideoBackendCapability = { configured: boolean; message: string }
export type VideoCapabilities = { models: Record<string, VideoBackendCapability> }
export type RendererCapability = {
  configured: boolean
  native: boolean
  streamable: boolean
  realtime: boolean
  runtime: string
  transport: string
  message: string
  requirements?: string[]
  accelerators?: Record<string, boolean>
}
export type RendererCapabilities = {
  accelerators?: Record<string, boolean>
  renderers: Partial<Record<ScenePanelTab, RendererCapability>>
  versions?: Record<string, string | null>
}

// ── Preset types ────────────────────────────────────────────────────────────
export type ScenePreset = {
  prompt: string
  negativePrompt: string
  strength: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
  modelPath: string
  loraPaths: string[]
}
export type SamplerPreset = {
  strength?: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
}

// ── Dialog / picker types ───────────────────────────────────────────────────
export type PickerOption = { label: string; value: string; meta?: string }
export type PickerDialog =
  | { kind: 'option'; label: string; value: string; options: PickerOption[]; onSelect: (value: string) => void }
  | { kind: 'lora'; label: string; selectedPaths: string[]; onToggle: (value: string) => void }

// ── Streaming types ─────────────────────────────────────────────────────────
export type StreamMessage = {
  client_frame_id?: number
  client_input_id?: number
  scene_id?: string
  image?: string
  fps: number
  latency_ms: number
  mode: string
  content_type?: string
  binary?: boolean
  error?: string | null
  debug_channels?: Record<string, string>
  timings?: StreamTimingMap
}

export type StreamTimingMap = Record<string, number | string | undefined>

// ── Editor snapshot / serialisation ────────────────────────────────────────
export type EditorSnapshot = {
  canvas: unknown
  mask: string
  channelMasks?: { scope: MaskScope; channel: MaskChannel; image: string }[]
  regions: RegionItem[]
  stageWidth: number
  stageHeight: number
  selectionRect?: SelectionRect
  selectedLayerId: string
}

export type ArchiveResourceEntry = {
  hash: string
  path: string
  url: string
  mediaType: string
  size: number
  storedAt: number
}

export type SerializedMask = { scope: MaskScope; url: string }
export type SerializedProjectState = {
  version: number
  savedAt: number
  archiveRoot: string
  options: StoredOptions
  canvas: unknown
  masks: SerializedMask[]
  regions: RegionItem[]
  generatedLayers: GeneratedLayer[]
  outputImage: string
  sourceRoot: string
  selectedSourceRel: string
  sourceImages: SourceImage[]
  stageWidth: number
  stageHeight: number
  selectionRect?: SelectionRect
  selectedLayerId: string
  selectedEntity: ActiveEntityTarget
  layerPanelTab: LayerPanelTab
  scenePanelTab: ScenePanelTab
  resources: ArchiveResourceEntry[]
}

export type StoredOptions = Partial<{
  version: number
  prompt: string
  negativePrompt: string
  strength: number
  cfg: number
  steps: number
  stageWidth: number
  stageHeight: number
  brushSize: number
  brushHardness: number
  fillEdgeStrategy: FillEdgeStrategy
  fillTolerance: number
  shapeDrawMode: ShapeDrawMode
  textFontSize: number
  brushColor: string
  secondaryBrushColor: string
  activeColorSlot: ColorSlot
  colorPlannerMode: ColorPlannerMode
  toolMode: ToolMode | 'mask'
  noDefaultMaskLayers: string[]
  inputZoom: number
  outputZoom: number
  inputPaintVisible: boolean
  inputMaskVisible: boolean
  inputActiveMaskOnly: boolean
  editorDest: EditorDest
  selectedModel: string
  selectedLoras: string[]
  layerVariationCount: number
  generationWidth: number
  generationHeight: number
  layerGenerationSteps: number
  layerGenerationCfg: number
  layerGenerationSampler: string
  layerGenerationScheduler: string
  layerGenerationSeed: string
  layerTransparentBackground: boolean
  layerTransparentMethod: string
  layerTransparentMode: string
  layerTransparentColor: string
  layerTransparentTolerance: number
  layerTransparentAlphaBlur: number
  layerTransparentAlphaThreshold: number
  realtimeDevice: string
  sessionDirectory: string
  layerDevice: string
  videoDevice: string
  sceneSampler: string
  sceneScheduler: string
  sceneSeed: string
  sceneSeedMode: SeedMode
  sceneSeedRotationMode: SeedRotationMode
  sceneSeedRotationIntervalMs: number
  sceneReuseLatent: boolean
  sceneReuseLatentDenoise: number
  sceneReuseLatentNoise: number
  sceneReuseLatentNoiseMode: LatentReuseNoiseMode
  sceneTransparentBackground: boolean
  sceneTransparentMode: string
  sceneTransparentColor: string
  sceneTransparentTolerance: number
  sceneTransparentAlphaBlur: number
  sceneTransparentAlphaThreshold: number
  streamTimestepIndices: string
  streamFrameBufferSize: number
  streamCfgType: string
  streamRuntimePreset: StreamRuntimePreset
  streamMotionMode: StreamMotionMode
  streamMotionIntensity: number
  streamMotionSpeed: number
  streamTritonCompile: boolean
  streamVaeMode: StreamVaeMode
  streamOutputTransport: StreamOutputTransport
  scenePanelTab: ScenePanelTab
  selectedEntity: EntityTarget
  sceneMaskNegated: boolean
  sceneMaskAbsolute: boolean
  regionShape: RegionShape
  generatorActive: boolean
  leftPanelWidth: number
  rightPanelWidth: number
}>

// ── Reusable config types (used by render helpers) ──────────────────────────
import type { TemplateResult } from 'lit'

export type SamplingControlsConfig = {
  prompt: string
  negativePrompt: string
  modelPath?: string
  loraPaths?: string[]
  strength?: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
  onPrompt: (value: string) => void
  onNegativePrompt: (value: string) => void
  onModelPath?: (value: string) => void
  onToggleLora?: (value: string) => void
  onStrength?: (value: number) => void
  onCfg: (value: number) => void
  onSteps: (value: number) => void
  onSampler: (value: string) => void
  onScheduler: (value: string) => void
}

export type GenerationControlsConfig = {
  deviceLabel: string
  deviceValue: string
  onDevice: (value: string) => void
  sampling: SamplingControlsConfig
  beforeSampling?: TemplateResult
  afterSampling?: TemplateResult
}
