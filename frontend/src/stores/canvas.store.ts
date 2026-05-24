import { map } from 'nanostores'
import type { ColorPlannerMode, ColorSlot, EditorDest, FillEdgeStrategy, MaskChannel, ShapeDrawMode, ToolMode } from '../types'

export interface CanvasState {
  toolMode: ToolMode
  editorDest: EditorDest
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
  colorPopupOpen: boolean
  colorPopupPosition: { x: number; y: number }
  inputZoom: number
  outputZoom: number
  inputPaintVisible: boolean
  inputMaskVisible: boolean
  inputChannelVisible: boolean
  inputActiveMaskOnly: boolean
  activeMaskChannel: MaskChannel
  maskBrushValue: number
  inputOverlayVisible: boolean
  outputOverlayVisible: boolean
  noDefaultMaskLayers: string[]
  cursor: { x: number; y: number; visible: boolean }
}

export const $canvas = map<CanvasState>({
  toolMode: 'brush',
  editorDest: 'paint',
  brushSize: 34,
  brushHardness: 100,
  fillEdgeStrategy: 'transparent_blend',
  fillTolerance: 32,
  shapeDrawMode: 'both',
  textFontSize: 48,
  brushColor: '#ffffff',
  secondaryBrushColor: '#000000',
  activeColorSlot: 'primary',
  colorPlannerMode: 'complementary',
  colorPopupOpen: false,
  colorPopupPosition: { x: 92, y: 92 },
  inputZoom: 1,
  outputZoom: 1,
  inputPaintVisible: true,
  inputMaskVisible: true,
  inputChannelVisible: true,
  inputActiveMaskOnly: false,
  activeMaskChannel: 'color',
  maskBrushValue: 255,
  inputOverlayVisible: true,
  outputOverlayVisible: true,
  noDefaultMaskLayers: [],
  cursor: { x: 0, y: 0, visible: false },
})
