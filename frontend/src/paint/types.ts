import type { Layer } from '../model/layer'

/** Which channel the active brush stroke writes into. */
export type PaintChannel = 'rgba' | 'cfg' | 'denoise' | { kind: 'prompt'; index: number }

export interface BrushSettings {
  size: number
  hardness: number     // 0..1; 1 = hard edge, 0 = full feather
  /** rgba color in [0,255]; ignored for non-rgba channels (value derives from opacity). */
  color: { r: number; g: number; b: number; a: number }
  /** Mask value (0..255) painted into cfg/denoise/prompt channels. */
  maskValue: number
}

export interface PointerPoint {
  x: number  // layer pixel coords
  y: number
  pressure: number  // 0..1
}

/** A tool consumes pointer events and mutates the active layer's channel. */
export interface Tool {
  readonly name: string
  /** Called on pointerdown. */
  begin(layer: Layer, channel: PaintChannel, point: PointerPoint, brush: BrushSettings): void
  /** Called on each pointermove between begin and end. */
  step(layer: Layer, channel: PaintChannel, point: PointerPoint, brush: BrushSettings): void
  /** Called on pointerup or pointercancel. */
  end(layer: Layer, channel: PaintChannel, brush: BrushSettings): void
}
