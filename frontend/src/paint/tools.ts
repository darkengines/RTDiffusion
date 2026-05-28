import type { Layer } from '../model/layer'
import { eraseDab, paintDab } from './stroke'
import type { BrushSettings, PaintChannel, PointerPoint, Tool } from './types'

/**
 * Walks from `from` to `to` placing dabs every `spacing` pixels. Without
 * interpolation a fast pointer move would leave a string of disconnected
 * blobs; the spacing is set so adjacent dabs overlap by half the radius.
 */
function _interpolateAndApply(
  fn: (l: Layer, c: PaintChannel, p: PointerPoint, b: BrushSettings) => void,
  layer: Layer,
  channel: PaintChannel,
  from: PointerPoint,
  to: PointerPoint,
  brush: BrushSettings,
): void {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const dist = Math.sqrt(dx * dx + dy * dy)
  const spacing = Math.max(1, brush.size * 0.35)
  const steps = Math.max(1, Math.ceil(dist / spacing))
  for (let i = 1; i <= steps; i++) {
    const t = i / steps
    fn(layer, channel, {
      x: from.x + dx * t,
      y: from.y + dy * t,
      pressure: from.pressure + (to.pressure - from.pressure) * t,
    }, brush)
  }
}

class BrushTool implements Tool {
  readonly name = 'brush'
  private _last: PointerPoint | null = null

  begin(layer: Layer, channel: PaintChannel, point: PointerPoint, brush: BrushSettings): void {
    paintDab(layer, channel, point, brush)
    this._last = point
  }
  step(layer: Layer, channel: PaintChannel, point: PointerPoint, brush: BrushSettings): void {
    if (this._last) {
      _interpolateAndApply(paintDab, layer, channel, this._last, point, brush)
    } else {
      paintDab(layer, channel, point, brush)
    }
    this._last = point
  }
  end(): void {
    this._last = null
  }
}

class EraserTool implements Tool {
  readonly name = 'eraser'
  private _last: PointerPoint | null = null

  begin(layer: Layer, channel: PaintChannel, point: PointerPoint, brush: BrushSettings): void {
    eraseDab(layer, channel, point, brush)
    this._last = point
  }
  step(layer: Layer, channel: PaintChannel, point: PointerPoint, brush: BrushSettings): void {
    if (this._last) {
      _interpolateAndApply(eraseDab, layer, channel, this._last, point, brush)
    } else {
      eraseDab(layer, channel, point, brush)
    }
    this._last = point
  }
  end(): void {
    this._last = null
  }
}

export const brushTool = new BrushTool()
export const eraserTool = new EraserTool()
