/**
 * Region editor component.
 *
 * Renders the region/mask list and per-region property controls.
 * Also manages the selection overlay drawn on the editor canvas.
 *
 * Planned interface:
 *   @property regions: RegionItem[]
 *   @property target: RegionTarget   ('scene' | 'layer')
 *   @property layerId: string
 *   @property selectionRect: SelectionRect | undefined
 *   @property regionShape: RegionShape
 *   @property stageWidth: number
 *   @property stageHeight: number
 *
 * Events emitted:
 *   region-add     — detail: { target: RegionTarget; layerId?: string }
 *   region-update  — detail: { id: string; patch: Partial<RegionItem> }
 *   region-delete  — detail: { id: string }
 *   region-select  — detail: { id: string }
 */

export {}
