// Edge feathering for painted softmaps.
//
// Per the user-defined model: feather is a SIGNED pixel count.
//
//   +N (positive) -> outward feather: the edge softens N pixels outside
//                    the painted area. Mask GROWS by ~N px, gradient
//                    extends outward.
//   -N (negative) -> inward feather: the edge softens |N| pixels inside
//                    the painted area. Mask SHRINKS by ~|N| px, gradient
//                    extends inward.
//    0           -> hard edge (no feathering).
//
// Implementation: separable box-blur 3-pass approximation of a Gaussian,
// because Canvas2D's CSS filter isn't available in OffscreenCanvas across
// all browsers and box blur is allocation-free and fast on Uint8 buffers.
// For negative feather we apply the blur then re-map ``clamp(2v - 1, 0, 1)``
// which shifts the 50% line inward by ~radius pixels.

const MIN_RADIUS = 0
const MAX_RADIUS = 256

/**
 * Apply signed feathering to a single-channel mask buffer.
 *
 * @param mask    HxW Uint8 buffer (0..255). NOT mutated; returns a copy.
 * @param width   buffer width
 * @param height  buffer height
 * @param feather signed pixel count. >0 outward, <0 inward, 0 no-op.
 */
export function featherMask(
  mask: Uint8ClampedArray,
  width: number,
  height: number,
  feather: number,
): Uint8ClampedArray {
  const radius = Math.min(MAX_RADIUS, Math.max(MIN_RADIUS, Math.round(Math.abs(feather))))
  if (radius === 0) return new Uint8ClampedArray(mask)
  const out = _boxBlur3(mask, width, height, radius)
  if (feather >= 0) return out
  // Negative: shift the gradient inward. ``clamp(2v - 1, 0, 1)`` keeps
  // only the inner half of the blur, effectively eroding by ~radius px.
  for (let i = 0; i < out.length; i++) {
    const v = (out[i] / 255) * 2 - 1
    out[i] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(v * 255)
  }
  return out
}

/**
 * 3-pass separable box blur (good Gaussian approximation, O(n) per pass).
 * Operates on a 0..255 single-channel buffer; allocates one scratch buffer.
 */
function _boxBlur3(src: Uint8ClampedArray, width: number, height: number, radius: number): Uint8ClampedArray {
  // Each box blur of radius r gives a stddev of sqrt((r*(r+1))/3). Three
  // passes with the same r give stddev ~= r. For visual feathering the
  // exact match isn't critical -- the perceived softness is r px.
  let a = new Uint8ClampedArray(src)
  let b = new Uint8ClampedArray(src.length)
  for (let pass = 0; pass < 3; pass++) {
    _boxBlurH(a, b, width, height, radius)
    _boxBlurV(b, a, width, height, radius)
  }
  return a
}

function _boxBlurH(src: Uint8ClampedArray, dst: Uint8ClampedArray, w: number, h: number, r: number): void {
  const window = r * 2 + 1
  for (let y = 0; y < h; y++) {
    const row = y * w
    let acc = 0
    for (let i = -r; i <= r; i++) acc += src[row + _clampIndex(i, w)]
    for (let x = 0; x < w; x++) {
      dst[row + x] = acc / window
      const out = src[row + _clampIndex(x - r, w)]
      const inc = src[row + _clampIndex(x + r + 1, w)]
      acc += inc - out
    }
  }
}

function _boxBlurV(src: Uint8ClampedArray, dst: Uint8ClampedArray, w: number, h: number, r: number): void {
  const window = r * 2 + 1
  for (let x = 0; x < w; x++) {
    let acc = 0
    for (let i = -r; i <= r; i++) acc += src[_clampIndex(i, h) * w + x]
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = acc / window
      const out = src[_clampIndex(y - r, h) * w + x]
      const inc = src[_clampIndex(y + r + 1, h) * w + x]
      acc += inc - out
    }
  }
}

function _clampIndex(i: number, size: number): number {
  if (i < 0) return 0
  if (i >= size) return size - 1
  return i
}
