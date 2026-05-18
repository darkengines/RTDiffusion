import base64
from io import BytesIO

from PIL import Image


def decode_data_url(data_url: str) -> Image.Image:
    _, _, payload = data_url.partition(",")
    raw = base64.b64decode(payload or data_url)
    return Image.open(BytesIO(raw)).convert("RGB")


def decode_data_url_rgba(data_url: str) -> Image.Image:
    _, _, payload = data_url.partition(",")
    raw = base64.b64decode(payload or data_url)
    return Image.open(BytesIO(raw)).convert("RGBA")


def encode_data_url(image: Image.Image, *, image_format: str = "JPEG", quality: int = 82) -> str:
    buffer = BytesIO()
    save_kwargs = {"format": image_format}
    if image_format.upper() in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    image.save(buffer, **save_kwargs)
    mime = "image/jpeg" if image_format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def data_url_bytes(data_url: str) -> tuple[str, bytes]:
    header, separator, payload = data_url.partition(",")
    mime = "image/jpeg"
    if separator and header.startswith("data:"):
        mime = header[5:].split(";", 1)[0] or mime
    return mime, base64.b64decode(payload if separator else data_url)


def mask_to_luma(mask: Image.Image) -> Image.Image:
    """White-act mask normalization (legacy entry point).

    Equivalent to `compose.io.decode_mask(mask, mask.size)` with the default
    threshold. Kept for callers in `engine.py` that still take a PIL Image
    directly; new code should call `compose.io.decode_mask` instead.
    """
    from .compose.io import decode_mask

    return decode_mask(mask, mask.size)


def rgba_to_neutral_rgb(image: Image.Image, neutral: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*neutral, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def alpha_to_empty_mask(image: Image.Image) -> Image.Image:
    """White-act mask built from an RGBA source's transparent pixels (legacy).

    Delegates to `compose.io.decode_empty_alpha_as_mask`. New code should call
    that helper directly.
    """
    from .compose.io import decode_empty_alpha_as_mask

    return decode_empty_alpha_as_mask(image, image.size)