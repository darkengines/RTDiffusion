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


def mask_to_luma(mask: Image.Image) -> Image.Image:
    return mask.convert("L").point(lambda value: value if value > 12 else 0)


def rgba_to_neutral_rgb(image: Image.Image, neutral: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*neutral, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def alpha_to_empty_mask(image: Image.Image) -> Image.Image:
    alpha = image.convert("RGBA").getchannel("A")
    return alpha.point(lambda value: 255 if value < 250 else 0)