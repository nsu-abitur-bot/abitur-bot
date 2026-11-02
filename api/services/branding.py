"""Bounded raster decoding and immutable, instance-owned branding media."""

import os
import re
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 2 * 1024 * 1024
MEDIA_PATTERN = r"/api/v1/branding/media/([0-9a-f]{32})\.png"


def media_root() -> Path:
    return Path(os.getenv("BRANDING_MEDIA_DIR", "./branding-media"))


def prepare_logo(content: bytes) -> tuple[bytes, bytes]:
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Логотип должен быть не больше 2 МБ")
    try:
        with Image.open(BytesIO(content), formats=["PNG", "WEBP", "JPEG"]) as image:
            width, height = image.size
            if not (1 <= width <= 4096 and 1 <= height <= 4096):
                raise ValueError("Размеры логотипа: от 1 до 4096 пикселей")
            if width * height > 8_000_000 or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Нужен статичный логотип до 8 мегапикселей")
            image.verify()
        with Image.open(BytesIO(content), formats=["PNG", "WEBP", "JPEG"]) as image:
            raster = ImageOps.exif_transpose(image).convert("RGBA")
            # New canvases deliberately discard EXIF, text and embedded profiles.
            clean = Image.new("RGBA", raster.size)
            clean.paste(raster)
            logo = BytesIO()
            clean.save(logo, format="PNG")
            clean.thumbnail((52, 52), Image.Resampling.LANCZOS)
            icon = Image.new("RGBA", (64, 64))
            icon.paste(clean, ((64 - clean.width) // 2, (64 - clean.height) // 2))
            output = BytesIO()
            icon.save(output, format="PNG")
            return logo.getvalue(), output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("Нужен корректный PNG, WebP или JPEG") from None


def store_logo(content: bytes) -> tuple[str, list[Path]]:
    logo, icon = prepare_logo(content)
    root = media_root()
    root.mkdir(parents=True, exist_ok=True)
    key = uuid4().hex
    paths = [root / f"{key}.png", root / f"{key}-icon.png"]
    try:
        for path, data in zip(paths, (logo, icon), strict=True):
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
    except OSError:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    return f"/api/v1/branding/media/{key}.png", paths


def favicon_url(logo_url: str) -> str:
    match = re.fullmatch(MEDIA_PATTERN, logo_url)
    return f"/api/v1/branding/media/{match[1]}-icon.png" if match else ""
