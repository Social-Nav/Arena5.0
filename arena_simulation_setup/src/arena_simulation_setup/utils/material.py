import re
import typing
from pathlib import Path

from PIL import Image, ImageColor


class _ParsedColor:
    def __init__(self, rgba: tuple[float, float, float, float]):
        self._rgba = rgba

    def get_rgba(self) -> tuple[float, float, float, float]:
        return self._rgba


def _normalize_channel(value: float, /) -> float:
    if value > 1.0:
        return max(0.0, min(255.0, value)) / 255.0
    return max(0.0, min(1.0, value))


def _parse_function_color(color: str) -> _ParsedColor:
    match = re.fullmatch(r'(?P<kind>rgba?|hsla?)\((?P<body>[^)]*)\)', color.strip())
    if match is None:
        raise ValueError(f"Unsupported color format: {color}")

    kind = match.group('kind').lower()
    values = [part.strip() for part in match.group('body').split(',') if part.strip()]
    if kind in {'rgb', 'rgba'}:
        if len(values) not in {3, 4}:
            raise ValueError(f"Invalid {kind} color: {color}")
        rgba = tuple(_normalize_channel(float(value)) for value in values[:3])
        alpha = _normalize_channel(float(values[3])) if len(values) == 4 else 1.0
        return _ParsedColor((rgba[0], rgba[1], rgba[2], alpha))

    raise ValueError(f"Unsupported color function: {color}")


def Color(color: typing.Any) -> _ParsedColor:
    """Convert a color representation to a parsed color object.

    Args:
        color (typing.Any): Color representation (e.g., hex string, RGB tuple).

    Returns:
        _ParsedColor: Parsed color with normalized RGBA channels.
    """
    if isinstance(color, _ParsedColor):
        return color

    if isinstance(color, (tuple, list)):
        if len(color) not in {3, 4}:
            raise ValueError(f"Invalid tuple color: {color}")
        rgba = tuple(_normalize_channel(float(value)) for value in color[:3])
        alpha = _normalize_channel(float(color[3])) if len(color) == 4 else 1.0
        return _ParsedColor((rgba[0], rgba[1], rgba[2], alpha))

    if isinstance(color, str):
        stripped = color.strip()
        if '(' in stripped and stripped.endswith(')'):
            return _parse_function_color(stripped)

        rgba = ImageColor.getcolor(stripped, 'RGBA')
        return _ParsedColor(tuple(channel / 255.0 for channel in rgba))

    raise ValueError(f"Unsupported color value: {color!r}")


class ImgUtil:
    @classmethod
    def tint(cls, img: Path | Image.Image, tint: typing.Any) -> Image.Image:
        """Apply a tint to an image.

        Args:
            img (Path | Image.Image): Image or path to image.
            tint (typing.Any): Tint color.

        Returns:
            Image.Image: Tinted image.
        """

        if isinstance(img, Path):
            img = Image.open(img).convert("RGBA")

        tint_color = Color(tint).get_rgba()

        strength = tint_color[3]
        orig_alpha = img.split()[3]
        base_rgb = img.convert("RGB")
        overlay_rgb = Image.new(
            "RGB", img.size, (
                int(tint_color[0] * 255),
                int(tint_color[1] * 255),
                int(tint_color[2] * 255),
            )
        )
        blended_rgb = Image.blend(base_rgb, overlay_rgb, strength)
        r, g, b = blended_rgb.split()

        return Image.merge("RGBA", (r, g, b, orig_alpha))


class MdlUtil:
    """.mdl material helper
    """

    def __init__(self, path: Path):
        self.path = path

    @property
    def diffuse_texture_paths(self):
        """Yields paths to diffuse texture files referenced by the .mdl file.

        Yields:
            Path: Path to a texture file.
        """
        with open(self.path, 'r') as f:
            for line in f:
                match = re.search(r'diffuse_texture:\s*texture_2d\("([^"]+)"', line)
                if match:
                    yield self.path.parent / Path(match.group(1))
