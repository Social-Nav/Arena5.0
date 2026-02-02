#!/usr/bin/env python3

import argparse
import math
import os
import sys
from pathlib import Path


def _find_world_dir(world: str, worlds_dir: Path | None) -> Path:
    world_path = Path(world)
    if world_path.exists() and world_path.is_dir():
        return world_path

    if worlds_dir is None:
        # Fallback to source checkout layout: <pkg>/scripts/.. -> <pkg>
        worlds_dir = Path(__file__).resolve().parents[1] / 'worlds'

    candidate = worlds_dir / world
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"Cannot find world '{world}'. Tried: {candidate} and path '{world_path}'."
    )


def _load_yaml(path: Path):
    import yaml

    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _world_to_pixel(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
    resolution: float,
    image_height_px: int,
) -> tuple[float, float]:
    # ROS map convention: origin is the world pose of the map's (lower-left) pixel.
    dx = x - origin_x
    dy = y - origin_y

    # Rotate world into map frame by -yaw.
    c = math.cos(-origin_yaw)
    s = math.sin(-origin_yaw)
    mx = c * dx - s * dy
    my = s * dx + c * dy

    px = mx / resolution
    py = (image_height_px - 1) - (my / resolution)
    return px, py


def _polygon_centroid_xy(corners: list[dict]) -> tuple[float, float] | None:
    # Corners are expected as [{x, y, (z)}...]. Uses area-weighted centroid.
    pts: list[tuple[float, float]] = []
    for c in corners:
        if not isinstance(c, dict):
            continue
        if 'x' not in c or 'y' not in c:
            continue
        pts.append((float(c['x']), float(c['y'])))

    if len(pts) < 3:
        return None

    area2 = 0.0
    cx6 = 0.0
    cy6 = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx6 += (x0 + x1) * cross
        cy6 += (y0 + y1) * cross

    if abs(area2) < 1e-9:
        # Degenerate polygon; fall back to average.
        ax = sum(p[0] for p in pts) / len(pts)
        ay = sum(p[1] for p in pts) / len(pts)
        return ax, ay

    area = area2 / 2.0
    cx = cx6 / (6.0 * area)
    cy = cy6 / (6.0 * area)
    return cx, cy


def _load_worlds_dir_from_ament() -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = Path(get_package_share_directory('arena_simulation_setup'))
        return share_dir / 'worlds'
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Overlay zone/room names from world.yaml onto the occupancy map image.'
        )
    )
    parser.add_argument(
        '--world',
        required=True,
        help='World name (e.g., hospital_1) or path to a world directory.',
    )
    parser.add_argument(
        '--world-yaml',
        default=None,
        help='Override path to world.yaml (default: <world>/world.yaml).',
    )
    parser.add_argument(
        '--map-yaml',
        default=None,
        help='Override path to map.yaml (default: <world>/map/map.yaml).',
    )
    parser.add_argument(
        '--out',
        default=None,
        help='Output image path (default: <world>/map/map_labeled.png).',
    )
    parser.add_argument('--font-size', type=int, default=12)
    parser.add_argument(
        '--color',
        default='red',
        help='Text color (Pillow color name or #RRGGBB). Default: red.',
    )
    parser.add_argument(
        '--with-box',
        action='store_true',
        help='Draw a semi-transparent box behind text for readability.',
    )

    args = parser.parse_args(argv)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        print(
            'Missing Pillow dependency. Install it with e.g. `pip install pillow`.',
            file=sys.stderr,
        )
        raise e

    worlds_dir = _load_worlds_dir_from_ament()
    world_dir = _find_world_dir(args.world, worlds_dir)

    world_yaml = Path(args.world_yaml) if args.world_yaml else (world_dir / 'world.yaml')
    map_yaml = Path(args.map_yaml) if args.map_yaml else (world_dir / 'map' / 'map.yaml')

    if not world_yaml.exists():
        raise FileNotFoundError(f'world.yaml not found: {world_yaml}')
    if not map_yaml.exists():
        raise FileNotFoundError(f'map.yaml not found: {map_yaml}')

    map_meta = _load_yaml(map_yaml)
    if not isinstance(map_meta, dict):
        raise ValueError(f'Unexpected map.yaml format: {map_yaml}')

    image_rel = map_meta.get('image')
    if not image_rel:
        raise ValueError(f"Missing 'image' field in {map_yaml}")

    resolution = float(map_meta.get('resolution', 0.05))
    origin = map_meta.get('origin', [0.0, 0.0, 0.0])
    if not isinstance(origin, (list, tuple)) or len(origin) < 3:
        raise ValueError(f"Invalid 'origin' in {map_yaml}: {origin}")

    origin_x = float(origin[0])
    origin_y = float(origin[1])
    origin_yaw = float(origin[2])

    image_path = (map_yaml.parent / str(image_rel)).resolve()
    if not image_path.exists():
        # Try relative to world/map folder if image path is weird.
        alt = (world_dir / 'map' / str(image_rel)).resolve()
        if alt.exists():
            image_path = alt
        else:
            raise FileNotFoundError(f"Map image not found: {image_path} (also tried {alt})")

    world = _load_yaml(world_yaml)
    if not isinstance(world, dict):
        raise ValueError(f'Unexpected world.yaml format: {world_yaml}')

    zones = world.get('zones', [])
    if not isinstance(zones, list):
        raise ValueError(f"Unexpected 'zones' format in {world_yaml}")

    img = Image.open(image_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Load a reasonable default font.
    font = None
    for font_name in (
        'DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ):
        try:
            font = ImageFont.truetype(font_name, args.font_size)
            break
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()

    placed = 0
    skipped = 0

    for zone in zones:
        if not isinstance(zone, dict):
            skipped += 1
            continue

        name = zone.get('name')
        corners = zone.get('corners')
        if not name or not isinstance(corners, list):
            skipped += 1
            continue

        centroid = _polygon_centroid_xy(corners)
        if centroid is None:
            skipped += 1
            continue

        wx, wy = centroid
        px, py = _world_to_pixel(
            wx,
            wy,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_yaw=origin_yaw,
            resolution=resolution,
            image_height_px=img.size[1],
        )

        text = str(name)

        # Measure text bbox
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        else:
            tw, th = draw.textsize(text, font=font)  # type: ignore[attr-defined]

        x0 = int(round(px - tw / 2))
        y0 = int(round(py - th / 2))

        if args.with_box:
            pad = 3
            box = (x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad)
            draw.rectangle(box, fill=(255, 255, 255, 160), outline=(0, 0, 0, 180))

        draw.text((x0, y0), text, fill=args.color, font=font)
        placed += 1

    out_path = Path(args.out) if args.out else (world_dir / 'map' / 'map_labeled.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    composed = Image.alpha_composite(img, overlay).convert('RGB')
    composed.save(out_path)

    print(f'World: {world_dir.name}')
    print(f'Map: {image_path}')
    print(f'Out: {out_path}')
    print(f'Placed labels: {placed}, skipped: {skipped}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
