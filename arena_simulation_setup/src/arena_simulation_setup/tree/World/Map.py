from __future__ import annotations

import logging
import io
import itertools
import math
import typing
from collections.abc import Iterable
from pathlib import Path

import PIL.Image
import PIL.ImageDraw
import shapely
import shapely.affinity
import yaml
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from arena_simulation_setup.tree import PathView


def _set_precision(shape, grid_size: float):
    setter = getattr(shapely, 'set_precision', None)
    if setter is None:
        return shape
    return setter(shape, grid_size)


def _make_valid(shape):
    make_valid = getattr(shapely, 'make_valid', None)
    if make_valid is None:
        return shape.buffer(0)
    return make_valid(shape)


def _remove_repeated_points(shape):
    remove_repeated_points = getattr(shapely, 'remove_repeated_points', None)
    if remove_repeated_points is None:
        return shape
    return remove_repeated_points(shape)


class Map(PathView):
    @property
    def map_yaml(self) -> Path:
        return self.path / 'map.yaml'

    @property
    def map_png(self) -> Path:
        return self.path / 'map.png'

    @classmethod
    def generate_png(
        cls,
        rooms: MultiPolygon,
        doors: MultiPolygon,
        walls: MultiLineString,
        resolution: float = 0.01,
        padding: int = 5,
        *,
        static_objects: Iterable[tuple[str, Polygon]] = (),
        asset_color: str | None = "grey",
        asset_name_color: str | None = "blue"
    ) -> tuple[bytes, tuple[float, float]]:
        """
        Generate a PNG image of the map with the given elements.

        Args:
            rooms (MultiPolygon): MultiPolygon representing the rooms in the map.
            doors (MultiPolygon): MultiPolygon representing the doors in the map.
            walls (MultiLineString): MultiLineString representing the walls in the map.
            resolution (float): Size of each pixel in meters.
            padding (int): Number of pixels to pad around the map.
            show_obj_name (bool): Whether to display object names on the map.
            static_objects (Optional[List[Tuple[str, Polygon]]]): Optional list of (name, Polygon) tuples for static objects to draw.
            asset_color (str | None): Color used to fill static objects.
            asset_name_color (str | None): Color used for static object names.
        """
        min_x, min_y, max_x, max_y = rooms.bounds

        width = max_x - min_x
        height = max_y - min_y

        img = PIL.Image.new(
            'RGB',
            (
                math.ceil(width / resolution) + 2 * padding,
                math.ceil(height / resolution) + 2 * padding,
            ),
            color='black'
        )

        scaling_factor = 1 / resolution

        def tf(shape):
            shape = shapely.affinity.translate(shape, -min_x, -min_y)
            shape = shapely.affinity.scale(shape, scaling_factor, -scaling_factor, origin=(0, 0))
            shape = shapely.affinity.translate(shape, 0, height * scaling_factor)
            shape = _set_precision(shape, 0.01)
            shape = _make_valid(shape)
            shape = _remove_repeated_points(shape)
            return shape

        def as_int(coords):
            return [(int(math.trunc(x) + padding), int(math.trunc(y) + padding)) for (x, y, *_) in coords]

        draw = PIL.ImageDraw.Draw(img)
        for cutout in itertools.chain(rooms.geoms, doors.geoms):
            poly = tf(Polygon(cutout))
            draw.polygon(as_int(poly.exterior.coords), fill='white')

        for wall in walls.geoms:
            line = tf(LineString(wall))
            draw.line(as_int(line.coords), fill='black', width=1)

        if asset_color is not None:
            for name, obj in static_objects:
                logging.debug(f"Drawing asset '{name}' with geometry: {obj} in color {asset_color}")
                poly = tf(obj)
                if len(poly.exterior.coords) < 3:
                    logging.warning(f"Skipping asset '{name}' because it has insufficient geometry to draw ({len(poly.exterior.coords)} coordinates).")
                    continue
                draw.polygon(as_int(poly.exterior.coords), fill=asset_color)
                if asset_name_color is not None:
                    _min_x, _min_y, _max_x, _max_y = poly.bounds
                    logging.debug(f"Drawing name for asset '{name}' at ({int(_max_x)}, {int(_max_y)}) color {asset_name_color}")
                    draw.text((int(_max_x), int(_max_y)), name, fill=asset_name_color)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue(), (min_x - padding * resolution, min_y - padding * resolution)

    @classmethod
    def generate_map_yaml(cls, resolution: float, filename: str, origin: tuple[float, float]) -> str:
        return typing.cast(
            str,
            yaml.safe_dump({
                'free_thresh': 0.1,
                'image': filename,
                'negate': 0,
                'occupied_thresh': 0.9,
                'origin': [*origin, 0],
                'resolution': resolution,
            })
        )
