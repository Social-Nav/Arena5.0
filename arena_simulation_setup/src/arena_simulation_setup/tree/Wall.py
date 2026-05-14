from __future__ import annotations

import abc
import itertools
import math
from pathlib import Path
import typing
from collections.abc import Iterable

import attrs
import yaml

from arena_simulation_setup.shared.entities import Obstacle
from arena_simulation_setup.tree import (
    DomainAssetIdentifier,
    DynamicPathResolver,
    DynamicPaths,
    NetResolver,
)
from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.utils.cattrs import Parseable, converter
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

###
# Parsing wall description
###


class PositionalNumber(Parseable):
    def __init__(self, *, absolute: typing.Optional[float] = None, relative: typing.Optional[float] = None):
        if absolute is not None:
            self._absolute = absolute
            self._relative = None
        elif relative is not None:
            self._absolute = None
            self._relative = relative
        else:
            raise ValueError("Must specify either absolute or relative.")

    def absolute(self, low: float, high: float) -> float:
        if self._absolute is not None:
            if math.copysign(1, self._absolute) < 0:
                return high + self._absolute
            return self._absolute
        if self._relative is not None:
            return low + (high - low) * self._relative
        raise ValueError("Neither absolute nor relative is set.")

    def realize(self, start: Position, end: Position) -> Position:
        return start + self.absolute(0.0, (end - start).norm()) * (end - start).normalized()

    @classmethod
    def parse(cls, value: typing.Any) -> PositionalNumber:
        if isinstance(value, str) and value.endswith('%'):
            return cls(relative=float(value[:-1]) / 100.0)
        return cls(absolute=float(value))


@attrs.define(kw_only=True)
class SubWall(abc.ABC):
    x: float = attrs.field(converter=float, default=0.0)  # x axis shift [m]
    y: float = attrs.field(converter=float, default=0.0)  # y axis shift [m]
    z: float = attrs.field(converter=float, default=0.0)  # z axis shift [m]

    def _shift(self, start: Position, end: Position) -> tuple[Position, Position]:
        external_orientation = (end - start).to_orientation()
        offset = external_orientation * Position(self.x, self.y, self.z)
        return offset + start, offset + end

    @abc.abstractmethod
    def realize(self, start: Position, end: Position) -> WallRealization:
        pass


@attrs.define(kw_only=True)
class TilingAsset(SubWall):
    """
    Place repeating asset along the wall.
    """
    tile: list[SubWallT]
    every: float  # place every N meters
    width: float = attrs.field(converter=float, default=0.0)  # width of the tile [m]

    def realize(self, start: Position, end: Position) -> WallRealization:
        start, end = self._shift(start, end)

        r_walls, r_obstacles = itertools.chain(()), itertools.chain(())
        if (divisor := (end - start).norm()) > 1e-6:
            every = self.every / divisor
            width = self.width / divisor / 2.0
        else:
            every = 1.0
            width = 0.0

        offset = every + width
        while (offset + width) < 1:
            for asset in self.tile:
                walls, obstacles = asset.realize(start + (offset - width) * (end - start), start + (offset + width) * (end - start))
                r_walls = itertools.chain(r_walls, walls)
                r_obstacles = itertools.chain(r_obstacles, obstacles)
            offset += every
        return (r_walls, r_obstacles)


@attrs.define(kw_only=True)
class FillAsset(SubWall):
    """
    Place along slice of the wall.
    """
    fill: list[SubWallT]
    start: PositionalNumber = PositionalNumber.parse(0.0)  # start at N meters along the wall
    end: PositionalNumber = PositionalNumber.parse(-0.0)  # end at N meters along the wall

    def realize(self, start: Position, end: Position) -> WallRealization:
        start, end = self._shift(start, end)

        r_start = self.start.realize(start, end)
        r_end = self.end.realize(start, end)

        return tuple(
            itertools.chain.from_iterable(
                zip(*(e.realize(r_start, r_end) for e in self.fill))
            )
        )


@attrs.define(kw_only=True)
class PlaceObstacleAsset(SubWall):
    """
    Place a single obstacle.
    """
    model: ObjectIdentifier = attrs.field(converter=ObjectIdentifier.converter)  # model

    at: PositionalNumber = PositionalNumber.parse('50%')  # place at position along the wall
    orientation: Orientation = attrs.field(factory=Orientation.identity)  # interior orientation

    name: str = ""  # asset name, defaults to model name

    def realize(self, start: Position, end: Position) -> WallRealization:

        exterior_orientation = (end - start).to_orientation()
        start, end = self._shift(start, end)

        return (), (
            Obstacle(
                name=self.name or self.model.name,
                model=self.model,
                pose=Pose(
                    position=self.at.realize(start, end),
                    orientation=self.orientation * exterior_orientation,
                ),
            ),
        )


@attrs.define(kw_only=True)
class PlaceWallSegmentAsset(SubWall):
    """
    Place a single wall segment.
    """
    material: MaterialIdentifier = attrs.field(
        converter=MaterialIdentifier.converter,
        default=Material.default('wall'),
    )
    height: float = attrs.field(converter=float, default=2.0)
    width: float = attrs.field(converter=float, default=0.05)
    name: str = ""

    def realize(self, start: Position, end: Position) -> WallRealization:
        start, end = self._shift(start, end)

        return (
            WallSegment(
                start=start,
                end=end,
                height=self.height,
                width=self.width,
                material=self.material,
            ),
        ), ()


# Now that all SubWall classes are defined, create the proper type alias
SubWallT = TilingAsset | FillAsset | PlaceObstacleAsset | PlaceWallSegmentAsset


def _is_subwall_union(type_) -> bool:
    if str(type_) == 'SubWallT':
        return True
    args = typing.get_args(type_)
    return bool(args) and all(
        isinstance(arg, type) and issubclass(arg, SubWall)
        for arg in args
    )


def _is_subwall_list(type_) -> bool:
    if str(type_) in {'list[SubWallT]', 'typing.List[SubWallT]'}:
        return True
    origin = typing.get_origin(type_)
    args = typing.get_args(type_)
    return origin is list and len(args) == 1 and _is_subwall_union(args[0])


def _structure_subwall(value, _type):
    if isinstance(value, SubWall):
        return value
    if not isinstance(value, dict):
        raise ValueError(f'Cannot structure SubWall from {value!r}')

    # The wall asset YAML schema is key-discriminated.  Older Ubuntu cattrs
    # versions do not understand ``list[SubWallT]`` / PEP-604 unions, so decode
    # the union explicitly instead of relying on cattrs' generic dispatch.
    if 'tile' in value:
        data = dict(value)
        data['tile'] = [_structure_subwall(item, SubWallT) for item in data.get('tile') or []]
        return TilingAsset(
            tile=data['tile'],
            every=float(data['every']),
            width=float(data.get('width', 0.0)),
            x=float(data.get('x', 0.0)),
            y=float(data.get('y', 0.0)),
            z=float(data.get('z', 0.0)),
        )
    if 'fill' in value:
        data = dict(value)
        data['fill'] = [_structure_subwall(item, SubWallT) for item in data.get('fill') or []]
        return FillAsset(
            fill=data['fill'],
            start=PositionalNumber.parse(data.get('start', 0.0)),
            end=PositionalNumber.parse(data.get('end', -0.0)),
            x=float(data.get('x', 0.0)),
            y=float(data.get('y', 0.0)),
            z=float(data.get('z', 0.0)),
        )
    if 'model' in value:
        data = dict(value)
        return PlaceObstacleAsset(
            model=ObjectIdentifier.converter(data['model']),
            at=PositionalNumber.parse(data.get('at', '50%')),
            orientation=(
                data.get('orientation')
                if isinstance(data.get('orientation'), Orientation)
                else Orientation.parse(float(data.get('orientation', 0.0)))
            ),
            name=str(data.get('name', '')),
            x=float(data.get('x', 0.0)),
            y=float(data.get('y', 0.0)),
            z=float(data.get('z', 0.0)),
        )
    return converter.structure_attrs_fromdict(value, PlaceWallSegmentAsset)


def _structure_subwall_list(value, type_):
    if value is None:
        return []
    args = typing.get_args(type_)
    item_type = args[0] if args else SubWallT
    return [_structure_subwall(item, item_type) for item in value]


converter.register_structure_hook_func(_is_subwall_union, _structure_subwall)
converter.register_structure_hook_func(_is_subwall_list, _structure_subwall_list)


###
# Realization of a wall description
###


@attrs.define
class WallSegment:
    start: Position
    end: Position
    height: float
    width: float
    material: MaterialIdentifier = attrs.field(
        converter=MaterialIdentifier.converter,
        default=Material.default('wall'),
    )


WallRealization = tuple[Iterable[WallSegment], Iterable[Obstacle]]


@attrs.define
class WallDescription:
    main: list[SubWallT]

    def realize(self, start: Position, end: Position) -> WallRealization:
        r_walls, r_obstacles = itertools.chain(()), itertools.chain(())
        for subwall in self.main:
            walls, obstacles = subwall.realize(start, end)
            r_walls = itertools.chain(r_walls, walls)
            r_obstacles = itertools.chain(r_obstacles, obstacles)

        return (r_walls, r_obstacles)

    @classmethod
    def simple(cls, material: typing.Optional[MaterialIdentifier] = None) -> WallDescription:
        if material is None:
            return cls(main=[PlaceWallSegmentAsset()])
        return cls(
            main=[
                PlaceWallSegmentAsset(
                    material=material,
                )
            ]
        )


class WallIdentifier(DomainAssetIdentifier[WallDescription]):
    _asset_type = 'Wall'

    def load(self, path: Path, /, **kwargs) -> WallDescription:
        del kwargs  # unused
        with open(path / f'{path.name}.yaml') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f'Invalid wall description in {path}: expected mapping')
        return WallDescription(main=[_structure_subwall(item, SubWallT) for item in data.get('main') or []])


WallIdentifier.use(*DynamicPaths.as_resolvers(WallIdentifier))
WallIdentifier.use(*NetResolver.all(WallIdentifier))
