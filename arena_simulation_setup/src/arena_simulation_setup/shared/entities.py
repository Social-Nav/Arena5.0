from __future__ import annotations

import warnings
from pathlib import Path

import attrs
import cattrs
from typing_extensions import Self

from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.assets.Pedestrian import PedestrianIdentifier
from arena_simulation_setup.utils.cattrs import (
    Parseable,
    Serializable,
    converter,
)
from arena_simulation_setup.utils.geometry import Pose, Position, Scale


@attrs.define(kw_only=True)
class Named(Parseable, Serializable):
    name: str
    extra: dict = attrs.field(factory=dict)

    @property
    def sim_path(self) -> str:
        return self.extra.get('sim_path', self.name)

    @sim_path.setter
    def sim_path(self, value: str) -> None:
        self.extra['sim_path'] = str(value)

    @classmethod
    def parse(cls, value: dict) -> Self:
        if 'pos' in value:
            value['pose'] = value['pos']
            del value['pos']
        value['extra'] = {**value}
        return converter.structure_attrs_fromdict(value, cls)

    def serialize(self) -> dict:
        result = cattrs.gen.make_dict_unstructure_fn(type(self), converter, _cattrs_omit_if_default=True)(self)
        for k in attrs.fields(type(self)):
            result.get('extra', {}).pop(k.name, None)
        if not result.get('extra', {}):
            result.pop('extra', None)
        return result


@attrs.define(kw_only=True)
class Entity(Named, Parseable, Serializable):
    pose: Pose = attrs.field(converter=Pose.converter)
    model: ObjectIdentifier = attrs.field(converter=ObjectIdentifier.converter)

    included_from: Path | None = attrs.field(default=None, repr=False)

    def asdict(self, expand_extra: bool = True) -> dict:
        if expand_extra:
            return {
                **self.extra,
                **attrs.asdict(self, filter=lambda a, v: a.name != 'extra'),
            }
        return attrs.asdict(self)


@attrs.define
class Obstacle(Entity):
    scale: Scale | None = None


@attrs.define
class DynamicObstacle(Entity):
    # Overrides Entity.pose to read the [x, y, yaw] form's yaw as DEGREES, matching how
    # scenario.yaml writes pedestrian headings (values across the shipped scenarios span
    # -180..+278, so they cannot be radians). Mirrors what RobotGoal already does for the
    # robot's own pose in tree/World/Scenario.py.
    #
    # This converter only fires for RAW values, i.e. direct construction such as
    # CustomDynamicObstacle.parse's `cls(**known_values)`. The scenario-file path goes through
    # `parse` below, because cattrs structures the field into a Pose (via Pose.parse, radians)
    # before any attrs converter runs.
    #
    # NOT applied to Entity/Obstacle: static + interactive obstacles share that field and no
    # scenario currently defines any, so widening it is a separate, untested change.
    pose: Pose = attrs.field(converter=Pose.converter_deg)
    model: PedestrianIdentifier = attrs.field(converter=PedestrianIdentifier.converter)
    waypoints: list[Position] = attrs.field(factory=list)
    velocity: float = attrs.field(converter=float, default=1.0)  # m/s

    @classmethod
    def parse(cls, value: dict) -> Self:
        if isinstance(value, dict):
            key = 'pose' if 'pose' in value else 'pos'
            raw = value.get(key)
            converted = Pose.xy_yaw_deg_to_rad(raw)
            if converted is not raw:
                value = {**value, key: converted}
        return super().parse(value)


@attrs.define
class CustomDynamicObstacle(DynamicObstacle):
    """
    DynamicObstacles but with properties can be define in runtime
    """

    def __getattr__(self, name):
        """
        Allow access to dynamic attributes "attr_name" via self.attr_name
        """
        if name in self.extra:
            return self.extra[name]
        raise AttributeError(f"{name} not found")

    @classmethod
    def parse(cls, value) -> Self:
        known_fields = set(f.name for f in attrs.fields(cls))

        if 'pos' in value:
            value['pose'] = value['pos']
            del value['pos']

        known_values = {k: v for k, v in value.items() if k in known_fields}
        custom_fields = {k: v for k, v in value.items() if k not in known_fields}

        warnings.warn(
            "CustomDynamicObstacle.parse is deprecated and will be removed in a future release. "
            "Call the constructor directly, e.g., CustomDynamicObstacle(**value).",
            FutureWarning,
            stacklevel=2
        )

        obj = cls(**known_values)
        obj.extra.update(custom_fields)
        value = obj.asdict(True)

        return converter.structure(value, cls)
