import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

try:
    from typing import Self
except ImportError:
    from typing import TypeVar
    Self = TypeVar('Self')

import attrs

from arena_simulation_setup import AB_DIR
from arena_simulation_setup.tree import Identifier, PathResolverBase


def _get_attrib(
    element: ET.Element,
    attribute: str,
    default: Optional[str] = None
) -> str:
    val = element.get(attribute)
    if val is not None:
        return str(val)

    sub_elem = element.find(attribute)
    if sub_elem is not None:
        return str(sub_elem.text)

    if default is not None:
        return default

    raise ValueError(f"attribute {attribute} not found in {element}")


@attrs.define()
class ParametrizedConfig:
    @attrs.define()
    class ObstacleConfig:
        min: int
        max: int
        type: str
        model: str

    STATIC: list[ObstacleConfig]
    INTERACTIVE: list[ObstacleConfig]
    DYNAMIC: list[ObstacleConfig]


class ParametrizedResolver(PathResolverBase):
    @property
    def path(self) -> Path:
        return AB_DIR / 'configs' / 'parametrized'


class ParametrizedIdentifier(Identifier[ParametrizedConfig]):
    @property
    def shortname(self) -> str:
        return str(Path(self.name).with_suffix(''))

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        if relpath.suffix == '.xml':
            return cls(name=str(relpath.with_suffix('')))
        raise FileNotFoundError(f"Invalid file {relpath} for parametrized identifier")

    def load(self, path: Path, /, **kwargs) -> ParametrizedConfig:
        del kwargs

        tree = ET.parse(path)
        root = tree.getroot()

        assert isinstance(
            root, ET.Element) and root.tag == "random", "not a random.xml desc"

        def xml_to_config(config) -> ParametrizedConfig.ObstacleConfig:
            return ParametrizedConfig.ObstacleConfig(
                min=int(_get_attrib(config, "min")),
                max=int(_get_attrib(config, "max")),
                type=_get_attrib(config, "type", ""),
                model=_get_attrib(config, "model")
            )

        return ParametrizedConfig(
            STATIC=list(map(xml_to_config, root.findall("./static/obstacle") or [])),
            INTERACTIVE=list(map(xml_to_config, root.findall("./static/interactive") or [])),
            DYNAMIC=list(map(xml_to_config, root.findall("./static/dynamic") or [])),
        )


ParametrizedIdentifier.use(ParametrizedResolver(ParametrizedIdentifier))
