from pathlib import Path

import yaml

try:
    from typing import Self
except ImportError:
    from typing import TypeVar
    Self = TypeVar('Self')

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.tree import Identifier, PathResolverBase


class EnvironmentDescription(dict):
    # TODO
    ...


class EnvironmentResolver(PathResolverBase):
    @property
    def path(self) -> Path:
        return ASS_DIR / 'configs' / 'environment'


class EnvironmentIdentifier(Identifier[EnvironmentDescription]):
    @property
    def shortname(self) -> str:
        return str(Path(self.name).with_suffix(''))

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        if relpath.suffix == '.yaml':
            return cls(name=str(relpath))
        raise FileNotFoundError(f"Invalid file {relpath} for environment identifier")

    def load(self, path: Path, /, **kwargs) -> EnvironmentDescription:
        del kwargs
        with open(path, 'r') as f:
            value = yaml.safe_load(f)
        assert isinstance(value, dict)
        return EnvironmentDescription(value)


EnvironmentIdentifier.use(EnvironmentResolver(EnvironmentIdentifier))
