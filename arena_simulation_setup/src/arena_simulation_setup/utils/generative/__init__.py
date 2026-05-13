import abc
import enum
import typing

import pydantic

from arena_simulation_setup.tree.World import WorldDescription


class WorldGeneratorType(enum.Enum):
    """
    Enum for world generator types.
    """
    EMPTY = "empty"
    HALLWAY = "hallway"


class BaseConfiguration(pydantic.BaseModel):
    width: float = 15.0  # m
    height: float = 15.0  # m
    resolution: float = 0.05  # m / px
    wall_gap: float = 0.05  # gap between adjacent walls

    @classmethod
    def validate_compat(cls, value: dict):
        model_validate = getattr(cls, 'model_validate', None)
        if callable(model_validate):
            return model_validate(value)
        return cls.parse_obj(value)


class WorldGeneratorImpl(abc.ABC):
    """
    Abstract base class for world generators.
    """

    def __init__(self, configuration: dict) -> None:
        super().__init__()
        self.configure(configuration)

    @abc.abstractmethod
    def configure(self, configuration: dict):
        ...

    @abc.abstractmethod
    def compute(self) -> WorldDescription:
        ...


class WorldGenerator:
    __registry: typing.ClassVar[dict[WorldGeneratorType, typing.Callable[[], typing.Type[WorldGeneratorImpl]]]] = {}
    _active: WorldGeneratorImpl

    @classmethod
    def register(cls, name: WorldGeneratorType):
        def wrap(impl: typing.Callable[[], typing.Type[WorldGeneratorImpl]]):
            cls.__registry[name] = impl
            return impl
        return wrap

    def compute(self) -> WorldDescription:
        return self._active.compute()

    def update_generator(self, generator: WorldGeneratorType, configuration: dict):
        if generator not in self.__registry:
            raise ValueError(f"Generator {generator} has no implementation")
        self._active: WorldGeneratorImpl = self.__registry[generator]()(configuration)

    def __init__(self, generator: WorldGeneratorType, configuration: dict):
        self.update_generator(generator, configuration)


@WorldGenerator.register(WorldGeneratorType.EMPTY)
def lazy_Empty():
    from .empty import WorldGeneratorEmpty
    return WorldGeneratorEmpty


@WorldGenerator.register(WorldGeneratorType.HALLWAY)
def lazy_Hallway():
    from .hallway import WorldGeneratorHallway
    return WorldGeneratorHallway
