from shapely.geometry import Polygon


from . import (
    WorldDescription,
    BaseConfiguration,
    WorldGeneratorImpl,
)

from .utils import to_corners, to_walls


class WorldGeneratorEmpty(WorldGeneratorImpl):

    class Configuration(BaseConfiguration):
        ...

    config: Configuration

    def configure(self, configuration: dict):
        self.config = self.Configuration.validate_compat(configuration)

    def compute(self) -> WorldDescription:

        room = Polygon([
            (0, 0),
            (self.config.width, 0),
            (self.config.width, self.config.height),
            (0, self.config.height),
        ])

        return WorldDescription(
            zones=[
                WorldDescription.Zone(
                    name="empty_zone",
                    corners=to_corners(room),
                    walls=to_walls(room),
                    description="An empty zone",
                )
            ]
        )
