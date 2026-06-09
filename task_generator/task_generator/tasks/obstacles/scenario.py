import os

from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import Scenario

from task_generator.tasks import identifier_to_available
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles


class TM_Scenario(TM_Obstacles):

    _config: ROSParamT[Scenario]

    def _world_name(self) -> str:
        configured = str(getattr(self.node.conf.Arena.WORLD, 'value', '') or '').strip()
        return configured or self.node._world_manager.world_name

    def _scenario_param(self) -> str:
        env_value = str(os.environ.get('ARENA_SCENARIO_FILE', '') or '').strip()
        if env_value:
            return env_value
        try:
            value = self.node.get_parameter(self.namespace('file')).value
        except Exception:
            value = self._config.param
        return str(value or '').strip()

    def _parse_scenario(self, scenario: str) -> Scenario:
        return WorldIdentifier(self._world_name()).resolve_sync().scenario(str(scenario)).resolve_sync().load()

    async def reset(self, **kwargs) -> Obstacles:
        scenario = self._parse_scenario(self._scenario_param())
        return scenario.static, scenario.dynamic

    def __init__(self, **kwargs):
        TM_Obstacles.__init__(self, **kwargs)

        default_scenario: str | None = 'default'
        if default_scenario not in (scenarios := list(identifier_to_available(WorldIdentifier(self._world_name()).resolve_sync().scenario))):
            default_scenario = next(iter(scenarios), None)
        if default_scenario is None:
            raise ValueError(f"No scenarios found in world {self._world_name()}")

        self._config = self.node.ROSParam[Scenario](
            self.namespace('file'),
            default_scenario,
            parse=self._parse_scenario,
        )
