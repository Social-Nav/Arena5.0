from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import Scenario

from task_generator.tasks import identifier_to_available
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles


class TM_Scenario(TM_Obstacles):

    _config: ROSParamT[Scenario]

    def _parse_scenario(self, scenario: str) -> Scenario:
        return WorldIdentifier(self.node._world_manager.world_name).resolve_sync().scenario(scenario).resolve_sync().load()

    async def reset(self, **kwargs) -> Obstacles:
        del kwargs

        # Re-read scenario.yaml from disk on EVERY reset, mirroring tasks/robots/scenario.py.
        # Previously this returned self._config.value, i.e. the ROSParam's PARSED cache, which is
        # only recomputed when the `file` parameter itself is set (ROSParam.param setter). Resetting
        # does not touch that parameter, so edits to the pedestrian/obstacle sections of
        # scenario.yaml were invisible until the node was restarted -- while robot start/goal edits
        # in the SAME file applied immediately. That asymmetry made "edit yaml, press reset" quietly
        # respawn stale pedestrians.
        #
        # Re-resolving is cheap and genuinely re-reads: resolve_sync() builds a fresh ScenarioView
        # per call (so its scenario_path cached_property is new), and ScenarioView.load() does
        # open() + yaml.safe_load(). The resolver's own cache only memoizes identifier -> Path,
        # never file contents.
        try:
            scenario = self._parse_scenario(self._config.param)
        except Exception as e:
            # Deliberately softer than the robot path, which lets this propagate and aborts the
            # reset. Live-editing a yaml means transiently invalid files (half-typed value, bad
            # indent); wedging the running sim on a typo is worse than reusing the last good
            # obstacle set, since the next reset picks up the fix. The warning names the cause.
            self._logger.warn(
                f"scenario reload failed, reusing last good obstacles: {e!r}")
            scenario = self._config.value

        return scenario.static, scenario.dynamic

    def __init__(self, **kwargs):
        TM_Obstacles.__init__(self, **kwargs)

        default_scenario: str | None = 'default'
        if default_scenario not in (scenarios := list(identifier_to_available(WorldIdentifier(self.node._world_manager.world_name).resolve_sync().scenario))):
            default_scenario = next(iter(scenarios), None)
        if default_scenario is None:
            raise ValueError(f"No scenarios found in world {self.node._world_manager.world_name}")

        self._config = self.node.ROSParam[Scenario](
            self.namespace('file'),
            default_scenario,
            parse=self._parse_scenario,
        )
