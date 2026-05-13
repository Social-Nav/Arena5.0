import asyncio
import traceback

import arena_robots.Robot
import arena_simulation_setup.tree.assets.Object
import arena_simulation_setup.tree.assets.Pedestrian
import arena_simulation_setup.tree.configs.environment
import arena_simulation_setup.tree.configs.parametrized
import arena_simulation_setup.tree.World as World
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import std_srvs.srv as std_srvs
import task_generator_msgs.srv
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.shared import Namespace
from std_msgs.msg import Empty, Int16, String
from std_srvs.srv import Empty as EmptySrv

from task_generator.constants import Constants
from task_generator.constants.runtime import Configuration
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.manager.robot_manager import RobotsManagerROS
from task_generator.manager.robot_manager.robots_manager_ros import RobotsManager
from task_generator.manager.world_manager.world_manager_ros import (
    WorldManagerROS as WorldManager,
)
from task_generator.shared import configure_node
from task_generator.simulators.human import BaseHumanSimulator, HumanSimulatorRegistry
from task_generator.simulators.human.utils import ObstacleLayer
from task_generator.simulators.sim import BaseSim, SimulatorRegistry
from task_generator.tasks import identifier_to_available
from task_generator.tasks.task import Task

from . import SafeCallbackNode


class TaskGenerator(ArenaMixinNode, SafeCallbackNode):
    """
    Task Generator Node
    Will initialize and reset all tasks. The task to use is read from the `/task_mode` param.
    """

    _world_manager: WorldManager
    _human_simulator: BaseHumanSimulator
    _environment_manager: EnvironmentManager
    _robots_manager: RobotsManager
    _simulator: BaseSim

    _initialized: bool

    def __init__(
        self,
        namespace: str = "task_generator_node",
    ):
        configure_node(self)

        super().__init__('task_generator')
        self.conf = Configuration(self)

        self._namespace = Namespace(namespace)

        Task.declare_parameters(self)

        self._auto_reset = self.rosparam[bool].get('auto_reset', False)
        self._train_mode = self.rosparam[bool].get('train_mode', False)

        self._reset_lock: asyncio.Lock = asyncio.Lock()
        self._start_time = self.time
        self._number_of_resets = 0
        self._completed_episodes = 0
        self._finished_published = False
        self._world_geometry_spawned = False
        self._task: Task

        # VLN instruction interface (published per-episode)
        self._vln_instruction = self.rosparam[str].get('vln_instruction', 'navigate')
        self._vln_instruction_file = self.rosparam[str].get('vln_instruction_file', '')
        self._pub_vln_instruction = self.create_publisher(
            String,
            self.service_namespace('vln_instruction'),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # Publishers
        self._pub_task_reset = self.create_publisher(
            Int16,
            self.service_namespace('task_reset'),
            1,
        )

        self._pub_finished = self.create_publisher(
            Empty,
            self.service_namespace('finished'),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._check_status_task: asyncio.Task

    async def setup(self):
        self._logger.info("Setting up Task Generator Node")

        await self._set_up_managers()
        await self._set_up_services()

        tm_modules = self.conf.TaskMode.TM_MODULES.value
        tm_modules.add(Constants.TaskMode.TM_Module.CLEAR_FORBIDDEN_ZONES)
        tm_modules.add(Constants.TaskMode.TM_Module.RVIZ_UI)

        self._logger.info("Creating task")
        self._logger.debug(f"Modules: {list(tm_modules)}")
        self._task = await Task.create(
            node=self,
            environment_manager=self._environment_manager,
            robots_manager=self._robots_manager,
            world_manager=self._world_manager,
            modules=list(tm_modules)
        )

        try:
            synced = await asyncio.wait_for(self._world_manager.sync(), timeout=30.0)
        except asyncio.TimeoutError:
            synced = False
            self.get_logger().warn(
                "Timed out waiting for world/map synchronization; continuing because the "
                "world-change callback may already have spawned geometry in Isaac."
            )
        if not synced:
            self.get_logger().warn(
                "World/map synchronization did not report success before task reset; "
                "continuing to avoid blocking robot/Nav2 bringup."
            )
        if not self._world_geometry_spawned:
            self.get_logger().warn(
                "World map synchronized before the world-change spawn callback completed; "
                "spawning static world geometry explicitly."
            )
            await self._spawn_current_world_geometry()
        await self.reset_task(first_map=True)

        self._check_status_task = asyncio.create_task(self._check_task_status())

        self.rosparam[bool].set('initialized', True)

    @classmethod
    async def create(cls, *, namespace: str = "task_generator_node", **kwargs):
        self = cls(namespace=namespace, **kwargs)
        await self.setup()

        return self

    async def _set_up_managers(self):
        self._logger.info("Setting up managers")

        self._logger.info("Setting up simulator")
        self._simulator = await SimulatorRegistry.get(
            self.conf.Arena.SIM.value,
            node=self,
            namespace=self._namespace,
        )

        self._logger.info("Setting up human simulator")
        self._human_simulator = await HumanSimulatorRegistry.get(
            self.conf.Arena.HUMAN.value,
            node=self,
            namespace=self._namespace,
            simulator=self._simulator,
        )

        self._logger.info("Setting up environment manager")
        self._environment_manager = EnvironmentManager(
            node=self,
            namespace=self._namespace,
            simulator=self._simulator,
            entity_manager=self._human_simulator,
        )

        self._logger.info("Setting up world manager")
        self._world_manager = WorldManager(
            node=self,
            environment_manager=self._environment_manager
        )

        async def world_change_cb():
            await self._spawn_current_world_geometry()

        self._world_manager.on_world_change(world_change_cb)
        await self._world_manager.start()

        self._logger.info("Setting up robots manager")
        self._robots_manager = RobotsManagerROS(
            node=self,
            environment_manager=self._environment_manager
        )

        self._logger.info("Managers set up")

    async def _spawn_current_world_geometry(self):
        self.get_logger().info("Spawning static world geometry into simulator")
        await self._environment_manager.reset(ObstacleLayer.WORLD)
        await self._environment_manager.spawn_world_obstacles(self._world_manager.world)
        self._world_geometry_spawned = True

    # RUNTIME
    async def _reset_task_unlocked(self, **kwargs):
        self._start_time = self.sim_time

        await self._simulator.before_reset_task()

        self.get_logger().info("resetting")

        await self._task.reset(**kwargs)

        self._pub_task_reset.publish(Int16(data=self._number_of_resets))

        # Publish instruction after reset so downstream consumers can latch it.
        instruction = self._vln_instruction
        if self._vln_instruction_file:
            try:
                with open(self._vln_instruction_file, 'r', encoding='utf-8') as f:
                    instruction = f.read().strip() or instruction
            except Exception as e:
                self.get_logger().warn(f"Failed to read vln_instruction_file='{self._vln_instruction_file}': {e}")

        self._pub_vln_instruction.publish(String(data=instruction))

        self._number_of_resets += 1

        await self._simulator.after_reset_task()

        self.get_logger().warn("=============")
        self.get_logger().warn("Task Reset!")
        self.get_logger().warn("=============")

    async def reset_task(self, **kwargs):
        async with self._reset_lock:
            await self._reset_task_unlocked(**kwargs)

    async def _check_task_status(self, *args, **kwargs):
        del args, kwargs
        if self._train_mode or not self._auto_reset:
            self.get_logger().info(
                "Auto-reset disabled (train_mode=%s, auto_reset=%s). "
                "Task resets are driven externally via the reset_task service.",
                self._train_mode, self._auto_reset,
            )
            return
        try:
            while True:
                await asyncio.sleep(0.5)
                should_reset = False
                async with self._reset_lock:
                    if await self._task.is_done:
                        self._completed_episodes += 1
                        self._send_end_message_on_end()

                        if self.conf.General.DESIRED_EPISODES.value >= 0 and \
                                self._completed_episodes >= self.conf.General.DESIRED_EPISODES.value:
                            continue

                        should_reset = True
                    if should_reset:
                        await self._reset_task_unlocked()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.get_logger().error(f"Error in task status check: {e}\n{traceback.format_exc()}")
            raise

    def _send_end_message_on_end(self):
        if self.conf.General.DESIRED_EPISODES.value < 0 or self._completed_episodes < self.conf.General.DESIRED_EPISODES.value:
            return
        if self._finished_published:
            return

        self._finished_published = True

        self.get_logger().warn(
            f"All {int(self.conf.General.DESIRED_EPISODES.value)} tasks completed. Publishing finished message.")
        self._pub_finished.publish(Empty())

        # Delay shutdown to allow data saving to complete
        def delayed_shutdown():
            self.get_logger().info("Shutting down after data save delay...")
            rclpy.shutdown()

        self.create_timer(10.0, delayed_shutdown)

    # SERVICES
    async def _cb_reset_task(
        self,
        request: std_srvs.Empty.Request,
        response: std_srvs.Empty.Response
    ):
        self.get_logger().debug("Task Generator received task-reset request!")
        await self.reset_task()
        return response

    async def _cb_pause_simulation(
        self,
        request: std_srvs.SetBool.Request,
        response: std_srvs.SetBool.Response,
    ):
        """Pause (request.data=True) or unpause (request.data=False) the simulator."""
        if request.data:
            result = await self._simulator.pause_simulation()
            response.message = "paused"
        else:
            result = await self._simulator.unpause_simulation()
            response.message = "unpaused"
        response.success = bool(result)
        return response

    async def _cb_get_configs_environments(
        self,
        request: task_generator_msgs.srv.GetEnvironments.Request,
        response: task_generator_msgs.srv.GetEnvironments.Response,
    ):
        response.environments = list(identifier_to_available(arena_simulation_setup.tree.configs.environment.EnvironmentIdentifier))
        return response

    async def _cb_get_configs_parametrized(
        self,
        request: task_generator_msgs.srv.GetParametrizeds.Request,
        response: task_generator_msgs.srv.GetParametrizeds.Response,
    ):
        response.parametrizeds = list(identifier_to_available(arena_simulation_setup.tree.configs.parametrized.ParametrizedIdentifier))
        return response

    async def _cb_get_obstacles(
        self,
        request: task_generator_msgs.srv.GetObstacles.Request,
        response: task_generator_msgs.srv.GetObstacles.Response,
    ):
        response.models_static_obstacles = list(identifier_to_available(arena_simulation_setup.tree.assets.Object.ObjectIdentifier, network=True))
        response.models_dynamic_obstacles = list(identifier_to_available(arena_simulation_setup.tree.assets.Pedestrian.PedestrianIdentifier, network=True))

        return response

    async def _cb_get_scenarios(
        self,
        request: task_generator_msgs.srv.GetScenarios.Request,
        response: task_generator_msgs.srv.GetScenarios.Response,
    ):
        response.scenarios = list(identifier_to_available(World.WorldIdentifier(request.world or self._world_manager.world_name).resolve_sync().scenario))
        return response

    async def _cb_get_worlds(
        self,
        request: task_generator_msgs.srv.GetWorlds.Request,
        response: task_generator_msgs.srv.GetWorlds.Response,
    ):
        response.worlds = list(identifier_to_available(World.WorldIdentifier))
        return response

    async def _cb_get_robots(
        self,
        request: task_generator_msgs.srv.GetRobots.Request,
        response: task_generator_msgs.srv.GetRobots.Response,
    ):
        response.robots = list(identifier_to_available(arena_robots.Robot.RobotIdentifier))
        return response

    async def _cb_wait_for_world(
        self,
        request: EmptySrv.Request,
        response: EmptySrv.Response,
    ):
        await self._world_manager.sync()
        return response

    async def _set_up_services(self):
        self._logger.info("Setting up services")

        # Services
        self.create_service(
            EmptySrv,
            self.service_namespace('reset_task'),
            self._cb_reset_task,
        )

        self.create_service(
            std_srvs.SetBool,
            self.service_namespace('pause_simulation'),
            self._cb_pause_simulation,
        )

        self.create_service(
            task_generator_msgs.srv.GetEnvironments,
            self.service_namespace('get_environments'),
            self._cb_get_configs_environments,
        )

        self.create_service(
            task_generator_msgs.srv.GetParametrizeds,
            self.service_namespace('get_parametrizeds'),
            self._cb_get_configs_parametrized,
        )

        self.create_service(
            task_generator_msgs.srv.GetObstacles,
            self.service_namespace('get_obstacles'),
            self._cb_get_obstacles,
        )

        self.create_service(
            task_generator_msgs.srv.GetScenarios,
            self.service_namespace('get_scenarios'),
            self._cb_get_scenarios,
        )

        self.create_service(
            task_generator_msgs.srv.GetRobots,
            self.service_namespace('get_robots'),
            self._cb_get_robots,
        )

        self.create_service(
            task_generator_msgs.srv.GetWorlds,
            self.service_namespace('get_worlds'),
            self._cb_get_worlds,
        )

        self.create_service(
            EmptySrv,
            self.service_namespace('wait_for_world'),
            self._cb_wait_for_world,
        )

        self._logger.info("Services set up")
