import asyncio
import json
import traceback
import time

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
from hunav_msgs.msg import Agents
from std_msgs.msg import Empty, Int16, String
from std_srvs.srv import Empty as EmptySrv

from task_generator.constants import Constants
from task_generator.constants.runtime import Configuration
from task_generator.episode_barrier import (
    BarrierCondition,
    BarrierReport,
    EpisodeStartBarrierTimeout,
    PedestrianEpisodeClock,
    await_episode_start_barrier,
)
from task_generator.latched_stage_topic import (
    EVAL_READY_STATUS_TOPIC,
    EVAL_READY_TOPIC,
    LatchedStageTopicContract,
)
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


#: Relative topic on which the task generator publishes the episode time origin.
#: The eval video recorder and any other consumer that needs a single, shared
#: ``t = 0`` subscribes to ``<task_generator_namespace>/<EPISODE_START_TOPIC>``.
EPISODE_START_TOPIC = 'episode_start'

#: Relative topic on which the eval video recorder reports that every enabled
#: video stream is past its warm-up gate and would be writing frames.
VIDEO_STREAMS_READY_TOPIC = 'video_streams_ready'


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

    # How often the world-geometry wait wakes up to re-check its producers, and
    # how often it says so.  The wait used to be a single opaque asyncio.wait_for
    # of the whole budget, which is why a startup failure looked like a 600 s hang.
    _WORLD_GEOMETRY_POLL_INTERVAL_SEC: float = 5.0
    _WORLD_GEOMETRY_PROGRESS_INTERVAL_SEC: float = 30.0

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
        # Static world geometry is the first eval barrier.  Keep it blocked
        # until WorldManager's initial world callback has loaded/spawned the
        # environment into Isaac.  Robot/model/video consumers are only allowed
        # to progress after this event is set, otherwise direct-control VLN can
        # start against an empty stage or contend with the same single-threaded
        # Isaac service loop used for static-world spawning.
        self._world_geometry_ready: asyncio.Event = asyncio.Event()
        self._world_geometry_error: str = ''
        # Distinguishes "the spawn ran and did not finish" from "the spawn never
        # started", which are different failures with different places to look.
        self._world_geometry_spawn_started: bool = False
        self._episode_entities_ready: asyncio.Event = asyncio.Event()
        self._human_states_ready: asyncio.Event = asyncio.Event()
        self._last_human_states_count = 0
        # Episode-start barrier state.  ``_episode_started`` is the public t=0
        # edge: the timeout origin, pedestrian motion, the recorded episode and
        # the model client all key off it, so nothing that constitutes the
        # episode may happen before it.  ``_pedestrian_clock`` is what actually
        # holds HuNav: it is handed to HuNav as the request header stamp, and
        # HuNav derives its integration step from consecutive stamps.
        self._episode_started: asyncio.Event = asyncio.Event()
        self._pedestrian_clock = PedestrianEpisodeClock()
        self._last_barrier_report: BarrierReport | None = None
        self._robot_navigation_ready = False
        self._video_streams_ready_episode: int | None = None
        self._task: Task

        # VLN instruction interface (published per-episode)
        self._vln_instruction = self.rosparam[str].get('vln_instruction', 'navigate')
        self._vln_instruction_file = self.rosparam[str].get('vln_instruction_file', '')
        self._vln_instruction_republish_task: asyncio.Task | None = None
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
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
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

        self._pub_episode_outcome = self.create_publisher(
            String,
            self.service_namespace('episode_outcome'),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # Which stages may ride ``eval_ready`` at all.  See
        # ``latched_stage_topic.py`` for the measured delivery semantics this
        # encodes; it must be set before the first ``_publish_eval_ready`` call.
        self._eval_ready_contract = LatchedStageTopicContract.eval_ready()

        self._pub_eval_ready = self.create_publisher(
            String,
            self.service_namespace(EVAL_READY_TOPIC),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # The full readiness lifecycle, including the stages the two external
        # eval_ready consumers discard.  It exists so that adding a publication
        # can never again starve them: this topic is depth-1 latched and both
        # consumers subscribe at depth 1, and *only the final sample on such a
        # topic is reliably obtainable* (measured: 0/20 deliveries of an earlier
        # sample once anything follows it, and raising the publisher's depth to
        # 10 or 50 does not change that, because the binding constraint is the
        # subscriber's own KEEP_LAST(1) cache).  So eval_ready carries only
        # stages every consumer accepts, and everything else is published here.
        self._pub_eval_ready_status = self.create_publisher(
            String,
            self.service_namespace(EVAL_READY_STATUS_TOPIC),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # The explicit episode time origin.  Published exactly once per episode,
        # at the barrier, after task_reset let the recorders open their writers
        # and after every stream reported that it is past its warm-up gate.
        # Consumers treat the frame that carries this edge as their t=0, so all
        # four video streams and the timeout share one origin.  Latched, because
        # a recorder must be able to learn about it even if DDS discovery of the
        # topic completes a moment late.
        self._pub_episode_start = self.create_publisher(
            Int16,
            self.service_namespace(EPISODE_START_TOPIC),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self._publish_eval_ready('startup', False, reason='task_generator_constructed')

        self.create_subscription(
            Agents,
            self.service_namespace('human_states'),
            self._human_states_callback,
            10,
        )

        # Video-stream readiness, reported by the eval video recorder once every
        # enabled stream is past its warm-up/discard/content gates.  The
        # recorder creates its publisher at construction, so
        # ``count_publishers`` on this topic is an observable-state answer to
        # "is a recorder attached at all", rather than a flag someone has to
        # remember to set.
        self._video_streams_ready_topic = str(self.service_namespace(VIDEO_STREAMS_READY_TOPIC))
        self.create_subscription(
            Int16,
            self._video_streams_ready_topic,
            self._video_streams_ready_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._check_status_task: asyncio.Task

    def _human_states_callback(self, msg: Agents) -> None:
        agent_count = len(getattr(msg, 'agents', []) or [])
        self._last_human_states_count = agent_count
        if agent_count > 0:
            self._human_states_ready.set()

    def _video_streams_ready_callback(self, msg: Int16) -> None:
        self._video_streams_ready_episode = int(getattr(msg, 'data', -1))

    # EPISODE-START BARRIER

    @property
    def episode_motion_released(self) -> bool:
        """Whether the episode has started, i.e. whether agents may move.

        Read by :class:`~task_generator.simulators.human.hunav.hunav.HunavHumanSimulator`
        to decide whether HuNav may advance its behaviour trees.  Kept as a
        public property so a rename shows up as a test failure rather than as a
        gate that silently stops gating.
        """
        return self._episode_started.is_set()

    @property
    def pedestrian_episode_clock(self) -> PedestrianEpisodeClock:
        """The gated clock handed to HuNav as its request header stamp."""
        return self._pedestrian_clock

    def _video_recorder_attached(self) -> bool:
        """Whether an eval video recorder exists, from observable ROS state.

        Uses ``count_publishers`` rather than a configuration flag so that a run
        launched without a recorder is not blocked, and a run launched *with* one
        cannot accidentally skip the stream-readiness condition because somebody
        forgot to set an environment variable.
        """
        try:
            return int(self.count_publishers(self._video_streams_ready_topic)) > 0
        except Exception:
            return False

    def _dual_vln_command_services(self) -> list[str]:
        """Model command services this run expects, from the robot managers."""
        services: list[str] = []
        for robot_manager in getattr(self._robots_manager, 'robots', {}).values():
            resolver = getattr(robot_manager, '_dual_vln_command_service_name', None)
            if not callable(resolver):
                continue
            try:
                if not bool(robot_manager._is_dual_vln_robot()):
                    continue
                name = str(resolver() or '')
            except Exception:
                continue
            if name:
                services.append(name)
        return services

    def _model_channels(self) -> dict[str, list[str]]:
        """The observable channels through which a model can report itself live.

        Two exist and which one is populated depends on the eval mode, so the
        barrier accepts either rather than hard-coding one:

        * ``command_services`` -- the Arena/external ``get_command`` service.
          Present in the wrapper mode, where ``robot_manager`` also waits for it
          before publishing the goal.
        * ``status_topics`` -- the InternNav ``.../internnav/status`` topic.  This
          is the channel the official-client (direct ``cmd_vel``) mode uses, and
          the one ``robot_manager`` already subscribes to.
        """
        channels: dict[str, list[str]] = {'command_services': [], 'status_topics': []}
        channels['command_services'] = self._dual_vln_command_services()
        for robot_manager in getattr(self._robots_manager, 'robots', {}).values():
            try:
                if not bool(robot_manager._is_dual_vln_robot()):
                    continue
            except Exception:
                continue
            topic = str(getattr(robot_manager, '_dual_vln_status_topic', '') or '')
            if topic:
                channels['status_topics'].append(topic)
        return channels

    def _model_ready_state(self, channels: dict[str, list[str]]) -> dict[str, object]:
        """Observable state of every model channel, never a call that can return None.

        A service call that fails yields nothing distinguishable from a model that
        is merely slow, so readiness is derived from the ROS graph (is the service
        advertised?) and from received status samples (did the model speak?).
        """
        try:
            advertised = {name for name, _types in self.get_service_names_and_types()}
        except Exception:
            advertised = set()
        services_seen = [name for name in channels['command_services'] if name in advertised]
        status_seen = []
        for robot_manager in getattr(self._robots_manager, 'robots', {}).values():
            topic = str(getattr(robot_manager, '_dual_vln_status_topic', '') or '')
            if not topic:
                continue
            if float(getattr(robot_manager, '_dual_vln_status_wall_time', 0.0) or 0.0) > 0.0:
                status_seen.append(topic)
        return {
            'command_services_advertised': services_seen,
            'status_topics_with_sample': status_seen,
            'ready': bool(services_seen or status_seen),
        }

    def _episode_start_barrier_conditions(self) -> list[BarrierCondition]:
        """The conditions that define "the episode may now begin".

        Every element is here because something that constitutes the episode
        depends on it:

        * ``world_geometry_loaded`` -- the scene USD is composed and spawned.
          Without it the first observation is of an empty or partial stage.
        * ``robot_spawned_and_reset`` -- the robot reached its start pose and its
          navigation goal is accepted, so the recorded trajectory starts at the
          episode's start pose rather than mid-teleport.
        * ``pedestrians_spawned`` -- HuNav reported at least one agent on
          ``human_states``.  Releasing motion before the agents exist would let
          the barrier pass an episode with no pedestrians in it.
        * ``video_streams_ready`` -- every enabled video stream is past its
          warm-up/discard/content gate.  This is the condition whose absence
          caused the measured defect: ``sim_top_down.mp4`` discards ~20 s of
          unsettled frames, so with pedestrians walking from ``task_reset`` the
          review video's frame 0 began after the walk had already finished.
        * ``model_reachable`` -- the policy that is being graded has announced
          itself on at least one of its two observable channels, so episode time
          does not start during the model's cold start (measured: the first HTTP
          request left 0.06 s after ego frame 0, median completion 1.23 s).
        """
        human_simulator = self.conf.Arena.HUMAN.value
        pedestrians_expected = human_simulator in (
            Constants.HumanSimulator.HUNAV,
            Constants.HumanSimulator.GRSCENES_REPLAY,
        )
        recorder_attached = self._video_recorder_attached()
        channels = self._model_channels()
        model_configured = bool(channels['command_services'] or channels['status_topics'])
        require_model = model_configured and bool(
            self.rosparam[bool].get('episode_start_require_model_ready', True)
        )

        return [
            BarrierCondition(
                name='world_geometry_loaded',
                check=self._world_geometry_ready.is_set,
                detail=lambda: f'ready={self._world_geometry_ready.is_set()} error={self._world_geometry_error!r}',
            ),
            BarrierCondition(
                name='robot_spawned_and_reset',
                check=lambda: bool(self._robot_navigation_ready),
                detail=lambda: f'navigation_ready={bool(self._robot_navigation_ready)}',
            ),
            BarrierCondition(
                name='pedestrians_spawned',
                check=self._human_states_ready.is_set,
                required=pedestrians_expected,
                skip_reason=f'human_simulator={getattr(human_simulator, "value", human_simulator)}',
                detail=lambda: f'human_states_agents={self._last_human_states_count}',
            ),
            BarrierCondition(
                name='video_streams_ready',
                check=lambda: self._video_streams_ready_episode == self._number_of_resets,
                required=recorder_attached,
                skip_reason=f'no_publisher_on={self._video_streams_ready_topic}',
                detail=lambda: (
                    f'ready_episode={self._video_streams_ready_episode} '
                    f'expected_episode={self._number_of_resets} '
                    f'publishers={self._video_recorder_attached()}'
                ),
            ),
            BarrierCondition(
                name='model_reachable',
                check=lambda: bool(self._model_ready_state(channels)['ready']),
                required=require_model,
                skip_reason=(
                    'no_dual_vln_robot' if not model_configured
                    else 'episode_start_require_model_ready=false'
                ),
                detail=lambda: f'channels={channels} observed={self._model_ready_state(channels)}',
            ),
        ]

    async def _await_video_recorder_discovery(self) -> bool:
        """Give DDS a bounded moment to expose an attached video recorder.

        Whether the stream-readiness condition is *required* is decided once, from
        ``count_publishers``.  Deciding that while discovery is still in flight
        would silently drop the condition -- the precise failure mode this project
        keeps re-encountering -- so poll briefly and log the answer either way.
        The wait is only paid by runs that genuinely have no recorder.
        """
        grace_s = max(0.0, self.rosparam[float].get('episode_start_recorder_discovery_sec', 5.0))
        deadline = time.monotonic() + grace_s
        while True:
            if self._video_recorder_attached():
                # warn, not info: production evals run --log-level warn, and this
                # line is Stage-0 evidence that the barrier's decisive condition
                # was actually required rather than silently skipped.  It had to
                # be reconstructed from artifacts in all four validation runs.
                self.get_logger().warn(
                    f'Video recorder detected on {self._video_streams_ready_topic}; '
                    'stream readiness is a required episode-start condition.'
                )
                return True
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.25)
        self.get_logger().warn(
            f'No video recorder publisher on {self._video_streams_ready_topic} after {grace_s:.1f}s; '
            'the episode-start barrier will NOT require video-stream readiness for this run. '
            'Every video stream then starts at task_reset, as it did before the barrier existed.'
        )
        return False

    async def _await_episode_start_barrier(self) -> BarrierReport:
        """Block until the episode may begin, or fail loudly.

        Returns:
            The passing barrier report.

        Raises:
            EpisodeStartBarrierTimeout: If a required condition never held.  The
                episode origin is never declared in that case; proceeding would
                stamp every artifact with a ``t = 0`` that was never reached.
        """
        timeout_s = max(0.0, self.rosparam[float].get('episode_start_barrier_timeout_sec', 300.0))
        await self._await_video_recorder_discovery()
        conditions = self._episode_start_barrier_conditions()
        self._publish_eval_ready(
            'episode_start',
            False,
            reason='barrier_waiting',
            required=[condition.name for condition in conditions if condition.required],
            skipped={
                condition.name: condition.skip_reason
                for condition in conditions
                if not condition.required
            },
            timeout_sec=timeout_s,
        )
        # warn, not info: this is the Stage-0 record of WHICH conditions the
        # barrier required for this episode.  See the note above on --log-level.
        self.get_logger().warn(
            'Waiting for the episode-start barrier before releasing pedestrian motion, the '
            'timeout origin and the model: required='
            f'{[condition.name for condition in conditions if condition.required]} '
            f'not_required={{{", ".join(f"{c.name}:{c.skip_reason}" for c in conditions if not c.required)}}} '
            f'timeout={timeout_s:.1f}s'
        )

        def _log_progress(report: BarrierReport) -> None:
            self.get_logger().info(
                f'Episode-start barrier progress after {report.waited_sec:.1f}s: '
                f'satisfied={report.satisfied} still_waiting_on={report.unsatisfied}'
            )

        try:
            report = await await_episode_start_barrier(
                conditions,
                timeout_sec=timeout_s,
                on_progress=_log_progress,
            )
        except EpisodeStartBarrierTimeout as exc:
            self._last_barrier_report = exc.report
            self.get_logger().error(exc.report.failure_message())
            self._publish_eval_ready(
                'episode_start',
                False,
                reason='barrier_timeout',
                barrier=exc.report.to_dict(),
            )
            raise

        self._last_barrier_report = report
        self.get_logger().warn(
            f'Episode-start barrier passed after {report.waited_sec:.1f}s; '
            f'satisfied={report.satisfied}'
        )
        return report

    def _release_episode_start(self, report: BarrierReport) -> None:
        """Declare the episode time origin and release everything that keys off it."""
        self._episode_started.set()
        self._pedestrian_clock.release()

        mark_episode_started = getattr(self._task, 'mark_episode_started', None)
        if callable(mark_episode_started):
            mark_episode_started()

        self._pub_episode_start.publish(Int16(data=self._number_of_resets))
        self._publish_eval_ready(
            'episode_start',
            True,
            reason='episode_start_published',
            barrier=report.to_dict(),
            episode_origin_sim_time_sec=self._time_to_seconds(getattr(self, 'sim_time', None)),
            episode_origin_wall_time=time.time(),
            pedestrian_clock_sec=self._pedestrian_clock.value,
        )
        # The external model client and the timing manager are released here, at
        # t=0, and not at task_reset.  Before the barrier existed those were the
        # same instant, so this is the faithful translation rather than a change
        # of contract: the client must not be able to command the robot during
        # the recorders' warm-up, which on a second episode it otherwise could
        # (its instruction is latched from the previous one).
        self._publish_eval_ready('episode', True, reason='episode_start_published')
        self.get_logger().warn(
            '============= EPISODE START (t=0) ============= '
            f'episode={self._number_of_resets} '
            f'barrier_wait_sec={report.waited_sec:.1f} '
            f'sim_time_sec={self._time_to_seconds(getattr(self, "sim_time", None))}'
        )

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

        await self._world_manager.sync()
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

        initial_geometry_timeout_s = float(
            self.rosparam[float].get('world_geometry_ready_timeout_sec', 600.0)
        )
        if not await self.wait_for_world_geometry_ready(timeout_s=initial_geometry_timeout_s):
            raise RuntimeError(
                'Static world geometry never became ready before robot manager '
                f'setup: {self._world_geometry_wait_diagnosis(initial_geometry_timeout_s)}'
            )

        self._logger.info("Setting up robots manager")
        self._robots_manager = RobotsManagerROS(
            node=self,
            environment_manager=self._environment_manager
        )

        self._logger.info("Managers set up")

    async def _spawn_current_world_geometry(self):
        self.get_logger().info("Spawning static world geometry into simulator")
        self._world_geometry_ready.clear()
        self._world_geometry_error = ''
        self._world_geometry_spawn_started = True
        self._publish_eval_ready('world_geometry', False, reason='spawn_started')
        await self._environment_manager.reset(ObstacleLayer.WORLD)
        success = await self._environment_manager.spawn_world_obstacles(self._world_manager.world)
        if not success:
            self._world_geometry_error = 'world_geometry_spawn_failed'
            self._publish_eval_ready('world_geometry', False, reason=self._world_geometry_error)
            raise RuntimeError(
                'World geometry failed to load/spawn completely; refusing to release task_reset or VLN instruction.'
            )
        self._world_geometry_spawned = True
        self._world_geometry_ready.set()
        self._publish_eval_ready('world_geometry', True, reason='spawn_complete')

    async def wait_for_world_geometry_ready(self, timeout_s: float) -> bool:
        """Wait for static world geometry, reporting progress and producer death.

        Waits in slices instead of one opaque ``asyncio.wait_for`` so that it can
        (a) say something while it waits and (b) notice that the thing it is
        waiting for can no longer arrive.  The producer runs concurrently on an
        executor thread, so a one-shot check *before* waiting -- which is all this
        used to do -- is guaranteed to be too early to see a failure.

        Args:
            timeout_s: Overall budget in seconds.

        Returns:
            bool: True if geometry reported ready within the budget.
        """
        if self._world_geometry_ready.is_set():
            return True

        budget = max(float(timeout_s), 0.0)
        started = time.monotonic()
        next_progress_at = self._WORLD_GEOMETRY_PROGRESS_INTERVAL_SEC

        while True:
            # Checked on EVERY slice, not once up front: either of these can be
            # populated by another thread after the wait begins, and each of them
            # means "ready" will never arrive, so waiting out the rest of the
            # budget only delays the report.
            failure, detail = self._world_geometry_producer_failure()
            if failure is not None:
                message = f'World geometry will never become ready: {failure}'
                if detail:
                    message += f'\n{detail}'
                self.get_logger().error(message)
                return False

            remaining = budget - (time.monotonic() - started)
            if remaining <= 0.0:
                return False

            slice_s = min(self._WORLD_GEOMETRY_POLL_INTERVAL_SEC, remaining)
            try:
                await asyncio.wait_for(
                    self._world_geometry_ready.wait(),
                    timeout=slice_s,
                )
                return True
            except asyncio.TimeoutError:
                pass

            elapsed = time.monotonic() - started
            if elapsed >= next_progress_at:
                next_progress_at = elapsed + self._WORLD_GEOMETRY_PROGRESS_INTERVAL_SEC
                self.get_logger().warn(
                    'Still waiting for static world geometry: '
                    f'elapsed={elapsed:.1f}s remaining={max(budget - elapsed, 0.0):.1f}s '
                    f'spawn_started={self._world_geometry_spawn_started} '
                    f'world={getattr(getattr(self, "_world_manager", None), "world_name", "?")!r}'
                )

    def _world_geometry_producer_failure(self) -> tuple[str | None, str]:
        """Why world geometry can no longer become ready.

        A pure query: it logs nothing, so it can be called from both the wait and
        the raise site without printing the same traceback twice.

        Two independent producers can die without setting the event, and neither
        used to be visible to the waiter:

        * ``_spawn_current_world_geometry`` records ``_world_geometry_error``;
        * the ``world`` parameter callback -- which is what *starts* the spawn --
          runs as an rclpy Task whose exception nothing retrieves, so a failure
          there used to leave no trace at all until it was reported at its source.

        Returns:
            tuple: ``(summary, detail)``.  ``summary`` is ``None`` when no producer
            has reported a failure; ``detail`` is a traceback when one is
            available and ``''`` otherwise.
        """

        if self._world_geometry_error:
            return f'geometry spawn reported {self._world_geometry_error!r}', ''

        for failure in getattr(self, 'param_callback_failures', ()):
            summary = failure.summary()
            if not self._world_geometry_spawn_started:
                summary += (
                    ' -- static world geometry spawn never started, so the '
                    'geometry timeout is a consequence of this, not a slow load'
                )
            return summary, failure.traceback_text

        return None, ''

    def _world_geometry_wait_diagnosis(self, timeout_s: float) -> str:
        """Human-readable cause for a failed world-geometry wait."""

        failure, _detail = self._world_geometry_producer_failure()
        if failure is not None:
            return failure
        if self._world_geometry_spawn_started:
            return (
                f'geometry spawn started but did not complete within {timeout_s:.1f}s'
            )
        return (
            f'the world load never reached geometry spawn within {timeout_s:.1f}s '
            '(no spawn_started), and no producer reported an error -- the world '
            'load itself is where to look, not geometry spawn'
        )

    def _publish_eval_ready(self, stage: str, ready: bool, **details) -> None:
        """Publish one readiness sample, routed by the topic's stage contract.

        Both external consumers of ``eval_ready`` filter on ``stage`` and both
        subscribe ``depth=1``, so only the *final* sample on that topic is
        reliably obtainable and any later sample their filter discards starves
        them.  That is what broke robot control in 4/4 runs when the episode-start
        barrier began publishing ``stage='episode_start'`` immediately after the
        ``stage='episode', ready=True`` sample the consumers wait for.

        The routing rule removes the ordering hazard rather than working around
        it: a stage rides ``eval_ready`` only if *every* registered consumer
        filter accepts it, so every sample on that topic is one they accept,
        whatever order callers publish in.  Every sample -- contracted or not --
        is also published on ``eval_ready_status``, so nothing is lost and the
        full lifecycle stays observable with ``ros2 topic echo``.

        Args:
            stage: Lifecycle stage of this sample.
            ready: Whether the stage is satisfied.
            **details: Stage-specific payload, recorded under ``details``.
        """
        try:
            msg = String()
            msg.data = json.dumps(
                {
                    'stage': stage,
                    'ready': bool(ready),
                    'episode': self._number_of_resets,
                    'world_geometry_ready': self._world_geometry_ready.is_set(),
                    'episode_entities_ready': self._episode_entities_ready.is_set(),
                    'details': details,
                },
                ensure_ascii=False,
            )
            self._pub_eval_ready_status.publish(msg)
            if self._eval_ready_contract.carries(stage):
                self._pub_eval_ready.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f'Failed to publish eval_ready status: {exc}')

    async def wait_for_episode_entities_ready(self, timeout_s: float) -> bool:
        if self._episode_entities_ready.is_set():
            return True
        try:
            await asyncio.wait_for(self._episode_entities_ready.wait(), timeout=max(float(timeout_s), 0.0))
            return True
        except asyncio.TimeoutError:
            return False

    async def _wait_for_human_states_ready_if_required(self) -> None:
        require_human_states_ready = self.rosparam[bool].get('require_human_states_ready', False)
        timeout_s = max(0.0, self.rosparam[float].get('human_states_ready_timeout_sec', 10.0))
        if not require_human_states_ready or timeout_s <= 0.0:
            return

        if self.conf.Arena.HUMAN.value not in (
            Constants.HumanSimulator.HUNAV,
            Constants.HumanSimulator.GRSCENES_REPLAY,
        ):
            return

        if self._human_states_ready.is_set():
            return

        human_label = self.conf.Arena.HUMAN.value.value
        self.get_logger().info(
            f"Waiting up to {timeout_s:.1f}s for non-empty {human_label} human_states before releasing episode start"
        )
        try:
            await asyncio.wait_for(self._human_states_ready.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self.get_logger().warn(
                f"Timed out waiting for non-empty {human_label} human_states before episode start; "
                "continuing to avoid hanging the eval."
            )

    def _current_vln_instruction(self) -> str:
        instruction = self._vln_instruction
        if self._vln_instruction_file:
            try:
                with open(self._vln_instruction_file, 'r', encoding='utf-8') as f:
                    instruction = f.read().strip() or instruction
            except Exception as e:
                self.get_logger().warn(f"Failed to read vln_instruction_file='{self._vln_instruction_file}': {e}")
        return instruction

    async def _republish_vln_instruction_window(self, instruction: str, reset_index: int) -> None:
        """Bridge discovery/QoS races for late-starting external VLN clients.

        The official InternNav ROS2 HTTP client can be started outside this
        launch graph.  Publishing the instruction once at episode release is
        not sufficient if DDS discovery completes just after that edge or if a
        consumer uses volatile QoS.  Keep re-publishing the same per-episode
        instruction for a short bounded window after the episode-start barrier,
        without moving the public episode boundary away from that barrier.
        """
        for _ in range(8):
            await asyncio.sleep(0.25)
            if reset_index + 1 != self._number_of_resets:
                return
            self._pub_vln_instruction.publish(String(data=instruction))

    def _publish_vln_instruction_for_episode(self, reset_index: int) -> None:
        instruction = self._current_vln_instruction()
        self._pub_vln_instruction.publish(String(data=instruction))
        if self._vln_instruction_republish_task is not None:
            self._vln_instruction_republish_task.cancel()
        self._vln_instruction_republish_task = asyncio.create_task(
            self._republish_vln_instruction_window(instruction, reset_index)
        )

    # RUNTIME
    async def _reset_task_unlocked(self, **kwargs):
        self._start_time = self.sim_time
        self._episode_entities_ready.clear()
        self._human_states_ready.clear()
        self._last_human_states_count = 0
        # Re-arm the barrier for this episode.  Pedestrians are held from here
        # until the barrier passes, so no part of their route can be consumed
        # while the scene loads, the robot is teleported and the video streams
        # converge.
        self._episode_started.clear()
        self._pedestrian_clock.hold()
        self._robot_navigation_ready = False
        self._video_streams_ready_episode = None
        self._last_barrier_report = None
        self._publish_eval_ready('episode', False, reason='reset_started')

        await self._simulator.before_reset_task()

        self.get_logger().info("resetting")

        await self._task.reset(**kwargs)

        await self._simulator.after_reset_task()

        await self._wait_for_human_states_ready_if_required()

        wait_navigation_ready = getattr(self._task, 'wait_navigation_ready', None)
        if callable(wait_navigation_ready):
            await wait_navigation_ready(
                timeout_s=max(0.0, self.rosparam[float].get('episode_navigation_ready_timeout_sec', 600.0))
            )
        self._robot_navigation_ready = True

        episode_start_delay_sec = max(0.0, self.rosparam[float].get('episode_start_delay_sec', 0.0))
        if episode_start_delay_sec > 0.0:
            self.get_logger().info(
                f"Delaying episode start by {episode_start_delay_sec:.1f}s after reset readiness"
            )
            await asyncio.sleep(episode_start_delay_sec)

        self._episode_entities_ready.set()
        # task_reset is the *stream-open* edge, not the episode origin.  The
        # recorders need it before they can open their writers and start running
        # their warm-up gates, and the barrier below waits for the result.
        #
        # No ``eval_ready(stage='episode', ready=True)`` here: that sample is the
        # release signal for the external model client and the timing manager, and
        # it belongs at t=0, which is the barrier below.  Publishing it here made
        # it the *second-to-last* sample on a depth-1 latched topic and both
        # consumers were starved by the barrier's own status message -- 4/4 runs
        # with zero control ticks.  It is now published in
        # ``_release_episode_start``.
        self._pub_task_reset.publish(Int16(data=self._number_of_resets))

        # The episode origin.  Everything that constitutes the episode -- the
        # timeout clock, HuNav commanding pedestrian motion, the recorded video
        # frames and the model client -- is released here and not before, so the
        # pedestrians' dynamic window falls inside the observed episode instead
        # of inside the recorders' warm-up.
        report = await self._await_episode_start_barrier()
        self._release_episode_start(report)

        # Publish instruction only after the episode-start barrier so the model's
        # cold start is charged to startup rather than to the episode's first
        # seconds.  Re-publish for a short bounded window to let external
        # InternNav clients that discover the topic slightly late still
        # synchronize before first inference.
        self._publish_vln_instruction_for_episode(self._number_of_resets)

        self._number_of_resets += 1

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
                        done_reason = str(getattr(self._task, 'last_done_reason', 'unknown') or 'unknown')
                        self._completed_episodes += 1
                        self._publish_episode_outcome(done_reason)
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

    def _publish_episode_outcome(self, reason: str) -> None:
        desired = int(self.conf.General.DESIRED_EPISODES.value)
        episode_index = max(int(self._completed_episodes) - 1, 0)
        sim_time_sec = self._time_to_seconds(getattr(self, 'sim_time', None))
        if sim_time_sec is None:
            sim_time_sec = self._time_to_seconds(getattr(self, 'time', None))
        payload = {
            'episode_index': episode_index,
            'completed_episodes': int(self._completed_episodes),
            'desired_episodes': desired,
            'reason': str(reason or 'unknown'),
            'finished': desired >= 0 and int(self._completed_episodes) >= desired,
            'sim_time_sec': sim_time_sec,
            'wall_time': time.time(),
        }
        try:
            self._pub_episode_outcome.publish(String(data=json.dumps(payload, sort_keys=True)))
        except Exception as exc:
            self.get_logger().warn(f'Failed to publish episode outcome: {exc}')

    @staticmethod
    def _time_to_seconds(value) -> float | None:
        if value is None:
            return None
        to_seconds = getattr(value, 'to_seconds', None)
        if callable(to_seconds):
            try:
                return float(to_seconds())
            except Exception:
                pass
        nanoseconds = getattr(value, 'nanoseconds', None)
        if isinstance(nanoseconds, (int, float)):
            return float(nanoseconds) / 1e9
        sec = getattr(value, 'sec', None)
        nanosec = getattr(value, 'nanosec', None)
        if isinstance(sec, (int, float)) and isinstance(nanosec, (int, float)):
            return float(sec) + float(nanosec) / 1e9
        clock = getattr(value, 'clock', None)
        if clock is not None:
            return TaskGenerator._time_to_seconds(clock)
        return None

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
