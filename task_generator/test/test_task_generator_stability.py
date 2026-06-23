import asyncio
import importlib.util
import json
import os
import struct
import sys
import time
import types
from types import SimpleNamespace

if importlib.util.find_spec('rclpy') is None:
    def _module(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    rclpy = _module('rclpy')
    rclpy.callback_groups = _module('rclpy.callback_groups')
    rclpy.impl = _module('rclpy.impl')
    rclpy.impl.rcutils_logger = _module('rclpy.impl.rcutils_logger')
    rclpy.node = _module('rclpy.node')
    rclpy.client = _module('rclpy.client')
    rclpy.action = _module('rclpy.action')
    rclpy.logging = _module('rclpy.logging')
    rclpy.publisher = _module('rclpy.publisher')
    rclpy.timer = _module('rclpy.timer')
    rclpy.qos = _module('rclpy.qos')
    rclpy.callback_groups.CallbackGroup = object
    rclpy.callback_groups.ReentrantCallbackGroup = object
    rclpy.node.Node = object
    rclpy.impl.rcutils_logger.RcutilsLogger = object
    rclpy.client.Client = object
    rclpy.publisher.Publisher = object
    rclpy.timer.Timer = object
    rclpy.timer.Rate = object

    class _QoSProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    rclpy.qos.QoSProfile = _QoSProfile
    rclpy.qos.DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL=1)
    rclpy.qos.ReliabilityPolicy = SimpleNamespace(RELIABLE=1)

    action_msgs = _module('action_msgs')
    action_msgs.msg = _module('action_msgs.msg')
    action_msgs.msg.GoalStatus = SimpleNamespace(STATUS_SUCCEEDED=4, STATUS_ABORTED=6, STATUS_CANCELED=5)
    action_msgs.msg.__getattr__ = lambda name: object

    geometry_msgs = _module('geometry_msgs')
    geometry_msgs.msg = _module('geometry_msgs.msg')

    class _Twist:
        def __init__(self):
            self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)

    geometry_msgs.msg.Twist = _Twist
    geometry_msgs.msg.__getattr__ = lambda name: object

    lifecycle_msgs = _module('lifecycle_msgs')
    lifecycle_msgs.msg = _module('lifecycle_msgs.msg')
    lifecycle_msgs.msg.__getattr__ = lambda name: object
    nav_msgs = _module('nav_msgs')
    nav_msgs.msg = _module('nav_msgs.msg')
    nav_msgs.msg.Odometry = object
    nav_msgs.msg.__getattr__ = lambda name: object
    sensor_msgs = _module('sensor_msgs')
    sensor_msgs.msg = _module('sensor_msgs.msg')
    sensor_msgs.msg.__getattr__ = lambda name: object
    std_msgs = _module('std_msgs')
    std_msgs.msg = _module('std_msgs.msg')
    class _String:
        def __init__(self, data=''):
            self.data = data

    std_msgs.msg.String = _String

    ament_index_python = _module('ament_index_python')
    ament_index_python.packages = _module('ament_index_python.packages')
    ament_index_python.packages.get_package_share_directory = lambda package: ''

    arena_rclpy_mixins = _module('arena_rclpy_mixins')
    arena_rclpy_mixins.shared = _module('arena_rclpy_mixins.shared')

    class _Namespace(str):
        def __new__(cls, value=''):
            return str.__new__(cls, value)

        def __call__(self, *path):
            return _Namespace('/'.join(str(part) for part in (self, *path) if str(part)))

        def ParamNamespace(self):
            return self

    arena_rclpy_mixins.shared.Namespace = _Namespace

    arena_robots = _module('arena_robots')
    arena_robots.Robot = _module('arena_robots.Robot')
    arena_robots.Robot.RobotView = object
    nav2_msgs = _module('nav2_msgs')
    nav2_msgs.action = _module('nav2_msgs.action')
    nav2_msgs.action.NavigateToPose = object
    nav2_msgs.srv = _module('nav2_msgs.srv')
    nav2_msgs.srv.ClearCostmapAroundRobot = object
    nav2_msgs.srv.ClearEntireCostmap = object
    rosnav_rl_msgs = _module('rosnav_rl_msgs')
    rosnav_rl_msgs.srv = _module('rosnav_rl_msgs.srv')
    rosnav_rl_msgs.srv.GetCommand = object
    rosgraph_msgs = _module('rosgraph_msgs')
    rosgraph_msgs.msg = _module('rosgraph_msgs.msg')
    rosgraph_msgs.msg.Clock = object

    launch = _module('launch')
    launch.actions = SimpleNamespace(IncludeLaunchDescription=object)
    launch.launch_description_sources = _module('launch.launch_description_sources')
    launch.launch_description_sources.PythonLaunchDescriptionSource = object
    arena_bringup = _module('arena_bringup')
    arena_bringup.extensions = _module('arena_bringup.extensions')
    arena_bringup.extensions.SetGlobalLogLevelAction = object
    arena_simulation_setup = _module('arena_simulation_setup')
    arena_simulation_setup.tree = _module('arena_simulation_setup.tree')
    arena_simulation_setup.tree.Identifier = type('Identifier', (), {'shortname': '', 'listall': classmethod(lambda cls, **kwargs: [])})

    task_generator_utils = _module('task_generator.utils')
    task_generator_utils_arena = _module('task_generator.utils.arena')
    task_generator_utils.arena = task_generator_utils_arena
    task_generator_environment_manager = _module('task_generator.manager.environment_manager')
    task_generator_environment_manager.EnvironmentManager = object
    task_generator_shared = _module('task_generator.shared')
    task_generator_shared.Orientation = object
    task_generator_shared.Pose = object
    task_generator_shared.Position = object
    task_generator_shared.Robot = object
    task_generator_robots_manager_ros = _module('task_generator.manager.robot_manager.robots_manager_ros')
    task_generator_robots_manager_ros.RobotsManager = object
    task_generator_robots_manager_ros.RobotsManagerROS = object
    task_generator_world_manager_ros = _module('task_generator.manager.world_manager.world_manager_ros')
    task_generator_world_manager_ros.WorldManager = object
    task_generator_tasks_task = _module('task_generator.tasks.task')

    class _TaskRegistry:
        @classmethod
        def register_module(cls, *_args, **_kwargs):
            return lambda func: func

        @classmethod
        def register_obstacles(cls, *_args, **_kwargs):
            return lambda func: func

        @classmethod
        def register_robots(cls, *_args, **_kwargs):
            return lambda func: func

    task_generator_tasks_task._TaskRegistry = _TaskRegistry

from task_generator.manager.robot_manager import robot_manager as robot_manager_module
from task_generator.manager.robot_manager.robot_manager import RobotManager
from task_generator.node import TaskGenerator
from task_generator.tasks.robots import TM_Robots


class _RosParamAccessor:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, _type):
        return self

    def get(self, name, default=None):
        return self._values.get(name, default)


class _Logger:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)

    def get_child(self, _name):
        return self


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _robot_manager_stub(*, rosparams=None):
    manager = RobotManager.__new__(RobotManager)
    logger = _Logger()
    manager._NodeInterface__node = SimpleNamespace(rosparam=_RosParamAccessor(rosparams or {}), get_logger=lambda: logger)
    manager._direct_dual_vln_map_bounds = None
    manager._direct_dual_vln_bounds_warning_emitted = False
    manager._direct_dual_vln_last_status_twist = None
    manager._direct_dual_vln_status_bridge_active = True
    manager._cmd_vel_pub = _Publisher()
    manager._is_goal_reached = False
    manager._nav_stop_ticks = 0
    manager._pose = SimpleNamespace(position=SimpleNamespace(x=1.0, y=1.0))
    manager._is_dual_vln_robot = lambda: True
    return manager


def test_compat_rosparam_prefers_non_default_legacy_alias_over_primary_default():
    manager = _robot_manager_stub(rosparams={
        'internnav_timing_mode': 'wall',
        'dual_vln_timing_mode': 'sim_time_realworld',
    })

    assert manager._get_compat_rosparam(
        str,
        'internnav_timing_mode',
        'dual_vln_timing_mode',
        'wall',
        empty_is_missing=True,
    ) == 'sim_time_realworld'


def test_compat_rosparam_keeps_primary_non_default_over_legacy_alias():
    manager = _robot_manager_stub(rosparams={
        'internnav_timing_mode': 'sim_time_realworld',
        'dual_vln_timing_mode': 'wall',
    })

    assert manager._get_compat_rosparam(
        str,
        'internnav_timing_mode',
        'dual_vln_timing_mode',
        'wall',
        empty_is_missing=True,
    ) == 'sim_time_realworld'


def test_eval_env_can_override_timing_mode_after_compat_lookup(monkeypatch):
    manager = _robot_manager_stub(rosparams={
        'internnav_timing_mode': 'wall',
        'dual_vln_timing_mode': 'wall',
    })
    monkeypatch.setenv('ARENA_EVAL_INTERNNAV_TIMING_MODE', 'sim_time_realworld')

    selected = manager._get_compat_rosparam(
        str,
        'internnav_timing_mode',
        'dual_vln_timing_mode',
        'wall',
        empty_is_missing=True,
    )

    assert os.environ.get('ARENA_EVAL_INTERNNAV_TIMING_MODE', selected) == 'sim_time_realworld'


def test_dual_vln_status_sensor_freshness_accepts_only_fresh_rgb_and_depth():
    manager = _robot_manager_stub()

    assert manager._dual_vln_status_has_fresh_sensors({
        'debug': {
            'stale_after_sec': 2.0,
            'sensor_ages_sec': {'rgb': 0.1, 'depth': 0.2, 'odom': 0.1, 'camera_info': None},
        }
    }) is True

    assert manager._dual_vln_status_has_fresh_sensors({
        'debug': {
            'stale_after_sec': 2.0,
            'sensor_ages_sec': {'rgb': 0.1, 'depth': 2.5},
        }
    }) is False

    assert manager._dual_vln_status_has_fresh_sensors({
        'debug': {'stale_after_sec': 2.0, 'sensor_ages_sec': {'rgb': 0.1}}
    }) is False


def test_direct_dual_vln_status_twist_filters_invalid_payloads_and_keeps_valid_command():
    manager = _robot_manager_stub()

    manager._update_direct_dual_vln_status_twist({'status': 'inference_in_progress', 'linear_x': 1.0, 'angular_z': 1.0})
    assert manager._direct_dual_vln_last_status_twist is None

    manager._update_direct_dual_vln_status_twist({'status': 'internnav_command', 'linear_x': 'nan', 'angular_z': 0.0})
    assert manager._direct_dual_vln_last_status_twist is None

    manager._update_direct_dual_vln_status_twist({'status': 'internnav_command', 'linear_x': 0.36, 'angular_z': -0.375})
    assert manager._direct_dual_vln_last_status_twist.linear.x == 0.36
    assert manager._direct_dual_vln_last_status_twist.angular.z == -0.375


def test_direct_dual_vln_status_twist_accepts_heuristic_command_statuses():
    manager = _robot_manager_stub()

    manager._update_direct_dual_vln_status_twist({'status': 'arc_to_goal', 'linear_x': 0.16, 'angular_z': 0.6})
    assert manager._direct_dual_vln_last_status_twist.linear.x == 0.16
    assert manager._direct_dual_vln_last_status_twist.angular.z == 0.6

    manager._publish_direct_dual_vln_status_command({'status': 'arc_to_goal', 'linear_x': 0.16, 'angular_z': 0.6})
    assert len(manager._cmd_vel_pub.messages) == 1
    assert manager._cmd_vel_pub.messages[0].linear.x == 0.16
    assert manager._cmd_vel_pub.messages[0].angular.z == 0.6


def test_direct_dual_vln_bridge_stops_instead_of_publishing_when_pose_leaves_safe_bounds():
    manager = _robot_manager_stub(rosparams={
        'direct_dual_vln_map_bounds_enabled': True,
        'direct_dual_vln_map_bounds_margin_m': 0.5,
    })
    manager._direct_dual_vln_map_bounds = (0.0, 2.0, 0.0, 2.0)
    manager._pose.position.x = 0.25
    manager._pose.position.y = 1.0
    manager._update_direct_dual_vln_status_twist({'status': 'internnav_command', 'linear_x': 0.36, 'angular_z': -0.375})

    manager._publish_direct_dual_vln_status_command({'status': 'internnav_command'})

    assert len(manager._cmd_vel_pub.messages) == 1
    assert manager._cmd_vel_pub.messages[0].linear.x == 0.0
    assert manager._cmd_vel_pub.messages[0].angular.z == 0.0
    assert manager._direct_dual_vln_last_status_twist is None
    assert manager._logger.warnings


def test_direct_dual_vln_bridge_ignores_bounds_guard_by_default():
    manager = _robot_manager_stub(rosparams={})
    manager._direct_dual_vln_map_bounds = (0.0, 2.0, 0.0, 2.0)
    manager._pose.position.x = 0.25
    manager._pose.position.y = 1.0
    manager._update_direct_dual_vln_status_twist({'status': 'internnav_command', 'linear_x': 0.36, 'angular_z': -0.375})

    manager._publish_direct_dual_vln_status_command({'status': 'internnav_command'})

    # Default: bounds guard is disabled, so the command should be published
    assert len(manager._cmd_vel_pub.messages) == 1
    assert manager._cmd_vel_pub.messages[0].linear.x == 0.36
    assert manager._cmd_vel_pub.messages[0].angular.z == -0.375


def test_map_bounds_loader_reads_png_metadata_and_caches_bounds(monkeypatch, tmp_path):
    world_map_dir = tmp_path / 'worlds' / 'unit_world' / 'map'
    world_map_dir.mkdir(parents=True)
    image_path = world_map_dir / 'map.png'
    image_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 8 + struct.pack('>II', 20, 10) + b'\x00' * 16)
    (world_map_dir / 'map.yaml').write_text(
        'image: map.png\nresolution: 0.05\norigin: [-0.25, -0.5, 0]\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(
        robot_manager_module.ament_index_python.packages,
        'get_package_share_directory',
        lambda package: str(tmp_path),
    )
    manager = _robot_manager_stub()
    manager.node.conf = SimpleNamespace(Arena=SimpleNamespace(WORLD=SimpleNamespace(value='unit_world')))

    assert manager._load_direct_dual_vln_map_bounds() == (-0.25, 0.75, -0.5, 0.0)

    image_path.unlink()
    assert manager._load_direct_dual_vln_map_bounds() == (-0.25, 0.75, -0.5, 0.0)


def test_goal_status_callback_ignores_terminal_status_for_stale_goal_uuid():
    manager = _robot_manager_stub()
    manager._active_navigation_goal_uuid = (1, 2, 3)
    manager._goal_tolerance_distance = 0.25
    manager._goal_republish_ticks = 7
    manager._distance_to_goal = lambda: 0.0
    bridge_stops = []
    manager._stop_direct_dual_vln_command_bridge = lambda: bridge_stops.append(True)

    stale_status = SimpleNamespace(
        goal_info=SimpleNamespace(goal_id=SimpleNamespace(uuid=[9, 9, 9])),
        status=robot_manager_module.action_msgs.msg.GoalStatus.STATUS_CANCELED,
    )

    manager._goal_status_callback(SimpleNamespace(status_list=[stale_status]))

    assert bridge_stops == []
    assert manager._cmd_vel_pub.messages == []
    assert manager._nav_stop_ticks == 0
    assert manager._goal_republish_ticks == 7
    assert manager._active_navigation_goal_uuid == (1, 2, 3)


def test_goal_status_callback_stops_only_for_active_goal_uuid():
    manager = _robot_manager_stub()
    manager._active_navigation_goal_uuid = (1, 2, 3)
    manager._goal_tolerance_distance = 0.25
    manager._goal_republish_ticks = 7
    manager._distance_to_goal = lambda: 0.0
    bridge_stops = []
    manager._stop_direct_dual_vln_command_bridge = lambda: bridge_stops.append(True)

    active_status = SimpleNamespace(
        goal_info=SimpleNamespace(goal_id=SimpleNamespace(uuid=[1, 2, 3])),
        status=robot_manager_module.action_msgs.msg.GoalStatus.STATUS_CANCELED,
    )

    manager._goal_status_callback(SimpleNamespace(status_list=[active_status]))

    assert bridge_stops == [True]
    assert len(manager._cmd_vel_pub.messages) == 1
    assert manager._cmd_vel_pub.messages[0].linear.x == 0.0
    assert manager._cmd_vel_pub.messages[0].angular.z == 0.0
    assert manager._nav_stop_ticks == 15
    assert manager._goal_republish_ticks == 0
    assert manager._active_navigation_goal_uuid is None


def test_tm_robots_done_uses_separate_wall_timeout_factor(monkeypatch):
    mode = TM_Robots.__new__(TM_Robots)
    mode._NodeInterface__node = SimpleNamespace(
        conf=SimpleNamespace(Robot=SimpleNamespace(TIMEOUT=SimpleNamespace(value=10))),
        rosparam=_RosParamAccessor({'timeout_wall_factor': 5.0, 'timeout_wall_sec': 0.0}),
    )
    mode._PROPS = SimpleNamespace(clock=SimpleNamespace(clock=SimpleNamespace(sec=1)), robots={})
    mode._last_reset = 0
    mode._last_reset_wall = 100.0
    monkeypatch.setattr(time, 'monotonic', lambda: 149.0)

    assert asyncio.run(mode.done) is False

    monkeypatch.setattr(time, 'monotonic', lambda: 251.0)
    assert asyncio.run(mode.done) is True


def test_tm_robots_done_records_sim_timeout_reason(monkeypatch):
    mode = TM_Robots.__new__(TM_Robots)
    mode._NodeInterface__node = SimpleNamespace(
        conf=SimpleNamespace(Robot=SimpleNamespace(TIMEOUT=SimpleNamespace(value=10))),
        rosparam=_RosParamAccessor({'timeout_wall_factor': 5.0, 'timeout_wall_sec': 0.0}),
    )
    mode._PROPS = SimpleNamespace(clock=SimpleNamespace(clock=SimpleNamespace(sec=11)), robots={})
    mode._last_reset = 0
    mode._last_reset_wall = 100.0
    monkeypatch.setattr(time, 'monotonic', lambda: 101.0)

    assert asyncio.run(mode.done) is True
    assert mode.last_done_reason == 'sim_timeout'


def test_tm_robots_done_records_goal_reached_reason():
    mode = TM_Robots.__new__(TM_Robots)
    mode._NodeInterface__node = SimpleNamespace(
        conf=SimpleNamespace(Robot=SimpleNamespace(TIMEOUT=SimpleNamespace(value=10))),
        rosparam=_RosParamAccessor({'timeout_wall_factor': 5.0, 'timeout_wall_sec': 0.0}),
    )
    robot = SimpleNamespace(is_done=_done_true())
    mode._PROPS = SimpleNamespace(clock=SimpleNamespace(clock=SimpleNamespace(sec=1)), robots={'robot': robot})
    mode._last_reset = 0
    mode._last_reset_wall = time.monotonic()

    assert asyncio.run(mode.done) is True
    assert mode.last_done_reason == 'goal_reached'


async def _done_true():
    return True


def test_task_generator_episode_outcome_payload_contains_reason(monkeypatch):
    node = TaskGenerator.__new__(TaskGenerator)
    node._completed_episodes = 1
    node._pub_episode_outcome = _Publisher()
    node.conf = SimpleNamespace(General=SimpleNamespace(DESIRED_EPISODES=SimpleNamespace(value=1)))
    node.get_logger = lambda: _Logger()
    monkeypatch.setattr(TaskGenerator, 'time', property(lambda _self: SimpleNamespace(nanoseconds=12_300_000_000)))
    monkeypatch.setattr('task_generator.node.time.time', lambda: 456.0)

    node._publish_episode_outcome('sim_timeout')

    assert len(node._pub_episode_outcome.messages) == 1
    payload = json.loads(node._pub_episode_outcome.messages[0].data)
    assert payload['episode_index'] == 0
    assert payload['completed_episodes'] == 1
    assert payload['desired_episodes'] == 1
    assert payload['finished'] is True
    assert payload['reason'] == 'sim_timeout'
    assert payload['sim_time_sec'] == 12.3
    assert payload['wall_time'] == 456.0
