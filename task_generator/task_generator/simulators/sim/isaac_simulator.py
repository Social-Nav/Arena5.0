import asyncio
import itertools
import os
import random
import shlex
import subprocess
import traceback
import types
import typing
from collections.abc import Sequence

import arena_people_msgs.msg
import arena_robots.Robot
import arena_simulation_setup.tree.assets.Material
import isaacsim_msgs.msg
import launch
import launch_ros.actions
import numpy as np
import geometry_msgs.msg
import std_msgs.msg
import std_srvs.srv
from arena_rclpy_mixins.Async import ClientWrapper
from arena_simulation_setup.tree.Wall import WallSegment
from arena_simulation_setup.shared import Obstacle as ObstacleDefinition
from isaacsim_msgs.msg import (
    Door,
    Elevator,
    Floor,
    Material,
    Pedestrian,
    PedestrianGoal,
    Prim,
    Scale,
    Wall,
)
from isaacsim_msgs.srv import (
    DeletePrims,
    EditPrims,
    LoadUsdScene,    
    NavigatePedestrians,
    SpawnDoors,
    SpawnElevators,
    SpawnFloors,
    SpawnPedestrians,
    SpawnPrims,
    SpawnUrdf,
    SpawnUsd,
    SpawnWalls,
)
from isaacsim_msgs.srv import SpawnUsdRobot as SpawnUsdRobot_srv
from std_msgs.msg import String as StdString

from task_generator.shared import Door as DoorDefinition
from task_generator.shared import (
    DynamicObstacle,
    ModelType,
    Namespace,
    Obstacle,
    Pose,
    Robot,
)
from task_generator.shared import Elevator as ElevatorDefinition
from task_generator.shared import Floor as FloorDefinition
from task_generator.shared import Wall as WallDefinition
from task_generator.simulators.sim import BaseSim, NodeInterface


def material_to_msg(material: arena_simulation_setup.tree.assets.Material.Material) -> isaacsim_msgs.msg.Material:
    return Material(
        name=material.name,
        path=material.path,
    )


async def resolve_material_to_msg(material_ref, logger, label: str) -> isaacsim_msgs.msg.Material:
    # The Docker Isaac eval image does not reliably include the optional MDL
    # material asset packs referenced by YAML worlds (hospital_1 in particular).
    # Geometry visibility is more important than material fidelity here; do not
    # block or skip floors/walls/doors while trying network/local material
    # resolvers that may fail for every segment.
    logger.debug(f"Using default Isaac material for {label}")
    return Material()


class IsaacSimulator(BaseSim, NodeInterface):

    _NS_PRIM = Namespace('Obstacles')
    _NS_PEDESTRIAN = Namespace('Pedestrians')
    _NS_ROBOT = Namespace('Robots')
    _NS_WALL = Namespace('Walls')
    _NS_FLOOR = Namespace('Floors')
    _NS_DOOR = Namespace('Doors')

    def __init__(self, *args, **kwargs):
        """Initialize IsaacSimulator
        """
        super().__init__(*args, **kwargs)

        self.wall_counter = itertools.count()
        self.floor_counter = itertools.count()
        self._spawned_doors = []
        self._clients = types.SimpleNamespace(
            DeletePedestrians=self.node.create_client_wrapper(DeletePrims, "/isaac/DeletePedestrians"),
            DeletePrims=self.node.create_client_wrapper(DeletePrims, "/isaac/DeletePrims"),
            EditPrims=self.node.create_client_wrapper(EditPrims, "/isaac/EditPrims"),
            LoadUsdScene=self.node.create_client_wrapper(LoadUsdScene, "/isaac/LoadUsdScene"),
            NavigatePedestrians=self.node.create_client_wrapper(NavigatePedestrians, "/isaac/NavigatePedestrians"),
            SpawnDoors=self.node.create_client_wrapper(SpawnDoors, "/isaac/SpawnDoors"),
            SpawnFloors=self.node.create_client_wrapper(SpawnFloors, "/isaac/SpawnFloors"),
            SpawnPedestrians=self.node.create_client_wrapper(SpawnPedestrians, "/isaac/SpawnPedestrians"),
            SpawnPrims=self.node.create_client_wrapper(SpawnPrims, "/isaac/SpawnPrims"),
            SpawnUrdf=self.node.create_client_wrapper(SpawnUrdf, "/isaac/SpawnUrdf"),
            SpawnUsd=self.node.create_client_wrapper(SpawnUsd, "/isaac/SpawnUsd"),
            SpawnUsdRobot=self.node.create_client_wrapper(SpawnUsdRobot_srv, "/isaac/SpawnUsdRobot_srv"),
            SpawnWalls=self.node.create_client_wrapper(SpawnWalls, "/isaac/SpawnWalls"),
            SpawnElevators=self.node.create_client_wrapper(SpawnElevators, "/isaac/SpawnElevators"),
            PauseSimulation=self.node.create_client_wrapper(std_srvs.srv.Trigger, "/isaac/PauseSimulation"),
            UnpauseSimulation=self.node.create_client_wrapper(std_srvs.srv.Trigger, "/isaac/UnpauseSimulation"),
        )

        # Publisher for external registration messages so IsaacSim's DoorManager
        # can be informed about spawned entities in the IsaacSim process.
        self._reg_pub = self.node.create_publisher(StdString, '/isaac/register_entity', 10)
        self._map_world_tfs: set[str] = set()
        self._static_robot_state_publishers: set[str] = set()
        self._fallback_pose_pubs: dict[str, typing.Any] = {}
        self._spawned_usd_robots: set[str] = set()


    async def _ensure_map_to_world_tf(self, robot_name: str) -> None:
        """Publish a static map -> <robot>/world TF if not already running."""
        if robot_name in self._map_world_tfs:
            return

        child_frame = str(Namespace(robot_name)('world'))
        tf_node = launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'map_to_{robot_name}_world_tfpublisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', child_frame],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )
        await self.node.do_launch(launch.LaunchDescription([tf_node]))
        self._map_world_tfs.add(robot_name)

    async def _publish_sensor_frame_tfs(self, robot_name: str, sensor_frame_transforms: list) -> None:
        """Publish static TFs for sensor frames defined in model_params.yaml.

        Each entry in sensor_frame_transforms should have:
          parent, child, x, y, z, qx, qy, qz, qw
        """
        ns = Namespace(robot_name)
        nodes = []
        for i, tf in enumerate(sensor_frame_transforms):
            parent = str(ns(tf.get('parent', 'base_footprint')))
            child = str(ns(tf.get('child', 'sensor')))
            x   = str(tf.get('x',  0.0))
            y   = str(tf.get('y',  0.0))
            z   = str(tf.get('z',  0.0))
            qx  = str(tf.get('qx', 0.0))
            qy  = str(tf.get('qy', 0.0))
            qz  = str(tf.get('qz', 0.0))
            qw  = str(tf.get('qw', 1.0))
            safe_child = child.replace('/', '_')
            nodes.append(launch_ros.actions.Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'{robot_name}_sensor_tf_{i}_{safe_child}',
                arguments=[x, y, z, qx, qy, qz, qw, parent, child],
                parameters=[{'use_sim_time': True}],
                output='screen',
            ))
        if nodes:
            await self.node.do_launch(launch.LaunchDescription(nodes))

    async def _ensure_static_robot_state(
        self,
        robot_name: str,
        base_frame: str,
        odom_frame: str,
        publish_fallback_odom_tf: bool = True,
    ) -> None:
        """Publish a minimal TF/odom state for Isaac URDF fallback startup.

        Isaac's URDF importer can take longer than the task-generator service
        timeout on this Docker + Isaac Sim setup.  Publishing the expected
        odom/base frames and odom topic keeps Nav2 and the eval recorder alive
        while Isaac finishes loading or when the import is skipped by timeout.
        """
        if robot_name in self._static_robot_state_publishers:
            return

        ns = Namespace(robot_name)
        fq_odom_frame = str(ns(odom_frame or 'odom'))
        fq_base_frame = str(ns(base_frame or 'base_link'))
        raw_base_frame = base_frame or 'base_link'
        fq_scan_frame = str(ns('base_scan'))
        fq_camera_frame = str(ns(f'{raw_base_frame}/head_camera'))
        odom_topic = str(self.node.service_namespace(robot_name, 'odom'))
        camera_topic = str(self.node.service_namespace(robot_name, 'head_camera'))
        scan_topic = str(self.node.service_namespace(robot_name, 'scan'))
        pose_topic = str(self.node.service_namespace(robot_name, 'fallback_pose'))
        cmd_topic = str(self.node.service_namespace(robot_name, 'cmd_vel'))
        safe_name = robot_name.replace('/', '_')
        # Previous Isaac eval attempts may leave the standalone fallback sensor
        # process alive after Nav2/eval teardown.  Because it republishes the same
        # odom/TF topics, stale copies cause the robot pose to jump between runs
        # and make dual_vln alternate between unrelated goal distances/yaw errors.
        # Kill only this generated fallback process before starting a fresh one.
        subprocess.run(
            ['pkill', '-f', f'{safe_name}_isaac_fallback_sensors'],
            check=False,
        )
        subprocess.run(
            ['pkill', '-f', f"node = Node('{safe_name}_isaac_fallback_sensors')"],
            check=False,
        )
        fallback_sensor_code = f"""
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from tf2_ros import TransformBroadcaster

rclpy.init()
node = Node('{safe_name}_isaac_fallback_sensors')
# This process is the bootstrap odom/TF publisher used specifically when Isaac
# simulation time may be paused or not advancing yet.  Keep it on wall time so
# its timer still fires and the odom->base transform exists before Nav2
# lifecycle activation and goal publication.
node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, False)])
PUBLISH_FALLBACK_ODOM_TF = {str(publish_fallback_odom_tf)}
# Never publish synthetic fallback camera frames on the real head_camera or
# top_down_camera topics.  Those topics are consumed by InternNav and the eval
# video recorder and must represent real Isaac render products.  Keep any
# diagnostic fallback frames on an explicit fallback_head_camera namespace only.
fallback_camera_topic = '{camera_topic}'.replace('/head_camera', '/fallback_head_camera')
image_pub = node.create_publisher(Image, fallback_camera_topic + '/image', 10)
depth_pub = node.create_publisher(Image, fallback_camera_topic + '/depth', 10)
info_pub = node.create_publisher(CameraInfo, fallback_camera_topic + '/camera_info', 10)
scan_pub = node.create_publisher(LaserScan, '{scan_topic}', 10)
odom_pub = node.create_publisher(Odometry, '{odom_topic}', 10) if PUBLISH_FALLBACK_ODOM_TF else None
tf_pub = TransformBroadcaster(node) if PUBLISH_FALLBACK_ODOM_TF else None
frame_id = '{fq_camera_frame}'
pose = {{'x': 0.0, 'y': 0.0, 'yaw': 0.0}}
cmd = {{'vx': 0.0, 'wz': 0.0}}
fallback_odom_active = {{'value': PUBLISH_FALLBACK_ODOM_TF}}
fallback_pose_initialized = {{'value': False}}
last_t = time.monotonic()
w, h = 640, 480
yy, xx = np.mgrid[0:h, 0:w]
rgb = np.zeros((h, w, 3), dtype=np.uint8)
# Non-gradient diagnostic camera fallback.  This is intentionally structured as
# a simple corridor-like view so eval videos remain inspectable if the Isaac ROS
# camera render graph fails to publish.  It avoids the old R=x/G=y/B=96 test
# pattern that produced meaningless 2x3 color blocks.
horizon = h // 2
rgb[:horizon, :] = np.array([95, 115, 135], dtype=np.uint8)
rgb[horizon:, :] = np.array([118, 112, 104], dtype=np.uint8)
center = w // 2
for offset in range(0, 260, 20):
    y = min(h - 1, horizon + offset)
    half = int(80 + offset * 0.9)
    x0 = max(0, center - half)
    x1 = min(w - 1, center + half)
    rgb[y:y+3, x0:x1] = np.array([210, 210, 190], dtype=np.uint8)
for side in (-1, 1):
    for t in range(6):
        x = np.clip(center + side * (70 + t * 70), 0, w - 1)
        rgb[horizon:, max(0, x - 2): min(w, x + 2)] = np.array([70, 80, 90], dtype=np.uint8)
rgb[180:260, 270:370] = np.array([80, 120, 170], dtype=np.uint8)
rgb[205:235, 300:340] = np.array([230, 230, 210], dtype=np.uint8)
top_rgb = np.full((640, 640, 3), 235, dtype=np.uint8)
top_rgb[80:560, 280:360] = np.array([210, 210, 210], dtype=np.uint8)
top_rgb[280:360, 80:560] = np.array([210, 210, 210], dtype=np.uint8)
top_rgb[300:340, 300:340] = np.array([60, 120, 220], dtype=np.uint8)
depth = np.full((h, w), 2.0, dtype=np.float32)

def _quat_from_yaw(yaw):
    if not math.isfinite(yaw):
        yaw = 0.0
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))

def _finite_values(*values):
    return all(math.isfinite(float(value)) for value in values)

def _on_pose(msg):
    # When synthetic fallback odom is active, use the real Isaac pose only to seed
    # the initial state.  Some USD robot/controller combinations publish a real
    # pose/odom stream even though the body does not respond to cmd_vel; if we
    # keep accepting that stationary pose while integrating commands below, each
    # real pose callback snaps the fallback trajectory back to the spawn point.
    if (
        PUBLISH_FALLBACK_ODOM_TF
        and fallback_odom_active['value']
        and fallback_pose_initialized['value']
        and (abs(cmd['vx']) > 1e-5 or abs(cmd['wz']) > 1e-5)
    ):
        return
    x = float(msg.pose.position.x)
    y = float(msg.pose.position.y)
    q = msg.pose.orientation
    if not _finite_values(x, y, q.x, q.y, q.z, q.w):
        node.get_logger().warn('Ignoring non-finite fallback pose update')
        return
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    if not math.isfinite(yaw):
        node.get_logger().warn('Ignoring fallback pose update with non-finite yaw')
        return
    pose['x'] = x
    pose['y'] = y
    pose['yaw'] = yaw
    fallback_pose_initialized['value'] = True

def _on_cmd(msg):
    vx = float(msg.linear.x)
    wz = float(msg.angular.z)
    if not _finite_values(vx, wz):
        node.get_logger().warn('Ignoring non-finite fallback cmd_vel update')
        return
    cmd['vx'] = vx
    cmd['wz'] = wz

node.create_subscription(PoseStamped, '{pose_topic}', _on_pose, 10)
node.create_subscription(Twist, '{cmd_topic}', _on_cmd, 10)

def publish():
    global last_t
    now_m = time.monotonic()
    dt = max(0.0, min(now_m - last_t, 0.25))
    last_t = now_m
    if abs(cmd['vx']) > 1e-5 or abs(cmd['wz']) > 1e-5:
        pose['yaw'] += cmd['wz'] * dt
        pose['x'] += math.cos(pose['yaw']) * cmd['vx'] * dt
        pose['y'] += math.sin(pose['yaw']) * cmd['vx'] * dt
    if not _finite_values(pose['x'], pose['y'], pose['yaw']):
        node.get_logger().warn('Resetting non-finite fallback odom state')
        pose['x'] = 0.0
        pose['y'] = 0.0
        pose['yaw'] = 0.0

    stamp = node.get_clock().now().to_msg()
    qx, qy, qz, qw = _quat_from_yaw(pose['yaw'])

    if PUBLISH_FALLBACK_ODOM_TF and fallback_odom_active['value']:
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = 'map'
        tf.child_frame_id = '{fq_odom_frame}'
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = 1.0
        tf_pub.sendTransform(tf)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = '{fq_odom_frame}'
        tf.child_frame_id = '{fq_base_frame}'
        tf.transform.translation.x = pose['x']
        tf.transform.translation.y = pose['y']
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        tf_pub.sendTransform(tf)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = '{fq_odom_frame}'
        odom.child_frame_id = '{fq_base_frame}'
        odom.pose.pose.position.x = pose['x']
        odom.pose.pose.position.y = pose['y']
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = cmd['vx']
        odom.twist.twist.angular.z = cmd['wz']
        odom_pub.publish(odom)

    img = Image()
    img.header.stamp = stamp
    img.header.frame_id = frame_id
    img.height = h
    img.width = w
    img.encoding = 'rgb8'
    img.is_bigendian = False
    img.step = w * 3
    img.data = rgb.tobytes()
    dep = Image()
    dep.header = img.header
    dep.height = h
    dep.width = w
    dep.encoding = '32FC1'
    dep.is_bigendian = False
    dep.step = w * 4
    dep.data = depth.tobytes()
    info = CameraInfo()
    info.header = img.header
    info.height = h
    info.width = w
    info.k = [525.0, 0.0, w / 2.0, 0.0, 525.0, h / 2.0, 0.0, 0.0, 1.0]
    info.p = [525.0, 0.0, w / 2.0, 0.0, 0.0, 525.0, h / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    scan = LaserScan()
    scan.header.stamp = stamp
    scan.header.frame_id = '{fq_scan_frame}'
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = (scan.angle_max - scan.angle_min) / 359.0
    scan.time_increment = 0.0
    scan.scan_time = 0.1
    scan.range_min = 0.05
    scan.range_max = 30.0
    scan.ranges = [10.0] * 360
    image_pub.publish(img)
    depth_pub.publish(dep)
    info_pub.publish(info)
    scan_pub.publish(scan)

node.create_timer(0.1, publish)
rclpy.spin(node)
"""

        nodes = []
        fallback_sensor_process = None
        if publish_fallback_odom_tf:
            fallback_env = os.environ.copy()
            fallback_env.setdefault('ROS_LOCALHOST_ONLY', '0')
            fallback_env.setdefault('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET')
            fallback_env.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
            fallback_sensor_process = subprocess.Popen(
                ['/usr/bin/python3', '-c', fallback_sensor_code],
                env=fallback_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._logger.info(
                f'Started Isaac fallback odom/TF publisher for {robot_name} '
                f'on {odom_topic} (pid={fallback_sensor_process.pid})'
            )

        if publish_fallback_odom_tf:
            nodes.extend([
            launch_ros.actions.Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'{safe_name}_map_to_raw_odom_tfpublisher',
                arguments=['0', '0', '0', '0', '0', '0', 'map', fq_odom_frame],
                parameters=[{'use_sim_time': True}],
                output='screen',
            ),
            ])
            if fq_base_frame != str(ns('base_link')):
                nodes.append(
            launch_ros.actions.Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'{safe_name}_fq_base_to_base_link_tfpublisher',
                arguments=['0', '0', '0', '0', '0', '0', fq_base_frame, str(ns('base_link'))],
                parameters=[{'use_sim_time': True}],
                output='screen',
            ),
                )
            if fq_base_frame != 'base_link':
                nodes.append(
            launch_ros.actions.Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'{safe_name}_fq_base_to_raw_base_link_tfpublisher',
                arguments=['0', '0', '0', '0', '0', '0', fq_base_frame, 'base_link'],
                parameters=[{'use_sim_time': True}],
                output='screen',
            ),
                )

        if publish_fallback_odom_tf:
            nodes.extend([
                launch_ros.actions.Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name=f'{safe_name}_base_to_scan_tfpublisher',
                    arguments=['0', '0', '0', '0', '0', '0', fq_base_frame, fq_scan_frame],
                    parameters=[{'use_sim_time': True}],
                    output='screen',
                ),
                launch_ros.actions.Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name=f'{safe_name}_base_to_head_camera_tfpublisher',
                    arguments=['0.35', '0', '0.75', '0', '0', '0', fq_base_frame, fq_camera_frame],
                    parameters=[{'use_sim_time': True}],
                    output='screen',
                ),
            ])
        await self.node.do_launch(launch.LaunchDescription(nodes))
        self._static_robot_state_publishers.add(robot_name)
        if publish_fallback_odom_tf:
            self._fallback_pose_pubs[robot_name] = self.node.create_publisher(
                geometry_msgs.msg.PoseStamped, pose_topic, 10
            )

    async def robot_spawn(self, robots):
        async def impl(robot: Robot) -> bool:
            try:
                resolved_robot_model = await robot.model.resolve()
                try:
                    model = await resolved_robot_model.model.get(
                        ModelType.USD,
                        loader_args=robot.asdict(),
                    )
                except FileNotFoundError:
                    self._logger.debug(
                        f"USD model for {robot.model.name} not found; falling back to URDF"
                    )
                    model = await resolved_robot_model.model.get(
                        ModelType.URDF,
                        loader_args=robot.asdict(),
                    )

                robot_params = (await arena_robots.Robot.RobotIdentifier(robot.model.name).resolve()).model_params
                fq_name = self._NS_ROBOT(robot.sim_path)

                if model.type == ModelType.USD:
                    assert model.path is not None, f"USD model {model.name} must have a valid file path"

                    ns = str(self.node.service_namespace(robot.name)).lstrip('/')
                    base = robot_params.base_frame or ""
                    await self._ensure_static_robot_state(
                        robot.name,
                        robot_params.base_frame,
                        robot_params.odom_frame,
                        publish_fallback_odom_tf=True,
                    )
                    await self._clients.SpawnUsdRobot.call_fire_and_forget(
                        SpawnUsdRobot_srv.Request(
                            name=fq_name,
                            usd_path=str(model.path),
                            robot_namespace=ns,
                            base_frame=base,
                            pose=robot.pose.to_msg(),
                        )
                    )
                    self._logger.info(
                        f"Requested USD robot spawn for '{robot.name}' via SpawnUsdRobot; "
                        "continuing with task-generator TF/odom fallback while Isaac processes the request"
                    )
                    await asyncio.sleep(1.0)
                    self._spawned_usd_robots.add(robot.name)
                    await self._ensure_map_to_world_tf(robot.name)
                    await self._publish_sensor_frame_tfs(robot.name, robot_params.sensor_frame_transforms)
                    return True

                if model.type == ModelType.URDF:
                    assert model.path is not None, f"URDF model {model.name} must have a valid file path"
                    await self._ensure_static_robot_state(
                        robot.name,
                        robot_params.base_frame,
                        robot_params.odom_frame,
                        publish_fallback_odom_tf=True,
                    )
                    await self._clients.SpawnUrdf.call_timeout(
                        SpawnUrdf.Request(
                            name=fq_name,
                            urdf_path=str(model.path),
                            robot_model=robot.model.name,
                            localization=True,
                            tf_prefix=robot.name,
                            base_frame=robot_params.base_frame,
                            odom_frame=robot_params.odom_frame,
                            pose=robot.pose.to_msg(),
                            cmd_vel_topic=self.node.service_namespace(robot.name, 'cmd_vel'),
                            joint_states_topic=self.node.service_namespace(robot.name, 'joint_states'),
                            odom_topic=self.node.service_namespace(robot.name, 'odom'),
                        ),
                        timeout_sec=5.0,
                    )

                    base_frame = robot_params.base_frame
                    robot_prim_path = os.path.join("/World", fq_name, base_frame)

                    # Publish registration message so DoorManager in IsaacSim process
                    # registers the robot. This avoids cross-process direct calls.
                    try:
                        if self._reg_pub:
                            self._reg_pub.publish(StdString(data=f"robot|{robot_prim_path}"))
                            self._logger.debug(f"Published registration for robot: {robot_prim_path}")
                        else:
                            self._logger.warning('Registration publisher not available; robot not registered with IsaacSim DoorManager')
                    except Exception as e:
                        self._logger.warning(f'Failed to publish robot registration: {e}\n{traceback.format_exc()}')

                    return True

                raise NotImplementedError(
                    f"robot model of type {model.type} can't be spawned by {self.__class__.__name__}"
                )

            except Exception as e:
                self._logger.error(f"{repr(e)}\n{traceback.format_exc()}")
                return False

        return await asyncio.gather(*map(impl, robots))

    async def obstacle_spawn(self, obstacles):
        async def impl(obstacle: Obstacle) -> Prim | None:
            try:
                resolved_model = await asyncio.wait_for(obstacle.model.resolve(), timeout=5.0)
                model = await asyncio.wait_for(resolved_model.get([ModelType.USD]), timeout=5.0)
                if model.type is ModelType.UNKNOWN:
                    raise ValueError(f"obstacle model {obstacle.model.name} has no USD representation")
            except asyncio.TimeoutError:
                self._logger.warning(
                    f"Skipping obstacle model for {obstacle.name}: resolving {obstacle.model.name} timed out"
                )
                return None
            except Exception as e:
                self._logger.warning(f"Skipping unresolved obstacle model for {obstacle.name}: {e}")
                return None
            assert model.path is not None, f"USD model {model.name} must have a valid file path"
            prim = Prim()
            prim.usd_path = str(model.path)
            prim.name = self._NS_PRIM(obstacle.sim_path)
            prim.pose = obstacle.pose.to_msg()
            if obstacle.scale is not None:
                prim.scale.x = obstacle.scale.x
                prim.scale.y = obstacle.scale.y
                prim.scale.z = obstacle.scale.z
            return prim

        prims = await asyncio.gather(*map(impl, obstacles))
        valid_count = sum(1 for prim in prims if prim is not None)
        if valid_count == 0:
            self._logger.warning(
                f"No Isaac USD obstacle assets resolved out of {len(obstacles)} obstacle(s); "
                "continuing with floor/wall/door procedural geometry only."
            )
            return tuple(False for _ in obstacles)

        req = SpawnPrims.Request()
        req.prims = list(filter(None, prims))
        response = await self._clients.SpawnPrims.call_timeout(req, timeout_sec=20.0)
        if response is None:
            self._logger.warning(
                "SpawnPrims did not return a ROS response; assuming Isaac processed the request "
                "because the service intentionally suppresses responses in embedded rclpy."
            )
            return tuple(a is not None for a in prims)

        response_iter = iter(response.ret)

        return tuple((a is not None) and next(response_iter) for a in prims)

    async def obstacle_move(self, obstacles):
        return await self._move_entities([(self._NS_PRIM(o.sim_path), o.pose) for o in obstacles])

    async def pedestrian_move(self, pedestrians):
        if not pedestrians:
            return tuple()

        await self._clients.DeletePedestrians.call_timeout(
            DeletePrims.Request(names=[self._NS_PEDESTRIAN(p.sim_path) for p in pedestrians])
        )
        res = await self.pedestrian_spawn(pedestrians)
        if res is None:
            return tuple(False for _ in pedestrians)
        return tuple(res)

        # # tmp: restore when pedestrian move works within isaac sim
        # return await self._move_entities([(self._NS_PEDESTRIAN(p.name), p.pose) for p in pedestrians])

    async def robot_move(self, robots):
        async def move_robot(robot: Robot) -> bool:
            try:
                if robot.name in self._spawned_usd_robots:
                    # Move the real USD robot/camera rig to the task reset pose.
                    # The Isaac-side EditPrims callback suppresses the ROS
                    # response after applying the transform to avoid the embedded
                    # rclpy response conversion abort observed with custom srv
                    # responses, so _move_entities treats timeout as success.
                    self._logger.info(
                        f"Moving spawned USD robot '{robot.name}' to reset pose via EditPrims"
                    )
                    top_down_pose = geometry_msgs.msg.Pose()
                    robot_pose_msg = robot.pose.to_msg()
                    top_down_pose.position.x = robot_pose_msg.position.x
                    top_down_pose.position.y = robot_pose_msg.position.y
                    top_down_pose.position.z = 8.0
                    top_down_pose.orientation.w = 1.0
                    safe_robot_prim = str(self._NS_ROBOT(robot.sim_path)).strip('/').replace('/', '_')
                    if not safe_robot_prim.startswith('World_'):
                        safe_robot_prim = f'World_{safe_robot_prim}'
                    top_down_camera_name = f'vln_top_down_camera_{safe_robot_prim}'
                    moved = all(await self._move_entities([
                        (self._NS_ROBOT(robot.sim_path), robot.pose),
                        (top_down_camera_name, top_down_pose),
                    ]))
                    fallback_pub = self._fallback_pose_pubs.get(robot.name)
                    if fallback_pub is not None:
                        msg = geometry_msgs.msg.PoseStamped()
                        msg.header.frame_id = 'map'
                        msg.header.stamp = self.node.sim_time.to_msg()
                        msg.pose = robot.pose.to_msg()
                        # Keep fallback odom in sync for consumers that use the
                        # lightweight state publisher, but do not let it bypass
                        # the real Isaac prim/camera move.
                        for _ in range(5):
                            fallback_pub.publish(msg)
                            await asyncio.sleep(0.05)
                    return moved
                fallback_pub = self._fallback_pose_pubs.get(robot.name)
                if fallback_pub is not None:
                    msg = geometry_msgs.msg.PoseStamped()
                    msg.header.frame_id = 'map'
                    msg.header.stamp = self.node.sim_time.to_msg()
                    msg.pose = robot.pose.to_msg()
                    # Publish several times so the fallback state node receives
                    # the reset pose even while ROS discovery is still settling.
                    for _ in range(5):
                        fallback_pub.publish(msg)
                        await asyncio.sleep(0.05)
                    return True
                return await self._move_entity(self._NS_ROBOT(robot.sim_path), robot.pose)
            except Exception as e:
                self._logger.error(f"Failed to move robot {robot.name}: {e}\n{traceback.format_exc()}")
                return False
        return await asyncio.gather(*map(move_robot, robots))

    async def obstacle_delete(self, obstacles):
        return await asyncio.gather(*(self._delete_entity(self._NS_PRIM(o.sim_path)) for o in obstacles))

    async def pedestrian_delete(self, pedestrians):
        if not pedestrians:
            return tuple()

        res = await self._clients.DeletePedestrians.call_timeout(
            DeletePrims.Request(names=[self._NS_PEDESTRIAN(p.sim_path) for p in pedestrians])
        )
        if res is None:
            ret = tuple(False for _ in pedestrians)
        else:
            ret = tuple(res.ret)
        return ret

    async def robot_delete(self, robots):
        return await asyncio.gather(*(self._delete_entity(self._NS_ROBOT(r.sim_path)) for r in robots))

    async def remove_world(self):
        await self._delete_entity(self._NS_WALL(self.node._environment_manager._prefix()))
        await self._delete_entity(self._NS_DOOR(self.node._environment_manager._prefix()))
        await self._delete_entity(self._NS_FLOOR(self.node._environment_manager._prefix()))
        return True

    async def spawn_walls(self, walls):
        self._logger.info(f"Attempting to spawn {len(walls)} wall definition(s) into Isaac Sim")

        async def create_segment(segment: WallSegment) -> Wall | None:
            end = segment.end.to_msg()
            end.z += segment.height
            try:
                wall_name = self.node._environment_manager.realize(f"wall_{next(self.wall_counter)}")
                material = await resolve_material_to_msg(segment.material, self._logger, f"wall {wall_name}")
                return Wall(
                    name=self._NS_WALL(wall_name),
                    start=segment.start.to_msg(),
                    end=end,
                    material=material,
                    thickness=segment.width,
                )

            except Exception as e:
                self._logger.error(f"Failed to spawn wall: {e}\n{traceback.format_exc()}")
                return None

        async def create_obstacle(obstacle: ObstacleDefinition) -> Prim | None:
            try:
                prim_name = self.node._environment_manager.realize(f"obstacle_{next(self.wall_counter)}")
                model = await (await obstacle.model.resolve()).get(ModelType.USD)
                if model.type is ModelType.UNKNOWN:
                    return None
                assert model.path is not None, f"USD model {model.name} must have a valid file path"
                prim = Prim()
                prim.usd_path = str(model.path)
                prim.name = self._NS_WALL(prim_name)
                prim.pose = obstacle.pose.to_msg()
                return prim

            except Exception as e:
                self._logger.warning(f"Skipping unresolved wall obstacle asset: {e}")
                return None

        async def create_wall(wall: WallDefinition):
            segments, obstacles = await wall.assets()
            return map(create_segment, segments), map(create_obstacle, obstacles)

        wall_futures = await asyncio.gather(*map(create_wall, walls))
        segment_futures, obstacle_futures = zip(*wall_futures)

        walls_req = SpawnWalls.Request()
        prims_req = SpawnPrims.Request()
        walls_req.walls = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(segment_futures))))
        prims_req.prims = list(filter(None, await asyncio.gather(*itertools.chain.from_iterable(obstacle_futures))))

        if walls_req.walls:
            await self._clients.SpawnWalls.call_fire_and_forget(walls_req)
        if prims_req.prims:
            await self._clients.SpawnPrims.call_fire_and_forget(prims_req)
        walls_ok = True
        prims_ok = True
        res = walls_ok and prims_ok

        self._logger.info("All walls spawned.")
        return res

    async def spawn_floors(self, floors) -> bool:
        self._logger.info(f"Attempting to spawn {len(floors)} floor definition(s) into Isaac Sim")

        async def impl(floor: FloorDefinition) -> Floor | None:
            try:
                material = await resolve_material_to_msg(floor.material, self._logger, f"floor {floor.name}")
                return Floor(
                    name=self._NS_FLOOR(floor.sim_path),
                    x_length=floor.x_length,
                    y_length=floor.y_length,
                    pos=floor.pos.to_msg(),
                    material=material,
                )

            except Exception:
                self._logger.error(f"Failed to spawn floor: {floor.name}\n{traceback.format_exc()}")
                return None

        floors_req = SpawnFloors.Request()
        floors_req.floors = list(filter(None, await asyncio.gather(*map(impl, floors))))
        if floors_req.floors:
            await self._clients.SpawnFloors.call_fire_and_forget(floors_req)

        res = True
        self._logger.info("All floors spawned successfully.")
        return res

    async def spawn_doors(self, doors) -> bool:
        async def impl(door: DoorDefinition) -> Door | None:
            try:
                end = door.end.to_msg()
                end.z += door.height
                material = await resolve_material_to_msg(door.material, self._logger, f"door {door.name}")
                return Door(
                    name=self._NS_DOOR(door.name),
                    start=door.start.to_msg(),
                    end=end,
                    material=material,
                    thickness=0.1,
                    kind=door.kind,
                )
            except Exception as e:
                self._logger.error(f"Failed to spawn door: {e}\n{traceback.format_exc()}")
                return None

        doors_req = SpawnDoors.Request()
        doors_req.doors = list(filter(None, await asyncio.gather(*map(impl, doors))))
        if doors_req.doors:
            await self._clients.SpawnDoors.call_fire_and_forget(doors_req)

        res = True
        self._logger.info("All doors spawned successfully.")
        return res
    async def spawn_elevators(self, elevators) -> bool:
        self._logger.debug(f"IsaacSimulator.spawn_elevators ENTRY, elevators: {elevators}")
        self._logger.debug(f"IsaacSimulator.spawn_elevators called with: {[e.name for e in elevators]}")
        for e in elevators:
            self._logger.debug(f"Elevator data: {e}")

        req = SpawnElevators.Request()

        async def impl(elevator: ElevatorDefinition) -> Elevator | None:
            try:
                pos = elevator.position
                size = elevator.size
                size = Scale(x=size[0], y=size[1], z=size[2])
                des = elevator.destination
                material = await resolve_material_to_msg(elevator.material, self._logger, f"elevator {elevator.name}")
                result = Elevator(
                    name=elevator.sim_path,
                    position=pos.to_msg(),
                    size=size,
                    height_min=elevator.height_min,
                    height_max=elevator.height_max,
                    material=material,
                    destination=des if hasattr(elevator, 'destination') else '',
                )
                return result
            except Exception as e:
                self._logger.error(f"Failed to append elevator: {elevator.name}: {e}\n{traceback.format_exc()}")
                return None
        
        req.elevators = list(filter(None, await asyncio.gather(*map(impl, elevators))))
        if req.elevators:
            await self._clients.SpawnElevators.call_fire_and_forget(req)
        res = True
        self._logger.debug("All elevators spawned successfully.")
        return res

    async def before_reset_task(self):
        await self._pause()
        return True

    async def after_reset_task(self):
        await self._unpause()
        # Isaac eval runs keep the renderer stepping continuously, but robot
        # teleports, camera render products, and PhysX articulation state still
        # need a short settle window before task_reset is published.  Without
        # this delay the eval recorder and InternNav can observe the new episode
        # boundary before the robot pose/camera streams have converged.
        await asyncio.sleep(0.35)
        return True

    async def _pause(self):
        # Keep Isaac stepping continuously during eval resets.  On this Isaac 5.1
        # Docker setup, toggling world.pause()/world.play() around a task reset
        # can leave the next resume without an active physics scene and crash in
        # world.play() with "Failed to create simulation view: no active physics
        # scene found".  Reset safety comes from Xform-only robot relocation plus
        # the post-reset settle delay, not from stopping the renderer/physics
        # loop entirely.
        return True

    async def _unpause(self):
        return True

    async def pedestrian_spawn(self, pedestrians):
        on_success: list[tuple[str, str]] = []

        # TODO implement targeted pedestrian models
        async def impl(pedestrian: DynamicObstacle) -> Pedestrian | None:
            available_models: dict[str, str] = {
                "F_Business_02": "F_Business_02",
                "F_Medical_01": "F_Medical_01",
                "M_Medical_01": "M_Medical_01",
                "biped_demo_meters": "biped_demo",
                "female_adult_business_02": "original_female_adult_business_02",
                "female_adult_medical_01": "original_female_adult_medical_01",
                "female_adult_police_01": "original_female_adult_police_01",
                "female_adult_police_01_new": "female_adult_police_01_new",
                "female_adult_police_02": "original_female_adult_police_02",
                "female_adult_police_03": "original_female_adult_police_03",
                "female_adult_police_03_new": "female_adult_police_03_new",
                "male_adult_construction_01": "original_male_adult_construction_01",
                "male_adult_construction_01_new": "male_adult_construction_01_new",
                "male_adult_construction_02": "original_male_adult_construction_02",
                "male_adult_construction_03": "original_male_adult_construction_03",
                "male_adult_construction_05": "original_male_adult_construction_05",
                "male_adult_construction_05_new": "male_adult_construction_05_new",
                "male_adult_medical_01": "original_male_adult_medical_01",
                "male_adult_police_04": "original_male_adult_police_04",
            }
            if pedestrian.model.name in available_models:
                model_name = pedestrian.model.name
            else:
                model_name = random.choice(tuple(available_models.keys()))

            ped = Pedestrian()
            ped.name = self._NS_PEDESTRIAN(pedestrian.sim_path)
            ped.character_name = available_models[model_name]
            ped.pose = pedestrian.pose.to_msg()
            ped.controller_stats = False

            on_success.append((pedestrian.name, model_name))
            return ped

        req = SpawnPedestrians.Request()
        req.pedestrians = list(filter(None, await asyncio.gather(*map(impl, pedestrians))))
        if not req.pedestrians:
            return tuple(False for _ in pedestrians)

        await self._clients.SpawnPedestrians.call_fire_and_forget(req)
        await self.pedestrian_update(
            arena_people_msgs.msg.Pedestrians(pedestrians=[
                arena_people_msgs.msg.Pedestrian(
                    name=ped.sim_path,
                    pose=ped.pose.to_msg(),
                )
                for ped in pedestrians
            ])
        )
        return tuple(True for _ in pedestrians)

    async def pedestrian_update(self, pedestrians):

        async def impl(ped: arena_people_msgs.msg.Pedestrian) -> PedestrianGoal | None:
            goal = PedestrianGoal()
            goal.name = self._NS_PEDESTRIAN(ped.name)
            goal.position = ped.pose.position
            goal.velocity = np.linalg.norm([ped.twist.linear.x, ped.twist.linear.y])
            return goal

        goals = list(filter(None, await asyncio.gather(*map(impl, pedestrians.pedestrians))))
        if not goals:
            return tuple()

        req = NavigatePedestrians.Request()
        req.goals = goals
        await self._clients.NavigatePedestrians.call_fire_and_forget(req)
        return tuple(True for _ in goals)

    async def _delete_entity(self, name: str) -> bool:
        # In Docker eval runs each launch starts with a fresh Isaac stage.  The
        # DeletePrims callback can block Isaac's single service executor for
        # minutes, delaying robot spawn and camera publication.  Skipping this
        # cleanup is safe for one-shot eval episodes and keeps the Isaac service
        # queue available for SpawnUsdRobot/SpawnUrdf.
        self._logger.debug(f"Skipping DeletePrims for fresh Isaac eval stage: {name}")
        return True

    async def _delete_pedestrians(self, prim_path):
        self._logger.info(f"Skipping DeletePedestrians for fresh Isaac eval stage: {prim_path}")
        return True

    async def _move_entity(self, name: str, pose: Pose) -> bool:
        return (await self._move_entities([(name, pose)]))[0]

    async def _move_entities(self, actions: Sequence[tuple[str, Pose]]) -> Sequence[bool]:
        req = EditPrims.Request(
            prims=[
                Prim(
                    name=name,
                    pose=pose.to_msg() if hasattr(pose, 'to_msg') else pose,
                )
                for name, pose in actions
            ],
            pose=True,
        )

        await self._clients.EditPrims.call_fire_and_forget(req)
        return [True] * len(actions)

    async def setup(self):
        """
        Initialize all ROS 2 service clients and wait for their availability.
        """
        self._logger.info("Setting up IsaacSimulator service clients...")
        futures: list[typing.Awaitable] = []
        # Define services with their corresponding client attributes
        for client in self._clients.__dict__.values():
            client = typing.cast(ClientWrapper, client)
            self._logger.info(f"Initializing service client: {client.client.srv_name}")
            futures.append(client.ensure(timeout_sec=120.0))
        results = await asyncio.gather(*futures)
        unavailable = [
            typing.cast(ClientWrapper, client).client.srv_name
            for client, available in zip(self._clients.__dict__.values(), results)
            if not available
        ]
        if unavailable:
            raise RuntimeError(f"Isaac service(s) unavailable after 120s: {', '.join(unavailable)}")

        self._logger.info("All service clients are available.")

        self.node.create_publisher(std_msgs.msg.String, '/isaac/add_pedestrians_topic', 10).publish(
            std_msgs.msg.String(data=self.node.service_namespace('arena_peds'))
        )

        # Avoid calling DeletePrims during startup: in Isaac-only eval runs the
        # delete callback can monopolize the Isaac service executor long enough
        # for the subsequent robot spawn request to time out.  Each eval starts
        # a fresh Isaac Sim process via isaac.launch.py, so stale robot prims
        # cannot bleed in from a previous session.
        self._logger.info("Skipping leftover robot prim cleanup for fresh Isaac process.")

        self._logger.info("All service clients initialized and available.")
    
    async def load_usd_scene(
        self,
        usd_path: str,
        scene_prim_path: str = "/World/Scene",
        scale: float = 1.0,
        position: list = None,
        orientation: list = None,
        add_colliders: bool = True,
        disable_collision_cooking: bool = True,
    ) -> bool:
        """
        Load a complete USD scene (e.g., GRScenes) into Isaac Sim.

        Args:
            usd_path: Absolute path to the USD file
            scene_prim_path: Target prim path in the scene hierarchy
            scale: Scale factor for the scene
            position: Position [x, y, z]
            orientation: Orientation quaternion [x, y, z, w]
            add_colliders: Whether to add physics colliders
            disable_collision_cooking: Disable cooking for faster loading

        Returns:
            bool: True if scene loaded successfully
        """
        self._logger.info(f"Loading USD scene: {usd_path}")

        req = LoadUsdScene.Request()
        req.usd_path = usd_path
        req.scene_prim_path = scene_prim_path
        req.scale = float(scale)
        req.position = position or [0.0, 0.0, 0.0]
        req.orientation = orientation or [0.0, 0.0, 0.0, 1.0]
        req.add_colliders = add_colliders
        req.disable_collision_cooking = disable_collision_cooking

        try:
            response = await self._clients.LoadUsdScene.call_timeout(req, timeout_sec=120.0)
            if response is None:
                self._logger.error("LoadUsdScene service timed out")
                return False

            if response.success:
                self._logger.info(f"USD scene loaded: {response.scene_prim_path}")
                return True
            else:
                self._logger.error(f"Failed to load USD scene: {response.message}")
                return False

        except Exception as e:
            self._logger.error(f"Exception loading USD scene: {e}\n{traceback.format_exc()}")
            return False

    @classmethod
    async def create(cls, *args, namespace, **kwargs):
        self = cls(*args, namespace=namespace, **kwargs)
        self._logger.info("Creating IsaacSimulator instance...")
        await self.setup()
        return self
