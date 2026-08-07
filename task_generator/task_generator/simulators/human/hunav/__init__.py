
import enum
import math
import os
import typing

import attrs
import geometry_msgs.msg
import hunav_msgs.msg
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_simulation_setup.tree.assets.Pedestrian import PedestrianIdentifier

from task_generator.shared import DynamicObstacle, Pose, Position

from .goal_traversal import (
    GoalTraversal,
    expand_goal_sequence,
    parse_goal_traversal,
    resolve_goal_traversal,
)


@attrs.define
class PositionH(Position):
    h: float = attrs.field(default=0., converter=float)


class Goals(list[Position]):
    @classmethod
    def parse(cls, obj: dict) -> "Goals":
        waypoints = [
            Position(
                x=waypoint.get('x', 0.),
                y=waypoint.get('y', 0.),
                z=waypoint.get('z', 0.),
            )
            for waypoint in (obj.get(wpname) for wpname in obj['goals'])
            if waypoint is not None
        ]
        return cls(waypoints)

    @classmethod
    def from_waypoint_values(cls, value) -> "Goals":
        """Parse legacy hospital scenario waypoint values.

        Some hospital_1 scenarios use ``waypoint`` (singular) with either a
        single ``[x, y, yaw]`` list or a list of such lists.  The canonical
        DynamicObstacle field is ``waypoints``; this helper bridges the legacy
        schema without requiring scenario file rewrites.
        """
        if value is None:
            return cls([])
        values = value
        if isinstance(values, dict):
            values = [values]
        elif isinstance(values, (tuple, list)) and values and all(isinstance(v, (int, float)) for v in values[:2]):
            values = [values]

        waypoints = []
        for waypoint in values or []:
            if isinstance(waypoint, dict):
                waypoints.append(Position(
                    x=waypoint.get('x', 0.),
                    y=waypoint.get('y', 0.),
                    z=waypoint.get('z', 0.),
                ))
            elif isinstance(waypoint, (tuple, list)) and len(waypoint) >= 2:
                waypoints.append(Position(x=waypoint[0], y=waypoint[1], z=waypoint[2] if len(waypoint) > 2 else 0.))
        return cls(waypoints)

    def as_poses(self) -> list[geometry_msgs.msg.Pose]:
        return [
            Pose(p).to_msg()
            for p
            in self
        ]


@attrs.define()
class HunavDynamicObstacle:

    @attrs.define()
    class Behavior:
        type: int
        state: int
        configuration: int
        duration: float = attrs.field(converter=float)
        vel: float = attrs.field(converter=float)
        dist: float = attrs.field(converter=float)
        social_force_factor: float = attrs.field(converter=float)
        goal_force_factor: float = attrs.field(converter=float)
        obstacle_force_factor: float = attrs.field(converter=float)
        other_force_factor: float = attrs.field(converter=float)
        once: bool = False

        _default: typing.ClassVar["HunavDynamicObstacle.Behavior"]

        @classmethod
        def parse(cls, obj: dict) -> "HunavDynamicObstacle.Behavior":
            return cls(
                type=obj.get('type', cls._default.type),
                state=obj.get('state', cls._default.state),
                configuration=obj.get(
                    'configuration', cls._default.configuration),
                duration=obj.get('duration', cls._default.duration),
                once=obj.get('once', cls._default.once),
                vel=obj.get('vel', cls._default.vel),
                dist=obj.get('dist', cls._default.dist),
                social_force_factor=obj.get(
                    'social_force_factor',
                    cls._default.social_force_factor),
                goal_force_factor=obj.get(
                    'goal_force_factor', cls._default.goal_force_factor),
                obstacle_force_factor=obj.get(
                    'obstacle_force_factor',
                    cls._default.obstacle_force_factor),
                other_force_factor=obj.get(
                    'other_force_factor', cls._default.other_force_factor),
            )

        def to_msg(self) -> hunav_msgs.msg.AgentBehavior:
            behavior_msg = hunav_msgs.msg.AgentBehavior()
            behavior_msg.type = self.type
            behavior_msg.configuration = self.configuration
            behavior_msg.duration = self.duration
            behavior_msg.once = self.once
            behavior_msg.vel = self.vel
            behavior_msg.dist = self.dist
            behavior_msg.goal_force_factor = self.goal_force_factor
            behavior_msg.obstacle_force_factor = self.obstacle_force_factor
            behavior_msg.social_force_factor = self.social_force_factor
            behavior_msg.other_force_factor = self.other_force_factor
            return behavior_msg

    id: int
    type: int = attrs.field(converter=lambda v: v if isinstance(v, int) else 1)
    skin: int
    name: str
    group_id: int
    init_pose: PositionH
    yaw: float
    model: PedestrianIdentifier
    goals: Goals
    extra: dict
    velocity: None
    desired_velocity: float
    radius: float
    linear_vel: float
    angular_vel: float

    behavior: Behavior
    behavior_tree: str

    cyclic_goals: bool
    goal_radius: float
    closest_obs: list

    #: How the waypoint list is consumed.  ``cyclic_goals`` above is retained
    #: because it is the wire field name and the legacy config spelling; this
    #: field is the semantic one and is what ``to_msg`` acts on.  See
    #: ``goal_traversal.py`` for why a third state needed a named mode rather
    #: than a second boolean.
    goal_traversal: GoalTraversal = attrs.field(
        default=GoalTraversal.ONCE,
        converter=parse_goal_traversal,
    )

    _default: typing.ClassVar["HunavDynamicObstacle"]

    @classmethod
    def from_dynamic_obstacle(
        cls,
        obj: DynamicObstacle,
        extra: dict | None = None,
        default_goal_traversal: "GoalTraversal | str | None" = None,
    ) -> "HunavDynamicObstacle":
        """Build a HuNav agent from a scenario entity.

        ``default_goal_traversal`` is the *run-level* traversal default (from a
        launch parameter).  It is a fallback, not an override: a scenario that
        names ``goal_traversal`` or ``cyclic_goals`` for a specific pedestrian
        still wins, because the more specific layer should not be silently
        overruled by a run-wide switch.
        """
        if extra is None:
            extra = {}
        extra = {**obj.extra, **extra}

        if 'goals' in extra:
            waypoints = Goals.parse(extra)
        elif 'waypoints' in extra:
            waypoints = Goals.from_waypoint_values(extra.get('waypoints'))
        elif 'waypoint' in extra:
            waypoints = Goals.from_waypoint_values(extra.get('waypoint'))
        else:
            waypoints = Goals([
                Position(
                    x=waypoint.x,
                    y=waypoint.y,
                )
                for waypoint
                in obj.waypoints
            ])

        if 'behavior' in extra:
            behavior = cls.Behavior.parse(extra['behavior'])
        else:
            behavior = cls.Behavior._default

        behavior_tree: str = cls._default.behavior_tree
        if 'behavior_tree' in extra:
            behavior_tree = extra['behavior_tree']
            if behavior_tree.startswith('./') and obj.included_from:
                behavior_tree = str(obj.included_from / behavior_tree)

        yaw = obj.pose.orientation.to_yaw()
        if abs(yaw) > math.tau:
            yaw = math.radians(yaw)

        # Per-pedestrian `goal_traversal` > per-pedestrian legacy `cyclic_goals`
        # > run-level default > config default.  `.get(k)` returning None means
        # the layer said nothing; `cyclic_goals: false` is an opinion, not
        # silence, so it still pins ONCE.
        goal_traversal = resolve_goal_traversal(
            explicit=extra.get('goal_traversal'),
            legacy_cyclic_goals=extra.get('cyclic_goals'),
            fallback=(
                default_goal_traversal
                if default_goal_traversal is not None
                else cls._default.goal_traversal
            ),
        )

        return cls(
            name=obj.name,
            init_pose=PositionH(
                x=extra.get('position', {}).get('x', obj.pose.position.x),
                y=extra.get('position', {}).get('y', obj.pose.position.y),
                z=extra.get('position', {}).get('z', cls._default.init_pose.z),
                h=extra.get('position', {}).get('h', cls._default.init_pose.h),
            ),
            yaw=yaw,
            model=obj.model,
            goals=waypoints,
            velocity=extra.get('velocity', cls._default.velocity),
            desired_velocity=extra.get('desired_velocity', cls._default.desired_velocity),
            radius=extra.get('radius', cls._default.radius),
            linear_vel=extra.get('linear_vel', cls._default.linear_vel),
            angular_vel=extra.get('angular_vel', cls._default.angular_vel),
            behavior=behavior,
            behavior_tree=behavior_tree,
            cyclic_goals=goal_traversal.wire_cyclic_goals,
            goal_traversal=goal_traversal,
            goal_radius=extra.get('goal_radius', cls._default.goal_radius),
            closest_obs=[],
            extra=extra,
            id=extra.get('id', cls._default.id),
            type=extra.get('type', cls._default.type),
            skin=extra.get('skin', cls._default.skin),
            group_id=extra.get('group_id', cls._default.group_id),
        )

    def to_msg(self) -> hunav_msgs.msg.Agent:
        agent_msg = hunav_msgs.msg.Agent()
        agent_msg.id = self.id
        agent_msg.name = self.name
        agent_msg.type = self.type
        agent_msg.skin = self.skin
        agent_msg.group_id = self.group_id
        agent_msg.desired_velocity = self.desired_velocity
        # self._logger.info(f"=== spawn_dynamic_obstacles_desired_velocity: {agent_msg.desired_velocity}===")
        agent_msg.radius = self.radius

        # Set position
        agent_msg.position = geometry_msgs.msg.Pose()
        agent_msg.position.position.x = self.init_pose.x
        agent_msg.position.position.y = self.init_pose.y
        agent_msg.position.position.z = 1.250000
        agent_msg.yaw = self.yaw

        # Set behavior
        agent_msg.behavior = self.behavior.to_msg()
        agent_msg.behavior_tree = self.behavior_tree

        # Set goals
        agent_msg.goal_radius = self.goal_radius

        # `cyclic_goals` is the wire field and `goal_traversal` is the semantic
        # one; they must agree or somebody has evolved one without the other and
        # their request would be silently dropped here.  Fail loudly instead.
        if bool(self.cyclic_goals) != self.goal_traversal.wire_cyclic_goals:
            raise ValueError(
                f'agent {self.name!r}: cyclic_goals={self.cyclic_goals!r} contradicts '
                f'goal_traversal={self.goal_traversal.value!r} (which implies '
                f'cyclic_goals={self.goal_traversal.wire_cyclic_goals!r}). '
                'Set goal_traversal; cyclic_goals is derived from it.'
            )

        if self.goals:
            # Reciprocation mirrors the AUTHORED route.
            transmitted, wire_cyclic_goals = expand_goal_sequence(
                list(self.goals), self.goal_traversal
            )
        else:
            # Pre-existing fallback for an agent with no authored waypoints.
            # These Positions serialise to exactly the Poses the previous literal
            # built (z defaults to 0.0), so `once` and `cyclic` are unchanged here.
            #
            # The mirror is deliberately NOT applied to this list.  There is no
            # authored route to reciprocate over -- these three coordinates are a
            # hardcoded leftover, not something any scenario asked for -- and
            # ping-ponging a route nobody wrote would be a surprising thing to
            # infer from `reciprocate`.  So on this path `reciprocate` behaves as
            # `cyclic`.  It stays visible because the per-agent log line reports
            # authored_waypoints=0 alongside transmitted_goals=3.
            transmitted = [
                Position(x=-3.133759, y=-4.166653),
                Position(x=0.997901, y=-4.131655),
                Position(x=-0.227549, y=-20.187146),
            ]
            wire_cyclic_goals = self.goal_traversal.wire_cyclic_goals

        agent_msg.cyclic_goals = wire_cyclic_goals
        agent_msg.goals = [Pose(p).to_msg() for p in transmitted]

        return agent_msg

    @classmethod
    def parse(cls, obj: dict) -> "HunavDynamicObstacle":

        if 'goals' in obj:
            waypoints = Goals.parse(obj)
        else:
            waypoints = cls._default.goals

        # Global default layer.  `goal_traversal` wins over the legacy
        # `cyclic_goals` boolean when both are present, so an operator can leave
        # the old key in place while switching modes.
        goal_traversal = resolve_goal_traversal(
            explicit=obj.get('goal_traversal'),
            legacy_cyclic_goals=obj.get('cyclic_goals'),
            fallback=cls._default.goal_traversal,
        )

        return cls(
            name=obj.get('name', cls._default.name),
            extra=obj.get('extra', cls._default.extra),
            model=obj.get('model', cls._default.model),
            goals=waypoints,
            init_pose=PositionH(
                x=obj.get('init_pose', {}).get('x', cls._default.init_pose.x),
                y=obj.get('init_pose', {}).get('y', cls._default.init_pose.y),
                z=obj.get('init_pose', {}).get('z', cls._default.init_pose.z),
                h=obj.get('init_pose', {}).get('h', cls._default.init_pose.h),
            ),
            yaw=obj.get('yaw', cls._default.yaw),
            id=obj.get("id", cls._default.id),
            behavior=cls.Behavior.parse(obj.get('behavior', {})),
            behavior_tree=obj.get('behavior_tree', cls._default.behavior_tree),
            type=obj.get('type', cls._default.type),
            skin=obj.get('skin', cls._default.skin),
            group_id=obj.get('group_id', cls._default.group_id),
            velocity=None,
            desired_velocity=obj.get('max_vel', cls._default.desired_velocity),
            radius=obj.get('radius', cls._default.radius),
            linear_vel=cls._default.linear_vel,
            angular_vel=cls._default.angular_vel,
            cyclic_goals=goal_traversal.wire_cyclic_goals,
            goal_traversal=goal_traversal,
            goal_radius=obj.get('goal_radius', cls._default.goal_radius),
            closest_obs=[],
        )


def _load_config(filename: str = "default.yaml") -> "HunavDynamicObstacle":
    """Load config from YAML file in arena_bringup configs."""

    # second priority: Install space
    config_path = os.path.join(
        get_package_share_directory("arena_bringup"),
        "configs",
        "hunav",
        filename
    )

    try:
        with open(config_path, 'r') as f:
            agent_config = yaml.safe_load(f)

        assert isinstance(agent_config, dict), f"Top-level structure in {config_path} must be a mapping"

        return HunavDynamicObstacle.parse(agent_config)

    except Exception as e:
        raise RuntimeError(f"Error loading config from {config_path}") from e


HunavDynamicObstacle.Behavior._default = HunavDynamicObstacle.Behavior(
    type=0,
    state=0,
    configuration=0,
    duration=0,
    once=False,
    vel=0.,
    dist=0.,
    social_force_factor=0.,
    goal_force_factor=0.,
    obstacle_force_factor=0.,
    other_force_factor=0.,


)

HunavDynamicObstacle._default = HunavDynamicObstacle(
    init_pose=PositionH(x=0, y=0),
    name='',
    model=PedestrianIdentifier(''),
    extra={},
    goals=Goals(),
    id=0,
    type=1,
    skin=0,
    group_id=0,
    yaw=0.,
    velocity=None,
    desired_velocity=0.,
    radius=0.,
    linear_vel=0.,
    angular_vel=0.,
    behavior=HunavDynamicObstacle.Behavior._default,
    behavior_tree='default.xml',
    cyclic_goals=False,
    goal_traversal=GoalTraversal.ONCE,
    goal_radius=0.,
    closest_obs=[],
)


HunavDynamicObstacle._default = _load_config()
HunavDynamicObstacle.Behavior._default = HunavDynamicObstacle._default.behavior


# Animation configuration (from WorldGenerator)
SKIN_TYPES: dict[int, str] = {
    0: 'elegant_man.dae',
    1: 'casual_man.dae',
    2: 'elegant_woman.dae',
    3: 'regular_man.dae',
    4: 'worker_man.dae',
    5: 'walk.dae'
}


class ANIMATION_TYPES(str, enum.Enum):
    WALK = '07_01-walk.bvh',
    WALK_FORWARD = '69_02_walk_forward.bvh',
    NORMAL_WAIT = '137_28-normal_wait.bvh',
    WALK_CHILDISH = '142_01-walk_childist.bvh',
    SLOW_WALK = '07_04-slow_walk.bvh',
    WALK_SCARED = '142_17-walk_scared.bvh',
    WALK_ANGRY = '17_01-walk_with_anger.bvh'
