import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from arena_bringup.internnav_timing_manager import InternNavTimingManager


def _clock_msg(sec: float) -> Clock:
    msg = Clock()
    whole = int(sec)
    msg.clock.sec = whole
    msg.clock.nanosec = int((sec - whole) * 1_000_000_000)
    return msg


def _ready_msg(ready: bool = True) -> String:
    msg = String()
    msg.data = json.dumps({"stage": "episode", "ready": ready, "episode": 0})
    return msg


def _twist(linear_x: float, angular_z: float) -> Twist:
    msg = Twist()
    msg.linear.x = linear_x
    msg.angular.z = angular_z
    return msg


def test_sim_time_realworld_delays_raw_commands(tmp_path):
    if not rclpy.ok():
        rclpy.init()

    node = InternNavTimingManager(
        parameter_overrides=[
            Parameter('timing_mode', Parameter.Type.STRING, 'sim_time_realworld'),
            Parameter('model_latency_sec', Parameter.Type.DOUBLE, 0.3),
            Parameter('record_data_dir', Parameter.Type.STRING, str(tmp_path)),
            Parameter('action_hold_policy', Parameter.Type.STRING, 'zero'),
        ]
    )
    published = []
    node.cmd_pub.publish = published.append

    try:
        node._on_clock(_clock_msg(10.0))
        node._on_eval_ready(_ready_msg(True))
        node._on_raw_cmd(_twist(0.4, 0.2))

        node._on_timer()
        assert published[-1].linear.x == 0.0

        node._on_clock(_clock_msg(10.29))
        node._on_timer()
        assert published[-1].linear.x == 0.0

        node._on_clock(_clock_msg(10.31))
        node._on_timer()
        assert published[-1].linear.x == 0.4
        assert published[-1].angular.z == 0.2
        assert node._summary()["released_cmd_count"] == 1
        assert node._summary()["timing_valid_for_realworld"] is True
    finally:
        node.destroy_node()


def test_measured_latency_policy_uses_status_elapsed(tmp_path):
    if not rclpy.ok():
        rclpy.init()

    node = InternNavTimingManager(
        parameter_overrides=[
            Parameter('timing_mode', Parameter.Type.STRING, 'sim_time_realworld'),
            Parameter('latency_policy', Parameter.Type.STRING, 'measured'),
            Parameter('model_latency_sec', Parameter.Type.DOUBLE, 0.3),
            Parameter('record_data_dir', Parameter.Type.STRING, str(tmp_path)),
        ]
    )

    try:
        status = String()
        status.data = json.dumps({"status": "planning_response_received", "debug": {"elapsed_sec": 0.47}})
        node._on_status(status)

        node._on_clock(_clock_msg(2.0))
        node._on_eval_ready(_ready_msg(True))
        node._on_raw_cmd(_twist(0.1, -0.2))

        assert node.pending[0].delay_sec == 0.47
        assert abs(node.pending[0].eligible_sim_time - 2.47) < 1e-9
    finally:
        node.destroy_node()
