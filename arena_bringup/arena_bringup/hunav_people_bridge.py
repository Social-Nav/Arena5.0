"""LISN-only bridge: real HuNav human tracks -> people_msgs/People.

The SocialMPC controller subscribes to /people with SensorDataQoS.  The
existing arena_peds_to_people_bridge consumes arena_peds, which is empty in
the HuNav lane; this bridge consumes the real hunav_msgs/Agents human_states
and publishes the two tracked humans only.  It never converts semantic
entities to people.
"""
from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PointStamped
from hunav_msgs.msg import Agents
from people_msgs.msg import People, Person


class HunavPeopleBridge(Node):
    def __init__(self) -> None:
        super().__init__('lisn_hunav_people_bridge')
        self.declare_parameter('input_topic', '/task_generator_node/human_states')
        self.declare_parameter('output_topic', '/people')
        self.declare_parameter('output_frame', 'Ai2_Bot2/odom')
        self.declare_parameter('receipt_path', '')
        input_topic = str(self.get_parameter('input_topic').value or '').strip()
        output_topic = str(self.get_parameter('output_topic').value or '/people').strip()
        self._output_frame = str(self.get_parameter('output_frame').value or '').strip()
        self._receipt_path = str(self.get_parameter('receipt_path').value or '').strip()
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=15.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(People, output_topic, qos)
        self._sub = self.create_subscription(Agents, input_topic, self._callback, 10)
        self._count = 0
        self.get_logger().warn(
            f'[lisn_hunav_people_bridge] {input_topic} -> {output_topic} '
            f'(real HuNav humans only)'
        )

    def _callback(self, msg: Agents) -> None:
        out = People()
        out.header = msg.header
        source_frame = str(msg.header.frame_id or 'map')
        output_frame = self._output_frame or source_frame
        out.header.frame_id = output_frame
        source_people = []
        transform_errors = []
        for agent in msg.agents:
            if int(getattr(agent, 'type', 0)) != 1:
                continue
            source_position = agent.position.position
            source_velocity = agent.velocity.linear
            source_people.append({
                'name': str(getattr(agent, 'name', '') or getattr(agent, 'id', '')),
                'position': {
                    'x': float(source_position.x),
                    'y': float(source_position.y),
                    'z': float(source_position.z),
                },
            })
            x = float(source_position.x)
            y = float(source_position.y)
            z = float(source_position.z)
            if source_frame != output_frame:
                try:
                    # lookup_transform avoids passing a PointStamped object;
                    # the /opt/venv tf2 buffer does not register that Python
                    # message type in this mixed-binding environment.
                    transform = self._tf_buffer.lookup_transform(
                        output_frame, source_frame, Time(),
                        timeout=Duration(seconds=0.2),
                    )
                    translation = transform.transform.translation
                    rotation = transform.transform.rotation
                    qx, qy, qz, qw = (
                        float(rotation.x), float(rotation.y),
                        float(rotation.z), float(rotation.w),
                    )
                    r00 = 1 - 2 * (qy * qy + qz * qz)
                    r01 = 2 * (qx * qy - qz * qw)
                    r02 = 2 * (qx * qz + qy * qw)
                    r10 = 2 * (qx * qy + qz * qw)
                    r11 = 1 - 2 * (qx * qx + qz * qz)
                    r12 = 2 * (qy * qz - qx * qw)
                    r20 = 2 * (qx * qz - qy * qw)
                    r21 = 2 * (qy * qz + qx * qw)
                    r22 = 1 - 2 * (qx * qx + qy * qy)
                    x, y, z = (
                        r00 * x + r01 * y + r02 * z + float(translation.x),
                        r10 * x + r11 * y + r12 * z + float(translation.y),
                        r20 * x + r21 * y + r22 * z + float(translation.z),
                    )
                except Exception as exc:
                    transform_errors.append({
                        'person': source_people[-1]['name'],
                        'type': type(exc).__name__,
                        'message': str(exc)[:200],
                    })
                    self.get_logger().warn(
                        f'[lisn_hunav_people_bridge] skip person '
                        f'{source_people[-1]["name"]}: transform {source_frame}->'
                        f'{output_frame} failed: {type(exc).__name__}'
                    )
                    continue
            person = Person()
            person.name = source_people[-1]['name']
            # Copy scalars instead of assigning message objects: hunav_msgs and
            # people_msgs may come from different Python message bindings.
            person.position.x = x
            person.position.y = y
            person.position.z = z
            person.velocity.x = float(source_velocity.x)
            person.velocity.y = float(source_velocity.y)
            person.velocity.z = float(source_velocity.z)
            person.reliability = 1.0
            out.people.append(person)
        self._pub.publish(out)
        self._count += 1
        if self._receipt_path and self._count % 10 == 1:
            receipt = {
                'schema': 'arena.lisn.hunav_people_bridge.v1',
                'input_topic': str(self.get_parameter('input_topic').value),
                'output_topic': str(self.get_parameter('output_topic').value),
                'published_count': self._count,
                'input_frame': source_frame,
                'output_frame': output_frame,
                'source_people': source_people,
                'transform_errors': transform_errors,
                'header': {
                    'frame_id': out.header.frame_id,
                    'stamp': {
                        'sec': int(out.header.stamp.sec),
                        'nanosec': int(out.header.stamp.nanosec),
                    },
                },
                'people': [
                    {
                        'name': person.name,
                        'position': {
                            'x': float(person.position.x),
                            'y': float(person.position.y),
                            'z': float(person.position.z),
                        },
                        'velocity': {
                            'x': float(person.velocity.x),
                            'y': float(person.velocity.y),
                            'z': float(person.velocity.z),
                        },
                    }
                    for person in out.people
                ],
            }
            path = Path(self._receipt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


def main(args=None) -> int:
    rclpy.init(args=args)
    node = HunavPeopleBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
