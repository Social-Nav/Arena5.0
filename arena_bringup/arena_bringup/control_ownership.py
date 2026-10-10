"""Read-only ROS graph check before starting an external velocity controller."""
import argparse
from pathlib import Path
import time


def main(argv=None):
    import rclpy
    from rclpy.node import Node
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', default='Ai2_Bot2')
    parser.add_argument('--evaluation', action='store_true', help='refuse an already running benchmark evaluation')
    args = parser.parse_args(argv)
    if args.evaluation:
        owners = []
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                command = path.read_bytes().split(bytes([0]))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(arg.rsplit(b'/', 1)[-1] == b'internnav_eval' for arg in command):
                owners.append(path.parent.name)
        if owners:
            print('Refusing to replace active Arena evaluation; container PIDs: ' + ', '.join(owners))
            return 1
        return 0
    rclpy.init()
    node = Node('arena_control_ownership_probe')
    topic = f'/task_generator_node/{args.robot}/cmd_vel'
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        owners = node.get_publishers_info_by_topic(topic)
        if owners:
            print('Refusing to replace active control on ' + topic + ': ' + ', '.join(
                owner.node_namespace + '/' + owner.node_name for owner in owners))
            return 1
        print('Available control channel: ' + topic)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
