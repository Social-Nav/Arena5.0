"""Relay node: re-stamp LaserScan to current sim-time before forwarding to nav2.

Nav2 costmap looks up TF using the scan's header.stamp.  After a robot
teleportation (e.g., episode reset in Isaac Sim), buffered scans carry
pre-teleport timestamps → nav2 fetches the old TF (robot at odom origin) →
obstacles are projected to the robot's old position instead of the current one.

This node re-publishes the LaserScan with header.stamp = now(), so nav2
always resolves the transform against the robot's *current* pose.

Keeping the message as LaserScan (not PointCloud2) preserves ray-trace
clearing: the costmap can free cells between the sensor and each detected
range endpoint, which is not possible with a plain PointCloud2.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LaserScanToCloud(Node):
    """Re-stamp and relay a LaserScan topic."""

    def __init__(self):
        super().__init__('laser_scan_to_cloud')

        self.declare_parameter('scan_topic', 'laser_scan')
        self.declare_parameter('cloud_topic', 'lidar_scan_cloud')

        scan_topic = self.get_parameter('scan_topic').value
        cloud_topic = self.get_parameter('cloud_topic').value

        self._pub = self.create_publisher(LaserScan, cloud_topic, 10)
        self._sub = self.create_subscription(
            LaserScan, scan_topic, self._callback, 10
        )

        self.get_logger().info(
            f'laser_scan_to_cloud (re-stamp): {scan_topic} → {cloud_topic}'
        )

    def _callback(self, scan: LaserScan) -> None:
        # Replace the original (potentially stale) timestamp with the current
        # sim-time so nav2 looks up the TF at the robot's *current* pose.
        scan.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanToCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
