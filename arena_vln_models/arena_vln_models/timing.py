"""Passive simulation clock observer. Never subscribes or publishes cmd_vel."""
import csv
import json
from pathlib import Path
import time
from statistics import median, quantiles


def main():
    import rclpy
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock

    class Observer(Node):
        def __init__(self):
            super().__init__('timing_observer')
            self.declare_parameter('record_data_dir', '')
            root = self.get_parameter('record_data_dir').value
            self.root = Path(root) if root else None
            self.previous = None
            self.samples = []
            if self.root:
                self.root.mkdir(parents=True, exist_ok=True)
                with (self.root / 'rtf.csv').open('w') as stream:
                    csv.writer(stream).writerow(['wall_time', 'sim_time', 'dt_wall_sec', 'dt_sim_sec', 'rtf'])
            self.create_subscription(Clock, '/clock', self.clock, 10)

        def clock(self, msg):
            wall = time.monotonic()
            sim = msg.clock.sec + msg.clock.nanosec / 1e9
            if self.previous is None or sim < self.previous[1]:
                self.previous = wall, sim
                return
            dw, ds = wall-self.previous[0], sim-self.previous[1]
            if dw < .5:
                return
            self.previous = wall, sim
            rtf = ds/dw
            self.samples.append(rtf)
            if self.root:
                with (self.root / 'rtf.csv').open('a') as stream:
                    csv.writer(stream).writerow([time.time(), sim, dw, ds, rtf])
                (self.root / 'timing_summary.json').write_text(json.dumps({
                    'execution_mode': 'nav2_follow_path', 'rtf_sample_count': len(self.samples),
                    'rtf_mean': sum(self.samples)/len(self.samples), 'rtf_min': min(self.samples),
                    'rtf_max': max(self.samples), 'rtf_p50': median(self.samples),
                    'rtf_p95': quantiles(self.samples, n=100, method='inclusive')[94] if len(self.samples) > 1 else rtf}, indent=2))

    rclpy.init()
    node = Observer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
