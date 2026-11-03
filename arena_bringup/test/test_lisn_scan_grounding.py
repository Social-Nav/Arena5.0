"""Focused CPU tests for the A-track scan-range grounding resolver."""
import math
import unittest

from arena_bringup.lisn_grounding import GroundingError, resolve_scan_grounded_goal


K = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
N = 1600
AMIN = -math.pi
INC = 2.0 * math.pi / N


def _goal(x=320.0, y=240.0):
    return {
        'kind': 'pixel_goal', 'units': 'pixel',
        'frame_id': 'cam', 'confidence': 0.9,
        'value': {'x': x, 'y': y},
    }


def _resolve(ranges, **overrides):
    kwargs = dict(
        goal=_goal(), width=640, height=480, camera_info_k=K,
        tf_map_camera=IDENTITY, scan_ranges=ranges,
        scan_angle_min=AMIN, scan_angle_increment=INC,
        scan_range_min=0.8, scan_range_max=200.0,
        tf_map_scan=IDENTITY, expected_camera_frame='cam',
        expected_scan_frame='scan', source_stamp_ns=100,
        scan_stamp_ns=90, scan_age_ns=10, max_scan_age_ns=600_000_000,
        min_confidence=0.5, max_scan_range_m=12.0, beam_search_radius=12,
    )
    kwargs.update(overrides)
    return resolve_scan_grounded_goal(**kwargs)


class ScanGroundingTest(unittest.TestCase):
    def test_center_beam_is_measured_range(self):
        result = _resolve([2.0] * N)
        self.assertTrue(result['trusted'])
        self.assertAlmostEqual(result['map_pose']['x'], 2.0, places=6)
        self.assertEqual(result['scan']['beam_count'], N)
        self.assertLess(abs(result['scan']['angular_offset_rad']), 1e-6)

    def test_invalid_center_searches_bounded_neighbour_beam(self):
        ranges = [float('nan')] * N
        ranges[N // 2] = float('nan')
        ranges[N // 2 + 1] = 2.0
        result = _resolve(ranges)
        self.assertTrue(result['trusted'])
        self.assertGreaterEqual(abs(result['scan']['angular_offset_rad']), INC / 2)

    def test_age_bound_rejects(self):
        with self.assertRaises(GroundingError) as ctx:
            _resolve([2.0] * N, scan_age_ns=700_000_000)
        self.assertIn('scan_age_exceeds_bound', str(ctx.exception))

    def test_frame_mismatch_rejects(self):
        with self.assertRaises(GroundingError) as ctx:
            _resolve([2.0] * N, expected_camera_frame='other')
        self.assertIn('camera_frame_mismatch', str(ctx.exception))

    def test_no_valid_range_rejects(self):
        with self.assertRaises(GroundingError) as ctx:
            _resolve([float('nan')] * N)
        self.assertIn('no_valid_scan_range', str(ctx.exception))

    def test_far_range_rejects(self):
        with self.assertRaises(GroundingError) as ctx:
            _resolve([20.0] * N)
        self.assertIn('no_valid_scan_range', str(ctx.exception))

    def test_pixel_out_of_bounds_rejects(self):
        with self.assertRaises(GroundingError) as ctx:
            _resolve([2.0] * N, goal=_goal(x=900.0, y=240.0))
        self.assertIn('point_out_of_image', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
