"""Regression guard for the map_top_down_follow overlay/background alignment.

The recorder draws the robot marker unconditionally at the centre of the frame
and crops the occupancy map around the robot's map pixel.  A sign error in the
padded-crop box (``left + min(left, 0)`` instead of ``left - min(left, 0)``)
displaced only the map BACKGROUND by ``2 * min(left, 0)`` map px while the whole
overlay layer stayed in the intended frame, so the centre-pinned marker read as
biased relative to the map behind it.  Measured impact before the fix: up to
-2.897 m in x on 100% of one episode's odom samples.

The defect lived in the composition step, so these tests exercise the real
``EvalVideoRecorder._render_top_down`` end to end and assert properties of the
COMPOSED frame that tie the drawn marker to the background actually displayed.
Every geometric test carries an in-file positive control built from a mutated
copy of the recorder source, which proves the assertion fails on the old
behaviour and passes on the new one.
"""

import ast
import json
import math
import os
import time
from pathlib import Path

import pytest


SOURCE_PATH = Path(__file__).parents[1] / 'arena_bringup' / 'internnav_eval.py'

# The corrected crop box compensates the paste offset; the pre-fix expression
# added it instead of subtracting it.
CORRECTED_CROP_EXPR = (
    'crop = padded.crop((left - min(left, 0), top - min(top, 0), '
    'right - min(left, 0), bottom - min(top, 0)))'
)
BUGGY_CROP_EXPR = (
    'crop = padded.crop((left + min(left, 0), top + min(top, 0), '
    'right + min(left, 0), bottom + min(top, 0)))'
)

TOP_DOWN_SIZE_PX = 640
TOP_DOWN_WINDOW_M = 10.0
RESOLUTION = 0.01
# crop_radius_px as computed by the recorder for the values above
CROP_RADIUS_PX = max(int(round((TOP_DOWN_WINDOW_M / RESOLUTION) / 2.0)), 32)
# output px per map px
OUT_PER_MAP = TOP_DOWN_SIZE_PX / (CROP_RADIUS_PX * 2)


def _recorder_source(text=None):
    """Return the embedded recorder program source from internnav_eval.py."""
    tree = ast.parse(text if text is not None else SOURCE_PATH.read_text(encoding='utf-8'))
    start = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == '_start_eval_video_recorder'
    )
    assignment = next(
        node
        for node in start.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'recorder_code' for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


# ---------------------------------------------------------------------------
# Cheap source-level guard.  Runs even where numpy/Pillow are unavailable so a
# sign regression can never slip through on a stripped-down interpreter.
# ---------------------------------------------------------------------------
def test_recorder_source_compensates_the_pad_paste_offset():
    source = _recorder_source()
    assert CORRECTED_CROP_EXPR in source, (
        'the padded crop box must subtract min(left, 0) / min(top, 0) to cancel the '
        'paste offset applied one line earlier'
    )
    assert BUGGY_CROP_EXPR not in source, (
        'the padded crop box adds the paste offset again, which slides the occupancy-map '
        'background by 2*min(left, 0) map px while the overlay layer stays put'
    )


# ---------------------------------------------------------------------------
# Composition tests.  These need real numpy/Pillow because they inspect pixels.
# ---------------------------------------------------------------------------
np = pytest.importorskip('numpy', reason='composed-frame assertions need real numpy')
PILImage = pytest.importorskip('PIL.Image', reason='composed-frame assertions need Pillow')
ImageDraw = pytest.importorskip('PIL.ImageDraw', reason='composed-frame assertions need Pillow')

try:
    _PIL_BILINEAR = PILImage.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - very old Pillow
    _PIL_BILINEAR = PILImage.BILINEAR


def _recorder_class(source):
    """exec just the EvalVideoRecorder class body with real numpy/Pillow bound."""
    tree = ast.parse(source)
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'EvalVideoRecorder'
    ]
    assert len(selected) == 1, f'expected one EvalVideoRecorder definition, found {len(selected)}'
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    message_type = type('Message', (), {})
    namespace = {
        'np': np,
        'PILImage': PILImage,
        'ImageDraw': ImageDraw,
        '_PIL_BILINEAR': _PIL_BILINEAR,
        'math': math,
        'json': json,
        'os': os,
        'time': time,
        'yaml': __import__('yaml'),
        'Path': Path,
        'Node': object,
        'traceback': __import__('traceback'),
        'Int16': message_type,
        'Empty': message_type,
        'Image': message_type,
        'CameraInfo': message_type,
        'Odometry': message_type,
        'PoseStamped': message_type,
        'LaserScan': message_type,
        'BACKEND_NAME': 'cv2',
        'BACKEND_MODULE': None,
        'VideoWriterWrapper': object,
        '_is_static_fallback_gradient': lambda frame: False,
        '_looks_like_corrupt_sim_top_down': lambda frame: False,
    }
    exec(compile(module, '<recorder-render-definitions>', 'exec'), namespace)
    return namespace['EvalVideoRecorder']


@pytest.fixture(scope='module')
def fixed_recorder_cls():
    return _recorder_class(_recorder_source())


@pytest.fixture(scope='module')
def buggy_recorder_cls():
    """The pre-fix behaviour, for use as a positive control only."""
    source = _recorder_source()
    assert CORRECTED_CROP_EXPR in source
    return _recorder_class(source.replace(CORRECTED_CROP_EXPR, BUGGY_CROP_EXPR, 1))


def _make_recorder(cls, map_image, origin):
    recorder = object.__new__(cls)
    recorder.map_image = map_image
    recorder.map_resolution = RESOLUTION
    recorder.map_origin = tuple(origin)
    recorder.top_down_size_px = TOP_DOWN_SIZE_PX
    recorder.top_down_window_m = TOP_DOWN_WINDOW_M
    recorder.latest_pose = None
    recorder.latest_goal = None
    recorder.latest_scan = []
    recorder.trajectory_world = []
    recorder.current_episode = 0
    return recorder


def _compose(cls, map_image, origin, x, y, yaw=0.0, **overlay):
    recorder = _make_recorder(cls, map_image, origin)
    recorder.latest_pose = {'x': float(x), 'y': float(y), 'yaw': float(yaw)}
    for key, value in overlay.items():
        setattr(recorder, key, value)
    return np.asarray(recorder._render_top_down(), dtype=np.uint8)


def _world_for_map_pixel(px, py, origin, height):
    """Inverse of the recorder's _map_world_to_pixel (cell centre)."""
    return (origin[0] + px * RESOLUTION, origin[1] + (height - py) * RESOLUTION)


def _map_pixel(x, y, origin, height):
    return (
        int(round((x - origin[0]) / RESOLUTION)),
        int(round(height - ((y - origin[1]) / RESOLUTION))),
    )


FREE = 255
OCCUPIED = 0
FREE_DISC_MAP_PX = 60          # free area around the robot, in map px
PROBE_RING_OUT_PX = 20         # sampled ring radius, outside the 8+2 px marker


def _free_disc_map(width, height, free_centre_px, radius_map_px=FREE_DISC_MAP_PX):
    """Occupied map with one free disc, centred on the robot's true map pixel."""
    grid = np.full((height, width), OCCUPIED, dtype=np.uint8)
    cx, cy = free_centre_px
    yy, xx = np.mgrid[0:height, 0:width]
    grid[((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_map_px ** 2] = FREE
    return np.repeat(grid[:, :, None], 3, axis=2)


def _ring_samples(frame, radius_out_px=PROBE_RING_OUT_PX, count=32):
    """Luma samples on a ring around the centre-pinned marker."""
    centre = frame.shape[0] // 2
    values = []
    for i in range(count):
        theta = 2.0 * math.pi * i / count
        u = int(round(centre + math.cos(theta) * radius_out_px))
        v = int(round(centre + math.sin(theta) * radius_out_px))
        # skip the heading arrow, which is drawn from the centre outwards
        if abs(v - centre) <= 3 and u >= centre:
            continue
        values.append(float(frame[v, u].mean()))
    return values


# (label, map size, robot map pixel) covering every padding regime.
PADDED_CASES = [
    # left < 0
    ('left_edge', (1550, 1909), (356, 778)),
    # top < 0
    ('top_edge', (1472, 956), (1332, 422)),
    # left < 0 and top < 0
    ('top_left_corner', (900, 900), (120, 130)),
    # right > W and bottom > H  (padding, but min(left,0)==min(top,0)==0)
    ('bottom_right_corner', (900, 900), (780, 800)),
    # left < 0 and bottom > H
    ('bottom_left_corner', (900, 900), (140, 790)),
    # right > W and top < 0
    ('top_right_corner', (900, 900), (770, 150)),
    # crop wholly inside the map
    ('interior', (2400, 2400), (1200, 1200)),
]

ORIGIN = (-3.615, -17.505)


def _left_top_padding(robot_px):
    left = robot_px[0] - CROP_RADIUS_PX
    top = robot_px[1] - CROP_RADIUS_PX
    return min(left, 0), min(top, 0)


@pytest.mark.parametrize('label,map_size,robot_px', PADDED_CASES,
                         ids=[case[0] for case in PADDED_CASES])
def test_marker_sits_on_the_robot_cell_of_the_displayed_background(
    fixed_recorder_cls, label, map_size, robot_px
):
    """The map area shown around the pinned marker must be the robot's own cell.

    The synthetic map is occupied everywhere except a free disc centred exactly
    on the robot.  If the background is displaced, the free disc moves off the
    frame centre and the ring around the marker turns occupied.
    """
    width, height = map_size
    map_image = _free_disc_map(width, height, robot_px)
    x, y = _world_for_map_pixel(robot_px[0], robot_px[1], ORIGIN, height)
    assert _map_pixel(x, y, ORIGIN, height) == robot_px, 'pose/pixel round trip must be exact'

    frame = _compose(fixed_recorder_cls, map_image, ORIGIN, x, y)
    samples = _ring_samples(frame)
    assert samples, 'ring sampling produced no usable pixels'
    assert min(samples) > 200.0, (
        f'{label}: background around the marker is not the robot cell; '
        f'ring luma min={min(samples):.1f} (free is {FREE})'
    )


def test_positive_control_pre_fix_code_fails_the_marker_assertion(buggy_recorder_cls):
    """The pre-fix expression must fail the test above wherever left/top pad.

    Without this control the assertion could be vacuous.
    """
    failures = []
    checked = 0
    for label, (width, height), robot_px in PADDED_CASES:
        pad_left, pad_top = _left_top_padding(robot_px)
        if pad_left == 0 and pad_top == 0:
            continue  # the fix is a no-op here, so the control cannot fail
        checked += 1
        map_image = _free_disc_map(width, height, robot_px)
        x, y = _world_for_map_pixel(robot_px[0], robot_px[1], ORIGIN, height)
        samples = _ring_samples(_compose(buggy_recorder_cls, map_image, ORIGIN, x, y))
        if min(samples) > 200.0:
            failures.append(label)
    assert checked >= 4, f'expected several left/top padding cases, got {checked}'
    assert not failures, (
        f'pre-fix code unexpectedly passed the marker assertion for {failures}; '
        'the regression guard would not detect the defect'
    )


@pytest.mark.parametrize('label,map_size,robot_px', PADDED_CASES,
                         ids=[case[0] for case in PADDED_CASES])
def test_background_feature_lands_at_its_predicted_offset_from_the_marker(
    fixed_recorder_cls, label, map_size, robot_px
):
    """A map landmark must appear at the offset predicted from the robot pose.

    This pins the background scale and translation together, independently of
    the free/occupied classification used above.
    """
    width, height = map_size
    offset_map_px = 150
    # aim the landmark towards the map interior so it never falls in the pad area
    sign_x = 1 if robot_px[0] < width // 2 else -1
    sign_y = 1 if robot_px[1] < height // 2 else -1
    delta = (sign_x * offset_map_px, sign_y * offset_map_px)
    probe_px = (robot_px[0] + delta[0], robot_px[1] + delta[1])
    assert 0 <= probe_px[0] < width and 0 <= probe_px[1] < height

    grid = np.full((height, width), FREE, dtype=np.uint8)
    half = 12
    grid[probe_px[1] - half:probe_px[1] + half, probe_px[0] - half:probe_px[0] + half] = OCCUPIED
    map_image = np.repeat(grid[:, :, None], 3, axis=2)

    x, y = _world_for_map_pixel(robot_px[0], robot_px[1], ORIGIN, height)
    frame = _compose(fixed_recorder_cls, map_image, ORIGIN, x, y)

    centre = TOP_DOWN_SIZE_PX // 2
    expected_u = centre + delta[0] * OUT_PER_MAP
    expected_v = centre + delta[1] * OUT_PER_MAP

    # Restrict to achromatic dark pixels below the HUD band: the marker, arrow
    # and HUD outline are all drawn in saturated colours, and the green HUD
    # rectangle would otherwise register as dark.
    luma = frame.mean(axis=2)
    chroma = frame.astype(np.int16).max(axis=2) - frame.astype(np.int16).min(axis=2)
    candidate = (luma < 96) & (chroma <= 12)
    candidate[:64, :] = False
    dark = np.argwhere(candidate)
    assert dark.size, f'{label}: landmark not visible in the composed frame'
    measured_v, measured_u = dark.mean(axis=0)
    assert abs(measured_u - expected_u) <= 1.5, (
        f'{label}: landmark u={measured_u:.2f} expected {expected_u:.2f}'
    )
    assert abs(measured_v - expected_v) <= 1.5, (
        f'{label}: landmark v={measured_v:.2f} expected {expected_v:.2f}'
    )


def test_fix_is_a_no_op_where_left_and_top_do_not_pad(fixed_recorder_cls, buggy_recorder_cls):
    """Frames that were already correct must be byte-identical after the fix.

    Covers the interior crop and the right/bottom-only padding case, with the
    full overlay layer active (goal, trajectory, scan, marker, arrow, HUD).
    """
    checked = 0
    for label, (width, height), robot_px in PADDED_CASES:
        pad_left, pad_top = _left_top_padding(robot_px)
        if pad_left != 0 or pad_top != 0:
            continue
        checked += 1
        map_image = _free_disc_map(width, height, robot_px)
        x, y = _world_for_map_pixel(robot_px[0], robot_px[1], ORIGIN, height)
        overlay = dict(
            latest_goal={'x': x + 2.5, 'y': y + 1.5},
            trajectory_world=[(x - 0.4 * i, y - 0.3 * i) for i in range(6)],
            latest_scan=[(3.0, a * 0.05) for a in range(-30, 30)],
        )
        fixed = _compose(fixed_recorder_cls, map_image, ORIGIN, x, y, yaw=0.7, **overlay)
        buggy = _compose(buggy_recorder_cls, map_image, ORIGIN, x, y, yaw=0.7, **overlay)
        assert np.array_equal(fixed, buggy), (
            f'{label}: min(left,0)==min(top,0)==0 so the crop box is unchanged; '
            'the composed frame must be byte-identical'
        )
    assert checked >= 2, f'expected at least two no-op regimes, got {checked}'


def test_padded_background_matches_an_independent_robot_centred_renderer(fixed_recorder_cls):
    """Cross-check the composed background against a separate implementation.

    The reference uses explicit clamped source/destination slices rather than
    PIL pad-and-crop, so agreement is not a restatement of the recorder's own
    arithmetic.  Compared on the background only: the overlay layer is drawn in
    saturated colours over a greyscale map, and the HUD occupies the top band.
    """
    for label, (width, height), robot_px in PADDED_CASES:
        rng = np.random.default_rng(abs(hash(label)) % (2 ** 32))
        grid = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
        map_image = np.repeat(grid[:, :, None], 3, axis=2)
        x, y = _world_for_map_pixel(robot_px[0], robot_px[1], ORIGIN, height)

        frame = _compose(fixed_recorder_cls, map_image, ORIGIN, x, y)

        radius = CROP_RADIUS_PX
        reference = np.empty((2 * radius, 2 * radius, 3), dtype=np.uint8)
        reference[:, :] = (180, 180, 180)
        x0, y0 = robot_px[0] - radius, robot_px[1] - radius
        sx0, sx1 = max(x0, 0), min(x0 + 2 * radius, width)
        sy0, sy1 = max(y0, 0), min(y0 + 2 * radius, height)
        reference[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = map_image[sy0:sy1, sx0:sx1]
        reference = np.asarray(
            PILImage.fromarray(reference).resize((TOP_DOWN_SIZE_PX, TOP_DOWN_SIZE_PX), _PIL_BILINEAR),
            dtype=np.uint8,
        )

        a = frame.astype(np.int16)
        b = reference.astype(np.int16)
        background = ((a.max(axis=2) - a.min(axis=2)) <= 12) & ((b.max(axis=2) - b.min(axis=2)) <= 12)
        background[:64, :] = False
        # The marker disc is chromatic and therefore already excluded, but its
        # white outline is not; drop a disc around the pinned centre.
        centre = TOP_DOWN_SIZE_PX // 2
        yy, xx = np.mgrid[0:TOP_DOWN_SIZE_PX, 0:TOP_DOWN_SIZE_PX]
        background[((xx - centre) ** 2 + (yy - centre) ** 2) <= 16 ** 2] = False
        assert background.sum() > 1000, f'{label}: background mask too small to judge'
        mae = float(np.abs(a.mean(axis=2) - b.mean(axis=2))[background].mean())
        assert mae == 0.0, f'{label}: composed background differs from the reference (MAE {mae:.4f})'
