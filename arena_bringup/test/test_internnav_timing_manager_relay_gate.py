"""Lane Y2 — the command-velocity relay gate on ``internnav_timing_manager``.

Defect under test (D5).  In production (``timing_mode='wall'`` with
``internnav_direct_cmd_vel: true``) the official InternNav client publishes **directly** onto the
robot's ``cmd_vel``.  The timing manager is launched unconditionally, its input topic
``internnav/raw_cmd_vel`` has **zero** publishers so it can never forward anything, and its only
output is a 20 Hz **zero-velocity** stream onto that same ``cmd_vel``.  Those zeros are consumed:
``ROS2SubscribeTwist.execOut`` fires only on a new message and gates both controllers, and
``IsaacArticulationController`` writes PhysX joint **velocity targets**, so a consumed zero is a
latched stop that persists until the next consumed message.

The fix gates the **relay** responsibility inside the node and leaves the **clock observation**
and every timing artifact untouched.  A launch-level ``condition=`` was rejected: this node is the
sole producer of ``rtf.csv``, ``internnav_timing_summary.json`` and
``internnav_timing_trace.jsonl``, and ``artifact_validation.json`` references none of them, so
suppressing the node would destroy the RTF record **silently**.

Two test-design constraints are deliberate and load-bearing:

1. **Emissions are captured publisher-agnostically.**  The fix does not create ``cmd_pub`` at all
   in ``wall`` mode, so the older idiom ``node.cmd_pub.publish = published.append`` would raise
   ``AttributeError`` — and an ``AttributeError`` in a test that expects zero emissions looks
   exactly like a pass.  ``_capture_emissions`` therefore never raises when no publisher exists,
   and it is proven to capture 20 emissions against the unmodified module.
2. **Every "emitted nothing" assertion is paired with a positive count.**  Asserting only
   ``== 0`` would also pass if the callback never ran.  ``suppressed_cmd_count`` proves the path
   executed, so the tests cannot pass vacuously.
"""

from __future__ import annotations

import ast
import itertools
import json
import pathlib

import pytest
import rclpy
from geometry_msgs.msg import Twist
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Int16, String

from arena_bringup.internnav_timing_manager import InternNavTimingManager

# `relay_enabled_for_timing_mode` is deliberately NOT imported at module scope.  It does not exist
# in the pre-fix module, and a module-scope import of it turns the entire file into a single
# ImportError collection error when run against pre-fix code — which would destroy the whole point
# of the defect tests below, whose value is that they fail with a captured emission COUNT.  It is
# imported inside the one test that is about the function itself.

_UNIQUE = itertools.count()


def _ensure_rclpy() -> None:
    if not rclpy.ok():
        rclpy.init()


def _clock_msg(sec: float) -> Clock:
    msg = Clock()
    whole = int(sec)
    msg.clock.sec = whole
    msg.clock.nanosec = int(round((sec - whole) * 1_000_000_000))
    return msg


def _ready_msg(ready: bool = True) -> String:
    msg = String()
    msg.data = json.dumps({'stage': 'episode', 'ready': ready, 'episode': 0})
    return msg


def _twist(linear_x: float = 0.0, angular_z: float = 0.0) -> Twist:
    msg = Twist()
    msg.linear.x = linear_x
    msg.angular.z = angular_z
    return msg


def _make_node(tmp_path, *, timing_mode: str, **params) -> InternNavTimingManager:
    """Build the node with per-test unique topic names.

    Unique names matter: pytest shares one rclpy context across tests, so a fixed topic name would
    let one test's publisher be counted by another test's ``count_publishers`` check.
    """
    tag = f'{next(_UNIQUE)}'
    overrides = [
        Parameter('timing_mode', Parameter.Type.STRING, timing_mode),
        Parameter('record_data_dir', Parameter.Type.STRING, str(tmp_path)),
        Parameter('input_cmd_vel_topic', Parameter.Type.STRING, f'y2_raw_cmd_vel_{tag}'),
        Parameter('output_cmd_vel_topic', Parameter.Type.STRING, f'y2_cmd_vel_{tag}'),
    ]
    for name, value in params.items():
        if isinstance(value, bool):
            overrides.append(Parameter(name, Parameter.Type.BOOL, value))
        elif isinstance(value, float):
            overrides.append(Parameter(name, Parameter.Type.DOUBLE, value))
        elif isinstance(value, int):
            overrides.append(Parameter(name, Parameter.Type.INTEGER, value))
        else:
            overrides.append(Parameter(name, Parameter.Type.STRING, str(value)))
    _ensure_rclpy()
    return InternNavTimingManager(parameter_overrides=overrides)


def _capture_emissions(node: InternNavTimingManager) -> list:
    """Capture output-topic emissions WITHOUT assuming a publisher exists.

    Returns a list that receives every emitted message.  When the node created no publisher the
    list simply stays empty; nothing raises.  Against the unmodified module a publisher does exist,
    so the same helper collects the 20-message zero flood — which is how these tests fail for a
    functional reason (a count) rather than with an ``AttributeError``.
    """
    emitted: list = []
    pub = getattr(node, 'cmd_pub', None)
    if pub is not None:
        pub.publish = emitted.append
    return emitted


def _suppressed(node: InternNavTimingManager) -> int:
    """Suppressed-emission count, tolerant of the pre-fix module (which has no counter).

    Returning ``-1`` rather than raising keeps the *first* failing assertion in every test the
    functional emission count, so the recorded pre-fix failure is a number and not an attribute
    error.
    """
    return int(getattr(node, 'suppressed_cmd_count', -1))


def _output_topic(node: InternNavTimingManager) -> str:
    """Read the output topic from the ROS parameter, not from a post-fix attribute.

    Using the parameter keeps the defect tests runnable against the pre-fix module, so they fail on
    a publisher count rather than on a missing attribute.
    """
    return str(node.get_parameter('output_cmd_vel_topic').value)


def _input_topic(node: InternNavTimingManager) -> str:
    return str(node.get_parameter('input_cmd_vel_topic').value)


# --------------------------------------------------------------------------------------------
# The predicate itself — one named, testable expression of a rule that is also written in bash
# (`_meta/docker/features/internnav/main`, the cmd_vel redirect).
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'mode,expected',
    [
        ('wall', False),
        ('WALL', False),
        ('  wall  ', False),
        ('sim_time_realworld', True),
        ('SIM_TIME_REALWORLD', True),
        ('', True),
        (None, True),
    ],
)
def test_relay_predicate_matches_the_client_redirect_rule(mode, expected):
    """`wall` means the client keeps the robot's cmd_vel; anything else means it was redirected.

    An empty/None mode resolves to "relay", matching the node's own default handling, so a
    mis-typed mode fails loudly at the topology check rather than silently disabling the only
    command source in `sim_time_realworld`.
    """
    from arena_bringup.internnav_timing_manager import relay_enabled_for_timing_mode

    assert relay_enabled_for_timing_mode(mode) is expected


# --------------------------------------------------------------------------------------------
# T1-T4b — must FAIL against the unmodified module, for a functional reason
# --------------------------------------------------------------------------------------------


def test_t1_wall_mode_emits_nothing_while_the_episode_runs(tmp_path):
    """T1 (primary regression test): the production flood.

    `wall` mode, clock delivered, episode started, no raw command ever arrives — exactly the
    measured production state (`raw_cmd_count = 0` in 4/4 runs, `episode_started` reaching True).
    Pre-fix this emits 20 zero Twists onto the robot's own command topic.
    """
    node = _make_node(tmp_path, timing_mode='wall')
    emitted = _capture_emissions(node)
    try:
        node._on_clock(_clock_msg(10.0))
        node._on_eval_ready(_ready_msg(True))
        assert node.episode_started is True, 'test must exercise the episode-started branch'

        for _ in range(20):
            node._on_timer()

        # Functional assertion first, so the pre-fix failure is a count.
        assert len(emitted) == 0, f'emitted {len(emitted)} commands onto the robot cmd_vel topic'
        # Anti-vacuity: prove the 20 timer ticks really reached the emission seam.  eval_ready with
        # ready=True does NOT emit (its zero sits inside the `not episode_started` branch), so 20 is
        # the whole expected count.
        assert _suppressed(node) == 20, 'expected all 20 timer ticks to reach the emission seam'
        assert node.emitted_cmd_count == 0
    finally:
        node.destroy_node()


def test_t2_wall_mode_emits_nothing_before_the_episode_starts(tmp_path):
    """T2: the other zero branch, taken when `episode_started` is still False."""
    node = _make_node(tmp_path, timing_mode='wall')
    emitted = _capture_emissions(node)
    try:
        node._on_clock(_clock_msg(5.0))
        assert node.episode_started is False, 'test must exercise the not-started branch'

        for _ in range(20):
            node._on_timer()

        assert len(emitted) == 0, f'emitted {len(emitted)} pre-episode zeros'
        assert _suppressed(node) == 20
    finally:
        node.destroy_node()


def test_t3_wall_mode_registers_no_publisher_on_the_robot_command_topic(tmp_path):
    """T3: the live publisher census must not see this node at all.

    This is the offline form of the run-time Stage-0 check.  It is asserted on the real ROS graph
    via `count_publishers`, not merely on the absence of an attribute, so an implementation that
    keeps a publisher and guards each `publish` call would fail here.
    """
    node = _make_node(tmp_path, timing_mode='wall')
    try:
        assert node.count_publishers(_output_topic(node)) == 0, (
            'timing manager still registers a publisher on the robot cmd_vel topic in wall mode'
        )
        assert getattr(node, 'cmd_pub', None) is None
        assert node._summary().get('output_publisher_created') is False
    finally:
        node.destroy_node()


@pytest.mark.parametrize('hold_policy', ['hold_last', 'zero'])
def test_t4_the_gate_dominates_action_hold_policy(tmp_path, hold_policy):
    """T4: `action_hold_policy` is a live parameter; the mode gate must dominate both settings.

    Exercised with a released command already in hand, so the `hold_last` republish path at the
    end of `_on_timer` is reached too — that is a third emission site, not just `_publish_zero`.
    """
    node = _make_node(tmp_path, timing_mode='wall', action_hold_policy=hold_policy)
    emitted = _capture_emissions(node)
    try:
        node._on_clock(_clock_msg(1.0))
        node._on_eval_ready(_ready_msg(True))
        node._on_raw_cmd(_twist(0.4, 0.2))       # in wall mode delay is 0, so it is eligible at once
        node._on_timer()                          # release path
        for _ in range(5):
            node._on_timer()                      # hold_last / zero path

        assert len(emitted) == 0, f'emitted {len(emitted)} commands with action_hold_policy={hold_policy}'
        assert _suppressed(node) == 6, 'expected 1 release + 5 hold/zero ticks to reach the seam'
        assert node.released_count == 1, 'release bookkeeping must still run so telemetry stays honest'
    finally:
        node.destroy_node()


def test_t4b_wall_mode_emits_nothing_on_reset_or_not_ready(tmp_path):
    """T4b: the two `_publish_zero` callers outside `_on_timer`."""
    node = _make_node(tmp_path, timing_mode='wall')
    emitted = _capture_emissions(node)
    try:
        node._on_clock(_clock_msg(2.0))
        node._on_task_reset(Int16(data=3))
        node._on_eval_ready(_ready_msg(False))

        assert len(emitted) == 0, f'emitted {len(emitted)} zeros from reset/not-ready callbacks'
        assert _suppressed(node) == 2
        assert node.reset_episode == 3, 'reset bookkeeping must still run'
    finally:
        node.destroy_node()


# --------------------------------------------------------------------------------------------
# T5-T7 — the legitimate mode must be untouched
# --------------------------------------------------------------------------------------------


def test_t5_sim_time_realworld_still_delays_and_releases(tmp_path):
    """T5: same contract as the pre-existing `test_sim_time_realworld_delays_raw_commands`.

    Deliberately asserts ONLY behaviour that exists in the pre-fix module, so this test passes
    against pre-fix and post-fix code alike — which is what makes it evidence that the legitimate
    mode's contract did not move.  The new relay-state telemetry is asserted separately, in
    `test_sim_time_realworld_records_its_relay_state_in_its_own_artifacts`.
    """
    node = _make_node(tmp_path, timing_mode='sim_time_realworld',
                      model_latency_sec=0.3, action_hold_policy='zero')
    emitted = _capture_emissions(node)
    try:
        assert getattr(node, 'cmd_pub', None) is not None, 'the legitimate mode must keep its publisher'
        node._on_clock(_clock_msg(10.0))
        node._on_eval_ready(_ready_msg(True))
        node._on_raw_cmd(_twist(0.4, 0.2))

        node._on_timer()
        assert emitted[-1].linear.x == 0.0
        node._on_clock(_clock_msg(10.29))
        node._on_timer()
        assert emitted[-1].linear.x == 0.0
        node._on_clock(_clock_msg(10.31))
        node._on_timer()
        assert emitted[-1].linear.x == 0.4
        assert emitted[-1].angular.z == 0.2

        summary = node._summary()
        assert summary['released_cmd_count'] == 1
        assert summary['timing_valid_for_realworld'] is True
    finally:
        node.destroy_node()


def test_t6_sim_time_realworld_hold_last_republishes_the_same_nonzero_command(tmp_path):
    """T6: `hold_last` must keep republishing a real command, not decay to zero.

    Pre-fix-compatible by design (see T5).
    """
    node = _make_node(tmp_path, timing_mode='sim_time_realworld',
                      model_latency_sec=0.0, action_hold_policy='hold_last')
    emitted = _capture_emissions(node)
    try:
        node._on_clock(_clock_msg(4.0))
        node._on_eval_ready(_ready_msg(True))
        node._on_raw_cmd(_twist(0.35, -0.15))
        node._on_timer()
        assert emitted[-1].linear.x == pytest.approx(0.35)

        for _ in range(4):
            node._on_timer()
        assert emitted[-1].linear.x == pytest.approx(0.35), 'hold_last decayed'
        assert emitted[-1].angular.z == pytest.approx(-0.15)
    finally:
        node.destroy_node()


def test_t7_sim_time_realworld_still_holds_the_robot_before_the_episode(tmp_path):
    """T7: the pre-episode zero is this node's job in the legitimate mode and must survive.

    Pre-fix-compatible by design (see T5).
    """
    node = _make_node(tmp_path, timing_mode='sim_time_realworld', model_latency_sec=0.0)
    emitted = _capture_emissions(node)
    try:
        node._on_clock(_clock_msg(1.0))
        node._on_task_reset(Int16(data=1))
        assert len(emitted) == 1 and emitted[-1].linear.x == 0.0

        node._on_eval_ready(_ready_msg(False))
        assert len(emitted) == 2 and emitted[-1].linear.x == 0.0

        node._on_timer()
        assert len(emitted) == 3 and emitted[-1].linear.x == 0.0
    finally:
        node.destroy_node()


def test_sim_time_realworld_records_its_relay_state_in_its_own_artifacts(tmp_path):
    """The legitimate mode must report itself as a command source, and suppress nothing.

    Split out of T5 so that T5 stays runnable against pre-fix code; this half is about the new
    telemetry and is expected to be meaningless before the fix.
    """
    node = _make_node(tmp_path, timing_mode='sim_time_realworld', model_latency_sec=0.0)
    emitted = _capture_emissions(node)
    try:
        node._on_clock(_clock_msg(9.0))
        node._on_eval_ready(_ready_msg(True))
        node._on_raw_cmd(_twist(0.25, 0.1))
        node._on_timer()

        summary = node._summary()
        assert summary['relay_enabled'] is True
        assert summary['output_publisher_created'] is True
        assert summary['suppressed_cmd_count'] == 0, 'nothing may be suppressed in the legitimate mode'
        assert summary['emitted_cmd_count'] == len(emitted)
    finally:
        node.destroy_node()


# --------------------------------------------------------------------------------------------
# T8 / T9 — the rejected launch-level fix, given a test in both behavioural and structural form
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize('timing_mode', ['wall', 'sim_time_realworld'])
def test_t8_timing_artifacts_are_produced_in_both_modes(tmp_path, timing_mode):
    """T8: the constraint that ruled out the obvious fix, as an executable test.

    This node is the SOLE producer of `rtf.csv`, `internnav_timing_summary.json` and
    `internnav_timing_trace.jsonl`, and `artifact_validation.json` checks none of them — so a
    launch-level `condition=` would delete the RTF record with nothing failing.  This test is what
    fails instead.
    """
    node = _make_node(tmp_path, timing_mode=timing_mode)
    try:
        rtf = tmp_path / 'rtf.csv'
        assert rtf.exists(), 'rtf.csv must be created at construction, in every mode'
        header = rtf.read_text(encoding='utf-8').splitlines()[0].strip()
        assert header == 'wall_time,sim_time,dt_wall_sec,dt_sim_sec,rtf'

        node._on_clock(_clock_msg(1.0))
        node._on_clock(_clock_msg(1.05))
        rows = [line for line in rtf.read_text(encoding='utf-8').splitlines()[1:] if line.strip()]
        assert len(rows) >= 1, 'no RTF sample recorded — the wall->sim bridge is gone'
        assert len(rows[0].split(',')) == 5

        node._write_summary()
        summary_path = tmp_path / 'internnav_timing_summary.json'
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        for key in ('raw_cmd_count', 'released_cmd_count', 'rtf_sample_count', 'timing_mode'):
            assert key in summary, f'summary lost pre-existing key {key!r}'
        assert summary['rtf_sample_count'] >= 1

        trace_path = tmp_path / 'internnav_timing_trace.jsonl'
        assert trace_path.exists()
        events = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        assert any(e.get('event') == 'timing_manager_started' for e in events)
    finally:
        node.destroy_node()
        events = [
            json.loads(line)
            for line in (tmp_path / 'internnav_timing_trace.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        assert any(e.get('event') == 'timing_manager_stopped' for e in events)


def _launch_file() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    candidates = [
        parent / 'arena_simulation_setup' / 'launch' / 'internnav_async_eval.launch.py'
        for parent in here.parents
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise AssertionError(
        'internnav_async_eval.launch.py not found relative to '
        f'{here} — run the suite from the source tree'
    )


def test_t9_the_timing_manager_launch_action_stays_unconditional():
    """T9: the rejected fix, caught structurally.

    Gating the node at launch is the change almost anyone would write first.  It is wrong because
    it destroys the timing artifacts silently.  This asserts the action is still unconditional and
    still points at the robot's own `cmd_vel`, so the fix cannot quietly migrate back to the launch
    layer.  The `condition=` that legitimately exists in this file belongs to `data_recorder`.
    """
    tree = ast.parse(_launch_file().read_text(encoding='utf-8'))
    calls = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and any(isinstance(t, ast.Name) and t.id == 'timing_manager' for t in node.targets)
    ]
    assert len(calls) == 1, 'expected exactly one timing_manager Node(...) assignment'
    kwargs = {kw.arg for kw in calls[0].keywords}
    assert 'condition' not in kwargs, (
        'a condition= on the timing_manager launch action is the REJECTED fix: this node is the '
        'sole producer of rtf.csv and the timing artifacts, and no artifact check would notice '
        'their loss. Gate the relay inside the node instead.'
    )
    source = ast.unparse(calls[0])
    assert "'output_cmd_vel_topic': 'cmd_vel'" in source, (
        'renaming the output topic is a rejected fix (it leaves a live publisher on a decoy topic)'
    )


# --------------------------------------------------------------------------------------------
# The paired observational precondition — proven able to ARM, in both directions
# --------------------------------------------------------------------------------------------


def test_the_topology_reporter_arms_when_the_relay_has_no_input_publisher(tmp_path):
    """`sim_time_realworld` with nothing publishing on the raw topic is a real misconfiguration.

    This project's recurring defect class is a capability probe that can never arm while a comment
    claims it does, so this test exists to prove the reporter arms.  It also proves the grace
    period works: before it elapses there must be no report, because the client's publisher
    legitimately appears tens of seconds late.
    """
    node = _make_node(tmp_path, timing_mode='sim_time_realworld')
    try:
        node._check_relay_topology()
        assert node.input_publisher_count == 0
        assert node.topology_mismatch is None, 'reported a mismatch inside the grace period'

        node._start_monotonic -= node.INPUT_TOPOLOGY_GRACE_SEC + 1.0
        node._check_relay_topology()
        assert node.topology_mismatch == 'relay_enabled_but_input_topic_has_no_publisher'
        assert node._summary()['topology_mismatch'] == 'relay_enabled_but_input_topic_has_no_publisher'
    finally:
        node.destroy_node()


def test_the_topology_reporter_arms_when_a_disabled_relay_sees_an_input_publisher(tmp_path):
    """`wall` mode plus a publisher on the raw topic means NOBODY drives the robot.

    That is the `--cmd-vel-topic` drift path: a mode-only gate would stay silent while the client
    publishes somewhere the robot does not listen.  The reporter must make it loud.
    """
    node = _make_node(tmp_path, timing_mode='wall')
    try:
        node._check_relay_topology()
        assert node.topology_mismatch is None

        stray = node.create_publisher(Twist, _input_topic(node), 1)
        try:
            node._check_relay_topology()
            assert node.input_publisher_count >= 1
            assert node.topology_mismatch == 'relay_disabled_but_input_topic_has_a_publisher'
        finally:
            node.destroy_publisher(stray)

        node._check_relay_topology()
        assert node.topology_mismatch is None, 'the reporter must not latch'
    finally:
        node.destroy_node()


def test_the_topology_observation_never_gates_an_emission(tmp_path):
    """The reporter must never be able to drop a real command.

    A transient discovery gap in `sim_time_realworld` — exactly the state this test sets up, with
    the mismatch already reported — must still release the queued command.
    """
    node = _make_node(tmp_path, timing_mode='sim_time_realworld', model_latency_sec=0.0)
    emitted = _capture_emissions(node)
    try:
        node._start_monotonic -= node.INPUT_TOPOLOGY_GRACE_SEC + 1.0
        node._check_relay_topology()
        assert node.topology_mismatch == 'relay_enabled_but_input_topic_has_no_publisher'

        node._on_clock(_clock_msg(7.0))
        node._on_eval_ready(_ready_msg(True))
        node._on_raw_cmd(_twist(0.5, 0.0))
        node._on_timer()

        assert emitted[-1].linear.x == pytest.approx(0.5), (
            'a reported topology mismatch suppressed a real command — it is a reporter, not a gate'
        )
        assert _suppressed(node) == 0
    finally:
        node.destroy_node()


def test_wall_mode_records_its_relay_state_in_its_own_artifacts(tmp_path):
    """The run's own artifacts must state that the relay was disabled.

    This is the offline form of the Stage-0 treatment check: a live publisher census showing 2
    publishers cannot distinguish "the fix ran" from "the node died", so the node records the fact
    itself, in machine-readable form, independent of any log capture.
    """
    node = _make_node(tmp_path, timing_mode='wall')
    try:
        node._on_clock(_clock_msg(3.0))
        node._on_eval_ready(_ready_msg(True))
        for _ in range(5):
            node._on_timer()
        node._write_summary()

        summary = json.loads((tmp_path / 'internnav_timing_summary.json').read_text(encoding='utf-8'))
        assert summary['relay_enabled'] is False
        assert summary['output_publisher_created'] is False
        assert summary['emitted_cmd_count'] == 0
        assert summary['suppressed_cmd_count'] == 5
        assert summary['timing_valid_for_realworld'] is False

        started = [
            json.loads(line)
            for line in (tmp_path / 'internnav_timing_trace.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip() and json.loads(line).get('event') == 'timing_manager_started'
        ]
        assert started and started[0]['relay_enabled'] is False
    finally:
        node.destroy_node()
