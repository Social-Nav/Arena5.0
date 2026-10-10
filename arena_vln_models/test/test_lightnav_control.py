import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_vln_models.lightnav_control import ControlState, FrameSequence, SimCadence, valid_waypoints


@pytest.mark.parametrize('rate', [.25, .5, 1., 2., 4.])
def test_control_cadence_uses_simulation_time_without_catching_up(rate):
    cadence = SimCadence(.1)
    stamps = [i / 240. * rate for i in range(2401)]
    issued = [t for t in stamps if cadence.due(t)]
    assert all(b - a >= .1 - 1e-9 for a, b in zip(issued, issued[1:]))
    assert not cadence.due(stamps[-1])
    assert cadence.due(stamps[-1] + 20.)
    assert not cadence.due(stamps[-1] + 20.)
    cadence.reset()
    assert cadence.due(0.)


def test_camera_sequence_drops_duplicates_and_reordered_frames_even_after_busy_drop():
    sequence = FrameSequence()
    assert sequence.observe(100)
    # Admission consumes the frame even if the native client subsequently says busy.
    assert not sequence.observe(100)
    assert not sequence.observe(99)
    assert sequence.observe(101)
    sequence.reset()
    assert sequence.observe(10)  # new episode / simulation clock rewind
    assert not sequence.observe(10)


def moving_state():
    state = ControlState()
    state.ready(0)
    state.active = True
    state.sequence = 4
    state.path_capture_wall = state.path_capture_sim = 10.
    state.odom_wall = state.rgb_wall = state.clock_wall = state.command_wall = 10.
    return state


def test_start_waits_for_episode_authorization():
    state = moving_state()
    state.ready_episode = None
    assert state.block_reason(10.1, 10.1) == 'episode_not_active'


def test_late_duplicate_reset_does_not_disarm_ready_episode():
    state = moving_state()
    assert not state.reset(0)
    assert state.block_reason(10.1, 10.1, require_command=True) == ''


def test_new_episode_invalidates_path_and_pending_solver_generation():
    state = moving_state()
    generation = state.generation
    assert state.reset(1)
    assert state.generation != generation
    assert state.sequence == 0 and not state.active
    state.ready(1); state.active = True
    assert state.block_reason(10.1, 10.1) == 'path_expired'


def test_delayed_prior_episode_events_cannot_rearm_the_old_session():
    state = moving_state()
    state.ready(2); state.active = True
    assert not state.reset(1)
    assert not state.ready(1)
    assert state.episode == state.ready_episode == 2
    assert state.active


@pytest.mark.parametrize('field, expected', [
    ('odom_wall', 'odom_stale'), ('clock_wall', 'clock_stalled'),
    ('rgb_wall', 'camera_stale'), ('path_capture_wall', 'path_expired'),
    ('command_wall', 'control_timeout'),
])
def test_watchdogs_stop_even_when_sim_time_is_paused(field, expected):
    state = moving_state()
    setattr(state, field, 0.)
    assert state.block_reason(10.1, 10., require_command=True) == expected


def test_path_age_uses_capture_time_and_rejects_time_rewind():
    state = moving_state()
    assert state.block_reason(10.1, 13.) == 'path_sim_time_invalid'
    assert state.block_reason(10.1, 9.) == 'path_sim_time_invalid'
    state.terminal = True
    assert state.block_reason(10.1, 10.1) == 'episode_not_active'


@pytest.mark.parametrize('points', [[], [[1, 2]], [[True, 0, 0]], [[math.nan, 0, 0]],
                                    [[0, 0, math.inf]], [[0, 0, 0]] * 4097])
def test_invalid_trajectories_cannot_authorize_motion(points):
    assert not valid_waypoints(points)


def test_short_paths_and_pure_rotation_are_preserved():
    assert valid_waypoints([(0., 0., .5)])
    assert valid_waypoints([[.1, 0, 0], [.2, .1, .3]])
