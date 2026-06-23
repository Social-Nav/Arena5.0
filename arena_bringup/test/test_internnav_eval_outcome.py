import json

from arena_bringup.internnav_eval import (
    _classify_end_reason,
    _episode_outcome_topic,
    _write_internnav_diagnostic_summary,
)


def test_episode_outcome_topic_uses_task_root_from_finished_topic():
    assert _episode_outcome_topic('/task_generator_node/finished', '/task_generator_node/task_reset') == (
        '/task_generator_node/episode_outcome'
    )


def test_classify_end_reason_prefers_explicit_episode_timeout():
    assert _classify_end_reason(
        finished_observed=True,
        launch_returncode=0,
        timed_out=False,
        internnav_status={},
        episode_outcome={'reason': 'sim_timeout'},
        internnav_diagnostic_summary={'final_goal_distance': {'min': 0.1}},
    ) == 'episode_sim_timeout'


def test_classify_end_reason_prefers_explicit_goal_reached():
    assert _classify_end_reason(
        finished_observed=True,
        launch_returncode=0,
        timed_out=False,
        internnav_status={},
        episode_outcome={'reason': 'goal_reached'},
        internnav_diagnostic_summary={'final_goal_distance': {'min': 2.0}},
    ) == 'episode_goal_reached'


def test_classify_end_reason_keeps_legacy_finished_without_outcome():
    assert _classify_end_reason(
        finished_observed=True,
        launch_returncode=0,
        timed_out=False,
        internnav_status={},
        episode_outcome=None,
        internnav_diagnostic_summary={'final_goal_distance': {'min': 2.0}},
    ) == 'finished_without_goal_reached'


def test_internnav_diagnostic_summary_derives_odom_goal_progress(tmp_path):
    trace_path = tmp_path / 'internnav_trace.jsonl'
    output_path = tmp_path / 'internnav_diagnostic_summary.json'
    records = [
        {
            'event_type': 'navigation_goal_seen',
            'parsed': {
                'status': 'navigation_goal_seen',
                'debug': {'goal': {'x': 3.0, 'y': 4.0}},
            },
        },
        {
            'event_type': 'planning_request_started',
            'parsed': {
                'status': 'planning_request_started',
                'desired_v': 0.1,
                'desired_w': 0.0,
                'debug': {'request_id': 1, 'odom': [0.0, 0.0, 0.0]},
            },
        },
        {
            'event_type': 'planning_request_started',
            'parsed': {
                'status': 'planning_request_started',
                'desired_v': 0.1,
                'desired_w': 0.0,
                'debug': {'request_id': 2, 'odom': [0.0, 4.0, 0.0]},
            },
        },
    ]
    trace_path.write_text('\n'.join(json.dumps(record) for record in records) + '\n', encoding='utf-8')

    summary = _write_internnav_diagnostic_summary(str(trace_path), str(output_path))

    assert summary['odom_goal_distance']['sample_count'] == 2
    assert summary['odom_goal_distance']['first'] == 5.0
    assert summary['odom_goal_distance']['last'] == 3.0
    assert summary['odom_goal_distance']['min'] == 3.0
    assert summary['odom_goal_distance']['progress_first_minus_last'] == 2.0
    assert summary['odom_goal_distance']['min_sample']['request_id'] == 2
