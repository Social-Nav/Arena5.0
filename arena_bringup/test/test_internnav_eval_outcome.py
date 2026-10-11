from arena_bringup.internnav_eval import (
    _classify_end_reason,
    _episode_outcome_topic,
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
        model_status={},
        episode_outcome={'reason': 'sim_timeout'},
        model_diagnostic_summary={'final_goal_distance': {'min': 0.1}},
    ) == 'episode_sim_timeout'


def test_classify_end_reason_prefers_explicit_goal_reached():
    assert _classify_end_reason(
        finished_observed=True,
        launch_returncode=0,
        timed_out=False,
        model_status={},
        episode_outcome={'reason': 'goal_reached'},
        model_diagnostic_summary={'final_goal_distance': {'min': 2.0}},
    ) == 'episode_goal_reached'


def test_classify_end_reason_preserves_model_stop_without_claiming_goal_reached():
    assert _classify_end_reason(
        finished_observed=True,
        launch_returncode=0,
        timed_out=False,
        model_status={'status': 'stop'},
        episode_outcome={'reason': 'model_stop'},
        model_diagnostic_summary={'final_goal_distance': {'min': 2.0}},
    ) == 'episode_model_stop'


def test_classify_end_reason_keeps_legacy_finished_without_outcome():
    assert _classify_end_reason(
        finished_observed=True,
        launch_returncode=0,
        timed_out=False,
        model_status={},
        episode_outcome=None,
        model_diagnostic_summary={'final_goal_distance': {'min': 2.0}},
    ) == 'finished_without_goal_reached'


def test_classify_model_error_is_not_success():
    assert _classify_end_reason(
        finished_observed=True, launch_returncode=0, timed_out=False,
        model_status={}, episode_outcome={'reason': 'model_error'},
    ) == 'episode_model_error'
