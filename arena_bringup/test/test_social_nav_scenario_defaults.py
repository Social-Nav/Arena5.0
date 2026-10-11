from arena_bringup.social_nav_scenario import SocialNavScenario


def test_scenario_defaults_use_generic_model_diagnostics():
    scenario = SocialNavScenario(data={})
    scenario.apply_defaults()

    diagnostics = scenario.data['evaluation']['metrics']['diagnostics']
    assert diagnostics == [
        'model_status',
        'model_trace',
        'path_execution',
        'artifact_validation',
        'real_time_factor',
    ]

    # Defaults belong to each scenario, so modifying one must not affect another.
    diagnostics.append('custom_trace')
    another = SocialNavScenario(data={})
    another.apply_defaults()
    assert 'custom_trace' not in another.data['evaluation']['metrics']['diagnostics']


def test_scenario_preserves_explicit_diagnostics():
    scenario = SocialNavScenario(data={
        'evaluation': {'metrics': {'diagnostics': ['internnav_status', 'custom_trace']}},
    })
    scenario.apply_defaults()

    assert scenario.data['evaluation']['metrics']['diagnostics'] == [
        'internnav_status', 'custom_trace',
    ]
