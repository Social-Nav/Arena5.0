import json
from pathlib import Path

import pytest
import yaml

from arena_bringup.density_contract import main, resolve_density_contract


@pytest.fixture(params=[0, 5])
def density_config(tmp_path, request):
    count = request.param
    path = tmp_path / f"task_generator_n{count:02d}.yaml"
    path.write_text(
        yaml.safe_dump({
            "/**": {
                "ros__parameters": {
                    "task": {"random": {"dynamic": {"n": [count, count]}}}
                }
            }
        }),
        encoding="utf-8",
    )
    return count, path


def test_scenario_consumer_rejects_n0_and_n5(density_config):
    count, path = density_config
    result = resolve_density_contract(
        path,
        requested_consumer="scenario",
        resolved_consumer="scenario",
        expected_human_count=count,
    )
    assert result["valid"] is False
    assert result["failure_reasons"] == ["density_consumer_must_be_random"]


def test_random_consumer_accepts_n0_and_n5(density_config):
    count, path = density_config
    result = resolve_density_contract(
        path,
        requested_consumer="random",
        resolved_consumer="random",
        expected_human_count=count,
    )
    assert result["valid"] is True
    assert result["configured_human_count"] == count
    assert result["expected_human_count"] == count
    assert result["config_sha256"]


def test_cli_exit_code_and_json_are_fail_closed(density_config, capsys):
    count, path = density_config
    rc = main([
        "--config", str(path),
        "--requested-consumer", "random",
        "--resolved-consumer", "scenario",
        "--expected-count", str(count),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["valid"] is False


def test_expected_count_must_match_config(density_config):
    count, path = density_config
    result = resolve_density_contract(
        path,
        requested_consumer="random",
        resolved_consumer="random",
        expected_human_count=count + 1,
    )
    assert "expected_count_mismatch" in result["failure_reasons"]
