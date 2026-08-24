import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena_bringup import internnav_eval


def _args(**overrides):
    values = {
        'world': 'grscenes_1_v1',
        'scenario_file': 'default',
        'robot': 'jackal',
        'vln_instruction': 'navigate',
        'vln_instruction_file': '',
        'vln_instruction_manifest': '',
        'vln_instruction_episode': 'episode_00',
        'vln_instruction_timestamp': '',
        'dual_vln_python_executable': '',
        'dual_vln_status_topic': '',
        'dual_vln_adapter_target': '',
        'dual_vln_mode': 'heuristic',
        'dual_vln_model_path': '',
        'dual_vln_http_url': '',
        'dual_vln_http_timeout_sec': 0.0,
        'dual_vln_require_real_backend': False,
        'dual_vln_rgb_topic': '',
        'dual_vln_depth_topic': '',
        'dual_vln_camera_info_topic': '',
        'eval_video_sim_top_down_topic': '',
        'eval_video_debug_overlay_topic': '',
        'save_eval_video': False,
        'dual_vln_enable_visualization': False,
        'dual_vln_device': 'cpu',
        'dual_vln_inference_timeout_sec': 0.2,
        'dual_vln_inference_rate_hz': 3.3333333333,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _instruction_paths(share: Path, world='grscenes_1_v1', scenario='default'):
    instruction_dir = share / 'worlds' / world / 'scenarios' / scenario / 'instruction'
    return instruction_dir / 'instruction.json', instruction_dir / 'instruction.txt'


def _write_instruction(share: Path, instruction: str):
    json_path, text_path = _instruction_paths(share)
    json_path.parent.mkdir(parents=True)
    payload = {
        'video_file': 'rgb_video.mp4',
        'model': 'gemini-2.5-pro',
        'timestamp': '2026-08-17T00:00:00',
        'raw_text': json.dumps({'instruction': instruction}),
        'parsed_result': {'instruction': instruction},
    }
    json_path.write_text(json.dumps(payload), encoding='utf-8')
    text_path.write_text(instruction, encoding='utf-8')
    return json_path, text_path


def _use_share(monkeypatch, share: Path):
    monkeypatch.setattr(
        internnav_eval,
        'get_package_share_directory',
        lambda package: str(share) if package == 'arena_simulation_setup' else None,
    )


def test_valid_scenario_instruction_is_discovered_with_provenance(tmp_path, monkeypatch):
    share = tmp_path / 'arena_simulation_setup'
    instruction = 'Walk past the desks, enter the meeting area, and stop at the bookshelf.'
    json_path, text_path = _write_instruction(share, instruction)
    _use_share(monkeypatch, share)
    monkeypatch.setattr(
        internnav_eval,
        '_resolve_existing_manifest_path',
        lambda *_args, **_kwargs: pytest.fail('legacy manifest lookup must not run'),
    )
    args = _args()

    adjustments = internnav_eval._apply_runtime_defaults(args)

    assert args.vln_instruction == instruction
    assert args.vln_instruction_file == str(text_path)
    provenance = adjustments['vln_instruction']
    assert provenance == {
        'source': 'arena_simulation_setup_scenario_instruction_json',
        'world': 'grscenes_1_v1',
        'scenario': 'default',
        'json_path': str(json_path),
        'sha256': hashlib.sha256(json_path.read_bytes()).hexdigest(),
        'schema_version': 1,
        'instruction_field': 'parsed_result.instruction',
        'instruction_file': str(text_path),
        'instruction_file_sha256': hashlib.sha256(text_path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    ('json_bytes', 'expected_error'),
    [
        (b'{not-json', 'Invalid scenario instruction JSON'),
        (
            json.dumps({
                'video_file': 'rgb_video.mp4',
                'model': 'gemini-2.5-pro',
                'timestamp': '2026-08-17T00:00:00',
                'raw_text': '{}',
                'parsed_result': {},
            }).encode('utf-8'),
            'parsed_result.instruction must be a non-empty string',
        ),
    ],
)
def test_present_corrupt_scenario_instruction_fails_loudly(
    tmp_path,
    monkeypatch,
    json_bytes,
    expected_error,
):
    share = tmp_path / 'arena_simulation_setup'
    json_path, text_path = _instruction_paths(share)
    json_path.parent.mkdir(parents=True)
    json_path.write_bytes(json_bytes)
    text_path.write_text('must not be used', encoding='utf-8')
    _use_share(monkeypatch, share)
    monkeypatch.setattr(
        internnav_eval,
        '_resolve_existing_manifest_path',
        lambda *_args, **_kwargs: pytest.fail('corrupt-present JSON must not fall through'),
    )

    with pytest.raises(RuntimeError) as caught:
        internnav_eval._apply_runtime_defaults(_args())

    message = str(caught.value)
    assert expected_error in message
    assert "world='grscenes_1_v1'" in message
    assert "scenario='default'" in message
    assert f"expected_path='{json_path}'" in message


def test_absent_scenario_instruction_uses_legacy_then_fail_closed_policy(tmp_path, monkeypatch):
    share = tmp_path / 'arena_simulation_setup'
    share.mkdir()
    _use_share(monkeypatch, share)
    monkeypatch.setattr(
        internnav_eval,
        '_resolve_existing_manifest_path',
        lambda *_args, **_kwargs: ('', ['/legacy/instruction/manifest.json']),
    )
    args = _args()

    adjustments = internnav_eval._apply_runtime_defaults(args)

    assert args.vln_instruction == 'navigate'
    assert args.vln_instruction_file == ''
    missing = adjustments['vln_instruction_scenario_json_missing']
    assert missing['reason'] == 'instruction_json_not_found'
    assert missing['policy'] == 'fall_back_to_legacy_manifest_then_fail_closed'
    assert missing['expected_json_path'] == str(_instruction_paths(share)[0])
    assert adjustments['vln_instruction_manifest_not_found'] == [
        '/legacy/instruction/manifest.json'
    ]


def test_explicit_instruction_wins_over_present_scenario_json(tmp_path, monkeypatch):
    share = tmp_path / 'arena_simulation_setup'
    json_path, _ = _instruction_paths(share)
    json_path.parent.mkdir(parents=True)
    json_path.write_bytes(b'{corrupt-on-purpose')
    _use_share(monkeypatch, share)
    explicit = 'Use the operator-provided instruction exactly.'
    args = _args(vln_instruction=explicit)

    adjustments = internnav_eval._apply_runtime_defaults(args)

    assert args.vln_instruction == explicit
    assert args.vln_instruction_file == ''
    assert 'vln_instruction' not in adjustments
    assert 'vln_instruction_scenario_json_missing' not in adjustments


def test_explicit_instruction_file_wins_over_present_scenario_json(tmp_path, monkeypatch):
    share = tmp_path / 'arena_simulation_setup'
    json_path, _ = _instruction_paths(share)
    json_path.parent.mkdir(parents=True)
    json_path.write_bytes(b'{corrupt-on-purpose')
    _use_share(monkeypatch, share)
    explicit_file = str(tmp_path / 'operator_instruction.txt')
    args = _args(vln_instruction_file=explicit_file)

    adjustments = internnav_eval._apply_runtime_defaults(args)

    assert args.vln_instruction == 'navigate'
    assert args.vln_instruction_file == explicit_file
    assert 'vln_instruction' not in adjustments
    assert 'vln_instruction_scenario_json_missing' not in adjustments
