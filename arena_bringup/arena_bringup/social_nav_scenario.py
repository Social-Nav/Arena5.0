"""Benchmark-level Dynamic Social VLN scenario configuration.

This module validates the *top-level* benchmark scenario YAML used by the
short-term Social Navigation benchmark work.  It intentionally does not replace
Arena's native world/scenario files.  Instead, it validates an overlay that adds
language, task predicates, metric gates, and required artifacts on top of the
existing Arena world geometry and HuNav actor definitions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TASK_METRICS = [
    'episode_result',
    'path_length_m',
    'goal_progress_m',
    'robot_moved',
    'timeout',
]
DEFAULT_SOCIAL_METRICS = [
    'social_success',
    'min_human_distance_m',
    'near_miss_count',
    'human_collision_count',
    'personal_space_violation_time_sec',
    'crowd_freezing_time_sec',
]
DEFAULT_DIAGNOSTICS = [
    'internnav_status',
    'internnav_trace_summary',
    'command_stats',
    'stale_camera_count',
    'artifact_validation',
]
DEFAULT_REQUIRED_ARTIFACTS = [
    'run_manifest.yaml',
    'params.yaml',
    'start_goal.csv',
    'odom.csv',
    'cmd_vel.csv',
    'human_states.csv',
    'metrics.csv',
    'social_metrics.json',
    'artifact_validation.json',
]
DEFAULT_SOCIAL_CONSTRAINTS = [
    'human_collision_count == 0',
    'near_miss_count == 0',
    'personal_space_violation_time_sec == 0.0',
    'min_human_distance_m >= 0.25',
]
DEFAULT_FAILURE_PREDICATES = [
    'timeout',
    'no_motion',
    'collision',
    'missing_humans',
    'missing_required_artifacts',
]


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding for a scenario config."""

    severity: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {'severity': self.severity, 'path': self.path, 'message': self.message}


@dataclass
class SocialNavScenario:
    """Top-level Dynamic Social VLN benchmark scenario config.

    The class owns three responsibilities:

    1. Load a YAML config into a normalized dictionary.
    2. Apply benchmark defaults for optional fields.
    3. Validate that referenced Arena world/scenario assets and benchmark-level
       fields are internally consistent.
    """

    data: dict[str, Any]
    source_path: Path | None = None
    issues: list[ValidationIssue] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path, *, validate: bool = True) -> 'SocialNavScenario':
        source_path = Path(path).expanduser().resolve()
        with source_path.open(encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f'Scenario config must be a YAML mapping: {source_path}')
        scenario = cls(data=loaded, source_path=source_path)
        scenario.apply_defaults()
        if validate:
            scenario.validate()
        return scenario

    @property
    def scenario_id(self) -> str:
        return str(self.data.get('id', '') or '')

    @property
    def world_name(self) -> str:
        return str(self.data.get('world', {}).get('name', '') or '')

    @property
    def robot_name(self) -> str:
        return str(self.data.get('robot', {}).get('name', '') or '')

    @property
    def local_planner(self) -> str:
        return str(self.data.get('robot', {}).get('local_planner', '') or '')

    def apply_defaults(self) -> None:
        """Fill optional benchmark-level fields with stable defaults."""
        self.data.setdefault('schema_version', 0.1)

        world = self.data.setdefault('world', {})
        if isinstance(world, dict):
            world.setdefault('native_scenario', {})
            world.setdefault('semantic_regions', {})

        robot = self.data.setdefault('robot', {})
        if isinstance(robot, dict):
            robot.setdefault('local_planner', 'dual_vln')
            robot.setdefault('command_interface', {})
            if robot.get('name'):
                ns = f"/task_generator_node/{robot['name']}"
                robot['command_interface'].setdefault('cmd_vel_topic', f'{ns}/cmd_vel')
                robot['command_interface'].setdefault('odom_topic', f'{ns}/odom')
                robot['command_interface'].setdefault('goal_topic', f'{ns}/episode_goal_pose')
            robot.setdefault('goal', {})
            if isinstance(robot['goal'], dict):
                robot['goal'].setdefault('tolerance_m', 0.45)

        language = self.data.setdefault('language', {})
        if isinstance(language, dict):
            language.setdefault('instruction_type', 'goal_only')
            language.setdefault('rephrases', [])

        humans = self.data.setdefault('humans', {})
        if isinstance(humans, dict):
            humans.setdefault('simulator', 'hunav')
            humans.setdefault('source', 'native_scenario')
            humans.setdefault('density_level', 'unspecified')

        task_spec = self.data.setdefault('task_spec', {})
        if isinstance(task_spec, dict):
            task_spec.setdefault('type', 'bddl_like_social_nav')
            predicates = task_spec.setdefault('predicates', {})
            if isinstance(predicates, dict):
                entities = predicates.setdefault('entities', {})
                if isinstance(entities, dict) and robot.get('name'):
                    entities.setdefault('robot', robot['name'])
                    entities.setdefault('humans', 'all_hunav_pedestrians')
                    entities.setdefault('goal', 'robot.goal')
                predicates.setdefault('success', ['goal_reached(robot, goal)', 'robot_moved(robot, min_path_length_m=0.1)'])
                predicates.setdefault('social_constraints', list(DEFAULT_SOCIAL_CONSTRAINTS))
                predicates.setdefault('failure', list(DEFAULT_FAILURE_PREDICATES))

        evaluation = self.data.setdefault('evaluation', {})
        if isinstance(evaluation, dict):
            evaluation.setdefault('timeout_sec', 120.0)
            evaluation.setdefault('repetitions', 1)
            evaluation.setdefault('random_seed', None)
            metrics = evaluation.setdefault('metrics', {})
            if isinstance(metrics, dict):
                metrics.setdefault('task', list(DEFAULT_TASK_METRICS))
                metrics.setdefault('social', list(DEFAULT_SOCIAL_METRICS))
                metrics.setdefault('diagnostics', list(DEFAULT_DIAGNOSTICS))
            pass_criteria = evaluation.setdefault('pass_criteria', {})
            if isinstance(pass_criteria, dict):
                pass_criteria.setdefault('artifact_validation_overall_pass', True)
                pass_criteria.setdefault('humans_present', True)
                pass_criteria.setdefault('min_humans_observed', 1)
                pass_criteria.setdefault('robot_moved_min_path_length_m', 0.1)
                pass_criteria.setdefault('require_goal_reached', True)
                pass_criteria.setdefault('require_social_success', True)

        self.data.setdefault('artifacts_required', list(DEFAULT_REQUIRED_ARTIFACTS))

    def validate(self) -> list[ValidationIssue]:
        """Validate required fields, referenced files, and common consistency rules."""
        self.issues = []
        self._validate_required_sections()
        self._validate_id()
        self._validate_world()
        self._validate_robot()
        self._validate_language()
        self._validate_humans()
        self._validate_task_spec()
        self._validate_evaluation()
        self._validate_artifacts()
        return self.issues

    def is_valid(self) -> bool:
        return not any(issue.severity == 'error' for issue in self.issues)

    def raise_if_invalid(self) -> None:
        errors = [issue for issue in self.issues if issue.severity == 'error']
        if errors:
            joined = '\n'.join(f'- {issue.path}: {issue.message}' for issue in errors)
            raise ValueError(f'Invalid social-nav scenario config:\n{joined}')

    def to_dict(self) -> dict[str, Any]:
        return self.data

    def validation_report(self) -> dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'source_path': str(self.source_path) if self.source_path else None,
            'valid': self.is_valid(),
            'issues': [issue.to_dict() for issue in self.issues],
            'normalized': self.data,
        }

    def native_scenario_name(self) -> str:
        world = self.data.get('world') if isinstance(self.data.get('world'), dict) else {}
        native = world.get('native_scenario') if isinstance(world.get('native_scenario'), dict) else {}
        return str(native.get('name') or '')

    def internnav_eval_argv(
        self,
        *,
        output_root: str | None = None,
        output_prefix: str | None = None,
        save_eval_video: bool = True,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Translate this scenario overlay into ``internnav_eval`` CLI args."""
        evaluation = self.data.get('evaluation') if isinstance(self.data.get('evaluation'), dict) else {}
        language = self.data.get('language') if isinstance(self.data.get('language'), dict) else {}
        humans = self.data.get('humans') if isinstance(self.data.get('humans'), dict) else {}
        scenario_name = self.native_scenario_name()
        argv = [
            '--sim', 'isaac',
            '--human', str(humans.get('simulator') or 'hunav'),
            '--world', self.world_name,
            '--robot', self.robot_name,
            '--local-planner', self.local_planner,
            '--episodes', str(int(evaluation.get('repetitions') or 1)),
            '--timeout', str(int(float(evaluation.get('timeout_sec') or 120.0))),
            '--tm-robots', 'scenario',
            '--tm-obstacles', 'scenario',
            '--social-eval',
            '--internnav-mode', 'internnav',
            '--vln-instruction', str(language.get('instruction') or 'navigate'),
        ]
        if self.scenario_id:
            argv.extend(['--scenario-config-id', self.scenario_id])
        if self.source_path is not None:
            argv.extend(['--scenario-config-path', str(self.source_path)])
        if scenario_name:
            argv.extend(['--scenario-file', scenario_name])
        if output_root:
            argv.extend(['--output-root', output_root])
        if output_prefix:
            argv.extend(['--output-prefix', output_prefix])
        if save_eval_video:
            argv.append('--save-eval-video')
            argv.append('--internnav-enable-visualization')
        if extra_args:
            argv.extend(extra_args)
        return argv

    def resolved_path(self, value: str | None) -> Path | None:
        return resolve_resource_path(value, base_path=self.source_path)

    def _issue(self, severity: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity=severity, path=path, message=message))

    def _validate_required_sections(self) -> None:
        for key in ('id', 'world', 'robot', 'language', 'humans', 'task_spec', 'evaluation'):
            if key not in self.data:
                self._issue('error', key, 'missing required top-level field')
        for key in ('world', 'robot', 'language', 'humans', 'task_spec', 'evaluation'):
            if key in self.data and not isinstance(self.data.get(key), dict):
                self._issue('error', key, 'field must be a mapping')

    def _validate_id(self) -> None:
        scenario_id = self.scenario_id
        if not scenario_id:
            self._issue('error', 'id', 'scenario id is required')
        elif not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', scenario_id):
            self._issue('error', 'id', 'scenario id must be filesystem-friendly')

    def _validate_world(self) -> None:
        world = self.data.get('world') if isinstance(self.data.get('world'), dict) else {}
        if not world.get('name'):
            self._issue('error', 'world.name', 'world name is required')
        self._require_existing_file(world.get('world_config'), 'world.world_config')
        self._require_existing_file(world.get('map_yaml'), 'world.map_yaml')

        native = world.get('native_scenario') if isinstance(world.get('native_scenario'), dict) else {}
        if not native.get('name'):
            self._issue('warning', 'world.native_scenario.name', 'native scenario name is not set')
        self._require_existing_file(native.get('file'), 'world.native_scenario.file')

    def _validate_robot(self) -> None:
        robot = self.data.get('robot') if isinstance(self.data.get('robot'), dict) else {}
        if not robot.get('name'):
            self._issue('error', 'robot.name', 'robot name is required')
        if not robot.get('local_planner'):
            self._issue('error', 'robot.local_planner', 'local planner is required')
        for key in ('start', 'goal'):
            block = robot.get(key) if isinstance(robot.get(key), dict) else {}
            pose = block.get('pose_xy_yaw')
            if not _is_pose_xy_yaw(pose):
                self._issue('error', f'robot.{key}.pose_xy_yaw', 'pose must be [x, y, yaw] with finite numeric values')
        goal = robot.get('goal') if isinstance(robot.get('goal'), dict) else {}
        tolerance = goal.get('tolerance_m')
        if not _is_positive_number(tolerance):
            self._issue('error', 'robot.goal.tolerance_m', 'goal tolerance must be positive')

    def _validate_language(self) -> None:
        language = self.data.get('language') if isinstance(self.data.get('language'), dict) else {}
        if not str(language.get('instruction', '') or '').strip():
            self._issue('error', 'language.instruction', 'language instruction is required')
        rephrases = language.get('rephrases', [])
        if not isinstance(rephrases, list) or not all(isinstance(item, str) for item in rephrases):
            self._issue('error', 'language.rephrases', 'rephrases must be a list of strings')

    def _validate_humans(self) -> None:
        humans = self.data.get('humans') if isinstance(self.data.get('humans'), dict) else {}
        if humans.get('simulator') != 'hunav':
            self._issue('warning', 'humans.simulator', 'short-term benchmark support is currently validated for hunav')
        expected = humans.get('expected_count')
        if not isinstance(expected, int) or expected < 0:
            self._issue('error', 'humans.expected_count', 'expected_count must be a non-negative integer')
            expected = None

        native_path = humans.get('native_scenario_file')
        if not native_path:
            native = self.data.get('world', {}).get('native_scenario', {}) if isinstance(self.data.get('world'), dict) else {}
            native_path = native.get('file') if isinstance(native, dict) else None
        resolved_native = self._require_existing_file(native_path, 'humans.native_scenario_file')
        if resolved_native is not None:
            dynamic_agents = _read_dynamic_agents(resolved_native)
            if dynamic_agents is None:
                self._issue('error', 'humans.native_scenario_file', 'could not parse native scenario YAML dynamic agents')
            else:
                if expected is not None and len(dynamic_agents) != expected:
                    self._issue(
                        'error',
                        'humans.expected_count',
                        f'expected_count={expected} does not match native dynamic agent count={len(dynamic_agents)}',
                    )
                self._validate_behavior_trees(dynamic_agents, resolved_native.parent)

    def _validate_behavior_trees(self, dynamic_agents: list[Any], scenario_dir: Path) -> None:
        for index, agent in enumerate(dynamic_agents):
            if not isinstance(agent, dict):
                self._issue('error', f'humans.dynamic[{index}]', 'dynamic agent entry must be a mapping')
                continue
            for key in ('name', 'pose', 'model'):
                if key not in agent:
                    self._issue('error', f'humans.dynamic[{index}].{key}', f'native dynamic agent is missing {key}')
            bt = agent.get('behavior_tree')
            if not bt:
                self._issue('warning', f'humans.dynamic[{index}].behavior_tree', 'behavior_tree is missing')
                continue
            bt_path = (scenario_dir / str(bt)).resolve()
            if not bt_path.exists():
                self._issue('error', f'humans.dynamic[{index}].behavior_tree', f'behavior tree file does not exist: {bt_path}')

    def _validate_task_spec(self) -> None:
        task_spec = self.data.get('task_spec') if isinstance(self.data.get('task_spec'), dict) else {}
        predicates = task_spec.get('predicates') if isinstance(task_spec.get('predicates'), dict) else {}
        for key in ('success', 'social_constraints', 'failure'):
            value = predicates.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
                self._issue('error', f'task_spec.predicates.{key}', 'predicate group must be a non-empty list of strings')

    def _validate_evaluation(self) -> None:
        evaluation = self.data.get('evaluation') if isinstance(self.data.get('evaluation'), dict) else {}
        if not _is_positive_number(evaluation.get('timeout_sec')):
            self._issue('error', 'evaluation.timeout_sec', 'timeout_sec must be positive')
        repetitions = evaluation.get('repetitions')
        if not isinstance(repetitions, int) or repetitions <= 0:
            self._issue('error', 'evaluation.repetitions', 'repetitions must be a positive integer')
        metrics = evaluation.get('metrics') if isinstance(evaluation.get('metrics'), dict) else {}
        for group in ('task', 'social', 'diagnostics'):
            values = metrics.get(group)
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                self._issue('error', f'evaluation.metrics.{group}', 'metrics group must be a list of strings')

        task_metrics = set(metrics.get('task') or [])
        for spl_like in ('SPL', 'spl', 'soft_spl', 'route_efficiency'):
            if spl_like in task_metrics:
                self._issue(
                    'warning',
                    'evaluation.metrics.task',
                    f'{spl_like} requires a static-map shortest-path oracle and is not a default Dynamic Social VLN metric',
                )

    def _validate_artifacts(self) -> None:
        artifacts = self.data.get('artifacts_required')
        if not isinstance(artifacts, list) or not all(isinstance(item, str) and item.strip() for item in artifacts):
            self._issue('error', 'artifacts_required', 'artifacts_required must be a non-empty list of paths')
            return
        for required in ('odom.csv', 'cmd_vel.csv', 'human_states.csv', 'social_metrics.json', 'artifact_validation.json'):
            if required not in artifacts:
                self._issue('warning', 'artifacts_required', f'recommended artifact is missing: {required}')

    def _require_existing_file(self, value: Any, path: str) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            self._issue('error', path, 'file path is required')
            return None
        resolved = self.resolved_path(value)
        if resolved is None:
            self._issue('error', path, f'could not resolve path: {value}')
            return None
        if not resolved.exists():
            self._issue('error', path, f'file does not exist: {resolved}')
            return None
        if not resolved.is_file():
            self._issue('error', path, f'path is not a file: {resolved}')
            return None
        return resolved


def resolve_resource_path(value: str | None, *, base_path: Path | None = None) -> Path | None:
    """Resolve absolute, relative, and package:// resource paths.

    The source-tree fallback keeps validation useful before colcon install by
    walking upward from the scenario YAML and looking for sibling ROS packages.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith('package://'):
        without_scheme = text[len('package://'):]
        package, _, rest = without_scheme.partition('/')
        if not package or not rest:
            return None
        try:
            from ament_index_python.packages import get_package_share_directory

            return Path(get_package_share_directory(package)) / rest
        except Exception:
            fallback = _resolve_package_source_path(package, rest, base_path=base_path)
            return fallback
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate
    if base_path is not None:
        base_dir = base_path if base_path.is_dir() else base_path.parent
        return (base_dir / candidate).resolve()
    return candidate.resolve()


def _resolve_package_source_path(package: str, rest: str, *, base_path: Path | None) -> Path | None:
    if base_path is None:
        return None
    current = base_path.resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        sibling = parent / package
        if (sibling / 'package.xml').exists():
            return sibling / rest
        if parent.name == package and (parent / 'package.xml').exists():
            return parent / rest
    return None


def _read_dynamic_agents(path: Path) -> list[Any] | None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    dynamic = loaded.get('dynamic', [])
    if dynamic is None:
        return []
    return dynamic if isinstance(dynamic, list) else None


def _is_pose_xy_yaw(value: Any) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except Exception:
        return False


def _is_positive_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validate a Dynamic Social VLN scenario config.')
    parser.add_argument('scenario_config', help='Path to benchmark-level scenario YAML config')
    parser.add_argument('--format', choices=('text', 'json'), default='text')
    args = parser.parse_args(argv)

    scenario = SocialNavScenario.from_file(args.scenario_config, validate=True)
    report = scenario.validation_report()
    if args.format == 'json':
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = 'valid' if report['valid'] else 'invalid'
        print(f"{scenario.scenario_id}: {status}")
        for issue in scenario.issues:
            print(f"[{issue.severity}] {issue.path}: {issue.message}")
    return 0 if scenario.is_valid() else 1


def eval_main(argv: list[str] | None = None) -> int:
    """Validate a scenario config and run ``internnav_eval`` with derived args."""
    parser = argparse.ArgumentParser(description='Run a Dynamic Social VLN scenario config via internnav_eval.')
    parser.add_argument('--scenario-config', required=True, help='Path to benchmark-level scenario YAML config')
    parser.add_argument('--output-root', default='', help='Optional internnav_eval --output-root')
    parser.add_argument('--output-prefix', default='', help='Optional internnav_eval --output-prefix')
    parser.add_argument('--no-save-eval-video', action='store_true', help='Do not request eval video artifacts')
    parser.add_argument('--dry-run', action='store_true', help='Print the derived internnav_eval argv without running')
    parser.add_argument('extra_internnav_eval_args', nargs='*', help='Extra args appended to internnav_eval; use -- before leading -- options')
    args, unknown_args = parser.parse_known_args(argv)

    scenario = SocialNavScenario.from_file(args.scenario_config, validate=True)
    scenario.raise_if_invalid()
    derived = scenario.internnav_eval_argv(
        output_root=args.output_root or None,
        output_prefix=args.output_prefix or scenario.scenario_id,
        save_eval_video=not args.no_save_eval_video,
        extra_args=[*(args.extra_internnav_eval_args or []), *unknown_args],
    )
    if args.dry_run:
        print('ros2 run arena_bringup internnav_eval ' + ' '.join(_shell_quote(item) for item in derived))
        return 0

    from arena_bringup import internnav_eval

    previous_argv = sys.argv[:]
    try:
        sys.argv = ['internnav_eval', *derived]
        return int(internnav_eval.main())
    finally:
        sys.argv = previous_argv


def _shell_quote(value: str) -> str:
    if re.fullmatch(r'[A-Za-z0-9_./:=+-]+', value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
