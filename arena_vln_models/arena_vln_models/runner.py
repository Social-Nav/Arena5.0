"""Shared benchmark entry point. External clients are opaque argv processes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_vln_models.model_adapter import DEFAULT_CONTROL_MODE, MODEL_SPECS, resolve_model_spec
from arena_vln_models.provenance import (
    collect_overlay_provenance,
    install_root_from_environment,
    summary as provenance_summary,
)
from arena_vln_models.profile import (
    load_model_profile,
    load_profile,
    profile_metadata,
    validate_machine_environment,
    write_resolved_snapshot,
)


def process_start(pid):
    try:
        # comm may contain spaces or parentheses; fields after its final ')'
        # start at field 3, so starttime (field 22) is index 19.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def owned_process(state):
    pid = state.get('pid', 0)
    if type(pid) is not int or pid <= 1 or process_start(pid) != state.get('start'):
        return False
    try:
        token = ('ARENA_VLN_RUN_TOKEN=' + state['token']).encode()
        return token in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') and os.getpgid(pid) == pid
    except (OSError, KeyError):
        return False


class Runner:
    def __init__(self, args):
        self.args = args
        self.arena = Path(__file__).resolve().parents[2]
        configured_ws = os.environ.get('ARENA_WS_DIR') or os.environ.get('HOST_ARENA_WS_DIR')
        if configured_ws:
            self.ws = Path(configured_ws).expanduser().resolve()
        elif self.arena.parent.name == 'src':
            self.ws = self.arena.parents[1]
        else:
            self.ws = next(
                (
                    parent
                    for parent in self.arena.parents
                    if (parent / 'src' / 'Arena').is_dir()
                ),
                self.arena.parents[1],
            )
        requested_model = str(getattr(args, 'model', '') or '').strip().lower()
        explicit_profile = getattr(args, 'profile', None)
        self.model = requested_model or 'internnav'
        resolve_model_spec(self.model)
        if explicit_profile:
            self.profile_path = Path(explicit_profile).expanduser().resolve()
            self.profile = load_profile(self.profile_path)
            if requested_model and str(self.profile.get('model', '')).strip().lower() != requested_model:
                raise ValueError(
                    f"--profile model '{self.profile.get('model')}' conflicts with --model '{requested_model}'"
                )
            metadata = profile_metadata(self.profile_path, self.profile)
            self.profile_identity = {
                **metadata,
                'base_profile_path': '',
                'base_profile_sha256': '',
                'overlay_profile_path': '',
                'overlay_profile_sha256': '',
                'resolved_profile_sha256': metadata['sha256'],
                'resolved_profile_snapshot': '',
            }
        else:
            base_profile = str(getattr(args, 'base_profile', '') or '').strip()
            self.profile, self.profile_identity = load_model_profile(
                self.model, self.arena,
                base_name=base_profile or 'base_follow_path_grscenes.yaml',
            )
            snapshot = write_resolved_snapshot(
                self.profile, self.profile_identity, self.ws / 'outputs' / 'benchmark_profiles'
            )
            self.profile_identity['resolved_profile_snapshot'] = str(snapshot)
            # The evaluator consumes the fully resolved profile, not a partial
            # overlay.  The source identity remains in profile_identity.
            self.profile_path = Path(snapshot)
        self.control_mode = str(getattr(args, 'control_mode', '') or DEFAULT_CONTROL_MODE)
        if self.control_mode != DEFAULT_CONTROL_MODE:
            raise ValueError(
                f"control_mode='{self.control_mode}' is not the unified path; use the isolated legacy "
                f"direct entry for native velocity. Unified mode is '{DEFAULT_CONTROL_MODE}'."
            )
        adapter = self.profile.get('adapter') if isinstance(self.profile.get('adapter'), dict) else {}
        self.model_adapter = str(getattr(args, 'model_adapter', '') or adapter.get('name', self.model))
        self.model_inference_hz = float(adapter.get('inference_hz', 4.0))
        self.model_request_timeout_sec = float(adapter.get('request_timeout_sec', 10.0))
        self.model_observation_max_age_sec = float(adapter.get('observation_max_age_sec', 2.0))
        self.model_server_url = str(adapter.get('server_url', '') or '')
        self.model_params_json = str(getattr(args, 'model_params_json', '') or '{}')
        self.model_trace_dir = str(adapter.get('trace_dir', '') or '')
        self.bfast_identity = None
        self.bfast_blocked = False
        self.bfast_blocked_reason = ''
        bfast_profile = str(getattr(args, 'bfast_profile', '') or '').strip()
        if bfast_profile:
            from arena_bringup.paper_pipeline import build_bfast_plan
            payload = json.loads(Path(bfast_profile).read_text(encoding='utf-8'))
            result = build_bfast_plan(
                payload,
                risk_threshold=getattr(args, 'bfast_risk_threshold', None),
                max_frame_gap_sec=getattr(args, 'bfast_max_frame_gap_sec', None),
                profile_path=bfast_profile,
            )
            if not result['valid']:
                raise ValueError('bfast profile invalid: ' + '; '.join(result['errors']))
            if not bool(getattr(args, 'lisn_slow_loop', False)):
                raise ValueError(
                    '--bfast-profile requires --lisn-slow-loop; the B profile must use '
                    'the existing Qwen/LISN producer path, not the InternNav adapter/server'
                )
            endpoint = str(getattr(args, 'lisn_producer_endpoint', '') or '').strip()
            if not endpoint:
                raise ValueError('--bfast-profile requires --lisn-producer-endpoint')
            if '/v1/chat/completions' not in endpoint:
                raise ValueError(
                    '--bfast-profile endpoint must be the Qwen/LISN /v1/chat/completions '
                    f'endpoint, got {endpoint!r}'
                )
            self.bfast_identity = result['identity']
            # The Qwen/LISN sidecar has no common metric FollowPath action
            # bridge.  Until that exists, a B plan is metadata only: it must
            # not emit executable commands or an InternNav top-level identity.
            self.bfast_blocked = True
            self.bfast_blocked_reason = 'qwen_lisn_common_metric_action_bridge_missing'
            # The actual B execution identity is the Qwen/LISN backend, not the
            # borrowed runner entry.  Never keep the InternNav adapter/server/
            # checkpoint in the generated command.
            self.model_adapter = 'lisn_qwen3vl_fp8'
            self.model_server_url = ''
            self.model_params_json = '{}'
        self.ros_env = validate_machine_environment(self.profile)
        if args.episodes not in (None, 1):
            raise ValueError('Use one episode per result directory; full-eval runs the scene loop')
        self.robot = args.robot or self.profile['evaluation']['robot']
        self.container = os.environ.get('ARENA_CONTAINER_NAME', f'arena-{self.ws.name}-arena-1')
        key = profile_metadata(self.profile_path)['sha256'][:16]
        self.root = self.ws / 'outputs' / 'waypoint_clients' / key
        self.state_file = self.root / 'process.json'
        self.log_file = self.root / 'client.log'

    def container_path(self, path):
        path = Path(path).resolve()
        if self.args.local:
            return str(path)
        try:
            return str(Path('/opt/arena_ws') / path.relative_to(self.ws))
        except ValueError as exc:
            raise ValueError('Docker evaluation files must be under the mounted Arena workspace; use --local otherwise') from exc

    def ros_command(self, argv, extra_env=None):
        env = {**self.ros_env, **(extra_env or {})}
        if self.args.local:
            return ['env', *[f'{k}={v}' for k, v in env.items()], *argv]
        # shlex.join preserves instructions and paths as single argv entries.
        command = ('source /opt/ros/jazzy/setup.bash; source /opt/arena_ws/install/setup.bash; '
                   'exec ' + shlex.join(['env', *[f'{k}={v}' for k, v in env.items()], *argv]))
        return ['docker', 'exec', '-i', self.container, 'bash', '--noprofile', '--norc', '-c', command]

    def configuration(self):
        adapter = self.profile.get('adapter') if isinstance(self.profile.get('adapter'), dict) else {}
        lisn = bool(getattr(self.args, 'lisn_slow_loop', False))
        prox = bool(getattr(self.args, 'prox_mpc', False))
        local_planner = 'prox_mpc' if prox else ('social_mpc' if lisn else 'regulated_pure_pursuit')
        control_path = (
            'nav2_follow_path_prox_mpc' if prox
            else 'nav2_follow_path_social_mpc' if lisn
            else 'nav2_follow_path_rpp'
        )
        return {
            **self.profile_identity,
            'model': (
                self.bfast_identity['model'] if self.bfast_identity
                else self.profile['model']),
            'model_backend': (
                'lisn_qwen3vl_fp8' if self.bfast_identity else self.profile['model']),
            'executable': not self.bfast_blocked,
            'blocked_reason': self.bfast_blocked_reason,
            'model_adapter': self.model_adapter,
            'control_mode': self.control_mode,
            'model_server': adapter.get('server_url', ''),
            'model_inference_hz': self.model_inference_hz,
            'model_request_timeout_sec': self.model_request_timeout_sec,
            'model_trace_dir': self.model_trace_dir,
            'bfast_profile': self.bfast_identity,
            'actual_model_backend': (
                'lisn_qwen3vl_fp8' if self.bfast_identity else self.profile.get('model')),
            'runner_entry': self.model,
            'checkpoint': '' if self.bfast_identity else adapter.get('checkpoint', ''),
            'execution_mode': 'nav2_follow_path',
            'controller': local_planner,
            'control_path': control_path,
            'local_planner': local_planner,
            'lisn_slow_loop': lisn,
            'action': f'/task_generator_node/{self.robot}/follow_path',
            'status_topic': f'/task_generator_node/{self.robot}/model/status',
            'external_command': self.profile['external']['command'],
            'client_start': 'configured command' if self.profile['external']['command'] else 'already running externally',
        }

    def state(self):
        try:
            return json.loads(self.state_file.read_text())
        except (OSError, ValueError):
            return {}

    def start_client(self):
        command = self.profile['external']['command']
        if not command:
            raise ValueError('serve requires external.command from the model repository; '
                             'use eval/run for an already running FollowPath client')
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'process.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if owned_process(self.state()):
                raise RuntimeError('this profile already owns an external client; use status/stop')
            token = str(uuid.uuid4())
            env = {**os.environ, **self.ros_env, 'ARENA_VLN_RUN_TOKEN': token,
                   'ARENA_ROBOT_NAMESPACE': f'/task_generator_node/{self.robot}',
                   'ARENA_TASK_NAMESPACE': '/task_generator_node'}
            with self.log_file.open('a') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            state = {'pid': child.pid, 'start': process_start(child.pid), 'token': token,
                     'command': command, 'log': str(self.log_file)}
            self.state_file.write_text(json.dumps(state, indent=2) + '\n')
        # Do not wait for sensors or a first path before starting the simulator.
        if child.poll() is not None:
            raise RuntimeError(f'external client exited; inspect {self.log_file}')
        return child, state

    def stop_client(self, state=None):
        state = self.state() if state is None else state
        if not owned_process(state):
            return
        os.killpg(state['pid'], signal.SIGINT)
        deadline = time.monotonic() + 10
        while owned_process(state) and time.monotonic() < deadline:
            time.sleep(.1)
        if owned_process(state):
            os.killpg(state['pid'], signal.SIGTERM)

    def infrastructure_command(self, verb):
        """Operate only the shared simulation services, never model containers."""
        compose = ['docker', 'compose', '--project-name', os.environ.get('ARENA_PROJECT_NAME', f'arena-{self.ws.name}'),
                   '--project-directory', str(self.ws), '-f', str(self.arena / '_meta/docker/docker-compose.yaml')]
        if verb in {'build', 'install'}:
            return [*compose, 'build', 'arena', 'isaac']
        if verb == 'up':
            return [*compose, 'up', '-d', '--no-recreate', '--no-build', 'arena', 'isaac']
        if verb == 'runtime':
            return self.ros_command(['bash', '-c',
                'cd /opt/arena_ws && '
                'export ARENA_WS_DIR=/opt/arena_ws ARENA_DIR=/opt/arena_ws/src/Arena ARENA_VENV_DIR=/opt/venv && '
                'exec bash "$ARENA_DIR/_meta/tools/colcon_build" '
                '--packages-select arena_bringup arena_vln_models task_generator arena_simulation_setup'])
        raise ValueError(verb)

    def overlay_provenance(self):
        """Overlay identity is captured once and reused for every case."""
        cached = getattr(self, '_overlay_provenance_cache', None)
        if cached is None:
            cached = collect_overlay_provenance(
                self.arena,
                install_root_from_environment(),
            )
            self._overlay_provenance_cache = cached
        return cached

    def evaluation_command(self, case=None, extra=(), timestamp=None):
        if self.bfast_blocked:
            raise ValueError('B_BLOCKED: ' + self.bfast_blocked_reason)
        adapter = self.profile.get('adapter') if isinstance(self.profile.get('adapter'), dict) else {}
        argv = ['ros2', 'run', 'arena_bringup', 'internnav_eval', '--benchmark-profile',
                self.container_path(self.profile_path), '--robot', self.robot,
                '--model-adapter', self.model_adapter, '--control-mode', self.control_mode]
        if getattr(self.args, 'lisn_slow_loop', False):
            # LISN uses a non-RPP FollowPath plugin; the RPP identity must never
            # be recorded for a LISN apply.  The experiment label is the
            # LISN/Qwen identity, never the borrowed internnav runner label.
            prox = bool(getattr(self.args, 'prox_mpc', False))
            argv += ['--local-planner', 'prox_mpc' if prox else 'social_mpc',
                     '--model-backend', 'lisn_qwen3vl_fp8',
                     '--log-level', 'info']
        if adapter.get('server_url') and not self.bfast_identity:
            argv += ['--model-server-url', str(adapter['server_url'])]
        argv += [
            '--model-inference-hz', str(self.model_inference_hz),
            '--model-request-timeout-sec', str(self.model_request_timeout_sec),
        ]
        if self.model_trace_dir:
            argv += ['--model-trace-dir', self.model_trace_dir]
        if self.model_params_json and self.model_params_json != '{}':
            argv += ['--model-params-json', self.model_params_json]
        case = case or self.args.case
        if case:
            parts = case.split('/')
            if len(parts) != 2 or any(p in ('', '.', '..') for p in parts):
                raise ValueError('--case must be WORLD/SCENARIO')
            argv += ['--world', parts[0], '--scenario-file', parts[1]]
        for name, flag in [('world', '--world'), ('scenario', '--scenario-file'), ('human', '--human'),
                           ('tm_obstacles', '--tm-obstacles'), ('episodes', '--episodes'), ('timeout', '--timeout'),
                           ('expected_human_count', '--expected-human-count'), ('pedestrian_seed', '--pedestrian-seed'),
                           ('pedestrian_pool_size', '--pedestrian-pool-size')]:
            value = getattr(self.args, name)
            if value is not None:
                argv += [flag, str(value)]
        if self.args.task_generator_parameter_file:
            argv += ['--task-generator-parameter-file', self.container_path(self.args.task_generator_parameter_file)]
        overlay = self.overlay_provenance()
        overlay_env = {
            # Only the identity/hash summary is passed through argv/env.  The
            # full per-entry inventory can exceed MAX_ARG_STRLEN and is not
            # required for run provenance; the summary keeps the same schema,
            # commits, counts and inventory hash.
            'ARENA_EVAL_OVERLAY_PROVENANCE_JSON': json.dumps(
                provenance_summary(overlay), sort_keys=True
            ),
            'ARENA_EVAL_OVERLAY_SOURCE_COMMIT': str(overlay.get('source_commit') or ''),
        }
        if timestamp:
            overlay_env['ARENA_EVAL_FIXED_TIMESTAMP'] = timestamp
        if (getattr(self.args, 'lisn_grounding_live', False)
                or getattr(self.args, 'lisn_grounding_scan', False)):
            overlay_env['ARENA_LISN_GROUNDING_LIVE'] = '1'
            overlay_env['ARENA_LISN_GROUNDED_GOAL_TOPIC'] = (
                self.args.lisn_grounded_goal_topic
                or '/task_generator_node/Ai2_Bot2/lisn_slow_loop/grounded_goal'
            )
            overlay_env['ARENA_LISN_CRUISE_LINEAR_VEL'] = str(self.args.lisn_cruise_vel)
            if getattr(self.args, 'lisn_grounding_scan', False):
                overlay_env['ARENA_LISN_GROUNDING_SCAN'] = '1'
            if bool(getattr(self.args, 'prox_mpc', False)):
                overlay_env['ARENA_LISN_CONTROL_PATH'] = 'nav2_follow_path_prox_mpc'
        launch_args = []
        if getattr(self.args, 'lisn_slow_loop', False):
            if not self.args.lisn_producer_endpoint:
                raise ValueError(
                    '--lisn-slow-loop requires --lisn-producer-endpoint; '
                    'there is no fixture fallback'
                )
            launch_args += [
                'lisn_slow_loop:=true',
                f'lisn_slow_loop_producer_endpoint:={self.args.lisn_producer_endpoint}',
                f'lisn_slow_loop_producer_timeout_sec:={self.args.lisn_producer_timeout_sec}',
                f'lisn_slow_loop_producer_deadline_sec:={self.args.lisn_producer_deadline_sec}',
                # The sidecar has no robot namespace; pass the absolute RGB topic.
                f'lisn_slow_loop_rgb_topic:=/task_generator_node/{self.robot}/head_camera/image',
                # LISN is a separate non-RPP identity; never mix it with the
                # default three-model RPP profile.
                'local_planner:=prox_mpc' if bool(getattr(self.args, 'prox_mpc', False))
                else 'local_planner:=social_mpc',
            ]
            if self.args.lisn_evidence:
                launch_args.append(f'lisn_slow_loop_evidence:={self.args.lisn_evidence}')
            if self.args.lisn_wiring_live:
                launch_args.append('lisn_slow_loop_wiring_live:=true')
            if self.args.lisn_policy:
                launch_args.append(f'lisn_slow_loop_policy:={self.args.lisn_policy}')
            if self.args.lisn_grounding_live:
                launch_args += ['lisn_slow_loop_grounding_live:=true']
                if self.args.lisn_depth_topic:
                    launch_args.append(f'lisn_slow_loop_depth_topic:={self.args.lisn_depth_topic}')
                if self.args.lisn_camera_info_topic:
                    launch_args.append(f'lisn_slow_loop_camera_info_topic:={self.args.lisn_camera_info_topic}')
                if self.args.lisn_grounded_goal_topic:
                    launch_args.append(f'lisn_slow_loop_grounded_goal_topic:={self.args.lisn_grounded_goal_topic}')
                if self.args.lisn_semantic_cost_topic:
                    launch_args.append(
                        f'lisn_slow_loop_semantic_cost_topic:={self.args.lisn_semantic_cost_topic}'
                    )
                if self.args.lisn_semantic_cost_frame:
                    launch_args.append(
                        f'lisn_slow_loop_semantic_cost_frame:={self.args.lisn_semantic_cost_frame}'
                    )
                if self.args.lisn_semantic_ttl_sec is not None:
                    launch_args.append(
                        f'lisn_slow_loop_semantic_ttl_sec:={self.args.lisn_semantic_ttl_sec}'
                    )
                if self.args.lisn_min_confidence is not None:
                    launch_args.append(f'lisn_slow_loop_min_confidence:={self.args.lisn_min_confidence}')
            if self.args.lisn_grounding_scan:
                launch_args += [
                    'lisn_slow_loop_grounding_scan:=true',
                    f'lisn_slow_loop_scan_topic:={self.args.lisn_scan_topic}',
                    f'lisn_slow_loop_scan_max_age_sec:={self.args.lisn_scan_max_age_sec}',
                    f'lisn_slow_loop_max_scan_range_m:={self.args.lisn_max_scan_range_m}',
                ]
                if self.args.lisn_grounded_goal_topic:
                    launch_args.append(
                        f'lisn_slow_loop_grounded_goal_topic:={self.args.lisn_grounded_goal_topic}'
                    )
                if self.args.lisn_min_confidence is not None:
                    launch_args.append(f'lisn_slow_loop_min_confidence:={self.args.lisn_min_confidence}')
        if self.bfast_identity:
            identity = self.bfast_identity
            launch_args += [
                f'--bfast-profile={self.container_path(identity["profile_path"])}',
                f'--bfast-guard-enabled={str(identity["guard_enabled"]).lower()}',
                f'--bfast-veto-topic={identity["veto_topic"]}',
                f'--bfast-guard-required={str(identity["guard_required"]).lower()}',
                f'--bfast-guard-timeout-sec={identity["guard_timeout_sec"]}',
            ]
            if identity.get('risk_threshold') is not None:
                launch_args.append(f'--bfast-risk-threshold={identity["risk_threshold"]}')
            if identity.get('max_frame_gap_sec') is not None:
                launch_args.append(
                    f'--bfast-max-frame-gap-sec={identity["max_frame_gap_sec"]}')
        return self.ros_command([*argv, *launch_args, *self.args.extra, *extra], overlay_env)

    def cases(self, verb):
        if verb not in {'full-eval', 'density-eval'}:
            return [(self.args.case, [])]
        if self.args.case or self.args.world or self.args.scenario:
            raise ValueError('full/density-eval selects cases exclusively from --test-split')
        if any(arg.split('=')[0] in {'--output-prefix', '--output-root'} for arg in self.args.extra):
            raise ValueError('use --full-output-prefix for full/density-eval output')
        sys.path.insert(0, str(self.arena / 'arena_bringup'))
        from arena_bringup.test_split import load_test_split
        setup = self.arena / 'arena_simulation_setup'
        split = self.args.test_split or str(setup / ('density_eval_split_10.json' if verb == 'density-eval' else 'test_split.json'))
        cases = load_test_split(split, setup)
        prefix = self.args.full_output_prefix or f'{self.profile["model"]}_{verb}'
        if Path(prefix).is_absolute() or '..' in Path(prefix).parts or prefix in ('', '.'):
            raise ValueError('--full-output-prefix must be a relative output directory')
        jobs = []
        for density in ([0, 5, 10] if verb == 'density-eval' else [None]):
            for case in cases:
                arm = f'{prefix}/n{density:02d}' if density is not None else prefix
                extra = ['--output-prefix', f'{arm}/{case.world}_{case.scenario}']
                if density is not None:
                    path = self.arena / f'arena_bringup/configs/benchmark/density/task_generator_n{density:02d}.yaml'
                    extra += ['--expected-human-count', str(density), '--task-generator-parameter-file', self.container_path(path)]
                jobs.append((f'{case.world}/{case.scenario}', extra))
        return jobs

    def evaluate(self, verb):
        # One evaluation per robot/domain, regardless of experiment label.
        lock_root = self.ws / 'outputs' / 'waypoint_clients'
        lock_root.mkdir(parents=True, exist_ok=True)
        with (lock_root / f'eval-{self.ros_env["ROS_DOMAIN_ID"]}-{self.robot}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child = state = None
            try:
                if verb != 'eval' and self.profile['external']['command']:
                    child, state = self.start_client()
                failed = False
                jobs = self.cases(verb)
                for case, extra in jobs:
                    if child and child.poll() is not None:
                        raise RuntimeError(f'external client exited; inspect {self.log_file}')
                    result = subprocess.call(self.evaluation_command(case, extra))
                    failed |= result != 0
                if verb in {'full-eval', 'density-eval'}:
                    prefix = self.args.full_output_prefix or f'{self.profile["model"]}_{verb}'
                    root = self.container_path(self.ws / 'outputs' / prefix)
                    failed |= subprocess.call(self.ros_command([
                        'ros2', 'run', 'arena_bringup', 'social_nav_metrics_aggregate', '--root', root,
                        '--output-csv', root + '/aggregate.csv', '--summary-json', root + '/summary.json',
                        '--min-runs', str(len(jobs)), '--require-ready'])) != 0
                return int(failed)
            finally:
                if state:
                    self.stop_client(state)
                    child.wait(timeout=15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb', choices=['config', 'plan', 'doctor', 'serve', 'eval', 'run',
                                       'full-eval', 'density-eval', 'status', 'logs', 'stop',
                                       'build', 'install', 'up', 'runtime', 'down'])
    parser.add_argument('target', nargs='?', default='run')
    parser.add_argument('--profile', default=None,
        help='Explicit v3 profile. Preserved for compatibility; must agree with --model when both are given.')
    parser.add_argument('--model', choices=sorted(MODEL_SPECS), default=None,
        help='Select a model overlay on top of the shared FollowPath base profile.')
    parser.add_argument('--base-profile', default=None,
        help='Optional base profile file name under arena_bringup/configs/benchmark/profiles '
             '(default base_follow_path_grscenes.yaml; Task34 uses base_follow_path_grscenes_task34.yaml).')
    parser.add_argument('--model-adapter', default=None,
        help='Override the adapter name selected by the profile overlay.')
    parser.add_argument('--model-trace-dir', default=None,
        help='Optional executor trace directory; the evaluator defaults to <run>/model_runtime.')
    parser.add_argument('--model-params-json', default='{}',
        help='Optional bounded adapter parameters (no credentials or checkpoints); forwarded to the evaluator.')
    parser.add_argument('--control-mode', default=DEFAULT_CONTROL_MODE,
        help=f'Unified control mode; only {DEFAULT_CONTROL_MODE} is supported. Native velocity uses the legacy entry.')
    parser.add_argument('--local', action='store_true', help='Run evaluation in the current sourced Jazzy environment')
    parser.add_argument('--lisn-slow-loop', action='store_true',
        help='Enable the LISN slow-loop sidecar (separate social_mpc identity; default RPP run is unchanged).')
    parser.add_argument('--lisn-producer-endpoint', default='',
        help='Real LISN VLM JSON-HTTP endpoint; required with --lisn-slow-loop.')
    parser.add_argument('--lisn-producer-timeout-sec', type=float, default=120.0,
        help='Explicit bounded HTTP socket timeout for the LISN producer.')
    parser.add_argument('--lisn-producer-deadline-sec', type=float, default=120.0,
        help='Global monotonic per-request deadline; the owned child is aborted on expiry.')
    parser.add_argument('--lisn-evidence', default='',
        help='Evidence JSONL path for the LISN sidecar.')
    parser.add_argument('--lisn-wiring-live', action='store_true',
        help='Wiring-only live producer run; never flips runtime_ready.')
    parser.add_argument('--lisn-policy', default='',
        help='Explicit LISN policy JSON; grounding-live selection is explicit.')
    parser.add_argument('--prox-mpc', action='store_true',
        help='A-track: select the ProxMPC controller (real LiDAR + geometry costmap) '
             'instead of SocialMPC for the LISN slow loop.')
    parser.add_argument('--lisn-grounding-live', action='store_true',
        help='Enable same-stamp RGB/depth/CameraInfo/TF grounding in the sidecar.')
    parser.add_argument('--lisn-grounding-scan', action='store_true',
        help='A-track: RGB bearing + same-source real LaserScan range (no depth).')
    parser.add_argument('--lisn-scan-topic',
        default='/task_generator_node/Ai2_Bot2/laser_scan')
    parser.add_argument('--lisn-scan-max-age-sec', type=float, default=0.6)
    parser.add_argument('--lisn-max-scan-range-m', type=float, default=12.0)
    parser.add_argument('--lisn-depth-topic', default='')
    parser.add_argument('--lisn-camera-info-topic', default='')
    parser.add_argument('--lisn-grounded-goal-topic', default='')
    parser.add_argument('--lisn-semantic-cost-topic', default='')
    parser.add_argument('--lisn-semantic-cost-frame', default='')
    parser.add_argument('--lisn-semantic-ttl-sec', type=float, default=None)
    parser.add_argument('--lisn-min-confidence', type=float, default=None)
    parser.add_argument('--lisn-cruise-vel', type=float, default=0.8,
        help='LISN-only SocialMPC desired cruise target; the optimizer ceiling is separate.')
    parser.add_argument('--bfast-profile', default='',
        help='Explicit B off/on identity JSON; validated before plan/dry-run/eval argv construction.')
    parser.add_argument('--bfast-risk-threshold', type=float, default=None,
        help='Explicit on-arm flow-risk threshold; must be provided by the profile or here.')
    parser.add_argument('--bfast-max-frame-gap-sec', type=float, default=None,
        help='Explicit on-arm max RGB frame gap; must be provided by the profile or here.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--yes', '-y', action='store_true', help=argparse.SUPPRESS)
    for name in ('case', 'robot', 'world', 'scenario', 'human', 'tm-obstacles', 'test-split',
                 'full-output-prefix', 'task-generator-parameter-file'):
        parser.add_argument('--' + name)
    for name in ('episodes', 'timeout', 'expected-human-count', 'pedestrian-seed', 'pedestrian-pool-size'):
        parser.add_argument('--' + name, type=int)
    raw = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in raw:
        split = raw.index('--'); raw, extra = raw[:split], raw[split + 1:]
    args = parser.parse_args(raw)
    args.extra = extra
    try:
        runner = Runner(args)
        if args.verb == 'config':
            print(json.dumps(runner.configuration(), indent=2)); return 0
        if args.verb == 'plan' or args.dry_run:
            target = args.target if args.verb == 'plan' else args.verb
            if getattr(runner, 'bfast_blocked', False):
                commands = []
            elif target in {'build', 'install', 'up', 'runtime'}:
                commands = [runner.infrastructure_command(target)]
            else:
                commands = [runner.evaluation_command(case, extra)
                            for case, extra in runner.cases(target)]
            print(json.dumps({**runner.configuration(), 'commands': commands}, indent=2))
            return 0
        if args.verb in {'build', 'install', 'up', 'runtime'}:
            env = {**os.environ, 'ARENA_WS_DIR': str(runner.ws), 'HOST_ARENA_WS_DIR': str(runner.ws),
                   'ARENA_DIR': str(runner.arena), 'HOST_UID': str(os.getuid()), 'HOST_GID': str(os.getgid()),
                   'HOST_DOCKER_GID': str(os.stat('/var/run/docker.sock').st_gid),
                   'ARENA_IMAGE': os.environ.get('ARENA_IMAGE', 'arena:dev')}
            return subprocess.call(runner.infrastructure_command(args.verb), env=env)
        if args.verb == 'doctor':
            code = ('from ament_index_python.packages import get_package_share_directory as p; '
                    '[p(n) for n in ["nav2_controller", "nav2_regulated_pure_pursuit_controller", '
                    '"nav2_velocity_smoother", "arena_vln_models"]]; '
                    'print("Arena/RPP packages available; external model readiness is checked at the episode barrier")')
            return subprocess.call(runner.ros_command(['/usr/bin/python3', '-c', code]))
        if args.verb == 'status':
            state = runner.state(); print(json.dumps({**state, 'running': owned_process(state)}, indent=2)); return 0
        if args.verb == 'logs':
            print(runner.log_file.read_text() if runner.log_file.exists() else 'No owned client log.'); return 0
        if args.verb in {'stop', 'down'}:
            runner.stop_client(); return 0
        if args.verb == 'serve':
            child, state = runner.start_client()
            try:
                return child.wait()
            finally:
                runner.stop_client(state)
        if getattr(runner, 'bfast_blocked', False):
            print(json.dumps({
                'executable': False,
                'blocked_reason': runner.bfast_blocked_reason,
                'commands': [],
            }), file=sys.stderr)
            return 2
        return runner.evaluate(args.verb)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'[benchmark] {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
