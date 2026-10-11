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
from arena_vln_models.profile import load_profile, profile_metadata, validate_machine_environment


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
        self.ws = self.arena.parents[1]
        self.profile_path = Path(args.profile).expanduser().resolve()
        self.profile = load_profile(self.profile_path)
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
        return {**profile_metadata(self.profile_path), 'model': self.profile['model'],
                'execution_mode': 'nav2_follow_path', 'controller': 'regulated_pure_pursuit',
                'action': f'/task_generator_node/{self.robot}/follow_path',
                'status_topic': f'/task_generator_node/{self.robot}/model/status',
                'external_command': self.profile['external']['command'],
                'client_start': 'configured command' if self.profile['external']['command'] else 'already running externally'}

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

    def evaluation_command(self, case=None, extra=(), timestamp=None):
        argv = ['ros2', 'run', 'arena_bringup', 'internnav_eval', '--benchmark-profile',
                self.container_path(self.profile_path), '--robot', self.robot]
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
        return self.ros_command([*argv, *self.args.extra, *extra],
                                {'ARENA_EVAL_FIXED_TIMESTAMP': timestamp} if timestamp else None)

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
    parser.add_argument('--profile', default=str(Path(__file__).resolve().parents[2] /
        'arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml'))
    parser.add_argument('--local', action='store_true', help='Run evaluation in the current sourced Jazzy environment')
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
            commands = ([runner.infrastructure_command(target)] if target in {'build', 'install', 'up', 'runtime'} else
                        [runner.evaluation_command(case, extra) for case, extra in runner.cases(target)])
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
        return runner.evaluate(args.verb)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'[benchmark] {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
