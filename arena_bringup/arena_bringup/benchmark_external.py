"""Shared lifecycle for standalone model services and external ROS clients."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_bringup.benchmark_profile import load_profile, profile_metadata, validate_machine_environment


def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def capture(argv):
    return run(argv, capture_output=True, text=True).stdout.strip()


def installed_features(path, backend):
    entries = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    selected = set(entries)
    if (len(entries) != len(selected) or 'isaac' not in selected or backend not in selected
            or not selected <= {'isaac', 'internnav', 'lightnav', 'navila'}):
        raise ValueError(f'invalid installed features for {backend}: {entries}')
    return sorted(selected)


class Deployment:
    def __init__(self, args):
        self.args = args
        self.arena = Path(__file__).resolve().parents[2]
        self.ws = Path(os.environ.get('HOST_ARENA_WS_DIR', self.arena.parents[1])).resolve()
        self.profile_path = Path(args.profile).expanduser().resolve()
        self.profile = load_profile(self.profile_path)
        self.backend = self.profile['server']['mode']
        if self.backend not in {'lightnav', 'navila'}:
            raise ValueError('this dispatcher requires an external-client profile')
        self.controller = {'lightnav': 'lightnav_mpc', 'navila': 'navila_action_velocity'}[self.backend]
        machine = validate_machine_environment(self.profile)
        self.project = os.environ.get('ARENA_PROJECT_NAME', f'arena-{self.ws.name}')
        self.container = f'{self.project}-arena-1'
        self.model_container = f'arena-{self.backend}-{self.backend}-1'
        self.source = Path(os.environ.get('LIGHTNAV_SOURCE_PATH', '/home/kaizhen/vln_ws/LightNav-0'))
        self.weights = Path(os.environ.get('LIGHTNAV_MODEL_PATH', '/media/kaizhen/data/vln_ws/LightNav-0/models/LightNav-0'))
        self.image = os.environ.get('LIGHTNAV_IMAGE', 'lightnav-inference:arena-497bebf')
        if self.backend == 'navila':
            self.source = Path(os.environ.get('NAVILA_SOURCE_PATH', '/home/kaizhen/vln_ws/NaVILA'))
            self.weights = Path(os.environ.get('NAVILA_MODEL_PATH', self.source / 'artifacts/checkpoints/navila-llama3-8b-8f'))
            self.image = os.environ.get('NAVILA_IMAGE', 'navila-isaac-vlm:cu122')
        self.root = self.ws / f'outputs/{self.backend}_deploy'
        self.state_file = self.root / 'service.json'
        self.venv = '/opt/arena_ws/deps/lightnav-client/venv/bin/python' if self.backend == 'lightnav' else '/usr/bin/python3'
        self.robot = args.robot or self.profile['evaluation']['robot']
        self.env = os.environ.copy()
        self.env.update({f'{self.backend.upper()}_{key}': value for key, value in {
            'IMAGE': self.image, 'SOURCE_PATH': str(self.source), 'MODEL_PATH': str(self.weights),
            'PORT': str(urlparse(self.profile['server']['endpoint']).port),
            'GPU': machine['device'].removeprefix('cuda:'),
        }.items()})
        if self.backend == 'navila':
            self.env.update(NAVILA_IMAGE=self.image, NAVILA_MODEL_PATH=str(self.weights),
                NAVILA_PORT=str(urlparse(self.profile['server']['endpoint']).port),
                NAVILA_GPU=machine['device'].removeprefix('cuda:'),
                NAVILA_UID=str(os.getuid()), NAVILA_GID=str(os.getgid()),
                NAVILA_CACHE_PATH=str(self.source / 'artifacts/huggingface'))
        for key in ('ros_domain_id', 'rmw_implementation', 'fastdds_builtin_transports',
                    'ros_localhost_only', 'ros_automatic_discovery_range'):
            self.env[key.upper()] = str(self.profile['runtime'][key])

    def compose(self, *args):
        return ['docker', 'compose', '--project-name', f'arena-{self.backend}', '--project-directory', str(self.ws),
                '-f', str(self.arena / f'_meta/docker/features/{self.backend}/compose.yaml'), *args]

    def ros_command(self, argv, *, client=False, extra_env=None):
        exports = {key: self.env[key] for key in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION',
            'FASTDDS_BUILTIN_TRANSPORTS', 'ROS_LOCALHOST_ONLY', 'ROS_AUTOMATIC_DISCOVERY_RANGE')}
        exports.update(extra_env or {})
        if client:
            exports.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        prefix = 'source /opt/ros/jazzy/setup.bash; source /opt/arena_ws/install/setup.bash; '
        if client:
            prefix += 'export PYTHONDONTWRITEBYTECODE=1; '
        return ['docker', 'exec', '-i', self.container, 'bash', '--noprofile', '--norc', '-c',
                prefix + 'exec ' + shlex.join(['env', *[f'{k}={v}' for k,v in exports.items()], *argv])]

    def client_command(self, *args):
        return self.ros_command([self.venv, '-m', f'arena_vln_models.{self.backend}_client',
            '--profile', str(self.profile_path), '--robot', self.robot, *args], client=True)

    def configuration(self):
        return {
            **profile_metadata(self.profile_path, self.profile), 'model_backend': self.backend,
            'execution_mode': 'external_direct_cmd_vel', 'controller': self.controller,
            'shared_project': self.project, 'model_project': f'arena-{self.backend}',
            'client_container': self.container, 'client_python': self.venv,
            'model_container': self.model_container, 'model_image': self.image,
            'model_source': str(self.source), 'weights': str(self.weights),
            'endpoint': self.profile['server']['endpoint'], 'robot': self.robot,
            'cleanup': f'only this serve client process group and its {self.backend} model container',
        }

    def doctor(self):
        features = installed_features(self.arena / '.installed', self.backend)
        print(f'[PASS] installed={features}; selected=arena,isaac + arena-{self.backend}/{self.backend}')
        capture(['docker', 'info', '--format', '{{.ServerVersion}}'])
        capture(['docker', 'compose', 'version'])
        capture(['nvidia-smi', '-L'])
        image_id = capture(['docker', 'image', 'inspect', '--format', '{{.Id}}', self.image])
        for name in (self.container, f'{self.project}-isaac-1'):
            if capture(['docker', 'inspect', '--format', '{{.State.Running}}', name]) != 'true':
                raise RuntimeError(f'{name} must be started before evaluation')
        if self.backend == 'navila':
            from arena_bringup.navila_support import check_weights
            marker = check_weights(self.weights)
            run(self.client_command('--smoke'))
            import socket
            address = ('127.0.0.1', urlparse(self.profile['server']['endpoint']).port)
            with socket.socket() as probe:
                occupied = probe.connect_ex(address) == 0
            if occupied and not self.health():
                raise RuntimeError(f'NaVILA port {address[1]} is owned by an unrelated service')
            print(f'[PASS] NaVILA weights/client and image={image_id}')
            return marker
        marker = json.loads((self.weights / '.lightnav-download.json').read_text())
        for name, size in marker['files'].items():
            path = self.weights / name
            if not path.is_file() or path.stat().st_size != size:
                raise RuntimeError(f'checkpoint file missing or size mismatch: {name}')
        if not (self.weights / 'eval_config.json').is_file() or not (self.weights / 'action_tokenizer').is_dir():
            raise RuntimeError('checkpoint/action tokenizer incomplete')
        run(self.client_command('--smoke'))
        print(f'[PASS] CUDA host, existing containers, native client/MPC, image={image_id}')
        print(f'[PASS] model marker present; endpoint={self.profile["server"]["endpoint"]}')
        return marker

    def install(self):
        """Stage only the two hardware-independent packages; never build robot_deploy."""
        if self.backend == 'navila':
            self.start_shared()
            self.doctor_dependencies()
            installed = self.arena / '.installed'
            entries = installed.read_text().splitlines()
            if 'navila' not in entries:
                installed.write_text('\n'.join([*entries, 'navila']) + '\n')
            self.doctor()
            return
        revision = capture(['git', '-C', self.source, 'rev-parse', 'HEAD'])
        stage = self.ws / 'deps/lightnav-client' / revision
        for package in ('vln_client', 'vln_mpc'):
            shutil.copytree(self.source / 'robot_deploy/src' / package, stage / package,
                            dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
        shutil.copy2(self.source / 'LICENSE', stage / 'LICENSE')
        files = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(stage.rglob('*')) if p.is_file() and p.name != 'source.json'}
        (stage / 'source.json').write_text(json.dumps({'source': str(self.source), 'revision': revision,
            'dirty': capture(['git', '-C', self.source, 'status', '--short']), 'files': files}, indent=2) + '\n')
        runtime = {'revision': revision, 'stage': f'/opt/arena_ws/deps/lightnav-client/{revision}'}
        (stage.parent / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
        setup = ('/usr/bin/python3 -m venv --system-site-packages /opt/arena_ws/deps/lightnav-client/venv && '
                 '/opt/arena_ws/deps/lightnav-client/venv/bin/python -m pip install '
                 "'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' 'casadi==3.7.2' 'aiohttp>=3.9,<4' Pillow PyYAML")
        run(['docker', 'exec', self.container, 'bash', '--noprofile', '--norc', '-c', setup])
        build_env = {k: v for k,v in self.env.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
        run(['docker', 'build', '--build-arg', f'APP_UID={os.getuid()}', '--label',
             f'org.opencontainers.image.revision={revision}', '-t', self.image, self.source], env=build_env)
        run(['docker', 'volume', 'create', 'arena-arena_ws_lightnav_cache'])
        installed = self.arena / '.installed'
        entries = installed.read_text().splitlines()
        if 'lightnav' not in entries:
            installed.write_text('\n'.join([*entries, 'lightnav']) + '\n')
        self.doctor()

    def doctor_dependencies(self):
        run(self.ros_command(['/usr/bin/python3', '-c', 'import rclpy, numpy, PIL, yaml']))

    def health(self):
        result = subprocess.run(['docker', 'inspect', '--format', '{{if .State.Health}}{{.State.Health.Status}}{{end}}',
                                 self.model_container], capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == 'healthy'

    def save_state(self, state):
        temporary = self.state_file.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(self.state_file)

    def owner_active(self, state):
        """A reused PID or another profile is never this service's owner."""
        pid = int(state.get('owner_pid') or 0)
        if pid <= 1 or state.get('model_backend') != self.backend:
            return False
        process = Path(f'/proc/{pid}')
        try:
            command = process.joinpath('cmdline').read_bytes().decode().split('\0')
            if not any(Path(arg).name in ('benchmark_external.py', 'benchmark_lightnav.py') for arg in command):
                return False
            profile = Path(command[command.index('--profile') + 1])
            if not profile.is_absolute():
                profile = process.joinpath('cwd').resolve() / profile
            if profile.resolve() != self.profile_path:
                return False
            ticks = process.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
            return state.get('owner_start_ticks', ticks) == ticks
        except (OSError, ValueError, IndexError):
            return False

    def start_shared(self):
        # Existing containers are reused verbatim, including their mounts.
        run(['docker', 'start', self.container, f'{self.project}-isaac-1'])

    def serve(self):
        installed_features(self.arena / '.installed', self.backend)
        self.start_shared()
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'serve.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Refuse a competing controller before allocating GPU memory.
            run(self.ros_command(['/usr/bin/python3', '-m', 'arena_bringup.control_ownership', '--robot', self.robot]))
            was_running = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}',
                self.model_container], capture_output=True, text=True).stdout.strip() == 'true'
            if was_running:
                raise RuntimeError('Selected model container already running; stop its existing owner first')
            token = f'{os.getpid()}-{time.time_ns()}'
            state = {**self.configuration(), 'owner_pid': os.getpid(), 'token': token, 'ready': False}
            state['owner_start_ticks'] = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
            self.save_state(state)
            child = None
            try:
                run(self.compose('up', '-d', '--no-build', '--pull', 'never', self.backend), env=self.env)
                deadline = time.monotonic() + float(os.environ.get(f'{self.backend.upper()}_READY_TIMEOUT_SEC', '900'))
                while not self.health():
                    status = capture(['docker', 'inspect', '--format', '{{.State.Status}}', self.model_container])
                    if status in ('exited', 'dead') or time.monotonic() > deadline:
                        run(self.compose('logs', '--tail', '80', self.backend), env=self.env)
                        raise RuntimeError('Model warmup failed or timed out')
                    time.sleep(2)
                if self.backend == 'navila':
                    from arena_bringup.navila_support import model_smoke
                    model_smoke(self.profile['server']['endpoint'])
                child = subprocess.Popen(self.client_command('--run-token', token), start_new_session=True)
                run(self.client_command('--ready-probe', '--run-token', token))
                state.update(ready=True, client_exec_pid=child.pid,
                    image_id=capture(['docker', 'inspect', '--format', '{{.Image}}', self.model_container]))
                self.save_state(state)
                print(f'[{self.backend}] Model and {self.controller} client ready. Leave this terminal open.', flush=True)
                returncode = child.wait()
                if returncode:
                    raise RuntimeError(f'External client exited with {returncode}')
            finally:
                # Signal the exact process identified by its unique token in the container.
                stop_code = ("import os,signal\nfrom pathlib import Path\n"
                    f"token={token.encode()!r}\nmodule={f'arena_vln_models.{self.backend}_client'.encode()!r}\n"
                    "for p in Path('/proc').glob('[0-9]*/cmdline'):\n"
                    " try:\n"
                    "  args=p.read_bytes().split(bytes([0]))\n"
                    "  if module in args and token in args: os.kill(int(p.parent.name),signal.SIGINT)\n"
                    " except (FileNotFoundError,ProcessLookupError,PermissionError): pass\n")
                subprocess.run(['docker', 'exec', self.container, '/usr/bin/python3', '-c', stop_code],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if child is not None:
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=5)
                subprocess.run(self.compose('stop', self.backend), env=self.env)
                state['ready'] = False
                self.save_state(state)

    def stop(self):
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
            pid = int(state.get('owner_pid') or 0)
            cmdline = Path(f'/proc/{pid}/cmdline')
            if self.owner_active(state):
                os.kill(pid, signal.SIGINT)
                deadline = time.monotonic() + 25
                while cmdline.exists() and time.monotonic() < deadline:
                    time.sleep(.2)
                if cmdline.exists():
                    raise RuntimeError('External service owner did not stop; inspect service.json and logs')
        run(self.compose('stop', self.backend), env=self.env)
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
            state['ready'] = False
            self.save_state(state)

    def eval_command(self):
        evaluation = self.profile['evaluation']
        world, scenario = evaluation['world'], evaluation['scenario']
        if self.args.case:
            # The shared evaluator resolves and hashes the scenario instruction.
            world, scenario = self.args.case.split('/', 1)
            if '/' in scenario or world in ('', '.', '..') or scenario in ('', '.', '..'):
                raise ValueError('case must be world/scenario')
            path = self.arena / 'arena_simulation_setup/worlds' / world / 'scenarios' / scenario / 'scenario.yaml'
            if not path.is_file():
                raise ValueError(f'scenario not found: {path}')
        argv = ['ros2', 'run', 'arena_bringup', 'internnav_eval', '--benchmark-profile', str(self.profile_path),
            '--robot', self.robot, '--world', world, '--scenario-file', scenario,
            '--episodes', str(self.args.episodes or evaluation['episodes']),
            '--timeout', str(self.args.timeout or evaluation['timeout_sec']), *self.args.extra]
        return self.ros_command(argv, extra_env={'ARENA_MODEL_BACKEND': self.backend,
            'ARENA_MODEL_EXECUTION_MODE': 'external_direct_cmd_vel'})

    def evaluate(self):
        run(self.ros_command(['/usr/bin/python3', '-m', 'arena_bringup.control_ownership', '--evaluation']))
        if not self.health():
            raise RuntimeError('Selected model service is not healthy; run benchmark serve with this profile first')
        state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        if not state.get('ready') or not self.owner_active(state):
            raise RuntimeError('selected service owner is absent; restart benchmark serve')
        run(self.client_command('--ready-probe', '--run-token', state['token']))
        return subprocess.call(self.eval_command())

    def run_case(self):
        self.start_shared()
        self.doctor()
        log_dir = self.root / 'setup'
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f'serve-{time.strftime("%Y%m%d_%H%M%S")}.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'serve',
                '--profile', str(self.profile_path), '--robot', self.robot], stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 960
                while True:
                    if child.poll() is not None:
                        raise RuntimeError(f'serve failed; inspect {log.name}')
                    state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
                    if state.get('owner_pid') == child.pid and state.get('ready') and self.health():
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError('serve readiness timed out')
                    time.sleep(2)
                return self.evaluate()
            finally:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=45)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in raw:
        split = raw.index('--')
        raw, extra = raw[:split], raw[split + 1:]
    plan_verb = 'run'
    if raw[:1] == ['plan'] and len(raw) > 1 and not raw[1].startswith('-'):
        plan_verb = raw.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb', choices=['doctor', 'serve', 'launch', 'eval', 'run', 'config', 'plan',
                                       'status', 'stop', 'logs', 'install', 'runtime'])
    parser.add_argument('--profile', required=True)
    parser.add_argument('--case')
    parser.add_argument('--robot')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--timeout', type=int)
    args = parser.parse_args(raw)
    args.extra = extra
    for value in extra:
        if value.startswith(('--internnav-', '--dual-vln-', '--benchmark-profile', '--local-planner')):
            parser.error(f'{value} cannot override the selected model backend')
    if args.episodes is not None and args.episodes < 1 or args.timeout is not None and args.timeout < 1:
        parser.error('episodes and timeout must be positive')
    deployment = Deployment(args)
    if args.verb in ('config', 'plan'):
        print(json.dumps(deployment.configuration(), indent=2))
        if args.verb == 'plan':
            print(f'plan {plan_verb}: check installed features and model/controller occupancy; start only selected services;')
            print(f'serve: {deployment.backend} warmup → model readiness → {deployment.controller} in Arena')
            print('eval:', shlex.join(deployment.eval_command()))
        return 0
    if args.verb == 'doctor':
        deployment.doctor()
    elif args.verb in ('install', 'runtime'):
        deployment.install()
    elif args.verb in ('serve', 'launch'):
        deployment.serve()
    elif args.verb == 'eval':
        return deployment.evaluate()
    elif args.verb == 'run':
        return deployment.run_case()
    elif args.verb == 'stop':
        deployment.stop()
    elif args.verb == 'status':
        run(deployment.compose('ps', '-a'), env=deployment.env)
        if deployment.state_file.exists():
            print(deployment.state_file.read_text())
    elif args.verb == 'logs':
        run(deployment.compose('logs', '--tail', '100', deployment.backend), env=deployment.env)
    return 0


if __name__ == '__main__':
    def interrupt(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'[benchmark] ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
