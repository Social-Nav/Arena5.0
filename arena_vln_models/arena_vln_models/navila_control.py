"""NaVILA text actions and simulation-time executor, independent of ROS/model deps."""
from dataclasses import dataclass, asdict
import base64
import io
import json
import math
import re
import socket
import threading


@dataclass(frozen=True)
class Action:
    kind: str
    magnitude: int
    linear_x: float
    angular_z: float
    duration_sec: float

    def record(self):
        return asdict(self)


def parse_action(text):
    """Accept one explicit native action; never default malformed output to motion."""
    if not isinstance(text, str):
        raise ValueError('model response must be a string')
    value = re.sub(r'^the next action is\s+', '', text.strip(), flags=re.I).strip(' .\n')
    # Native NaVILA also emits explanatory STOP sentences. Do not require the
    # fixed action prefix, but reject mixed STOP/movement rather than choosing one.
    if re.search(r'\bstop\b', value, re.I) and not re.search(r'\b(?:move|turn)\b', value, re.I):
        return Action('stop', 0, 0., 0., 0.)
    match = re.fullmatch(r'move forward\s+(25|50|75)\s+cm', value, re.I)
    if match:
        n = int(match[1])
        return Action('forward', n, .5, 0., n / 50.)
    match = re.fullmatch(r'turn (left|right)\s+(15|30|45)\s+degrees?', value, re.I)
    if match:
        side, n = match[1].lower(), int(match[2])
        return Action(side, n, 0., math.pi / 6 * (1 if side == 'left' else -1), n / 30.)
    raise ValueError(f'unrecognized or conflicting NaVILA action: {text!r}')


def check_velocity_limits(robot):
    # These are the direct velocity limits; max_vel_x is a Nav2 planner setting.
    limits = robot['velocity_smoother_max_velocity']
    minimum = robot['velocity_smoother_min_velocity']
    if float(limits[0]) < .5 or float(limits[2]) < math.pi / 6 or float(minimum[2]) > -math.pi / 6:
        raise ValueError('robot direct velocity limits cannot execute native NaVILA actions without clipping')


def sample_indices(count):
    if count < 1:
        raise ValueError('at least one image is required')
    padded = max(8, count)
    pad = padded - count
    return [None if i < pad else i - pad for i in
            [int(i * (padded - 1) / 7) for i in range(7)] + [padded - 1]]


def encode_image(image):
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


class FrameHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_seen = None
        self.next_sample = None
        self.frames = []
        self.latest = None
        self.duplicates = 0

    def offer(self, stamp, image):
        if self.last_seen is not None and stamp <= self.last_seen:
            self.duplicates += 1
            return False
        self.last_seen = stamp
        self.latest = (stamp, image.copy())
        if self.next_sample is None or stamp >= self.next_sample:
            self.frames.append((stamp, encode_image(image)))
            self.next_sample = stamp + 500_000_000
        return True

    def sample(self):
        from PIL import Image
        if self.latest is None:
            raise ValueError('no current image')
        current, image = self.latest
        frames = list(self.frames)
        if not frames or frames[-1][0] != current:
            frames.append((current, encode_image(image)))
        black = encode_image(Image.new('RGB', image.size, (0, 0, 0)))
        selected = [frames[i] if i is not None else (None, black) for i in sample_indices(len(frames))]
        return [s[1] for s in selected], [s[0] for s in selected]


class TimedAction:
    def __init__(self, action, start):
        self.action = action
        self.start = start
        self.end = start + action.duration_sec
        self.next_publish = start

    def command(self, sim):
        if sim < self.start or sim >= self.end - 1e-9:
            return (0., 0.)
        if sim + 1e-9 < self.next_publish:
            return None
        self.next_publish = sim + .05
        return (self.action.linear_x, self.action.angular_z)


class TCPModel:
    """One bounded request per connection; abort is safe during a ROS reset."""
    def __init__(self, host, port, timeout=10.):
        self.host, self.port, self.timeout = host, port, timeout
        self.lock = threading.Lock()
        self.connection = None

    def abort(self):
        with self.lock:
            if self.connection is not None:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    @staticmethod
    def receive(connection, size):
        data = bytearray()
        while len(data) < size:
            chunk = connection.recv(min(size - len(data), 1048576))
            if not chunk:
                raise ConnectionError('NaVILA disconnected before completing response')
            data.extend(chunk)
        return bytes(data)

    def generate(self, images, query):
        payload = json.dumps({'images': images, 'query': query}).encode()
        with socket.create_connection((self.host, self.port), timeout=min(3., self.timeout)) as connection:
            connection.settimeout(self.timeout)
            with self.lock:
                self.connection = connection
            try:
                connection.sendall(len(payload).to_bytes(8, 'big') + payload)
                size = int.from_bytes(self.receive(connection, 8), 'big')
                if not 0 < size <= 65536:
                    raise ValueError('invalid NaVILA response size')
                result = json.loads(self.receive(connection, size).decode())
                if not isinstance(result, str) or result.lower().startswith('server error:'):
                    raise ValueError(f'NaVILA server error: {result!r}')
                return result
            finally:
                with self.lock:
                    self.connection = None
