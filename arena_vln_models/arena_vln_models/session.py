"""Model-independent episode identity and heartbeat contract (no ROS imports)."""
from dataclasses import dataclass
import math


@dataclass
class Session:
    session_id: str = ''
    episode: int = -1
    status: str = ''
    received_at: float = float('-inf')
    reason: str = ''
    terminal: bool = False
    released: bool = False
    heartbeat_timeout: float = 2.5

    def reset(self, session_id, episode):
        self.session_id, self.episode = session_id, episode
        self.status, self.reason = '', ''
        self.received_at = float('-inf')
        self.terminal = self.released = False

    def receive(self, payload, now):
        if not isinstance(payload, dict) or self.terminal:
            return False
        if (not self.session_id or payload.get('session_id') != self.session_id
                or type(payload.get('episode')) is not int or payload['episode'] != self.episode):
            return False
        status = payload.get('status')
        if not isinstance(status, str) or status not in {'ready', 'running', 'hold', 'stop', 'error'} or not math.isfinite(now):
            return False
        # A STOP before release cannot finish a newly reset episode.
        if status == 'stop' and not self.released:
            return False
        self.status, self.received_at = status, now
        self.reason = str(payload.get('reason', ''))
        self.terminal = status in {'stop', 'error'}
        return True

    def ready(self, now):
        return (not self.terminal and self.status in {'ready', 'running'}
                and 0 <= now - self.received_at <= self.heartbeat_timeout)

    def stop_reason(self, now):
        if self.status == 'error':
            return 'model_error'
        if self.status == 'stop':
            return 'model_stop'
        if not self.released:
            return 'episode_inactive'
        if not self.ready(now):
            return 'model_hold' if self.status == 'hold' else 'model_heartbeat_timeout'
        return ''
