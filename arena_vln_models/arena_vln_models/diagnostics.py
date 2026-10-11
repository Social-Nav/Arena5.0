"""Generic model-status trace summary; no inference or motion interpretation."""
import json
from collections import Counter
from pathlib import Path


def execution_evidence(directory):
    """Count current-session native action evidence, or a legitimate no-motion STOP."""
    root = Path(directory)
    def records(name):
        path = root / name
        if not path.exists():
            return []
        result = []
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                continue
        return result
    # The outcome file is formatted JSON, not necessarily JSONL.
    try:
        outcome = json.loads((root / 'episode_outcome.json').read_text())
    except (OSError, ValueError):
        outcome = {}
    session = outcome.get('session_id')
    statuses = [r.get('parsed', {}) for r in records('model_trace.jsonl')]
    statuses = [r for r in statuses if isinstance(r, dict) and session and r.get('session_id') == session]
    goals = {r.get('goal_id') for r in records('path_execution.jsonl')
             if session and r.get('goal_id') and r.get('session_id') == session and r.get('event') == 'goal_status' and r.get('status') in (1, 2, 4)}
    stopped = outcome.get('reason') == 'model_stop' and any(r.get('status') == 'stop' for r in statuses)
    return {'pass': bool(statuses) and bool(goals or stopped), 'path_count': len(goals),
            'trace_present': (root / 'model_trace.jsonl').is_file(),
            'trace_record_count': len(statuses), 'direct_control_event_count': len(goals),
            'model_status_count': len(statuses), 'model_stop': stopped,
            'interface': 'nav2_msgs/action/FollowPath'}


def summarize_status(trace, output):
    path = Path(trace)
    if not path.is_file():
        return None
    counts, sessions = Counter(), set()
    for line in path.read_text().splitlines():
        try:
            payload = json.loads(line).get('parsed', {})
            if isinstance(payload, dict):
                counts[str(payload.get('status', 'unknown'))] += 1
                if payload.get('session_id'):
                    sessions.add(payload['session_id'])
        except (ValueError, TypeError):
            continue
    result = {'interface': 'nav2_msgs/action/FollowPath', 'status_counts': dict(counts),
              'sessions': sorted(sessions)}
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result
