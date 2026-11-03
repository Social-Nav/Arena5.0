"""Generic model-status trace summary; no inference or motion interpretation."""
import json
import os
from collections import Counter
from pathlib import Path


def _records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            result.append(record)
    return result


def _unified_executor_evidence(root, session):
    """Current-session evidence from the unified ArenaModelExecutor trace.

    The unified executor writes ``executor_<pid>_<ns>.jsonl`` with
    ``inference_submitted``/``decision``/``follow_path_goal``/``goal_accepted``
    events.  Only one file carrying the current episode session may qualify;
    a status heartbeat alone never does.
    """
    empty = {
        'executor_trace_present': False,
        'executor_trace_file': None,
        'unified_control_loop': False,
        'inference_submitted_count': 0,
        'lisn_grounded_goal_count': 0,
        'follow_path_decision_count': 0,
        'follow_path_goal_count': 0,
        'goal_accepted_count': 0,
        'trajectory_event_count': 0,
        'model_control_event_count': 0,
        'control_path': 'nav2_follow_path_rpp',
    }
    if not session:
        return empty
    best = None
    for path in sorted(root.glob('executor_*.jsonl')):
        records = _records(path)
        counts = Counter()
        decisions = 0
        for record in records:
            if record.get('session_id') != session:
                continue
            event = record.get('event')
            counts[event] += 1
            if event == 'decision':
                decision = record.get('decision')
                action = decision.get('action') if isinstance(decision, dict) else None
                if action == 'follow_path':
                    decisions += 1
        if not counts:
            continue
        inference = counts.get('inference_submitted', 0)
        lisn_grounded = counts.get('lisn_grounded_goal_received', 0)
        goals = counts.get('follow_path_goal', 0)
        accepted = counts.get('goal_accepted', 0)
        control_path = (
            os.environ.get('ARENA_LISN_CONTROL_PATH')
            or 'nav2_follow_path_social_mpc'
        ) if lisn_grounded else 'nav2_follow_path_rpp'
        candidate = {
            'executor_trace_present': True,
            'executor_trace_file': path.name,
            # LISN has no fast-model inference_submitted event: its goal comes
            # from the slow-loop grounded-goal consumer.  Both are real
            # current-session control loops.
            'unified_control_loop': bool(
                (inference or lisn_grounded) and decisions and goals and accepted
            ),
            'inference_submitted_count': inference,
            'lisn_grounded_goal_count': lisn_grounded,
            'follow_path_decision_count': decisions,
            'follow_path_goal_count': goals,
            'goal_accepted_count': accepted,
            'trajectory_event_count': decisions,
            'model_control_event_count': (
                inference + lisn_grounded + decisions + goals + accepted
            ),
            'control_path': control_path,
        }
        score = candidate['model_control_event_count']
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best else empty


def _external_client_traces(root):
    """Locate an external FollowPath client trace (LightNav / NaVILA).

    The client runs in its own container and writes to the session output root,
    not the evaluation run directory, so accepted locations are ordered from the
    most explicit to the session-layout sibling:

    1. ``<run>/external_client_trace/*.jsonl`` (copied derived evidence)
    2. ``<run>/client_trace/*.jsonl``
    3. ``<session_root>/model_runtime/client/*.jsonl`` where ``session_root`` is
       an ancestor of ``<session_root>/evaluation/<control_path>/<case>/<run>``
    """
    candidates = []
    for sub in ('external_client_trace', 'client_trace'):
        directory = root / sub
        if directory.is_dir():
            candidates.extend(sorted(directory.glob('*.jsonl')))
    if candidates:
        return candidates
    for parent in list(root.parents)[:6]:
        directory = parent / 'model_runtime' / 'client'
        if directory.is_dir():
            candidates.extend(sorted(directory.glob('*.jsonl')))
            break
    return candidates


def _external_client_evidence(root, session, outcome):
    """Strict current-session evidence from an external FollowPath client.

    The client trace and the Nav2 ``path_execution.jsonl`` logger are separate
    producers and may assign different producer-local session strings.  The
    binding that actually proves one episode is therefore:

    * every counted client event carries the authoritative episode session
      (``episode_outcome.session_id``);
    * every ``goal_accepted`` UUID is also observed by ``path_execution`` with
      ACCEPTED/EXECUTING/SUCCEEDED status, and ``path_execution`` contains no
      goal UUID that the client did not accept (a foreign/old-run goal fails);
    * ``path_execution`` carries exactly one lifecycle session and a real
      ``controller_active`` transition.

    A status heartbeat alone, a missing accepted goal, or a mixed session never
    qualifies.
    """
    empty = {
        'external_client_trace_present': False,
        'external_client_trace_files': [],
        'external_client_session_match': False,
        'external_client_event_counts': {},
        'external_inference_count': 0,
        'external_inference_result_count': 0,
        'external_path_submitted_count': 0,
        'external_goal_accepted_count': 0,
        'external_joined_goal_count': 0,
        'external_control_loop': False,
        'path_execution_session_ids': [],
        'path_execution_controller_active': False,
        'external_client_terminal_reason': None,
    }
    if not session:
        return empty
    paths = _external_client_traces(root)
    if not paths:
        return empty
    records = []
    for path in paths:
        records.extend(_records(path))
    current = [record for record in records if record.get('session_id') == session]
    counts = Counter(record.get('event') for record in current)
    accepted = {
        record.get('goal_uuid') for record in current
        if record.get('event') == 'goal_accepted' and record.get('goal_uuid')
    }
    path_records = _records(root / 'path_execution.jsonl')
    observed_goals = {
        record.get('goal_id') for record in path_records
        if record.get('event') == 'goal_status' and record.get('goal_id')
        and record.get('status') in (1, 2, 4)
    }
    lifecycle_sessions = sorted({
        str(record.get('session_id')) for record in path_records if record.get('session_id')
    })
    controller_active = any(record.get('event') == 'controller_active' for record in path_records)
    joined = accepted & observed_goals
    exact_goal_join = bool(accepted) and accepted == observed_goals
    inference = counts.get('inference_started', 0)
    results = counts.get('inference_result', 0)
    submitted = counts.get('path_submitted', 0)
    accepted_count = counts.get('goal_accepted', 0)
    loop = bool(
        current
        and inference > 0 and results > 0 and submitted > 0 and accepted_count > 0
        and exact_goal_join and controller_active
        and len(lifecycle_sessions) == 1
        and counts.get('episode_released', 0) > 0
    )
    return {
        **empty,
        'external_client_trace_present': True,
        'external_client_trace_files': [path.name for path in paths],
        'external_client_session_match': bool(current),
        'external_client_event_counts': dict(counts),
        'external_inference_count': inference,
        'external_inference_result_count': results,
        'external_path_submitted_count': submitted,
        'external_goal_accepted_count': accepted_count,
        'external_joined_goal_count': len(joined),
        'external_control_loop': loop,
        'path_execution_session_ids': lifecycle_sessions,
        'path_execution_controller_active': controller_active,
        'external_client_terminal_reason': outcome.get('reason'),
    }


def execution_evidence(directory):
    """Count current-session native action evidence, or a legitimate no-motion STOP."""
    root = Path(directory)
    # The outcome file is formatted JSON, not necessarily JSONL.
    try:
        outcome = json.loads((root / 'episode_outcome.json').read_text())
    except (OSError, ValueError):
        outcome = {}
    session = outcome.get('session_id')
    statuses = [r.get('parsed', {}) for r in _records(root / 'model_trace.jsonl')]
    statuses = [r for r in statuses if isinstance(r, dict) and session and r.get('session_id') == session]
    goals = {r.get('goal_id') for r in _records(root / 'path_execution.jsonl')
             if session and r.get('goal_id') and r.get('session_id') == session and r.get('event') == 'goal_status' and r.get('status') in (1, 2, 4)}
    stopped = outcome.get('reason') == 'model_stop' and any(r.get('status') == 'stop' for r in statuses)
    unified = _unified_executor_evidence(root, session)
    external = _external_client_evidence(root, session, outcome)
    legacy_pass = bool(statuses) and bool(goals or stopped)
    if unified['executor_trace_present']:
        path_count = unified['goal_accepted_count']
        trajectory_count = unified['trajectory_event_count']
        control_event_count = unified['model_control_event_count']
    elif external['external_client_trace_present']:
        path_count = external['external_joined_goal_count']
        trajectory_count = external['external_path_submitted_count']
        control_event_count = (
            external['external_inference_count']
            + external['external_inference_result_count']
            + external['external_path_submitted_count']
            + external['external_goal_accepted_count']
        )
    else:
        path_count = len(goals)
        trajectory_count = len(goals)
        control_event_count = len(goals)
    evidence = {'pass': legacy_pass or unified['unified_control_loop'] or external['external_control_loop'],
                'trace_present': (root / 'model_trace.jsonl').is_file(),
                'trace_record_count': len(statuses), 'direct_control_event_count': len(goals),
                'model_status_count': len(statuses), 'model_stop': stopped,
                'control_path': 'nav2_follow_path_rpp',
                'interface': 'nav2_msgs/action/FollowPath',
                **unified, **external}
    # The per-mode counters must win over the shared empty defaults that the
    # unified/external evidence blocks carry.
    evidence['path_count'] = path_count
    evidence['trajectory_event_count'] = trajectory_count
    evidence['model_control_event_count'] = control_event_count
    return evidence



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
    result = {'interface': 'nav2_msgs/action/FollowPath',
              'control_path': 'nav2_follow_path_rpp', 'status_counts': dict(counts),
              'sessions': sorted(sessions)}
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result
