"""
Node Interface Consistency Tests

Verifies that the ROS2 interfaces (subscriptions, publishers, services, actions)
documented in docs/nodes/*.md match the actual code.

For each documented interface, this test checks that:
1. The message/service/action type appears in the node's source code
2. The topic/service/action name (or its key component) appears in the source code

This catches "interface drift" where code changes but documentation doesn't get updated.

Usage:
    cd /opt/arena_ws
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    python3 -m pytest src/Arena/arena_bringup/test/test_node_interfaces.py -v
"""

import os
import re
from typing import Dict, List, Tuple

import pytest

_ARENA_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOCS_DIR = os.path.join(_ARENA_SRC, 'docs', 'nodes')

# Source file paths (relative to _ARENA_SRC)
_SOURCE_PATHS = {
    'task_generator': 'task_generator/task_generator/node.py',
    'internnav_server': 'arena_vln_models/arena_vln_models/internnav_server.py',
    'isaac_controller': 'arena_isaac/arena_isaac/arena_isaac/run_isaacsim.py',
    'robot_manager': 'task_generator/task_generator/manager/robot_manager/robot_manager.py',
    'eval_video_recorder': 'arena_bringup/arena_bringup/internnav_eval.py',
    'data_recorder': 'arena_evaluation/arena_evaluation/arena_evaluation/data_recorder_node.py',
    'pedestrian_marker_publisher': 'utils/rviz_utils/rviz_utils/scripts/pedestrian_marker_publisher.py',
    'raycast_obstacle_publisher': 'arena_isaac/arena_isaac/arena_isaac/run_isaacsim.py',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_markdown_interfaces(filepath: str) -> Dict[str, List[Tuple[str, str]]]:
    """Parse markdown tables to extract documented interfaces.

    Returns dict with keys: subscriptions, publishers, services, action_clients.
    Each value is a list of (topic_name, message_type) tuples.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    interfaces = {
        'subscriptions': [],
        'publishers': [],
        'services': [],
        'action_clients': [],
    }

    section_map = {
        '## Subscriptions': 'subscriptions',
        '### Subscriptions': 'subscriptions',
        '## Publishers': 'publishers',
        '### Publishers': 'publishers',
        '## Services Provided': 'services',
        '### Services Provided': 'services',
        '## Action Clients': 'action_clients',
        '### Action Clients': 'action_clients',
    }

    current_section = None
    type_col_idx = 1  # default: second column is the type
    header_cols = []
    for line in content.split('\n'):
        line = line.strip()
        if line in section_map:
            current_section = section_map[line]
            type_col_idx = 1
            header_cols = []
            continue
        # Stop parsing when we hit a non-interface heading
        if (line.startswith('## ') or line.startswith('### ')) and line not in section_map:
            current_section = None
            type_col_idx = 1
            header_cols = []
            continue
        # Detect header row (before the --- separator)
        if line.startswith('|') and current_section:
            cols = [c.strip() for c in line.split('|')[1:-1]]
            if len(cols) >= 2:
                first_col = cols[0].lower()
                if any(first_col.startswith(h) for h in ('topic', 'service', 'action', '产物')):
                    # This is a header row — find the type column
                    header_cols = [c.lower() for c in cols]
                    for i, h in enumerate(header_cols):
                        if h in ('type', 'message type'):
                            type_col_idx = i
                            break
                    continue
        # Skip table separator rows
        if '---' in line and line.startswith('|'):
            continue
        if line.startswith('|') and current_section:
            cols = [c.strip() for c in line.split('|')[1:-1]]
            if len(cols) > type_col_idx:
                # Skip rows where first column looks like a header label
                first_col = cols[0].lower()
                if any(first_col.startswith(h) for h in ('topic', 'service', 'action', '产物')):
                    continue
                # Strip backticks from parsed values
                name = cols[0].strip('`')
                type_str = cols[type_col_idx].strip('`')
                interfaces[current_section].append((name, type_str))

    return interfaces


def _read_source(key: str) -> str:
    """Read source code from a file by key."""
    filepath = os.path.join(_ARENA_SRC, _SOURCE_PATHS[key])
    with open(filepath, 'r') as f:
        return f.read()


def _msg_type_short_name(type_str: str) -> str:
    """Extract the short message type name from a full type string.
    'geometry_msgs/msg/PoseStamped' -> 'PoseStamped'
    'rosnav_rl_msgs/srv/GetCommand' -> 'GetCommand'
    'nav2_msgs/action/NavigateToPose' -> 'NavigateToPose'
    """
    return type_str.split('/')[-1]


def _check_type_in_source(source: str, type_str: str, label: str) -> bool:
    """Check if a message/service/action type appears in source code."""
    short = _msg_type_short_name(type_str)
    return short in source


def _check_name_in_source(source: str, name: str) -> bool:
    """Check if a topic/service/action name (or its key component) appears in source.

    Handles dynamic names like '{ns}/human_states' by checking for the suffix.
    """
    # Strip namespace placeholders
    clean = name.replace('{ns}/', '').replace('{ns}', '')
    clean = clean.replace('{robot}', '').strip('/')
    if not clean:
        return True  # empty after stripping means it's purely dynamic
    return clean in source


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTaskGenerator:
    """TaskGenerator node: docs/nodes/task_generator.md"""

    DOC = _parse_markdown_interfaces(os.path.join(DOCS_DIR, 'task_generator.md'))

    @classmethod
    def _source(cls):
        return _read_source('task_generator')

    def test_subscription_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['subscriptions']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Subscription type '{msg_type}' for '{name}' not found in TaskGenerator code"

    def test_publisher_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['publishers']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Publisher type '{msg_type}' for '{name}' not found in TaskGenerator code"

    def test_service_types_in_code(self):
        source = self._source()
        for name, srv_type in self.DOC['services']:
            assert _check_type_in_source(source, srv_type, name), \
                f"Service type '{srv_type}' for '{name}' not found in TaskGenerator code"

    def test_topic_names_in_code(self):
        source = self._source()
        for name, _ in self.DOC['subscriptions'] + self.DOC['publishers']:
            assert _check_name_in_source(source, name), \
                f"Topic name component '{name}' not found in TaskGenerator code"


class TestInternNavServer:
    """InternNavServer node: docs/nodes/internnav_server.md"""

    DOC = _parse_markdown_interfaces(os.path.join(DOCS_DIR, 'internnav_server.md'))

    @classmethod
    def _source(cls):
        return _read_source('internnav_server')

    def test_subscription_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['subscriptions']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Subscription type '{msg_type}' for '{name}' not found in BaseModelSimServer code"

    def test_publisher_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['publishers']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Publisher type '{msg_type}' for '{name}' not found in BaseModelSimServer code"

    def test_service_types_in_code(self):
        source = self._source()
        for name, srv_type in self.DOC['services']:
            assert _check_type_in_source(source, srv_type, name), \
                f"Service type '{srv_type}' for '{name}' not found in BaseModelSimServer code"

    def test_get_command_service_exists(self):
        """The get_command service is the critical interface for Nav2."""
        source = self._source()
        assert 'GetCommand' in source, "GetCommand service not found in BaseModelSimServer"
        assert 'create_service' in source, "No create_service call in BaseModelSimServer"


class TestIsaacController:
    """IsaacController node: docs/nodes/isaac_controller.md"""

    DOC = _parse_markdown_interfaces(os.path.join(DOCS_DIR, 'isaac_controller.md'))

    @classmethod
    def _source(cls):
        return _read_source('isaac_controller')

    def test_service_types_in_code(self):
        source = self._source()
        for name, srv_type in self.DOC['services']:
            assert _check_type_in_source(source, srv_type, name), \
                f"Service type '{srv_type}' for '{name}' not found in IsaacController code"

    def test_service_names_in_code(self):
        source = self._source()
        for name, _ in self.DOC['services']:
            assert _check_name_in_source(source, name), \
                f"Service name '{name}' not found in IsaacController code"


class TestRobotManager:
    """RobotManager: docs/nodes/robot_manager.md"""

    DOC = _parse_markdown_interfaces(os.path.join(DOCS_DIR, 'robot_manager.md'))

    @classmethod
    def _source(cls):
        return _read_source('robot_manager')

    def test_publisher_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['publishers']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Publisher type '{msg_type}' for '{name}' not found in RobotManager code"

    def test_action_client_types_in_code(self):
        source = self._source()
        for name, action_type in self.DOC['action_clients']:
            assert _check_type_in_source(source, action_type, name), \
                f"Action type '{action_type}' for '{name}' not found in RobotManager code"

    def test_navigate_to_pose_action_exists(self):
        source = self._source()
        assert 'NavigateToPose' in source, "NavigateToPose action not found in RobotManager"


class TestEvalVideoRecorder:
    """EvalVideoRecorder: docs/nodes/eval_video_recorder.md"""

    DOC = _parse_markdown_interfaces(os.path.join(DOCS_DIR, 'eval_video_recorder.md'))

    @classmethod
    def _source(cls):
        return _read_source('eval_video_recorder')

    def test_subscription_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['subscriptions']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Subscription type '{msg_type}' for '{name}' not found in EvalVideoRecorder code"


class TestDataRecorder:
    """DataRecorder nodes: docs/nodes/data_recorder.md"""

    DOC = _parse_markdown_interfaces(os.path.join(DOCS_DIR, 'data_recorder.md'))

    @classmethod
    def _source(cls):
        return _read_source('data_recorder')

    def test_recorder_service_types_in_code(self):
        source = self._source()
        for name, srv_type in self.DOC['services']:
            assert _check_type_in_source(source, srv_type, name), \
                f"Service type '{srv_type}' for '{name}' not found in Recorder code"


class TestPedestrianMarkerPublisher:
    """PedestrianMarkerPublisher: docs/nodes/pedestrian_marker_publisher.md"""

    DOC = _parse_markdown_interfaces(os.path.join(DOCS_DIR, 'pedestrian_marker_publisher.md'))

    @classmethod
    def _source(cls):
        return _read_source('pedestrian_marker_publisher')

    def test_subscription_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['subscriptions']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Subscription type '{msg_type}' for '{name}' not found in PedestrianMarkerPublisher code"

    def test_publisher_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['publishers']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Publisher type '{msg_type}' for '{name}' not found in PedestrianMarkerPublisher code"


class TestRaycastObstaclePublisher:
    """RaycastObstaclePublisher: docs/nodes/raycast_obstacle_publisher.md"""

    DOC = _parse_markdown_interfaces(os.path.join(DOCS_DIR, 'raycast_obstacle_publisher.md'))

    @classmethod
    def _source(cls):
        return _read_source('raycast_obstacle_publisher')

    def test_subscription_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['subscriptions']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Subscription type '{msg_type}' for '{name}' not found in RaycastObstaclePublisher code"

    def test_publisher_types_in_code(self):
        source = self._source()
        for name, msg_type in self.DOC['publishers']:
            assert _check_type_in_source(source, msg_type, name), \
                f"Publisher type '{msg_type}' for '{name}' not found in RaycastObstaclePublisher code"


class TestAllDocsExist:
    """Ensure all documented nodes have their markdown files and vice versa."""

    def test_all_doc_files_exist(self):
        index_path = os.path.join(DOCS_DIR, 'index.md')
        with open(index_path, 'r') as f:
            content = f.read()

        matches = re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', content)
        existing = set(os.listdir(DOCS_DIR))
        for name, filename in matches:
            assert filename in existing, \
                f"Document '{filename}' referenced in index but not found"

    def test_all_doc_files_in_index(self):
        index_path = os.path.join(DOCS_DIR, 'index.md')
        with open(index_path, 'r') as f:
            content = f.read()

        for filename in sorted(os.listdir(DOCS_DIR)):
            if filename == 'index.md' or not filename.endswith('.md'):
                continue
            assert filename in content, \
                f"docs/nodes/{filename} exists but is not referenced in index.md"
