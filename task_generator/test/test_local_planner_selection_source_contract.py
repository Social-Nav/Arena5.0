from pathlib import Path


def test_explicit_node_local_planner_precedes_robot_config():
    source = Path(__file__).resolve().parents[1] / 'task_generator/manager/robot_manager/robot_manager.py'
    text = source.read_text()
    marker = "str(self.node.rosparam[str].get('local_planner', '') or '').strip()"
    assert marker in text
    assert "'local_planner': (" in text


def test_waypoints_branch_forwards_local_planner_to_follow_path():
    source = Path(__file__).resolve().parents[1] / 'task_generator/manager/robot_manager/robot_manager.py'
    text = source.read_text()
    start = text.index('if self._uses_waypoints():')
    end = text.index('await self.node.do_launch(launch_description)', start)
    branch = text[start:end]
    assert 'follow_path.launch.py' in branch
    assert "'local_planner': (" in branch

