# Node Interface Reference

本文档记录 Arena 项目中每个 ROS 2 node 的完整接口定义，包括 subscriptions、publishers、services 和 actions。

## 接口约定

- 所有 topic 和 service 名称使用 `{ns}` 表示 namespace 前缀
- 默认 namespace 为 `/task_generator_node`
- Robot-specific topic 使用 `{ns}/{robot_name}` 格式（如 `/task_generator_node/Ai2_Bot2`）
- QoS 标注格式：`depth=N, RELIABLE/BEST_EFFORT, VOLATILE/TRANSIENT_LOCAL`

## Node 列表

| Node | Package | 说明 |
|------|---------|------|
| [TaskGenerator](task_generator.md) | `task_generator` | Episode 生命周期管理、VLN 指令发布、配置查询 |
| [InternNavServer](internnav_server.md) | `arena_vln_models` | VLN/VLA 模型 ROS 桥接、get_command service |
| [IsaacController](isaac_controller.md) | `arena_isaac` | Isaac Sim 仿真控制（Pause/Unpause/Step） |
| [RaycastObstaclePublisher](raycast_obstacle_publisher.md) | `arena_isaac` | 行人障碍物信息转换 |
| [RobotManager](robot_manager.md) | `task_generator` | 机器人生命周期、Nav2 导航、dual_vln 命令 |
| [EvalVideoRecorder](eval_video_recorder.md) | `arena_bringup` | 多路视频录制（RGB/Depth/Debug/TopDown） |
| [DataRecorder](data_recorder.md) | `arena_evaluation` | CSV 数据记录、rosbag 录制 |
| [PedestrianMarkerPublisher](pedestrian_marker_publisher.md) | `rviz_utils` | 行人 RViz 可视化 |

## 接口一致性验证

运行以下命令验证代码中的接口与文档一致：

```bash
cd /opt/arena_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 -m pytest src/Arena/arena_bringup/test/test_node_interfaces.py -v
```
