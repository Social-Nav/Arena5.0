# RobotManager

**Package:** `task_generator`  
**Source:** `task_generator/task_generator/manager/robot_manager/robot_manager.py`

## 职责

RobotManager 管理单个机器人的完整生命周期：
- 发布 episode start/goal pose
- 通过 Nav2 action 发送导航目标
- 在 dual_vln 模式下通过 `get_command` service 获取模型命令
- 监控 InternNav status 以判断 goal_reached
- 管理 camera readiness（dual_vln 模式）

## Subscriptions

| Topic | Message Type | QoS | Callback | 说明 |
|-------|-------------|-----|----------|------|
| `{ns}/odom` | `nav_msgs/msg/Odometry` | depth=10, RELIABLE | `_robot_pos_callback` | 机器人里程计 |
| `{ns}/navigate_to_pose/_action/status` | `action_msgs/msg/GoalStatusArray` | depth=10, RELIABLE | `_goal_status_callback` | Nav2 导航状态 |
| `{ns}/head_camera/image` | `sensor_msgs/msg/Image` | depth=10, BEST_EFFORT | camera ready marker | RGB 相机（dual_vln 条件订阅） |
| `{ns}/head_camera/depth` | `sensor_msgs/msg/Image` | depth=10, BEST_EFFORT | camera ready marker | 深度图（dual_vln 条件订阅） |
| `{ns}/head_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | depth=10, BEST_EFFORT | camera ready marker | 相机内参（dual_vln 条件订阅） |
| `{ns}/internnav/status` | `std_msgs/msg/String` | depth=10, VOLATILE | `_on_dual_vln_status` | InternNav 状态（dual_vln 条件订阅） |

> `{ns}` = `/task_generator_node/{robot_name}`（如 `/task_generator_node/Ai2_Bot2`）

## Publishers

| Topic | Message Type | QoS | 说明 |
|-------|-------------|-----|------|
| `{ns}/episode_start_pose` | `geometry_msgs/msg/PoseStamped` | depth=1, RELIABLE, TRANSIENT_LOCAL | Episode 起始位姿 |
| `{ns}/episode_goal_pose` | `geometry_msgs/msg/PoseStamped` | depth=1, RELIABLE, TRANSIENT_LOCAL | Episode 目标位姿 |
| `{ns}/episode_goal_pose_metadata` | `geometry_msgs/msg/PoseStamped` | depth=1, RELIABLE, TRANSIENT_LOCAL | 目标位姿元数据 |
| `{ns}/cmd_vel` | `geometry_msgs/msg/Twist` | depth=10, RELIABLE | 速度命令（dual_vln 模式） |

## Action Clients

| Action | Type | 说明 |
|--------|------|------|
| `{ns}/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | Nav2 导航目标 |

## Service Clients

| Service | Type | 说明 |
|---------|------|------|
| `{ns}/local_costmap/clear_entirely_local_costmap` | `nav2_msgs/srv/ClearEntireCostmap` | 清除 local costmap |
| `{ns}/local_costmap/clear_around_local_costmap` | `nav2_msgs/srv/ClearCostmapAroundRobot` | 清除机器人周围 costmap |
| `{ns}/get_command` | `rosnav_rl_msgs/srv/GetCommand` | 获取模型速度命令（dual_vln 模式） |

## 接口约定

- 所有 topic 使用 `{ns}` 前缀（robot namespace）
- `episode_goal_pose` 和 `episode_start_pose` 使用 TRANSIENT_LOCAL 以支持 late-joining subscribers
- InternNav status 订阅使用 VOLATILE 以匹配发布端的 QoS
- camera 类订阅使用 BEST_EFFORT 以兼容 Isaac Sim
