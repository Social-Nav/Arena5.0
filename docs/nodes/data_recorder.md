# DataRecorder (Recorder / BagRecorder / DataCollector)

**Package:** `arena_evaluation`  
**Source:** `arena_evaluation/arena_evaluation/arena_evaluation/data_recorder_node.py`

## 职责

DataRecorder 系列节点负责记录 eval 运行时的 ROS 数据：
- **Recorder**: 将指定 topic 数据写入 CSV 文件
- **BagRecorder**: 录制 rosbag
- **DataCollector**: 通用数据收集器

## Recorder

**Node name:** `data_recorder_node`

### Subscriptions

| Topic | Message Type | QoS | Callback | 说明 |
|-------|-------------|-----|----------|------|
| `/clock` | `rosgraph_msgs/msg/Clock` | depth=10, RELIABLE | `clock_callback` | 仿真时钟 |
| `/scenario_reset` | `std_msgs/msg/Int16` | depth=10, RELIABLE | `scenario_reset_callback` | Scenario reset 信号 |

### Services Provided

| Service | Type | 说明 |
|---------|------|------|
| `change_directory` | `arena_evaluation_msgs/srv/ChangeDirectory` | 切换输出目录 |

## BagRecorder

**Node name:** `bag_recorder_node`

### Subscriptions

| Topic (param) | Default | Message Type | QoS | Callback | 说明 |
|---------------|---------|-------------|-----|----------|------|
| `start_topic` | `episode_start_pose` | `geometry_msgs/msg/PoseStamped` | depth=10, RELIABLE | `_start_callback` | Episode 起始位姿 |
| `goal_topic` | `goal_pose` | `geometry_msgs/msg/PoseStamped` | depth=10, RELIABLE | `_goal_callback` | Goal pose |
| `/clock` | (hardcoded) | `rosgraph_msgs/msg/Clock` | depth=10, RELIABLE | `clock_callback` | 仿真时钟 |
| `scenario_reset_topic` | `/scenario_reset` | `std_msgs/msg/Int16` | depth=10, RELIABLE | `scenario_reset_callback` | Scenario reset |

### Services Provided

| Service | Type | 说明 |
|---------|------|------|
| `change_directory` | `arena_evaluation_msgs/srv/ChangeDirectory` | 切换输出目录 |

## DataCollector

**Node name:** `data_collector{unique_name}`

### Subscriptions

| Topic | Message Type | QoS | Callback | 说明 |
|-------|-------------|-----|----------|------|
| (dynamic) | (dynamic) | depth=10, RELIABLE | `callback` | 动态订阅，根据配置决定 topic 和类型 |

> DataCollector 的订阅 topic 和类型由运行时配置决定，支持 `LaserScan`、`Odometry`、`Twist`、`Agents` 等类型。

## 接口约定

- Recorder 和 BagRecorder 通过 `change_directory` service 切换输出目录
- 输出 CSV 文件包含 timestamp 和数据字段
- BagRecorder 使用 `rosbag2_py` 进行录制
