# InternNavServer (dual_vln_server)

**Package:** `arena_vln_models`  
**Node name:** `internnav_server`  
**Base class:** `BaseModelSimServer`  
**Source:** `arena_vln_models/arena_vln_models/internnav_server.py`

## 职责

InternNavServer 是 VLN/VLA 模型与 ROS 2 Nav2 之间的桥接节点。它：
- 订阅 RGB/Depth/CameraInfo/Odom/Goal/Instruction
- 调用 InternNav backend 进行推理
- 通过 `get_command` service 向 Nav2 controller 提供 `Twist` 命令
- 发布 status、model_output、debug_image、action_image 用于监控和录像

## Subscriptions

| Topic (param) | Default | Message Type | QoS | Callback | 说明 |
|---------------|---------|-------------|-----|----------|------|
| `pose_topic` | `pose` | `geometry_msgs/msg/PoseStamped` | depth=10, RELIABLE | `_on_pose` | 机器人位姿 |
| `odom_topic` | `odom` | `nav_msgs/msg/Odometry` | depth=10, BEST_EFFORT | `_on_odom` | 里程计（Isaac 用 BEST_EFFORT 发布） |
| `goal_topic` | `goal_pose` | `geometry_msgs/msg/PoseStamped` | depth=10, RELIABLE | `_on_goal` | Episode goal pose |
| `subgoal_topic` | `subgoal` | `geometry_msgs/msg/PoseStamped` | depth=10, RELIABLE | `_on_subgoal` | 中间 subgoal |
| `instruction_topic` | `vln_instruction` | `std_msgs/msg/String` | depth=1, RELIABLE, TRANSIENT_LOCAL | `_on_instruction` | VLN 语义指令 |
| `rgb_topic` | `""` (disabled) | `sensor_msgs/msg/Image` | depth=10, BEST_EFFORT | `_on_rgb` | RGB 相机（条件订阅） |
| `depth_topic` | `""` (disabled) | `sensor_msgs/msg/Image` | depth=10, BEST_EFFORT | `_on_depth` | 深度图（条件订阅） |
| `camera_info_topic` | `""` (disabled) | `sensor_msgs/msg/CameraInfo` | depth=10, BEST_EFFORT | `_on_camera_info` | 相机内参（条件订阅） |

> RGB、depth、camera_info 仅在对应参数非空时创建订阅。  
> odom 和 camera 类 topic 使用 BEST_EFFORT QoS 以兼容 Isaac Sim 的发布端。

## Publishers

| Topic (param) | Default | Message Type | QoS | 说明 |
|---------------|---------|-------------|-----|------|
| `status_topic` | `internnav/status` | `std_msgs/msg/String` | depth=1, RELIABLE, TRANSIENT_LOCAL | 当前状态（JSON），如 `backend_ready`、`internnav_command`、`goal_reached` |
| `model_output_topic` | `internnav/model_output` | `std_msgs/msg/String` | depth=10, RELIABLE | 模型原始输出（JSONL trace） |
| `visualization_topic` | `internnav/debug_image` | `sensor_msgs/msg/Image` | depth=1, RELIABLE | Debug overlay 图像（条件发布） |
| `action_visualization_topic` | `internnav/action_image` | `sensor_msgs/msg/Image` | depth=1, RELIABLE | 动作可视化图像（条件发布） |

> visualization 类 publisher 仅在 `enable_visualization=true` 时创建。

## Services Provided

| Service | Type | 说明 |
|---------|------|------|
| `get_command` | `rosnav_rl_msgs/srv/GetCommand` | Nav2 controller 调用，返回 `geometry_msgs/msg/Twist` |

## 关键参数

| Parameter | Type | Default | 说明 |
|-----------|------|---------|------|
| `mode` | string | `"heuristic"` | 后端模式：`heuristic` / `internnav` |
| `model_path` | string | `""` | 模型权重路径 |
| `device` | string | `"cpu"` | 推理设备：`cpu` / `cuda:0` |
| `inference_rate_hz` | float | 10.0 | 推理频率 |
| `inference_timeout_sec` | float | 0.2 | 单次推理超时 |
| `camera_ready_timeout_sec` | float | 120.0 | 初始相机就绪等待超时 |
| `camera_stale_after_sec` | float | 2.0 | 传感器数据过期阈值 |
| `require_real_backend` | bool | false | 是否要求真实模型后端 |
| `strict_device` | bool | false | 是否严格匹配 device |
| `enable_visualization` | bool | false | 是否发布 debug/action 图像 |
| `adapter_target` | string | `""` | Python adapter target；`mode=internnav` 且为空时默认 `arena_vln_models.internnav:load_internnav_adapter`；配置 `internnav_http_url` 且为空时自动选择 HTTP adapter `arena_vln_models.internnav:load_internvla_realworld_http_adapter` |
| `internnav_http_url` | string | `""` | InternVLA realworld HTTP `/eval_dual` URL；非空时自动切换 `mode=internnav`；空值时 HTTP adapter 默认 `http://127.0.0.1:5801/eval_dual` |
| `internnav_http_timeout_sec` | float | 0.0 | HTTP 请求超时；0.0 时继承 `inference_timeout_sec` |
| `model_output_policy` | string | `"trajectory"` | 模型输出策略 |
| `discrete_arc_turn` | bool | false | 离散转向是否使用前进弧线；默认原地转向 |
| `goal_tolerance` | float | 0.45 | 目标到达容差（米） |
| `angle_tolerance` | float | 0.25 | 角度容差（弧度） |
| `max_linear` | float | 0.6 | 最大线速度 |
| `max_angular` | float | 1.5 | 最大角速度 |

## 接口约定

- 所有 topic 名称相对于 node namespace（如 `/task_generator_node/Ai2_Bot2`）
- `get_command` service 返回的 `Twist` 直接用于 Nav2 cmd_vel
- status 消息为 JSON 格式，包含 `status`、`degraded`、`linear_x`、`angular_z`、`debug` 字段
- TF frame 名称不加 namespace 前缀（TF 是数据面标识符，不是 ROS topic）
- 环境变量可覆盖大部分参数（前缀 `ARENA_EVAL_INTERNNAV_`）
