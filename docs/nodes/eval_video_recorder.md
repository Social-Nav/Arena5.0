# EvalVideoRecorder

**Package:** `arena_bringup`  
**Node name:** `internnav_eval_video_recorder`  
**Source:** `arena_bringup/arena_bringup/internnav_eval.py`

## 职责

EvalVideoRecorder 是 eval 管线的视频录制节点。它订阅多个 ROS topic，将数据编码为 H.264 MP4 视频文件。支持 4 路视频输出：
- Ego RGB（机器人视角）
- Ego Depth（深度图）
- Debug Overlay（InternNav 可视化叠加）
- Sim Top Down（仿真俯视图）
- Map Top Down Follow（地图俯视跟随）

## Subscriptions

| Topic (CLI arg) | Message Type | QoS | Callback | 说明 |
|-----------------|-------------|-----|----------|------|
| `task_reset_topic` | `std_msgs/msg/Int16` | depth=10, RELIABLE | `_on_task_reset` | Episode reset 信号 |
| `scenario_reset_topic` | `std_msgs/msg/Int16` | depth=10, RELIABLE | `_on_task_reset` | Scenario reset（条件订阅） |
| `finished_topic` | `std_msgs/msg/Empty` | depth=10, RELIABLE | `_on_finished` | Eval 完成信号 |
| `ego_topic` | `sensor_msgs/msg/Image` | depth=10, BEST_EFFORT | `_on_ego_image` | RGB 相机 |
| `depth_topic` | `sensor_msgs/msg/Image` | depth=10, BEST_EFFORT | `_on_depth_image` | 深度图（条件订阅） |
| `camera_info_topic` | `sensor_msgs/msg/CameraInfo` | depth=10, BEST_EFFORT | `_on_camera_info` | 相机内参（条件订阅） |
| `debug_overlay_topic` | `sensor_msgs/msg/Image` | depth=10, BEST_EFFORT | `_on_debug_overlay_image` | Debug overlay（条件订阅） |
| `sim_top_down_topic` | `sensor_msgs/msg/Image` | depth=10, BEST_EFFORT | `_on_sim_top_down_image` | 仿真俯视图（条件订阅） |
| `odom_topic` | `nav_msgs/msg/Odometry` | depth=10, BEST_EFFORT | `_on_odom` | 里程计 |
| `goal_topic` | `geometry_msgs/msg/PoseStamped` | depth=10, RELIABLE | `_on_goal` | Goal pose |
| `scan_topic` | `sensor_msgs/msg/LaserScan` | depth=10, BEST_EFFORT | `_on_scan` | 激光扫描 |

## 输出产物

| 产物 | 路径 | 格式 |
|------|------|------|
| Ego RGB | `videos/episode_XXXX/ego_observation.mp4` | H.264 MP4 |
| Ego Depth | `videos/episode_XXXX/ego_depth.mp4` | H.264 MP4 |
| Debug Overlay | `videos/episode_XXXX/ego_debug_overlay.mp4` | H.264 MP4 |
| Sim Top Down | `videos/episode_XXXX/sim_top_down.mp4` | H.264 MP4 |
| Map Top Down Follow | `videos/episode_XXXX/map_top_down_follow.mp4` | H.264 MP4 |
| Video Index | `video_index.json` | JSON |

## 接口约定

- 所有 topic 名称通过 CLI 参数传入，由 `internnav_eval.py` 组装
- 相机类 topic 使用 BEST_EFFORT QoS 以兼容 Isaac Sim
- 视频编码优先使用 `libx264`（H.264），fallback 到 `mp4v`
- 该节点作为独立子进程运行，通过 `--ros-args` 传递参数
