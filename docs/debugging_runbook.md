# Debugging Runbook

本指南用于 Arena eval 管线的系统性调试。目标：**在改代码之前先定位根因**。

## 停止阈值

**连续 3 次 rerun 出现相同症状 → 立即停止，不要继续改参数重跑。**

每次 rerun 之前必须回答：
1. 上一次失败的直接原因是什么？（引用具体日志行）
2. 这次改动针对的是哪个根因？
3. 如果这次改动无效，备选诊断路径是什么？

如果无法回答以上三个问题，说明还没有足够信息进行下一次 rerun。

## 日志分诊清单

拿到 eval 日志后，按以下顺序 grep：

### 1. TF 时间戳问题

```bash
grep -i "TF_OLD_DATA\|Lookup would require extrapolation\|tf2" <log>
```

| 信号 | 根因 | 修复 |
|------|------|------|
| `TF_OLD_DATA` 洪水（30Hz+） | TF tree 从未更新，sim time 与 wall time 不匹配 | 检查 `/clock` topic 是否发布、`use_sim_time` 是否一致 |
| `Lookup would require extrapolation` | TF buffer 中的最新 transform 比请求时间旧 | 检查 TF 发布频率、时间戳是否正确 |
| `"timed out waiting for transform"` | 某个 frame 从未发布 | `ros2 run tf2_tools view_frames` 检查 TF tree |

### 2. QoS 不匹配

```bash
grep -i "QoS\|incompatible\|RELIABLE\|BEST_EFFORT\|TRANSIENT_LOCAL\|VOLATILE" <log>
```

| 信号 | 根因 | 修复 |
|------|------|------|
| Publisher BEST_EFFORT, Subscriber RELIABLE | 数据静默丢失，无错误日志 | 订阅端改为 BEST_EFFORT（Isaac Sim 常见） |
| Publisher TRANSIENT_LOCAL, Subscriber VOLATILE | late-joiner 收不到历史消息 | 订阅端改为 TRANSIENT_LOCAL |
| Publisher VOLATILE, Subscriber TRANSIENT_LOCAL | 收到过期/旧 episode 的消息 | 订阅端改为 VOLATILE |

### 3. Topic 未发布

```bash
grep -i "topic.*not.*published\|no publisher\|waiting for\|timed out waiting" <log>
```

| 信号 | 根因 | 修复 |
|------|------|------|
| `"waiting for topic X"` 超时 | 上游节点未启动或 topic 名称不匹配 | `ros2 topic list` 确认 topic 存在 |
| `"no publisher"` | 发布端未创建或已销毁 | 检查发布端节点是否存活 |

### 4. 传感器数据过期

```bash
grep -i "stale\|camera.*timeout\|camera.*ready\|sensor.*timeout" <log>
```

| 信号 | 根因 | 修复 |
|------|------|------|
| `"camera stale"` / `camera_ready_timeout` | 相机数据未到达或时间戳不更新 | 检查相机 topic 发布频率、`camera_stale_after_sec` 参数 |
| `get_command camera gate response: linear=0.000 angular=0.000` | 传感器过期检查阻塞了命令 | 增大 `camera_stale_after_sec` 或修复传感器发布 |

### 5. 模型推理问题

```bash
grep -i "backend\|adapter\|model.*error\|inference.*fail\|cuda\|device" <log>
```

| 信号 | 根因 | 修复 |
|------|------|------|
| `"backend_init_error"` | 模型加载失败 | 检查 `model_path`、`device`、CUDA 可用性 |
| `"backend not ready"` | 模型初始化未完成 | 等待 `backend_ready` status |
| `yaw_sign_mismatch` / 连续左转 | 模型输出方向映射错误 | 检查 `invert_discrete_turns` 参数 |

### 6. Eval 挂起

```bash
grep -i "running\|finished\|timeout\|hang\|deadlock" <log>
```

| 信号 | 根因 | 修复 |
|------|------|------|
| `"running"` 状态永不结束 | `finished` topic 从未发布 | 检查 TaskGenerator 是否正常完成所有 episode |
| `timeout` 后无反应 | 进程未正确处理超时信号 | 检查 signal handler 和 shutdown 逻辑 |

## 改代码前的验证步骤

在修改任何参数或代码之前，先用 ROS 2 CLI 工具验证假设：

```bash
# 在 Docker 容器内执行
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
cd /opt/arena_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# 1. 确认 topic 存在且有发布者
ros2 topic list
ros2 topic info <topic_name>

# 2. 确认 topic 有数据流入
ros2 topic hz <topic_name> --window 10

# 3. 确认 TF tree 完整
ros2 run tf2_tools view_frames

# 4. 确认 service 可用
ros2 service list
ros2 service type <service_name>

# 5. 检查 QoS 配置
ros2 topic info --verbose <topic_name>
'
```

## Rosbag 作为调试工具

当问题难以在线复现时，录制 rosbag 离线分析：

```bash
# 录制所有 topic
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
cd /opt/arena_ws
source /opt/ros/jazzy/setup.bash
ros2 bag record -a -o /tmp/debug_bag
'

# 离线回放分析
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
cd /opt/arena_ws
source /opt/ros/jazzy/setup.bash
ros2 bag info /tmp/debug_bag
ros2 bag play /tmp/debug_bag --topics <topic1> <topic2>
'
```

## 调试决策树

```
Eval 失败
├── 日志中有 TF_OLD_DATA？
│   └── YES → 检查 /clock 和 use_sim_time → 不要改模型参数
├── 日志中有 QoS incompatible？
│   └── YES → 对齐 QoS profile → 不要改模型参数
├── 日志中有 topic not published？
│   └── YES → ros2 topic list 确认 → 检查 launch 文件
├── 日志中有 camera stale？
│   └── YES → ros2 topic hz 确认频率 → 检查相机发布端
├── 日志中有 backend error？
│   └── YES → 检查 model_path/device/CUDA → 这是模型问题
├── 日志中无错误但 eval 挂起？
│   └── YES → ros2 topic echo /task_generator_node/finished → 检查 episode 完成逻辑
└── 以上都不是？
    └── 录制 rosbag → 离线回放 → 逐 topic 检查数据流
```
