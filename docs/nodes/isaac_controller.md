# IsaacController

**Package:** `arena_isaac`  
**Node name:** `isaac`  
**Source:** `arena_isaac/arena_isaac/arena_isaac/run_isaacsim.py`

## 职责

IsaacController 是 Isaac Sim 的 ROS 2 桥接节点。负责：
- 提供仿真控制服务（Pause/Unpause/Step）
- 监听 task reset/finished 信号以同步仿真状态
- 管理 Isaac 世界的主循环（world.step）

## Subscriptions

| Topic | Message Type | QoS | Callback | 说明 |
|-------|-------------|-----|----------|------|
| `/task_generator_node/task_reset` | `std_msgs/msg/Int16` | depth=10, RELIABLE | `_on_task_reset` | Episode reset 信号 |
| `/task_generator_node/{robot}/navigate_to_pose/_action/status` | `action_msgs/msg/GoalStatusArray` | depth=10, RELIABLE | `_on_nav_status` | Nav2 导航状态 |
| `/task_generator_node/finished` | `std_msgs/msg/Empty` | depth=10, RELIABLE | `_on_finished` | Eval 完成信号 |

## Services Provided

| Service | Type | 说明 |
|---------|------|------|
| `/isaac/PauseSimulation` | `std_srvs/srv/Trigger` | 暂停仿真 |
| `/isaac/UnpauseSimulation` | `std_srvs/srv/Trigger` | 恢复仿真 |
| `/isaac/StepSimulation` | `std_srvs/srv/Trigger` | 单步推进仿真 |

## 接口约定

- 所有 service 使用 `/isaac/` 前缀
- `PauseSimulation` 和 `UnpauseSimulation` 控制 `controller.running` 标志
- `StepSimulation` 执行单次 `world.step()` 并立即返回
- 该节点必须在 Isaac Sim 进程内运行（依赖 `omni.isaac.core`）
