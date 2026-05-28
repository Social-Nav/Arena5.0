# TaskGenerator Node

**Package:** `task_generator`  
**Node name:** `task_generator`  
**Default namespace:** `/task_generator_node`  
**Source:** `task_generator/task_generator/node.py`

## 职责

TaskGenerator 是 Arena eval 管线的核心编排节点。负责：
- 管理 episode 生命周期（reset、start、finish）
- 发布 VLN 指令和 goal pose
- 协调 world geometry spawn、robot spawn、human agent 就绪
- 提供配置查询服务（worlds、robots、scenarios 等）

## Subscriptions

| Topic | Message Type | QoS | Callback | 说明 |
|-------|-------------|-----|----------|------|
| `{ns}/human_states` | `hunav_msgs/msg/Agents` | depth=10, RELIABLE | `_human_states_callback` | HuNav agent 状态，用于 readiness barrier |

> `{ns}` = `/task_generator_node`（默认）

## Publishers

| Topic | Message Type | QoS | 说明 |
|-------|-------------|-----|------|
| `{ns}/vln_instruction` | `std_msgs/msg/String` | depth=1, RELIABLE, TRANSIENT_LOCAL | 每 episode 的 VLN 语义指令 |
| `{ns}/task_reset` | `std_msgs/msg/Int16` | depth=1, RELIABLE, TRANSIENT_LOCAL | Episode reset 信号（含 episode 编号） |
| `{ns}/finished` | `std_msgs/msg/Empty` | depth=1, RELIABLE, TRANSIENT_LOCAL | 所有 episode 完成信号 |

## Services Provided

| Service | Type | 说明 |
|---------|------|------|
| `{ns}/reset_task` | `std_srvs/srv/Empty` | 手动触发 task reset |
| `{ns}/pause_simulation` | `std_srvs/srv/SetBool` | 暂停/恢复仿真 |
| `{ns}/get_environments` | `task_generator_msgs/srv/GetEnvironments` | 查询可用环境列表 |
| `{ns}/get_parametrizeds` | `task_generator_msgs/srv/GetParametrizeds` | 查询可用参数化配置 |
| `{ns}/get_obstacles` | `task_generator_msgs/srv/GetObstacles` | 查询可用障碍物类型 |
| `{ns}/get_scenarios` | `task_generator_msgs/srv/GetScenarios` | 查询可用场景 |
| `{ns}/get_robots` | `task_generator_msgs/srv/GetRobots` | 查询可用机器人类型 |
| `{ns}/get_worlds` | `task_generator_msgs/srv/GetWorlds` | 查询可用世界 |
| `{ns}/wait_for_world` | `std_srvs/srv/Empty` | 等待 world 加载完成 |

## 关键参数

| Parameter | Type | Default | 说明 |
|-----------|------|---------|------|
| `auto_reset` | bool | false | 是否自动 reset episode |
| `train_mode` | bool | false | 训练模式（跳过某些 eval 检查） |
| `vln_instruction` | string | `"navigate"` | 默认 VLN 指令文本 |
| `vln_instruction_file` | string | `""` | VLN 指令文件路径 |
| `episodes` | int | 1 | 总 episode 数 |
| `timeout` | float | 180.0 | 单 episode 超时（秒） |
| `require_human_states_ready` | bool | false | 是否等待 HuNav agent 就绪 |
| `human_states_ready_timeout_sec` | float | 20.0 | HuNav 就绪等待超时 |

## 接口约定

- 所有 topic 和 service 使用 `self.service_namespace(name)` 生成，默认前缀为 `/task_generator_node`
- `task_reset` 发布 `Int16` 值表示 episode 编号（从 0 开始）
- `finished` 在所有 episode 完成后发布一次
- `vln_instruction` 在每个 episode 开始时发布
