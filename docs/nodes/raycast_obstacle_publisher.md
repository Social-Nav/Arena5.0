# RaycastObstaclePublisher

**Package:** `arena_isaac`  
**Node name:** `raycast_obstacle_publisher`  
**Source:** `arena_isaac/arena_isaac/arena_isaac/run_isaacsim.py`

## 职责

将 Isaac Sim 中的行人位置转换为 HuNav 可用的最近障碍物信息，供 social navigation 使用。

## Subscriptions

| Topic | Message Type | QoS | Callback | 说明 |
|-------|-------------|-----|----------|------|
| `/task_generator_node/arena_peds` | `arena_people_msgs/msg/Pedestrians` | depth=10, RELIABLE | `_peds_callback` | 行人位置信息 |

## Publishers

| Topic | Message Type | QoS | 说明 |
|-------|-------------|-----|------|
| `/task_generator_node/hunav_closest_obstacles` | `hunav_msgs/msg/Agents` | depth=10, RELIABLE | 最近障碍物信息（供 HuNav 使用） |

## 接口约定

- 输入 `arena_peds` 来自 Isaac Sim 的行人管理
- 输出 `hunav_closest_obstacles` 被 HuNav agent manager 消费
- 该节点必须在 Isaac Sim 进程内运行
