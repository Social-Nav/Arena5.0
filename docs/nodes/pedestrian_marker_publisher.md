# PedestrianMarkerPublisher

**Package:** `rviz_utils`  
**Node name:** `pedestrian_marker_publisher`  
**Source:** `utils/rviz_utils/rviz_utils/scripts/pedestrian_marker_publisher.py`

## 职责

将行人位置信息转换为 RViz 可视化 markers。

## Subscriptions

| Topic | Message Type | QoS | Callback | 说明 |
|-------|-------------|-----|----------|------|
| `{ns}/arena_peds` | `arena_people_msgs/msg/Pedestrians` | depth=10, RELIABLE | `pedestrians_callback` | 行人位置信息 |

> `{ns}` = `/task_generator_node`（默认）

## Publishers

| Topic | Message Type | QoS | 说明 |
|-------|-------------|-----|------|
| `{ns}/pedestrian_markers` | `visualization_msgs/msg/MarkerArray` | depth=10, RELIABLE | 行人可视化 markers |

## 接口约定

- 每个行人渲染为一个圆柱体 marker（绿色表示正常，红色表示碰撞风险）
- Marker 使用 `base_footprint` frame
