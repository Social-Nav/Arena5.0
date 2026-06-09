# Episodic Data Annotation for Dynamic Social VLN Evaluation

本文档定义 Arena 中每条 Dynamic Social VLN / DualVLN 端到端评测 episode 需要交付的**标注字段、运行产物、metric 框架和验收口径**。目标读者是下游数据生产、评测聚合、模型分析和论文结果复现实验的同学。

本文档中的 “episode” 指一次完整的机器人导航尝试：从 task reset / scenario reset 后机器人、目标、人群和传感器就绪开始，到到达目标、超时、碰撞、人工终止或评测进程结束为止。

!!! warning "真实传感器约束"
    Dynamic Social VLN 评测必须使用真实 Isaac / ROS topic 输出的 TF、odom、RGB、depth、camera_info、HuNav human states 等数据。不要用 dummy / fallback 数据填补缺失传感器；缺失时应 fail fast，并在 artifact validation 中标注失败原因。

## 1. 设计目标

Episodic data annotation 需要同时回答四类问题：

1. **任务是否完成**：机器人是否按语言指令到达目标，是否移动，是否超时或碰撞。
2. **社交导航是否安全**：是否撞人、near miss、侵犯 personal space、在人群中卡死。
3. **端到端 VLN 模型是否真实工作**：DualVLN / InternNav 是否接收到有效视觉输入，是否输出有效控制或轨迹，是否走到官方期望的 coordinate grounding / trajectory 主路径，还是依赖 symbolic fallback。
4. **产物是否可复现、可审计**：是否有 manifest、scenario config、CSV、trace、metrics、视频和人工/自动视觉复查记录。

因此每条 episode 的标注不是单个 success/fail，而是一个分层 schema：

```text
scenario static annotation
  └── episode runtime annotation
        ├── task metrics
        ├── social metrics
        ├── VLN/model diagnostics
        ├── artifact quality checks
        └── aggregate-ready labels / failure taxonomy
```

## 2. 文件组织约定

每个 eval run directory 应至少包含以下文件。对于 `episodes=1` 的短期 pipeline，run directory 通常等价于 episode directory；对于多 episode run，必须在 `episodes/episode_XXXX/` 或 manifest 中明确 episode 边界。

```text
<run_dir>/
  run_manifest.yaml                 # 运行参数、场景、模型、artifact 索引
  params.yaml                       # 机器人半径、topic、planner 等运行参数快照
  start_goal.csv                    # 每个 episode 的 start / goal pose
  episode.csv                       # episode id / reset timeline
  odom.csv                          # robot pose / velocity time series
  cmd_vel.csv                       # control command time series
  scan.csv                          # base collision / laser sanity check
  human_states.csv                  # HuNav agents time series
  metrics.csv                       # base navigation metrics
  social_metrics.json               # HuNav social metrics
  artifact_validation.json          # artifact / social-nav acceptance gate
  internnav_status.json             # latest model server status
  internnav_trace.jsonl             # per-decision model/observation/control trace
  internnav_diagnostic_summary.json # trace-level aggregate diagnostics
  video_index.json                  # video file index and frame counts
  videos/episode_0000/
    ego_observation.mp4
    ego_debug_overlay.mp4
    sim_top_down.mp4
    map_top_down_follow.mp4
  frame_analysis/                  # optional but recommended for release-quality data
    video_frame_analysis.json
    contact_sheet.jpg
```

## 3. Scenario-level annotation

Scenario-level annotation 是 episode 运行前的静态标注，通常来自 `arena_bringup/configs/social_nav_scenarios/*.yaml` 这类 overlay。它不替代 Arena 原生 world / scenario YAML，而是把 world、robot、HuNav 人群、语言指令和评测门槛绑定成 benchmark case。

### 3.1 必填字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | number/string | 是 | 标注 schema 版本，例如 `0.1`。 |
| `id` | string | 是 | 全局唯一 scenario id，例如 `hospital_1_demo_001`。 |
| `name` | string | 是 | 人类可读名称。 |
| `description` | string | 推荐 | 说明场景意图、动态人群、目标区域。 |
| `world.name` | string | 是 | Arena world 名称，例如 `hospital_1`。 |
| `world.world_config` | URI/path | 是 | 原生 world config。 |
| `world.map_yaml` | URI/path | 是 | 2D map yaml。 |
| `world.native_scenario.name` | string | 是 | 原生 scenario 名称，例如 `default`。 |
| `world.native_scenario.file` | URI/path | 是 | 原生 HuNav / obstacle scenario 文件。 |
| `robot.name` | string | 是 | robot id，例如 `Ai2_Bot2`。 |
| `robot.local_planner` | string | 是 | 被测方法，例如 `dual_vln`。 |
| `robot.start.pose_xy_yaw` | `[x,y,yaw]` | 是 | map frame 下起点。 |
| `robot.goal.pose_xy_yaw` | `[x,y,yaw]` | 是 | map frame 下目标。 |
| `robot.goal.tolerance_m` | number | 是 | goal reached 判定半径。 |
| `language.instruction_id` | string | 是 | 指令 id。 |
| `language.instruction` | string | 是 | 给 VLN/VLA model 的自然语言指令。 |
| `language.instruction_type` | enum | 是 | 例如 `goal_only`、`social_distance_goal`、`route_instruction`。 |
| `humans.simulator` | enum | 是 | 当前为 `hunav`。 |
| `humans.expected_count` | int | 是 | 期望动态人类数量；用于 coverage check。 |
| `humans.source` | enum/string | 是 | `native_scenario` 或其它来源。 |
| `task_spec.predicates.success` | list[string] | 是 | 任务成功 predicate。 |
| `task_spec.predicates.social_constraints` | list[string] | 是 | 社交安全约束。 |
| `evaluation.timeout_sec` | number | 是 | episode 超时阈值。 |
| `evaluation.metrics` | object | 是 | 任务、社交、模型诊断 metric 列表。 |
| `evaluation.pass_criteria` | object | 是 | pass/fail gate。 |
| `artifacts_required` | list[string] | 是 | 下游验收必须存在的文件。 |

### 3.2 推荐字段

| 字段 | 说明 |
| --- | --- |
| `world.semantic_regions` | 标注 start / goal / interaction region，用于论文分组和错误归因。 |
| `language.rephrases` | 同一 episode 的等价自然语言重述，用于 language robustness。 |
| `humans.density_level` | `low` / `medium` / `high` / `crowded`。 |
| `humans.behavior_tags` | `crossing`、`queueing`、`oncoming_flow`、`group_motion` 等。 |
| `difficulty` | `easy` / `medium` / `hard`，需要说明判定依据。 |
| `known_risk_zones` | 走廊交叉口、门口、窄通道等 social interaction 区域。 |
| `oracle_path` | 可选静态几何 oracle path；动态社交评测默认不依赖 SPL。 |

## 4. Episode-level runtime annotation

每次运行后，需要把静态 scenario 和实际运行结果合并成 episode runtime annotation。建议生成一个下游消费用的 `episode_annotation.yaml` 或 `episode_annotation.json`，即使当前 pipeline 的原始来源分散在 `run_manifest.yaml`、CSV、trace 和 metrics 中。

### 4.1 顶层 schema

```yaml
schema_version: 1
episode_id: hospital_1_demo_001__seed0__rep0
run_dir: /home/ubuntu/arena_jazzy_ws/outputs/<prefix>/<timestamp>_hospital_1_Ai2_Bot2_internnav
scenario:
  scenario_id: hospital_1_demo_001
  scenario_config_path: src/Arena/arena_bringup/configs/social_nav_scenarios/hospital_1_demo_001.yaml
  world: hospital_1
  native_scenario: default
  simulator: isaac_eval
  human_simulator: hunav
robot:
  name: Ai2_Bot2
  local_planner: dual_vln
  start_xy_yaw: [2.85, 26.75, -0.694]
  goal_xy_yaw: [7.0, 29.55, -0.318]
  goal_tolerance_m: 0.45
language:
  instruction_id: hospital_1_demo_001_keep_distance
  instruction: Navigate to the target room through the hospital corridor while keeping a safe distance from pedestrians.
  instruction_type: social_distance_goal
episode:
  seed: 0
  repetition_index: 0
  start_time_sim_sec: 0.0
  end_time_sim_sec: 120.0
  duration_sec: 120.0
  end_reason: timeout
  valid_for_aggregate: true
labels:
  task_success: false
  social_success: true
  overall_success: false
  primary_failure: timeout
  failure_tags: [timeout, task_failure]
```

### 4.2 Episode identity fields

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `episode_id` | scenario id + seed + repetition | 必须稳定、可复现、可 join。 |
| `run_dir` | output path | 原始产物目录。 |
| `scenario.scenario_id` | `run_manifest.yaml.parameters.scenario_config_id` 或 overlay `id` | 场景 id。 |
| `scenario.scenario_config_path` | manifest / runner arg | 静态 overlay 路径。 |
| `episode.seed` | overlay / runner | 随机种子；没有则为 null，但不建议长期缺失。 |
| `episode.repetition_index` | runner | 同 scenario 多次重复的 index。 |
| `episode.end_reason` | manifest / episode.csv / validation | `goal_reached`、`timeout`、`collision`、`human_collision`、`launch_failure`、`invalid_artifact` 等。 |
| `episode.valid_for_aggregate` | artifact validation | 缺关键传感器、dummy data、空视频、teleport 时应为 false。 |

## 5. Required raw signal annotation

这些字段不一定都写在单个 annotation 文件里，但必须能从 run directory 中恢复。

### 5.1 Robot state

| 数据 | 文件/topic | 最低要求 |
| --- | --- | --- |
| start pose | `start_goal.csv` / manifest | map frame 下 `[x,y,yaw]`。 |
| goal pose | `start_goal.csv` / `/episode_goal_pose` | map frame 下 `[x,y,yaw]` 和 tolerance。 |
| odometry | `odom.csv` | time、position、orientation/yaw、linear/angular velocity。 |
| command | `cmd_vel.csv` / trace `command` | `linear_x`、`angular_z`、时间戳。 |
| scan | `scan.csv` | 用于 base collision / obstacle proximity sanity。 |
| reset events | `episode.csv` / `scenario_reset.csv` | episode 边界和 reset index。 |

### 5.2 Human state

`human_states.csv` 每行应能解析为 agent list。每个 human agent 至少包含：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` 或 `name` | 是 | 稳定 human id；必须过滤 HuNav synthetic robot agent。 |
| `position` | 是 | `[x,y,z]` 或至少 `[x,y]`。 |
| `orientation` | 推荐 | yaw 或 quaternion。 |
| `velocity` | 推荐 | `[vx,vy,vz]`，用于 crowd freezing / flow analysis。 |
| `behavior_tag` | 可选 | crossing / walking / standing / queueing 等。 |
| `goal` / `waypoints` | 可选 | 用于预测 human intent，不作为必需 social metric 输入。 |

必须记录 human coverage：

- `human_sample_count`
- `max_humans_observed`
- `observed_human_ids`
- `humans_present`

如果 `humans.expected_count > 0` 但 `max_humans_observed == 0`，该 episode 必须标为 `missing_humans`，不能进入有效 social-nav success 统计。

### 5.3 Visual observations

端到端 VLN/VLA model 必须审计模型实际看到的输入：

| 数据 | 文件/topic | 说明 |
| --- | --- | --- |
| ego RGB | `ego_observation.mp4` / RGB topic | 模型主视觉输入，必须有非零帧。 |
| depth | depth topic / optional `ego_depth.mp4` | 若模型或 fallback 使用 depth，应记录 freshness。 |
| camera info | camera_info topic | intrinsics 是否可用。 |
| debug overlay | `ego_debug_overlay.mp4` | 叠加 action、cmd_vel、goal distance、sensor age。 |
| sim top-down | `sim_top_down.mp4` | 审查机器人、人群、场景几何和异常下陷。 |
| map top-down | `map_top_down_follow.mp4` | 复查轨迹和目标接近过程。 |

视频验收最低要求：

- `video_index.json` 存在；
- required videos 路径存在；
- 每个 required video frame count > 0；
- codec 可播放；
- 关键场景的 `sim_top_down.mp4` 不被天花板完全遮挡；
- release / paper 使用的数据应有 `frame_analysis/video_frame_analysis.json` 或人工复查记录。

## 6. Metric framework

Dynamic Social VLN 的 metric 分四层：task、social、model diagnostics、artifact quality。最终 aggregate row 只保留若干核心字段，但原始 annotation 应保留全部可追溯信息。

### 6.1 Task metrics

| Metric | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| `episode_result` | enum | `metrics.csv` / manifest | `GOAL_REACHED`、`TIMEOUT`、`COLLISION` 等。 |
| `task_success` | bool | derived | 通常 `episode_result == GOAL_REACHED`。 |
| `duration_sec` | float | odom / episode | episode 持续时间。 |
| `path_length_m` | float | odom / social_metrics | 机器人实际路径长度。 |
| `goal_distance_initial_m` | float | start/goal 或 trace | 初始离目标距离。 |
| `goal_distance_final_m` | float | trace / odom goal | 结束时离目标距离。 |
| `goal_progress_m` | float | diagnostic summary | `initial - final`。 |
| `min_goal_distance_m` | float | trace | 曾经最接近目标的距离。 |
| `robot_moved` | bool | path length | dual-vln social eval 建议 `path_length_m >= 0.1`。 |
| `base_collision_count` | int | `metrics.csv` / scan | 机器人与静态障碍碰撞次数。 |
| `timeout` | bool | manifest / result | 是否超时。 |

#### SPL / path-efficiency policy

默认 **不把 SPL 作为 Dynamic Social VLN 的核心 gate**，原因是动态行人会改变有效可行路径，当前 pipeline 没有提供对 moving-human constraints 有效的动态 shortest-path oracle。

如需报告效率，可使用：

- `path_length_m`
- `goal_progress_m / path_length_m`
- `time_to_goal_sec`
- 可选静态 `oracle_path_length_m`，但必须明确是 static map oracle，不应和动态社交约束混为一谈。

### 6.2 Social metrics

默认阈值：

| Threshold | Default | 说明 |
| --- | --- | --- |
| `personal_space_radius_m` | `1.0` | personal space violation 半径。 |
| `near_miss_radius_m` | `0.5` | near miss 进入半径。 |
| `human_collision_radius_m` | `0.25` | human collision 半径。 |
| `crowd_radius_m` | `1.5` | crowd proximity 分析半径。 |
| `crowd_freezing_speed_mps` | `0.05` | 近人且机器人低速视作 freezing。 |

必报 social metrics：

| Metric | 类型 | 通过条件建议 | 说明 |
| --- | --- | --- | --- |
| `humans_present` | bool | true | 有非空 human states 并能和 odom 对齐。 |
| `max_humans_observed` | int | `>= min_humans_observed` | 人群 coverage sanity。 |
| `observed_human_ids` | list | 非空 | 确认不是只录到 synthetic robot。 |
| `min_human_distance_m` | float | `>= human_collision_radius_m`，严格场景可 `>= near_miss_radius_m` | 全 episode 最小人机距离。 |
| `personal_space_violation_time_sec` | float | scenario 决定；严格 social-distance 场景为 `0.0` | 进入 personal space 的累计时间。 |
| `near_miss_count` | int | `0` | 进入 near miss 半径的 rising-edge 次数。 |
| `human_collision_count` | int | `0` | 进入 human collision 半径的 rising-edge 次数。 |
| `crowd_freezing_time_sec` | float | 越低越好 | 近人且机器人速度过低的累计时间。 |
| `large_teleports` | list | empty | odom 大跳变，出现时 episode invalid。 |
| `social_success` | bool | true | 当前实现要求 humans present、无 human collision、无 near miss、无 teleport。 |

### 6.3 VLN / model diagnostics

对于 DualVLN / InternNav 这类端到端模型，必须把“模型是否真的按预期协议工作”作为一等 metric，而不是只看机器人是否移动。

| Metric | 来源 | 说明 |
| --- | --- | --- |
| `model_backend` | manifest/status | `internnav`、mock、external server 等。 |
| `model_checkpoint` | status / env | checkpoint 路径和版本。 |
| `instruction` | manifest / scenario overlay | 实际传入模型的语言指令。 |
| `instruction_type` | scenario overlay | goal-only / route / social-distance。 |
| `rgb_available_ratio` | trace | 决策时 RGB 可用比例。 |
| `depth_available_ratio` | trace | 决策时 depth 可用比例。 |
| `camera_info_available_ratio` | trace | camera intrinsics 可用比例。 |
| `stale_record_count` | diagnostic summary | sensor stale 次数。 |
| `model_result_count` | trace summary | 有效模型输出次数。 |
| `fallback_command_count` | trace summary | fallback command 次数。 |
| `forward_count` / `rotate_count` / `stop_count` | diagnostic summary | 控制分布；`stop_count=0` 对 stop-required task 是风险。 |
| `coordinate_output_count` | trace / model debug | System 2 输出 pixel coordinate 的次数。 |
| `symbolic_output_count` | trace / model debug | 输出 `←/→/↑/↓/STOP` 等 symbolic action 的次数。 |
| `trajectory_output_count` | trace / response | 返回 `output_trajectory` 的次数。 |
| `official_path_ratio` | derived | `coordinate_output_count / model_result_count` 或 trajectory 主路径比例。 |
| `fallback_policy` | debug | `official_discrete`、`goal_guided`、`synthetic_trajectory` 等。 |
| `infer_time_sec_p50/p95` | trace | 推理延迟。 |

#### DualVLN-specific protocol labels

DualVLN episode 必须区分以下标签：

| Label | 含义 | 对结论的影响 |
| --- | --- | --- |
| `dualvln_protocol: coordinate_trajectory` | System 2 稳定输出 pixel coordinate，并触发 latent / trajectory 主路径。 | 可认为在测官方 DualVLN 主路径。 |
| `dualvln_protocol: symbolic_discrete` | 长期输出 arrows / STOP，按官方 symbolic action protocol 执行。 | 可用于协议诊断，但不能声称完整 trajectory 主路径成功。 |
| `dualvln_protocol: goal_guided_fallback` | Arena 使用真实 goal 将 symbolic output 转为 goal-bearing trajectory。 | 只能作为 recovery / ablation，不应用作官方主结果。 |
| `dualvln_protocol: synthetic_trajectory_fallback` | 将 symbolic action 伪造成局部 trajectory。 | 只能用于兼容性诊断。 |
| `dualvln_protocol: degraded_or_mock` | mock、缺模型、缺相机、fallback-only。 | 不能进入端到端模型主表。 |

建议在 episode annotation 中加入：

```yaml
model_diagnostics:
  backend: internnav
  checkpoint: /opt/arena_ws/deps/models/InternVLA-N1-DualVLN
  external_server: true
  model_result_count: 607
  coordinate_output_count: 0
  symbolic_output_count: 607
  trajectory_output_count: 0
  fallback_policy: goal_guided
  dualvln_protocol: goal_guided_fallback
  official_path_ratio: 0.0
  rgb_available_ratio: 1.0
  depth_available_ratio: 1.0
  stale_record_count: 0
```

### 6.4 Artifact quality metrics

| Metric | 说明 |
| --- | --- |
| `artifact_validation_pass` | 所有 required artifact、metrics、人群、视频、model loop 检查是否通过。 |
| `social_nav_ready` | 是否可进入 social-nav aggregate。 |
| `recording_present` | rosbag 是否存在。 |
| `video_frame_counts` | 每路视频帧数。 |
| `video_visual_pass` | 人工或多模态抽帧检查是否通过。 |
| `no_dummy_sensor_data` | 是否确认没有 dummy / fallback sensor data。 |
| `no_large_teleports` | odom 是否没有大跳变。 |
| `human_coverage_pass` | 是否录到期望人群。 |

## 7. Pass/fail labels and failure taxonomy

每条 episode 至少输出以下 labels：

```yaml
labels:
  task_success: false
  social_success: true
  artifact_validation_pass: true
  model_protocol_success: false
  overall_success: false
  valid_for_aggregate: true
  primary_failure: timeout
  failure_tags:
    - timeout
    - task_failure
```

### 7.1 `overall_success`

推荐定义：

```text
overall_success =
  task_success
  AND social_success
  AND artifact_validation_pass
  AND valid_for_aggregate
```

如果论文或报告重点是“官方 DualVLN 主协议”，则还应增加：

```text
official_dualvln_success =
  overall_success
  AND dualvln_protocol == coordinate_trajectory
  AND official_path_ratio >= scenario_defined_threshold
```

### 7.2 Failure tags

标准 failure tags：

| Tag | 触发条件 |
| --- | --- |
| `artifact_failure` | required artifact 缺失、空视频、metrics 缺失、validation fail。 |
| `missing_humans` | 期望有 HuNav humans，但 `humans_present=false`。 |
| `no_motion` | `path_length_m < robot_moved_min_path_length_m`。 |
| `timeout` | 超时结束。 |
| `task_failure` | 未到达目标，但不是更具体错误。 |
| `base_collision` | 静态障碍 / base collision。 |
| `human_collision` | `human_collision_count > 0`。 |
| `near_miss` | `near_miss_count > 0`。 |
| `personal_space_violation` | personal space violation 超过场景允许阈值。 |
| `crowd_freezing` | crowd freezing 超过场景允许阈值。 |
| `large_teleport` | odom 大跳变。 |
| `stale_observation_candidate` | 相机/odom/depth stale 且 episode 未成功。 |
| `model_no_output` | 没有有效 model result。 |
| `model_symbolic_only` | DualVLN 长期只输出 symbolic actions。 |
| `fallback_dominated` | 主要依赖 goal-guided / synthetic fallback。 |
| `invalid_visual_recording` | sim top-down / ego video 不可审计。 |

## 8. Aggregate table schema

下游聚合 CSV 每行对应一个 episode/run。建议字段：

| Field | 说明 |
| --- | --- |
| `run_id` | run directory 名称。 |
| `episode_id` | 稳定 episode id。 |
| `scenario_id` | scenario overlay id。 |
| `world` / `robot` / `planner` / `human` | 运行环境。 |
| `instruction_id` / `instruction_type` | 语言任务分组。 |
| `episode_result` | base result。 |
| `task_success` / `social_success` / `overall_success` | 三层成功标签。 |
| `official_dualvln_success` | 是否满足官方 DualVLN 主协议成功。 |
| `path_length_m` / `goal_progress_m` / `duration_sec` | 基础导航。 |
| `min_human_distance_m` / `near_miss_count` / `human_collision_count` / `personal_space_violation_time_sec` / `crowd_freezing_time_sec` | 社交安全。 |
| `humans_present` / `max_humans_observed` | 人群 coverage。 |
| `model_result_count` / `coordinate_output_count` / `symbolic_output_count` / `trajectory_output_count` | 模型协议诊断。 |
| `dualvln_protocol` / `fallback_policy` / `official_path_ratio` | DualVLN 主路径判定。 |
| `stale_camera_count` / `rgb_available_ratio` / `depth_available_ratio` | 传感器质量。 |
| `artifact_validation_pass` / `video_visual_pass` | 产物质量。 |
| `primary_failure` / `failure_tags` | 错误归因。 |

## 9. Example annotation object

```yaml
schema_version: 1
episode_id: hospital_1_demo_001__seed0__rep0
run_dir: /home/ubuntu/arena_jazzy_ws/outputs/social_nav_video_rerun_20260517/20260517_081450_hospital_1_Ai2_Bot2_internnav
scenario:
  scenario_id: hospital_1_demo_001
  world: hospital_1
  native_scenario: default
  simulator: isaac_eval
  human_simulator: hunav
robot:
  name: Ai2_Bot2
  local_planner: dual_vln
  start_xy_yaw: [2.85, 26.75, -0.694]
  goal_xy_yaw: [7.0, 29.55, -0.318]
  goal_tolerance_m: 0.45
language:
  instruction_id: hospital_1_demo_001_keep_distance
  instruction_type: social_distance_goal
  instruction: Navigate to the target room through the hospital corridor while keeping a safe distance from pedestrians.
metrics:
  task:
    episode_result: GOAL_REACHED
    task_success: true
    path_length_m: 4.2
    goal_progress_m: 3.8
    robot_moved: true
  social:
    humans_present: true
    max_humans_observed: 14
    min_human_distance_m: 0.83
    near_miss_count: 0
    human_collision_count: 0
    personal_space_violation_time_sec: 1.2
    crowd_freezing_time_sec: 0.0
    social_success: true
  model_diagnostics:
    backend: internnav
    model_result_count: 142
    coordinate_output_count: 80
    symbolic_output_count: 20
    trajectory_output_count: 80
    dualvln_protocol: coordinate_trajectory
    official_path_ratio: 0.56
  artifact_quality:
    artifact_validation_pass: true
    social_nav_ready: true
    required_videos_present: true
    video_visual_pass: true
labels:
  task_success: true
  social_success: true
  artifact_validation_pass: true
  model_protocol_success: true
  overall_success: true
  official_dualvln_success: true
  valid_for_aggregate: true
  primary_failure: success
  failure_tags: []
```

## 10. Release checklist for downstream data users

发布一批 Dynamic Social VLN episodic data 前，至少检查：

- [ ] 每个 scenario overlay 通过 validation，且 `id` 唯一。
- [ ] 每个 episode 有 `run_manifest.yaml`、`params.yaml`、`start_goal.csv`、`odom.csv`、`human_states.csv`。
- [ ] 每个 episode 有 `metrics.csv`、`social_metrics.json`、`artifact_validation.json`。
- [ ] 每个 DualVLN episode 有 `internnav_trace.jsonl` 和 `internnav_diagnostic_summary.json`。
- [ ] 所有 required videos 存在且 frame count > 0。
- [ ] 没有使用 dummy / fallback sensor data 填补真实 topic 缺失。
- [ ] `humans.expected_count > 0` 的 episode 中 `humans_present=true`。
- [ ] 所有失败 episode 都有 `primary_failure` 和 `failure_tags`。
- [ ] 聚合 CSV 可由 run directory 重建，且字段与本文档第 8 节一致。
- [ ] 如果报告“官方 DualVLN 主路径成功”，必须同时报告 coordinate / symbolic / trajectory / fallback 相关诊断，而不仅是 `overall_success`。

## 11. Relation to existing Arena artifacts

当前 Arena 已有的产物和本文档字段对应关系：

| 本文档字段 | 当前来源 |
| --- | --- |
| scenario identity | `run_manifest.yaml.parameters.scenario_config_id` / social scenario overlay |
| world / robot / planner / timeout | `run_manifest.yaml.parameters` |
| instruction | `run_manifest.yaml.parameters.vln_instruction` / scenario overlay `language.instruction` |
| start / goal | `start_goal.csv` / `run_manifest.yaml.parameters` / goal topic |
| robot trajectory | `odom.csv` |
| commands | `cmd_vel.csv` / `internnav_trace.jsonl.command` |
| human coverage | `human_states.csv` / `social_metrics.json` |
| base task result | `metrics.csv` |
| social safety | `social_metrics.json` |
| model protocol | `internnav_trace.jsonl` / `internnav_status.json` / `internnav_diagnostic_summary.json` |
| artifact quality | `artifact_validation.json` / `video_index.json` |
| visual review | `videos/episode_XXXX/*.mp4` / `frame_analysis/*` |

