# LightNav 原生客户端接入

LightNav 和 InternNav 使用相同的 benchmark 入口、场景指令、回合开始屏障、录像和评分。
LightNav 在独立 `arena-lightnav` Compose 项目中运行模型服务；Arena Jazzy 容器内的独立 Python 环境运行轻量客户端和原生 CasADi MPC，直接发布机器人 `Twist`。

## 当前部署

- 算法源码：`/home/kaizhen/vln_ws/LightNav-0`，提交 `497bebf8bd432a34a5de8eff2a9c2b217f2a8a92`。原目录保持不变。
- 镜像：`lightnav-inference:arena-497bebf`，保留原 `:local` 镜像。
- 权重：`/media/kaizhen/data/vln_ws/LightNav-0/models/LightNav-0`，只读挂载到 `/models/LightNav-0`。
- 原生客户端源码和许可证：`arena_ws/deps/lightnav-client/<提交>/`，`source.json` 记录来源、修改状态和每个文件的 SHA-256；启动时验证哈希。
- 客户端解释器：Arena 容器内 `/opt/arena_ws/deps/lightnav-client/venv/bin/python`，Python 3.12，复用 Jazzy 的系统 ROS 包。
- 模型解释器：独立推理容器内 Python 3.11、vLLM 0.19.1；WebSocket `ws://127.0.0.1:8050`。
- `.installed` 同时包含 `isaac`、`internnav`、`lightnav`。一次只选择一个算法控制机器人。

模型默认 batch=1、显式 KV cache=2 GiB、`GPU_MEM_UTIL=0.55`。显式 KV 容量不代表模型总显存；模型与 Isaac 的实测记录见部署输出的 `setup/gpu-during-eval.csv`。

## 启动

终端一启动模型和客户端，保持运行：

```bash
cd /home/kaizhen/arena_ws
src/Arena/_meta/docker/features/benchmark/main serve \
  --profile src/Arena/arena_bringup/configs/benchmark/profiles/lightnav_grscenes.yaml
```

终端二启动无窗口场景、一个回合、300 秒仿真超时以及录像：

```bash
cd /home/kaizhen/arena_ws
src/Arena/_meta/docker/features/benchmark/main eval \
  --profile src/Arena/arena_bringup/configs/benchmark/profiles/lightnav_grscenes.yaml \
  --case grscenes_20_v1/default_2 --episodes 1 --timeout 300 \
  -- --output-prefix lightnav_deploy
```

也可以直接用 `run` 完成服务启动、单回合和服务清理：

```bash
cd /home/kaizhen/arena_ws
src/Arena/_meta/docker/features/benchmark/main run \
  --profile src/Arena/arena_bringup/configs/benchmark/profiles/lightnav_grscenes.yaml \
  --case grscenes_20_v1/default_2 --episodes 1 --timeout 300 \
  -- --output-prefix lightnav_deploy
```

先用同一入口的 `doctor --profile ...` 检查环境，`plan run --profile ...` 查看实际启动范围。`config`、`status`、`logs` 和 `stop` 也接受同一 profile。

```bash
src/Arena/_meta/docker/features/benchmark/main stop \
  --profile src/Arena/arena_bringup/configs/benchmark/profiles/lightnav_grscenes.yaml
```

`serve` 拒绝占用已有速度控制通道。`stop` 停止该 LightNav 服务拥有的客户端及推理容器，保留 Arena、Isaac、InternNav 容器、镜像和缓存。切换到 InternNav 前先结束当前评测并停止 LightNav；InternNav 原有 v1 profile 和省略 `--profile` 的默认命令继续有效。

## 数据与控制约定

1. 场景加载并产生 RGB、里程计，录像四路流就绪，模型状态可见后，Arena 才声明回合开始并发送场景文件中的文字指令。评测目标位姿只进入评分和录像元数据，不发送给 LightNav。
2. 适配层直接调用所复制的 `VlnClient`。每个回合建立新的会话，重复指令不会重启；重置取消旧会话，旧推理和旧求解结果不能进入新回合。
   新图像到达时才提交推理，同一会话按相机时间戳去重并丢弃乱序帧；客户端忙时丢弃该帧，不缓存重发。时钟回退后清空帧序列，等待新回合。
3. 模型路径为拍摄时刻基座坐标系中的累计局部位姿。使用带时间戳的里程计历史及原生坐标函数投影到 odom，保留朝向，不把每个点当成独立位移。
4. 使用原生 `MPCController`：按仿真时间以 10 Hz 求解、20 Hz 发布控制，horizon=5，dt=0.1 秒，Q=(10,10,1)，R=(0.1,0.1)。dt 是控制器约定，模型路径没有时间戳。墙钟看门狗独立运行，仿真暂停仍停车，不补算积压控制周期。速度与加速度上限读取 `arena_robots/robots/Ai2_Bot2/model_params.yaml`。
5. 路径默认从相机帧接收时刻起有效 2 秒，并检查仿真时间年龄；里程计、控制与时钟停滞看门狗为 0.5 秒。看门狗使用墙钟。STOP、断连、过期路径和求解失败使客户端发布零速。模型 STOP 是算法终止信号，不等于到达评测目标。
6. 两种外部直控模式均由算法客户端独占 `cmd_vel`。Arena 在回合结束时发布 `eval_ready(stage=episode, ready=false, reason=episode_finished)`，由客户端停止运动；LightNav 的终止状态不被晚到的同回合开始信号解除。

profile 中保留的 `execution.timing_mode: wall` 表示不启用 Arena 的延迟命令转发器；它不决定 LightNav MPC 的调度时钟。正常 MPC 与控制发布使用上述仿真时间周期。

四路视频是头部相机、Isaac 俯视相机、评分俯视图和客户端实际输入图。最后一路保留接口名称 `debug_overlay`，当前内容是模型收到的 RGB；真实推理路径和控制关联保存在 JSONL 中。

## 输出与复现

每轮输出在 `arena_ws/outputs/lightnav_deploy/<时间>_grscenes_20_v1_Ai2_Bot2_lightnav/`：

- `run_manifest.yaml`：profile 快照、实际后端、镜像、原生源码版本与控制器语义。
- `lightnav_trace.jsonl`：真实模型响应、接收路径及控制命令，通过回合、原生会话和请求编号关联。
- `lightnav_diagnostic_summary.json`：推理次数、有效路径、非零控制、关联完整性，以及按回合/模型会话统计的唯一图像数和重复时间戳数。正常运行的 `duplicate_capture_count` 应为 0。
- `odom.csv`、`cmd_vel.csv`、`start_goal.csv`：轨迹、控制和评测目标。
- `benchmark_result.json`、`social_metrics.json`、`vln_task_metrics.json`、`artifact_validation.json`：链路有效性、任务结果及社交评分。
- `videos/` 和 `video_index.json`：四路录像及检查信息。

安装、构建、测试和硬件版本记录保存在 `outputs/lightnav_deploy/setup/`。不要把接入通过、导航到达和社交评分通过合并成一个“成功”。

GRScenes 原生 `scenario.yaml` 的机器人 start/goal 和行人 pose 三元组使用角度制朝向，由场景加载器转换为弧度。导入器转换版本 3 保留原生 YAML 的角度制，目录清单中明确带 `_yaw_rad` 的分析字段仍使用弧度；行人 waypoint 的第三项是高度，不是朝向。验证新场景时须同时核对实际初始位置与朝向。

新增 feature 的 `install` 入口只复制 `vln_client` 与 `vln_mpc`，不启动原项目的硬件 launch。重新安装可设置 `LIGHTNAV_SOURCE_PATH`、`LIGHTNAV_MODEL_PATH`、`LIGHTNAV_IMAGE`，应使用新的镜像 tag 保留旧版本。模型部署固定为本机 loopback WebSocket；client profile v2 的控制语义不接受 InternNav 参数覆盖。

回退先停止自己的评测与 LightNav 服务，再按部署备份恢复 `.installed` 并反向应用本次接入补丁。不要使用 `git reset --hard`；原有 RViz 改动与 `arena_robots` 工作区内容必须保留。模型、依赖和 Docker 缓存可保留供再次启用。
