# arena_vln_models

Arena 的通用模型路径执行与评测支持包。模型仓库负责推理、观测处理、动作解析和路径生成；本包提供原生 Nav2 `FollowPath` 执行、回合生命周期和公共 benchmark runner。

模型支持状态、整体架构、接口约定、NaVILA、InternNav 与 LightNav 下载部署、评测命令及结果说明统一维护在 [模型接入与评估总文档](../docs/Model_Evaluation.md)。本页只说明包内入口和开发验证。

## 模块与入口

| 入口 | 职责 |
| --- | --- |
| `arena_vln_models/runner.py`、`profile.py` | 公共 benchmark 命令、v3 profile 加载与外部进程管理 |
| `arena_vln_models/session.py`、`lifecycle.py` | 回合身份、状态心跳、controller 启停、取消与停车确认 |
| `arena_vln_models/tracking_config.py` | 合成专用 RPP 和 velocity smoother 参数，复用机器人限制 |
| `arena_vln_models/diagnostics.py`、`timing.py` | 路径执行证据和控制时序诊断 |
| `launch/follow_path.launch.py` | 启动 controller server、RPP、velocity smoother 及平滑器 lifecycle manager |
| `arena_vln_models/example_client.py` | 无模型的原生路径客户端示例，演示心跳、路径、抢占、回合切换和取消 |

安装后的可执行入口为 `benchmark`、`example_client` 和 `timing_observer`。任务管理器负责 controller 生命周期；单独启动控制 launch 不等于放行评测回合。launch 必需参数为 `robot`、`namespace`、`frame`，完整启动流程见总文档。

示例可用 `ros2 run arena_vln_models example_client --help` 查看参数。`--points` 接受相对于本回合首次提交路径时保存的 odom 位姿的局部 `[x, y, yaw]` 偏移，`--replacement-points` 与 `--replace-after` 演示路径抢占。它不调用真实模型；路径成功后主动发送 STOP 是示例自己的行为，不能视作 Arena 将所有路径成功转换为回合结束。

## 开发验证

修改源码后，在宿主机 Arena 工作空间更新安装：

```bash
cd ~/arena_ws
src/Arena/_meta/docker/features/benchmark/main runtime
```

以下测试在已经准备好的 Arena Jazzy 容器终端中运行。先加载安装环境，并进入 **Arena 仓库根目录**：

```bash
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
cd /opt/arena_ws/src/Arena
```

常规测试覆盖接口契约、profile、公共 runner、生命周期失败边界及结果生成。显式关闭真实 RPP 用例，不启动模型服务、Isaac 或控制 launch：

```bash
ARENA_RUN_RPP_TESTS=0 /opt/venv/bin/python3 -m pytest \
  -p no:cacheprovider arena_vln_models/test -q
```

真实 RPP 集成测试单独运行，使用独立 ROS domain、Nav2 控制组件和模拟里程计，不依赖模型服务、Isaac 或 GPU。测试会启动并清理自己的 ROS 子进程：

```bash
ROS_DOMAIN_ID=97 ARENA_RUN_RPP_TESTS=1 /opt/venv/bin/python3 -m pytest \
  -p no:cacheprovider arena_vln_models/test/test_rpp_integration.py -v
```

运行前确保 domain 97 未被其他测试占用；测试要求 `ROS_DOMAIN_ID >= 70`，不能使用实际 benchmark 默认的 domain 1。上述绝对路径属于 Arena 容器；本地 Jazzy 环境应加载自己的 Arena 安装，并使用具备相同依赖的 Python。

## 历史验证记录

**2026-10-10，通用接口迁移阶段**：四个修改包在临时 colcon 覆盖层构建成功；接口、录像、episode barrier、评测及普通导航相关回归共 494 项通过，真实 RPP 集成测试单独通过。该记录的验证对象是通用执行链，不代表三个模型均已完成接入，也不是本次文档整理重新运行的测试结果。

当时的 RPP 测试使用独立 domain 97 和模拟里程计，覆盖前进、左右曲线、短路径、末端旋转、抢占、取消、错误结果、心跳暂停与恢复、仿真时钟暂停后的停车、旧 session 过滤、STOP/error、最终速度发布者、CSV 记录及示例客户端回合；该阶段未运行真实模型或完整 Isaac 场景。

随后完成的 NaVILA、InternNav 与 LightNav 真实模型餐厅场景及其验收边界，见总文档的 [验证记录](../docs/Model_Evaluation.md#validation)。三个模型的部署与任务结果均以总文档为准。
