# SONIC 单卡最小复现工具

该工具面向以下已确认环境：Ubuntu 24.04、RTX 4090 24GB、GLIBC 2.39、约 62GB 内存，并使用 `uv` 管理 Python 环境。

它固定以下版本和源码状态：

| 组件 | 固定值 |
|---|---|
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| PyTorch | 2.7.0 + cu128 |
| Isaac Lab | v2.3.2 / `37ddf626871758333d6ed89cf64ad702aef127d0` |
| SONIC | `c374bae5b9039cd0ee71377e654d11ce1bc69e1d`，2026-08-18 官方 main HEAD |

## 一、传到 Ubuntu

把整个 `sonic-repro-kit` 目录复制到 Ubuntu，例如：

```bash
~/bly/sonic-repro-kit/
```

然后：

```bash
cd ~/bly/sonic-repro-kit
chmod +x sonic_repro.sh
```

脚本默认把环境与源码放在 `~/bly/sonic-repro`，把所有新实验产物放在
`~/bly/runs/<run_id>`。`SONIC_RUNS_ROOT` 只允许指向 `~/bly/runs` 或其子目录；脚本在写入前会解析并验证绝对路径。它不会删除已有环境、仓库、数据集或运行结果。

## 二、先做预检

```bash
./sonic_repro.sh preflight
```

报告保存到：

```text
~/bly/sonic-repro/state/preflight_时间戳.log
```

如果显存占用仍接近 10GB，环境安装可以继续，但不要运行 Isaac Sim、评测或训练。先用报告中的 PID 确认任务归属，并在对应终端正常停止任务。

## 三、安装环境

逐阶段执行，任一步失败即停止，不要跳过失败项：

```bash
./sonic_repro.sh install-system
./sonic_repro.sh install-env
./sonic_repro.sh verify-isaac
```

`verify-isaac` 首次运行可能要求阅读并接受 NVIDIA EULA，也可能花数分钟下载首次运行缓存。看到空场景成功启动后按 `Ctrl+C`；这是该阶段的预期结束方式。

## 四、安装 SONIC 与样例数据

```bash
./sonic_repro.sh clone-sonic
./sonic_repro.sh download-sample
```

该阶段只下载默认 SONIC 论文权重和小型样例动作，不下载约 30GB 的完整 SMPL 包，也不下载 BONES-SEED。

## 五、完成最小闭环

确保 `nvidia-smi` 中总显存占用低于 3GB，再执行：

```bash
./sonic_repro.sh eval
./sonic_repro.sh smoke-train
./sonic_repro.sh render-offline
```

`render-offline` 是当前机器的推荐渲染入口。它先用已跑通的 Isaac Lab headless
物理链路记录一段轨迹，再在独立进程中使用 MuJoCo OSMesa 生成 MP4，不会启用
Isaac Sim cameras、Hydra RTX 或 Vulkan。原 `render` 命令仍保留为官方 Isaac Sim
RTX 渲染入口，但驱动 595.84 环境下预期会在 RTX 插件初始化阶段失败。
`render-offline` 会按规范创建独立运行目录，不会把旧 eval/smoke 的 marker 复制成
本次证据；`verify-minimal` 仍用于审核确实在同一运行目录完成了 eval、render 和
smoke-train 的完整组合。

如果指标评测 OOM，只调整并行环境数量：

```bash
EVAL_ENVS=16 ./sonic_repro.sh eval
EVAL_ENVS=8 ./sonic_repro.sh eval
```

不要同时更改 checkpoint、动作路径或网络配置，否则无法判断问题来自显存还是实验配置。

## 六、验收与回传

最新实验路径记录在：

```text
~/bly/sonic-repro/state/latest_run_dir.txt
```

该文件中的值必须解析到 `~/bly/runs` 内；脚本不会信任移动运行目录前遗留的旧值。

最小闭环的必要证据如下：

| 证据 | 验收条件 |
|---|---|
| `logs/metrics.log` | 正常退出，无 OOM、NaN、缺失 checkpoint 或动作文件 |
| `videos/*.mp4` | 至少一个可播放 MP4 |
| `data/*.trajectory.pkl` | 离线渲染输入；包含根位姿、29 维关节角与 FPS |
| `logs/train_smoke.log` | 完成 5 次迭代并输出 reward/error/FPS |
| `manifests/` | 保存依赖、GPU、Git commit、配置、退出码与渲染元数据 |
| `markers/` | 对应阶段成功后才原子生成 `.ok` 标记 |
| `manifests/verification_report.json` | 顶层 `passed` 为 `true` |

若任一步失败，请回传以下命令输出，不需要复制整个环境：

```bash
cd ~/bly/sonic-repro-kit
./sonic_repro.sh status

RUN_DIR=$(cat ~/bly/sonic-repro/state/latest_run_dir.txt 2>/dev/null || true)
echo "$RUN_DIR"
test -n "$RUN_DIR" && tail -n 150 "$RUN_DIR/logs/metrics.log" 2>/dev/null
test -n "$RUN_DIR" && tail -n 150 "$RUN_DIR/logs/dump_trajectory.log" 2>/dev/null
test -n "$RUN_DIR" && tail -n 150 "$RUN_DIR/logs/render_mujoco.log" 2>/dev/null
test -n "$RUN_DIR" && tail -n 150 "$RUN_DIR/logs/train_smoke.log" 2>/dev/null
```

## 七、MuJoCo 离线渲染

Windows 修改推送后，Ubuntu 执行端先记录实际状态；如果外层工作树不适合
fast-forward，同步前停止并回传状态，不要 reset 或手工改源码：

```bash
cd ~/bly
git status --short --branch
git rev-parse HEAD
git -C sonic-repro/GR00T-WholeBodyControl status --short --branch
git -C sonic-repro/GR00T-WholeBodyControl rev-parse HEAD
git -C sonic-repro/IsaacLab status --short --branch
git -C sonic-repro/IsaacLab rev-parse HEAD
nvidia-smi

# 仅在外层工作树适合 ff-only 同步时执行
git pull --ff-only origin ubantu

cd ~/bly/sonic-repro-kit
bash -n sonic_repro.sh
OFFLINE_RENDER_ENVS=1 OFFLINE_WIDTH=960 OFFLINE_HEIGHT=540 \
  ./sonic_repro.sh render-offline

RUN_DIR="$(cat ~/bly/sonic-repro/state/latest_run_dir.txt)"
test -f "$RUN_DIR/markers/trajectory_dump.ok"
test -f "$RUN_DIR/markers/render.ok"
find "$RUN_DIR/data" -maxdepth 1 -type f -name '*.trajectory.pkl' -ls
find "$RUN_DIR/videos" -maxdepth 1 -type f -name '*.mp4' -size +0c -ls
tail -n 100 "$RUN_DIR/logs/dump_trajectory.log"
tail -n 100 "$RUN_DIR/logs/render_mujoco.log"
```

可分阶段运行：`dump-trajectory` 总是创建新的 `offline_render_时间戳` 目录；
`render-mujoco` 默认读取 latest run，也可用 `OFFLINE_RUN_DIR=/绝对路径` 指定已有
轨迹目录；`render-offline` 将两阶段绑定到同一个新目录。可调参数包括
`OFFLINE_RENDER_ENVS`、`OFFLINE_FRAME_SKIP`、`OFFLINE_WIDTH`、`OFFLINE_HEIGHT`、
`OFFLINE_GL` 和 `OFFLINE_CAMERA_DISTANCE`。默认 `OFFLINE_GL=osmesa`。

渲染脚本不会改动 SONIC 中的 G1 XML。默认使用仓库已有、确实包含 free root +
29 DOF 的 `decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml`；计划中
最初提到的 `sim2mujoco/resources/robots/g1/g1.xml` 实际注释了 waist roll/pitch，只有
34 维 qpos，不能直接承载 36 维轨迹。脚本会在本次 `manifests/` 生成运行时副本，
把机器人 mesh 目录改为绝对路径；若模型包含缺失的 `terrain_mesh`/`terrain_body`，
只从副本移除。外层 recorder 适配器按 SONIC 官方关节名从 dex 模型中筛出 29 个
身体关节，再按官方映射转换为 MuJoCo 顺序；四元数保持 wxyz。

## 八、完整数据阶段的门禁

在取得 BONES-SEED 许可并准备至少约 300GB 可用空间前，不执行完整数据下载、转换和评测。当前 99% 满的数据盘不满足条件。

完整数据阶段包括：下载完整 SMPL、下载 BONES-SEED G1 CSV、转换为 motion library、过滤动作、用官方 checkpoint 跑完整评测，最后才考虑单卡小规模微调。单张 4090 不等价于论文建议的 64+ GPU 全量训练配置。
