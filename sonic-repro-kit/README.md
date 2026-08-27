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
./sonic_repro.sh collect-state-action
./sonic_repro.sh render-offline
```

`collect-state-action` 使用 released checkpoint 的确定性 `action_mean`，强制 G1
encoder，并在不启用 cameras/RTX 的情况下记录 50 Hz 单帧转移
`(state_t, goal_t, actions, state_tp1)`。环境数不再写死，而是严格派生为
`本批动作数 × 每动作变体数`；默认每动作 4 个 startup 变体。

首次在 Ubuntu 同步外层仓库后，先通过补丁把 Windows 侧的 SONIC 修改应用到嵌套
仓库。应用前要求 SONIC 工作树干净且 HEAD 为固定基线
`c374bae5b9039cd0ee71377e654d11ce1bc69e1d`：

```bash
cd ~/bly/sonic-repro/GR00T-WholeBodyControl
git status --short --branch
git rev-parse HEAD
PATCH_DIR=~/bly/sonic-repro-kit/patches
PATCH_1="$PATCH_DIR/0001-feat-record-minimal-SONIC-state-goal-action-data.patch"
PATCH_2="$PATCH_DIR/0002-fix-preserve-per-environment-joint-defaults.patch"
PATCH_3="$PATCH_DIR/0003-feat-collect-repeated-motion-randomized-dataset.patch"
PATCH_4="$PATCH_DIR/0004-fix-record-runtime-termination-terms.patch"
PATCH_5="$PATCH_DIR/0005-feat-add-external-action-physics-replay.patch"
PATCH_6="$PATCH_DIR/0006-feat-collect-physics-state-action-v3.patch"
PATCH_7="$PATCH_DIR/0007-fix-lazy-physics-diagnostics-initialization.patch"
git apply --check "$PATCH_1" "$PATCH_2" "$PATCH_3" "$PATCH_4" "$PATCH_5" "$PATCH_6" "$PATCH_7"
git switch -c codex/minimal-state-action-recorder
git am "$PATCH_1" "$PATCH_2" "$PATCH_3" "$PATCH_4" "$PATCH_5" "$PATCH_6" "$PATCH_7"
```

不要在 Ubuntu 手工编辑补丁内容。如果配置文件已存在，应先检查当前提交和工作树，
不要重复应用补丁。已经应用前三个补丁的执行端只对 `PATCH_4` 分别执行
`git apply --check` 和 `git am`。

已经应用前四个补丁且当前位于 `codex/minimal-state-action-recorder` 的执行端，只需：

```bash
cd ~/bly/sonic-repro/GR00T-WholeBodyControl
git status --short --branch
git rev-parse HEAD
PATCH_5=~/bly/sonic-repro-kit/patches/0005-feat-add-external-action-physics-replay.patch
git apply --check "$PATCH_5"
git am "$PATCH_5"
```

`0005` 只在显式设置 `external_action_replay_path` 时覆盖 eval 的 raw Action；所有既有
SONIC eval、采集与训练行为默认不变。补丁不改 Isaac Lab。

已经应用 `0001`–`0005` 的执行端只应用新增 v3 补丁：

```bash
cd ~/bly/sonic-repro/GR00T-WholeBodyControl
git status --short --branch
git rev-parse HEAD
PATCH_6=~/bly/sonic-repro-kit/patches/0006-feat-collect-physics-state-action-v3.patch
git apply --check "$PATCH_6"
git am "$PATCH_6"
```

`0006` 只增加默认关闭的 Physics v3 recorder 和 4 帧 contact history 配置；旧 collector、
SONIC 策略输入与训练行为不变，也不修改 Isaac Lab。

若首次 v3 smoke 在创建 `RecorderManager` 时报告
`AttributeError: ManagerBasedRLEnv has no attribute action_manager`，说明还缺少初始化顺序
修复。已经应用 `0006` 的执行端继续应用：

```bash
cd ~/bly/sonic-repro/GR00T-WholeBodyControl
PATCH_7=~/bly/sonic-repro-kit/patches/0007-fix-lazy-physics-diagnostics-initialization.patch
git apply --check "$PATCH_7"
git am "$PATCH_7"
```

```bash
./sonic_repro.sh collect-state-action
RUN_DIR="$(cat ~/bly/sonic-repro/state/latest_run_dir.txt)"
test -f "$RUN_DIR/markers/collect_state_action.ok"
test -s "$RUN_DIR/data/sonic_minimal_sa.hdf5"
cat "$RUN_DIR/manifests/collection_summary.json"
```

bundled sample 中只有 2 个动作，因此默认调度自动变成 `2 动作 × 4 变体 = 8 env`，
验收要求精确得到 8 条 `attempt_id=0` canonical episode。无需也不应再把
`COLLECT_ENVS` 手工改成动作数；若显式设置，它必须等于派生乘积。

HDF5 中的 `state_t` 和 `state_tp1` 均为 93 维分字段状态，`goal_t` 为 63 维，
`actions` 为实际送入 Isaac Lab ActionManager 的 29 维 raw action。运行时解析出的
29 关节顺序、默认关节角、action scale/offset/clip、wrapper clip、仿真步长和控制
步长保存在 `manifests/state_action_schema.json`，不依赖 YAML 默认值推断。受 startup
校准随机化影响的默认关节角、offset、material、restitution 与 body COM 会保留逐环境
实际值。HDF5 另外记录 stable global motion ID、variant/batch/attempt ID、全部 runtime
active termination terms（runtime `time_out` 稳定导出为 `motion_time_out`）和
episode 内恒定的 reset delta。主索引只包含 `attempt_id=0`；自动 reset 产生的额外
attempt 保留在独立索引。只有精确覆盖目标 `(motion, variant)` 且全部校验通过才生成
成功 marker。

旧版 recorder 因 runtime 名称为 `time_out` 而可能遗漏 `motion_time_out`。无需重跑或
改写 HDF5；当 schema 明确包含 `time_out` 时，验证器可严格使用同一帧的 `truncated`
恢复该标签，并在 summary 中记录兼容模式及无法恢复的额外 runtime term：

```bash
COLLECT_RUN_DIR=~/bly/runs/collect_state_action_YYYYMMDD_HHMMSS \
COLLECT_MOTION_COUNT=256 \
COLLECT_VARIANTS_PER_MOTION=4 \
COLLECT_VARIANT_OFFSET=0 \
COLLECT_RANDOMIZATION_PROFILE=startup \
bash ./sonic_repro.sh verify-state-action
```

该命令保留原验证 summary/index 的带时间戳副本，不改写 HDF5；通过后才生成 marker。
`context_t/reset_joint_pos_delta` 是最终写入仿真的关节角相对未扰动参考角的差值，包含
soft joint limit 裁剪，因此即使 startup 未启用 reset 随机化也可能非零。验证器要求它
在 episode 内恒定且有限，并在 summary 审计最大绝对值；root pose/velocity 与 joint
velocity 仍按所选随机化 profile 的配置范围严格验收。

## Physics State–Action v3 正式采集

v3 使用新命令 `collect-physics-state-action`，每条 episode 最终只保存
`states[T+1,70]` 与 `actions[T,29]`，并另存 raw/processed Action、控制周期平均力矩、
足端接触冲量、reset delta、root/body replay 状态及按 `context_id` 引用的 runtime
物理参数。它强制平面地形、关闭 camera/RTX 与 eval push；旧 HDF5 保持只读且不会混入。

```bash
cd /home/helloworld/bly/sonic-repro-kit
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

PREP_256=/home/helloworld/bly/runs/bones_seed_ingest_20260823_203201
PREP_512=/home/helloworld/bly/runs/bones_seed_ingest_512_20260823_223831

COLLECT_MOTION_FILE="$PREP_256/data/robot_filtered" \
COLLECT_MOTION_MANIFEST="$PREP_256/manifests/motion_manifest.jsonl" \
COLLECT_MOTION_COUNT=256 COLLECT_BATCH_MOTIONS=8 \
COLLECT_VARIANTS_PER_MOTION=4 COLLECT_VARIANT_OFFSET=0 \
COLLECT_RANDOMIZATION_PROFILE=startup COLLECT_SEED=20260825 \
bash ./sonic_repro.sh collect-physics-state-action
RUN_256_STARTUP="$(cat ~/bly/runs/latest_collect_physics_sa_run_dir.txt)"

COLLECT_MOTION_FILE="$PREP_256/data/robot_filtered" \
COLLECT_MOTION_MANIFEST="$PREP_256/manifests/motion_manifest.jsonl" \
COLLECT_MOTION_COUNT=256 COLLECT_BATCH_MOTIONS=8 \
COLLECT_VARIANTS_PER_MOTION=4 COLLECT_VARIANT_OFFSET=4 \
COLLECT_RANDOMIZATION_PROFILE=initial_state_mild COLLECT_SEED=20260826 \
bash ./sonic_repro.sh collect-physics-state-action
RUN_256_MILD="$(cat ~/bly/runs/latest_collect_physics_sa_run_dir.txt)"

COLLECT_MOTION_FILE="$PREP_512/data/robot_filtered" \
COLLECT_MOTION_MANIFEST="$PREP_512/manifests/motion_manifest.jsonl" \
COLLECT_MOTION_COUNT=512 COLLECT_BATCH_MOTIONS=8 \
COLLECT_VARIANTS_PER_MOTION=4 COLLECT_VARIANT_OFFSET=0 \
COLLECT_RANDOMIZATION_PROFILE=startup COLLECT_SEED=20260827 \
bash ./sonic_repro.sh collect-physics-state-action
RUN_512_STARTUP="$(cat ~/bly/runs/latest_collect_physics_sa_run_dir.txt)"

COLLECT_MOTION_FILE="$PREP_512/data/robot_filtered" \
COLLECT_MOTION_MANIFEST="$PREP_512/manifests/motion_manifest.jsonl" \
COLLECT_MOTION_COUNT=512 COLLECT_BATCH_MOTIONS=8 \
COLLECT_VARIANTS_PER_MOTION=4 COLLECT_VARIANT_OFFSET=4 \
COLLECT_RANDOMIZATION_PROFILE=initial_state_mild COLLECT_SEED=20260828 \
bash ./sonic_repro.sh collect-physics-state-action
RUN_512_MILD="$(cat ~/bly/runs/latest_collect_physics_sa_run_dir.txt)"

printf '%s\n' \
  "RUN_256_STARTUP=$RUN_256_STARTUP" \
  "RUN_256_MILD=$RUN_256_MILD" \
  "RUN_512_STARTUP=$RUN_512_STARTUP" \
  "RUN_512_MILD=$RUN_512_MILD"
```

四个路径必须立即保存，不能依赖后一次运行覆盖后的 latest pointer。成功 run 包含
`data/sonic_physics_sa_v3.hdf5`、schema、canonical/additional-attempt index、summary 和
`collect_physics_state_action.ok`。禁止使用已删除且错误的 `512_bones_seed_insight`；512
动作只允许来自上面的 `bones_seed_ingest_512_20260823_223831`。

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
`OFFLINE_GL` 和 `OFFLINE_CAMERA_DISTANCE`。默认 `OFFLINE_GL=egl`；只有系统已安装
`libOSMesa` 时才应显式使用 `OFFLINE_GL=osmesa`。

渲染脚本不会改动 SONIC 中的 G1 XML。默认使用仓库已有、确实包含 free root +
29 DOF 的 `decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml`；计划中
最初提到的 `sim2mujoco/resources/robots/g1/g1.xml` 实际注释了 waist roll/pitch，只有
34 维 qpos，不能直接承载 36 维轨迹。脚本会在本次 `manifests/` 生成运行时副本，
把机器人 mesh 目录改为绝对路径；若模型包含缺失的 `terrain_mesh`/`terrain_body`，
只从副本移除。外层 recorder 适配器按 SONIC 官方关节名从 dex 模型中筛出 29 个
身体关节，再按官方映射转换为 MuJoCo 顺序；四元数保持 wxyz。

运行时 XML 还会按 `OFFLINE_WIDTH`/`OFFLINE_HEIGHT` 写入匹配的离屏 framebuffer，
因此默认 960×540 以及 1920×1080 等正偶数分辨率不再受 MuJoCo 默认 640×480
framebuffer 限制。渲染场景默认在世界坐标 `z=0` 注入 50×50 m 的深浅灰棋盘地面；
该地面、纹理和材质只存在于本次运行的 `manifests/g1_offline_render_*.xml`，不会修改
SONIC 原始机器人模型。相机仍跟踪 pelvis。

脚本拒绝覆盖已有同名 MP4。若旧 run 已经有视频、但希望复用其中的轨迹验证新地面
或分辨率，请复制轨迹到新的 run；无需重新运行 Isaac Sim：

```bash
SOURCE_RUN=~/bly/runs/offline_render_20260823_123453
TARGET_RUN=~/bly/runs/offline_render_ground_$(date +%Y%m%d_%H%M%S)
mkdir -p "$TARGET_RUN"/{data,logs,videos,manifests,markers}
cp "$SOURCE_RUN"/data/*.trajectory.pkl "$TARGET_RUN/data/"

OFFLINE_RUN_DIR="$TARGET_RUN" OFFLINE_WIDTH=960 OFFLINE_HEIGHT=540 \
  ./sonic_repro.sh render-mujoco
```

## 八、BONES-SEED 256 动作正式采集  

BONES-SEED 是门控数据集，必须先在网页接受许可。token 只能通过 `hf auth login`
进入 Hugging Face 凭据存储，不得写入命令、日志或 manifest。下载前脚本要求至少
70 GiB 可用，下载、转换和采集各阶段结束后要求至少保留 40 GiB：

```bash
cd ~/bly/sonic-repro-kit
./sonic_repro.sh bones-download-preflight
source ~/bly/sonic-repro/.venv-sonic/bin/activate
hf auth login
hf auth whoami

INGEST_RUN=~/bly/runs/bones_seed_ingest_$(date +%Y%m%d_%H%M%S)
mkdir -p "$INGEST_RUN"/{logs,data/source,data/extracted,data/converted,data/robot_filtered,manifests,markers}

hf download bones-studio/seed \
  g1.tar.gz \
  metadata/seed_metadata_v004.parquet \
  metadata/seed_metadata_v004.csv \
  --repo-type dataset \
  --local-dir "$INGEST_RUN/data/source"

BONES_INGEST_RUN="$INGEST_RUN" ./sonic_repro.sh prepare-bones-subset
test -f "$INGEST_RUN/markers/prepare_bones_subset.ok"
cat "$INGEST_RUN/manifests/bones_subset_report.json"
```

CSV metadata 是 146 MB 的无新增依赖回退；4.5 MB parquet 仍同时保留。准备器固定种子
`20260823`，排除镜像、2 秒以下、20 秒以上及 SONIC 官方关键词命中的动作；每个顶层
package 先选择 40 个候选，只从 23.5 GB G1 归档中解压这 320 个 CSV，以 120→30 FPS
转换并再次运行官方过滤器，最终每类严格保留 32 个。任一类别不足即失败，不跨类别
补齐，不覆盖失败 run，也不自动删除归档、Hugging Face 缓存或中间产物。当前 v004
metadata 的 G1 路径列 `move_g1_path` 与旧版/文档中的 `move_g1_mujoco_path` 均受支持；
准备报告会记录本次实际采用的源列名。

第一阶段采集 256 个动作的 4 个 startup 变体：

```bash
PREP_RUN="$INGEST_RUN"
COLLECT_MOTION_FILE="$PREP_RUN/data/robot_filtered" \
COLLECT_MOTION_MANIFEST="$PREP_RUN/manifests/motion_manifest.jsonl" \
COLLECT_MOTION_COUNT=256 \
COLLECT_BATCH_MOTIONS=8 \
COLLECT_VARIANTS_PER_MOTION=4 \
COLLECT_RANDOMIZATION_PROFILE=startup \
COLLECT_SEED=20260823 \
./sonic_repro.sh collect-state-action

STARTUP_RUN="$(cat ~/bly/sonic-repro/state/latest_run_dir.txt)"
test -f "$STARTUP_RUN/markers/collect_state_action.ok"
cat "$STARTUP_RUN/manifests/collection_summary.json"
```

这会使用 32 env 分 32 批完成精确的 1024 条 canonical episode。失败轨迹不会丢弃；
`status=failed` 与具体 termination term 会进入索引。若总体完成率不低于 80%，且八个
package 各自不低于 60%，才允许第二阶段的 2 个半幅初始状态扰动变体：

```bash
COLLECT_MOTION_FILE="$PREP_RUN/data/robot_filtered" \
COLLECT_MOTION_MANIFEST="$PREP_RUN/manifests/motion_manifest.jsonl" \
COLLECT_MOTION_COUNT=256 \
COLLECT_BATCH_MOTIONS=8 \
COLLECT_VARIANTS_PER_MOTION=2 \
COLLECT_VARIANT_OFFSET=4 \
COLLECT_RANDOMIZATION_PROFILE=initial_state_mild \
COLLECT_BASELINE_SUMMARY="$STARTUP_RUN/manifests/collection_summary.json" \
COLLECT_SEED=20260823 \
./sonic_repro.sh collect-state-action
```

第二阶段验收精确增加 512 条 canonical episode。两阶段均使用
`COLLECT_SMPL_MOTION_FILE=zeros`，不会下载完整 SOMA/SMPL 数据，也不会启用随机推力、
camera 或 RTX。BONES-SEED 许可与归属链接会写入 ingest run 的 manifests；数据与衍生
文件仍受其原许可约束。

## 九、CVAE Action Mask 通用物理重放

通常从 `state-action-cvae/cvae_repro.sh validate-action-mask-replay` 统一调用以下三个
内部阶段，无需手工分步执行：

```bash
ACTION_MASK_RUN_DIR=~/bly/runs/cvae_action_mask_eval_YYYYMMDD_HHMMSS \
  bash ./sonic_repro.sh capture-action-mask-source
ACTION_MASK_RUN_DIR=~/bly/runs/cvae_action_mask_eval_YYYYMMDD_HHMMSS \
  bash ./sonic_repro.sh replay-action-mask
ACTION_MASK_RUN_DIR=~/bly/runs/cvae_action_mask_eval_YYYYMMDD_HHMMSS \
  bash ./sonic_repro.sh render-action-mask
```

源采集与重放均为 headless Isaac 物理运行，固定使用 `plane` 地形，并关闭 camera、RTX、
observation corruption、startup/interval 随机化和跟踪失败 reset，只保留 motion timeout。
源采集和批量重放必须显式使用相同平面，不能继承 SONIC release 的随机 `trimesh`；否则
1-env 源运行和多 env 重放会落在不同地形块上，并在 Mask 开始前产生物理分歧。

为避免 GPU PhysX 在不同 env 世界坐标下的接触数值差异被开环回放放大，original 与每个
补全场景默认按清单顺序各启动一次隔离的单环境 Isaac 重放。批量 Action 会先拆成可审计的
`data/replay_action_slices/*.actions.npz`，输出仍以原场景编号写入同一个 `data/replay/`，
因此 CVAE 推理、指标和视频接口不变。该模式比 17-env 并行模式慢，但所有场景使用相同
seed、平面、单环境原点和控制配置，物理对比才有效。每次重放记录绝对 root/joint/body
状态和 runtime Action 映射。recorder 同时输出旧版 64 维字段与 Physics v4 所需的
`physics_state_v3 [T+1,70]`、canonical Action、nominal、
`joint_robot_information [29,11]`、actuator type、global RobotInfo 和 648 维 dynamics
context；关闭随机化时 nominal 明确定义为未扰动 runtime default。渲染阶段再用 MuJoCo
对这些完整物理状态做 50 FPS 公共世界相机可视化，不会把 CVAE 预测 State 当成物理结果。
`ACTION_MASK_GL`、宽高和相机距离仅影响离线视频，不影响 Isaac 轨迹。

## 十、CVAE State Mask 运动学视频

`state-action-cvae`的`validate-state-mask-video`会直接读取已索引的Physics v4 HDF5并生成
`data/state_predictions.npz`。渲染子阶段只进行MuJoCo forward kinematics，不启动Isaac：

```bash
STATE_MASK_RUN_DIR=~/bly/runs/cvae_state_mask_eval_YYYYMMDD_HHMMSS \
  bash ./sonic_repro.sh render-state-mask
```

左、中、右三栏分别为记录轨迹、真值70维State积分重建和预测State积分重建。相机始终
跟随记录轨迹，避免独立相机隐藏预测漂移。该阶段使用独立的`STATE_MASK_*`环境变量、日志、
marker和视频名，不读取或改写Action补全的`completed_actions.npz`与物理重放产物。
