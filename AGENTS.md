# SONIC 复现与 State–Action 数据研究：代理交接说明

本文档是本工作区的唯一代理交接入口。后续 Codex 会话开始工作前必须完整阅读本文档；事实、已验证结果与待办不得混淆。

## 1. 总体目标

当前目标不是从零复现 SONIC 的大规模训练，而是利用 SONIC 已有成熟权重，在 Isaac Sim/Isaac Lab 仿真中执行策略并采集时间对齐的 state–action 轨迹。随后以这些轨迹训练轻量生成模型（优先从 CVAE 基线开始），研究以下任务：

- 根据当前或历史 state 预测 action。
- 根据 action 或动作片段补全缺失 state。
- 补全完整轨迹中被遮挡的 state/action 区间。
- 给定过去半段轨迹与动作/语义条件，预测未来 state–action 序列。
- 比较不同 condition、表示方式和缺失模式对生成质量的影响。

第一阶段只要求建立可靠、可复查的数据采集闭环；在数据定义、时序语义和质量检查完成前，不直接扩展到复杂生成模型。

## 2. 强制协作方式

- **Windows 是唯一代码修改端。** 只允许当前工作区中的 Codex 修改源码、脚本和文档。
- **Ubuntu 是执行端。** Ubuntu 只执行拉取、依赖调用、仿真、评测、训练和数据采集；不得用 OpenCode、VS Code 或终端直接修改源码。
- 所有修复都必须先在 Windows 完成并保留 Git 差异，再同步到 Ubuntu 执行。Ubuntu 发生运行错误时，只回传命令、日志、环境状态和堆栈。
- 禁止自动更新 NVIDIA 驱动、CUDA、Isaac Sim、Isaac Lab、PyTorch 或批量修复依赖。驱动必须保持不动，除非用户之后明确授权。
- 不得删除两个源码仓库中的 `.git`。不得对当前工作目录之外的文件执行删除；确需删除时必须先说明绝对路径、原因和影响并获得用户许可。
- 若仓库根目录存在 `.codegraph/`，理解或定位源码时先使用 CodeGraph；不存在时再使用 `rg` 和常规只读检查。

## 3. 两端目录约定

| 用途 | Windows | Ubuntu |
|---|---|---|
| `bly` 主工作区 | `C:\Users\86136\Desktop\code\RL\bly` | `/home/helloworld/bly` |
| SONIC 源码 | `bly/sonic-repro/GR00T-WholeBodyControl` | `~/bly/sonic-repro/GR00T-WholeBodyControl` |
| Isaac Lab 源码 | `bly/sonic-repro/IsaacLab` | `~/bly/sonic-repro/IsaacLab` |
| 复现工具 | `bly/sonic-repro-kit` | `~/bly/sonic-repro-kit` |
| Python 环境 | Windows 不运行 | `~/bly/sonic-repro/.venv-sonic` |
| uv 缓存 | 不纳入同步 | `~/bly/sonic-repro/.uv-cache` |
| 历史状态记录 | `bly/sonic-repro/state` | `~/bly/sonic-repro/state` |
| **全部新实验产物** | `bly/runs` | `~/bly/runs` |

`.venv-sonic` 与 `.uv-cache` 不复制、不提交，由 Ubuntu 本机保留。模型 checkpoint、样例动作和大规模数据是否同步应单独决定，不得误纳入普通源码提交。

## 4. 强制运行产物规范

所有未来 eval、render、smoke-train、正式训练和数据采集均必须写入：

```text
/home/helloworld/bly/runs/<run_id>/
```

每次运行至少应形成以下结构：

```text
runs/<run_id>/
├── logs/                 # 控制台、评测、训练、采集日志
├── videos/               # MP4 或离线渲染结果
├── data/                 # state-action 分片或索引；大文件可改为外部路径清单
├── checkpoints/          # 仅本次新生成的 checkpoint（如有）
├── manifests/            # Git、环境、GPU、配置和数据 schema
└── markers/              # 成功/失败阶段标记
```

要求：

- 不得再在 `GR00T-WholeBodyControl/runs/`、`IsaacLab/runs/` 或源码树其他位置创建运行结果。
- 日志和 output 生成的视频必须位于同一个 `~/bly/runs/<run_id>/` 下，禁止依赖当前工作目录决定输出位置。
- `run_id` 建议采用 `<task>_YYYYMMDD_HHMMSS`；任何阶段复用同一运行目录时必须记录清楚。
- 每次运行必须保存 SONIC commit、IsaacLab commit、完整命令/配置、`uv pip freeze`、`nvidia-smi`、随机种子和退出码。
- `latest_run_dir.txt` 必须更新为新的绝对路径；移动历史目录后旧值可能仍指向 `GR00T-WholeBodyControl/runs`，不得直接信任。
- 普通 Git 是否提交视频和数据是另一项决策。大视频、checkpoint 和轨迹数据不得未经体积检查直接推送；必要时使用 Git LFS、对象存储或独立数据盘。

### 当前路径缺口（尚未修复）

`sonic-repro-kit/sonic_repro.sh` 当前的 `new_run_dir()` 仍使用：

```bash
local dir="$SONIC_DIR/runs/phase1_$(timestamp)"
```

`verify_minimal.py` 也仍假设运行目录位于 `GR00T-WholeBodyControl/runs`。因此另一个会话在再次运行前，必须先在 Windows 修改并同步至少以下内容：

- 增加独立的 `RUNS_ROOT`，默认值为 `${SONIC_RUNS_ROOT:-$HOME/bly/runs}`。
- `new_run_dir()`、日志、视频、marker 和 manifest 全部基于 `RUNS_ROOT`。
- `verify_minimal.py` 接受或推导同一个 runs 根目录，不再要求位于 SONIC 仓库内部。
- README 中所有验收和日志命令改为 `~/bly/runs/...`。
- 写入前检查最终解析路径确实处于 `~/bly/runs` 内。

## 5. 已确认环境与固定版本

Ubuntu 执行机已确认：

| 项目 | 当前值 |
|---|---|
| OS | Ubuntu 24.04.3 LTS |
| Kernel | Linux 7.0.0-28-generic |
| GPU | NVIDIA GeForce RTX 4090 24 GB |
| NVIDIA Driver | 595.84；不得自动变更 |
| `nvidia-smi` CUDA 上限 | 13.2 |
| 系统内存 | 62 GiB，Swap 31 GiB |
| Python / uv | Python 3.11.14，uv 0.9.13 |

复现环境已确认：

| 组件 | 已使用版本/记录 |
|---|---|
| Isaac Sim | 5.1.0.0 |
| PyTorch | 2.7.0+cu128；CUDA 可用 |
| TensorDict | 0.9.1；CUDA smoke test 通过 |
| Isaac Lab 源码 | `37ddf626871758333d6ed89cf64ad702aef127d0` |
| SONIC 已跑通源码 | `c374bae5b9039cd0ee71377e654d11ce1bc69e1d` |
| SONIC checkpoint | `sonic_release/last.pt`，约 448 MB |
| 样例动作 | `sample_data`，约 4.1 MB、6 个 PKL |

`uv pip check` 曾报告 12 项依赖版本不一致，但 Isaac Sim、SONIC 评测和 smoke training 已实际跑通。不要仅为让 `uv pip check` 变绿而升级/降级包；任何依赖调整都必须建立新环境或先完整记录并做回归验证。

## 6. 当前已完成与实验证据

- 系统预检、uv 环境创建、Isaac Sim 5.1 与 Isaac Lab 安装/修复已完成；Git LFS 可用。
- SONIC 固定版本、默认 checkpoint 和 quick-start 样例数据已下载，`check_environment.py --training` 显示 `All checks passed.`。
- Isaac Lab 官方 headless smoke test 已启动成功，日志出现 `[INFO]: Setup complete...`。
- SONIC checkpoint 评测已跑通。有效运行 `runs/phase1_20260818_081830` 的 `eval.ok` 存在，`Success Rate=1.0`、`Progress Rate=1.0`、`mpjpe_g=101.360`、`mpjpe_l=17.569`、`mpjpe_pa=11.389`。
- 5 iteration smoke training 已完成；`smoke_train.ok` 存在，最终记录 `40` episodes、`960` timesteps、mean reward `0.83912`、约 `10.77s`。
- 历史运行已经从 SONIC 子仓库移到工作区根目录 `bly/runs/`；现有四个 `phase1_*` 目录均未包含 MP4。
- 环境安装和历史诊断证据保存在 `sonic-repro/state/`，最新有效运行的评测与训练日志保存在 `runs/phase1_20260818_081830/`。

## 7. 当前未完成与已知问题

- MP4 渲染尚未跑通。`render.log` 显示 Isaac Sim RTX 渲染阶段在 `librtx.scenedb.plugin.so`/Hydra 销毁路径崩溃并产生 dump；当前没有 MP4。
- 用户明确不希望更新 Ubuntu NVIDIA 驱动。后续渲染应研究不改驱动的离线方案，且不能影响已跑通的 headless 评测。
- 尚未实现 state–action 采集器、数据 schema、episode 边界记录、时间对齐验证或数据导出。
- 尚未训练 CVAE；当前只验证了 SONIC 推理与最小训练链路。
- `sonic-repro-kit` 的输出路径仍指向 SONIC 子仓库，必须按第 4 节先修复。
- `sonic-repro/state/latest_run_dir.txt` 是移动前生成的，可能是过期绝对路径。

## 8. Git 现状与同步规则

截至 2026-08-23 Windows 只读检查：

- `bly` 是外层同步仓库，远端为 `https://github.com/Vamperd/bly.git`，当前分支字面值为 `ubantu`（拼写如此，不得擅自更名）。
- 外层仓库已跟踪 `sonic-repro-kit`、`sonic-repro/state` 和 `runs` 中的历史日志；根目录暂时没有 `.gitignore`。
- `sonic-repro/GR00T-WholeBodyControl` 与 `sonic-repro/IsaacLab` 是各自独立的嵌套 Git 仓库，外层仓库把它们显示为未跟踪目录；外层 `git push` **不会同步其中的源码修改**。
- GR00T 本地仓库跟随 NVIDIA 官方远端；检查时本地 `main` 比官方 `origin/main` 落后 2 个提交。用户已人工确认 Windows/Ubuntu 两端工作版本一致，因此不得自动拉取官方最新 main。
- IsaacLab 当前处于 detached HEAD，并显示 9 个零行数/二进制差异，表现像 Windows 拷贝导致的换行或文件元数据差异；在逐文件核验前不得提交、恢复或重置。
- 永远不要删除嵌套仓库的 `.git`。修改嵌套源码前先记录 `git status --short --branch` 与 `git rev-parse HEAD`。

### 推荐同步流程

修改外层脚本/文档时：

1. Codex 只在 Windows 修改并检查 `git diff`。
2. Windows 在外层 `bly` 仓库提交并推送 `ubantu` 分支。
3. Ubuntu 确认工作树干净后，只执行 `git pull --ff-only origin ubantu`。
4. Ubuntu 执行并把新的日志/产物写到 `~/bly/runs/<run_id>`。

修改嵌套 SONIC/IsaacLab 源码时，不能假设外层 Git 会携带修改。由于当前不使用 Fork，优先把嵌套仓库的提交导出为 patch 或 Git bundle，存入外层仓库后由 Ubuntu `git am`/`git fetch` 应用；应用前必须用 `git apply --check` 或在临时分支验证。不得在 Ubuntu 手工重复修改。

## 9. 下一阶段建议顺序

1. 先修复 `sonic-repro-kit` 的统一 `~/bly/runs` 输出路径，并更新验证器与 README。
2. 在 SONIC eval rollout 中定位策略 observation 构造、action 输出和 simulation step 的唯一对齐点。
3. 定义数据 schema：原始/归一化 joint position、joint velocity、root pose/velocity、command/目标、policy action、实际执行 action、next state、termination、motion id 与时间戳。
4. 实现只记录不改变策略行为的数据采集钩子，采用分片写入和原子完成 marker，避免长轨迹占满内存。
5. 用少量 episode 验证 shape、单位、关节顺序、action scale、时序偏移、NaN、episode 边界和可重放性。
6. 建立简单监督基线后，再分别实现 action prediction、masked completion 和 future prediction 的 CVAE；不同任务必须使用不同 condition 与 mask 定义。
7. 最后再扩展样本规模、离线渲染和更复杂的 latent sequence model。

## 10. 后续代理的启动检查

开始任何改动前必须执行并报告：

```bash
# Windows 外层仓库
git status --short --branch

# Ubuntu 执行前
cd ~/bly
git status --short --branch
git rev-parse HEAD
git -C sonic-repro/GR00T-WholeBodyControl status --short --branch
git -C sonic-repro/GR00T-WholeBodyControl rev-parse HEAD
git -C sonic-repro/IsaacLab status --short --branch
git -C sonic-repro/IsaacLab rev-parse HEAD
nvidia-smi
```

不得因为本文件记录了旧提交就自动 checkout、reset 或 pull；先以两端实际状态为准，并保护用户已有修改和运行结果。

