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

脚本默认把环境、代码和日志放在 `~/bly/sonic-repro`。它不会删除已有环境、仓库、数据集或运行结果。

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
./sonic_repro.sh render
./sonic_repro.sh smoke-train
./sonic_repro.sh verify-minimal
```

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

最小闭环的必要证据如下：

| 证据 | 验收条件 |
|---|---|
| `metrics.log` | 正常退出，无 OOM、NaN、缺失 checkpoint 或动作文件 |
| `renders/*.mp4` | 至少一个可播放 MP4 |
| `train_smoke.log` | 完成 5 次迭代并输出 reward/error/FPS |
| `nvidia-smi.txt` | 记录运行时 GPU/驱动状态 |
| `environment_freeze.txt` | 保存完整 Python 依赖版本 |
| 两个 commit 文件 | 与本页固定提交一致 |
| `verification_report.json` | 顶层 `passed` 为 `true` |

若任一步失败，请回传以下命令输出，不需要复制整个环境：

```bash
cd ~/bly/sonic-repro-kit
./sonic_repro.sh status

RUN_DIR=$(cat ~/bly/sonic-repro/state/latest_run_dir.txt 2>/dev/null || true)
echo "$RUN_DIR"
test -n "$RUN_DIR" && tail -n 150 "$RUN_DIR/metrics.log" 2>/dev/null
test -n "$RUN_DIR" && tail -n 150 "$RUN_DIR/render.log" 2>/dev/null
test -n "$RUN_DIR" && tail -n 150 "$RUN_DIR/train_smoke.log" 2>/dev/null
```

## 七、完整数据阶段的门禁

在取得 BONES-SEED 许可并准备至少约 300GB 可用空间前，不执行完整数据下载、转换和评测。当前 99% 满的数据盘不满足条件。

完整数据阶段包括：下载完整 SMPL、下载 BONES-SEED G1 CSV、转换为 motion library、过滤动作、用官方 checkpoint 跑完整评测，最后才考虑单卡小规模微调。单张 4090 不等价于论文建议的 64+ GPU 全量训练配置。
