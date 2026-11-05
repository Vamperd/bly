# LEI Group 服务器使用

本文记录本项目使用的远端服务器连接、文件存储和任务提交方式。

## 1. SSH 连接

```bash
ssh bailinying@10.130.140.103
```

当前服务器主机名为 `leis01`。登录后的 Home 目录为：

```text
/home/bailinying
```

项目当前位于远端数据盘：

```text
/data1/users/bailinying/FlexSeqWBC/bly
```

进入项目：

```bash
cd /data1/users/bailinying/FlexSeqWBC/bly
```

## 2. 文件存储

服务器上的 `~/data` 是符号链接，目标为：

```text
/data1/users/bailinying
```

按用途存储：

- Home：代码、小文件和 Conda/venv 环境。
- `~/data`：数据集、模型、checkpoint 和训练输出。
- `~/data/FlexSeqWBC/bly/runs`：本项目实验运行记录。
- `/data1/shared/datasets`：公共数据集，只读使用。
- `/data1/shared/bailinying`：需要与同盘用户协作的共享文件。

不要把大型数据集、模型或训练输出长期放在 Home。不要删除他人文件或修改系统环境。

检查个人配额和磁盘空间：

```bash
quota -s
df -h
du -sh ~/data
du -sh /data1/shared/bailinying
```

## 3. GPU 任务

先激活环境并进入项目：

```bash
conda activate <environment>
cd /data1/users/bailinying/FlexSeqWBC/bly
```

### 3.1 提交训练

默认申请 1 张 GPU：

```bash
gpu python train.py
```

申请 N 张 GPU：

```bash
gpu N python train.py
```

例如申请 2 张 GPU，并设置 CPU、内存、时长和任务名：

```bash
gpu 2 -c 16 -m 128G -t 8h -n baseline python -u train.py
```

长参数写法：

```bash
gpu --cpus 16 --mem 96G --time 2h --name debug python train.py
```

GPU 程序必须自行支持多卡；`gpu N` 不会自动修改单卡代码。PyTorch DDP 示例：

```bash
gpu N python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=N \
  train.py
```

GPU 任务默认资源为 1 张 GPU、8 个逻辑 CPU、48 GiB RAM、4 小时。CPU 和 RAM 是整个任务的总量，不会按 GPU 数量自动倍增。GPU、CPU 和 RAM 必须同时有可用资源，任务才会启动。

### 3.2 查看日志和停止任务

```bash
gpu ps                  # 活动任务及历史
gpu ps -0               # 仅活动任务
gpu ps -10 --json       # 活动任务及最近 10 条历史，JSON 格式
gpu logs <ID>           # 持续查看日志；Ctrl-C 不停止任务
gpu stop <ID>           # 取消排队或结束运行任务
```

提交后返回任务编号不代表已经开始运行；只有状态为 `RUNNING` 才表示任务已启动。排队时不要重复提交。

### 3.3 交互式会话

申请交互式 GPU 会话：

```bash
gpu shell
```

申请 2 张 GPU：

```bash
gpu shell 2
```

会话中可运行训练、调试器或后台进程：

```bash
python train.py > train.log 2>&1 &
python -m pdb debug.py
```

暂时离开但保持任务运行：按 `Ctrl-b`，松开后按 `d`。重新连接：

```bash
gpu ps
gpu attach <ID>
```

不用时释放资源：

```bash
gpu stop <ID>
```

同一 `gpu shell` 会话中的进程共享一次 GPU、CPU 和 RAM 申请；再次执行 `gpu python ...` 会创建新的独立任务。

### 3.4 低利用率回收

GPU 任务启动后有 15 分钟宽限期。宽限期结束后，如果所有已分配 GPU 的计算利用率连续低于 5% 达 30 分钟，任务会被自动回收。判断依据是计算利用率，不是显存占用；只把模型留在显存中也可能被回收。

因此应当：

- 预处理和数据准备尽量在申请 GPU 前完成；
- 调试长时间暂停时及时停止或退出会话；
- 定期将 checkpoint 写入 `~/data`；
- 在任务到期前保存结果并主动结束任务。

## 4. CPU 任务

CPU 预处理不应占用 GPU。默认申请 4 个逻辑 CPU、8 GiB RAM、4 小时：

```bash
cpu python preprocess.py
```

指定资源：

```bash
cpu -c 8 -m 32G -t 8h -n "dataset v2" python preprocess.py
```

常用命令：

```bash
cpu ps
cpu logs <ID>
cpu attach <ID>
cpu stop <ID>
```

CPU 会话的 tmux 分离、重连和资源释放方式与 GPU 会话相同。
