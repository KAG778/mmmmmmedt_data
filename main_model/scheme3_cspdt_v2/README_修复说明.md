# CSP-DT WorldModel 修复 & 训练指南

## 1. 问题描述

Stage 1 训练 `g_hat` 预测的是 **reward** (`rtg[t] - rtg[t+1] ≈ 0`)，而 Stage 2 计算 `advantage = g_doc (RTG) - g_hat (reward)`，导致单位不匹配，筛选机制失效。

## 2. 修复内容

**文件**: `train_stage1_coldstart.py` 第 132-133 行

```python
# 修改前（错误）
g_next = (rtgs[:, :-1, 0] - rtgs[:, 1:, 0]).reshape(-1, 1)  # 预测 reward

# 修改后（正确）
g_next = rtgs[:, 1:, 0].reshape(-1, 1)  # 预测 rtg[t+1]
```

修复后 Stage 2 的 advantage 计算：
```
advantage = g_doc - g_hat = rtg[t] - rtg[t+1] = reward  ✓
```

## 3. 修复验证

测试训练输出：
```
[Stage1] step 100/5000  pi=1.4097  O=1.5054
```
- `O=1.5054` 是预测 **rtg[t+1]** 的 MSE loss（正确量级）
- 原版预测 reward 时 O loss ≈ 0.1

## 4. 当前卡点：GPU 设备节点缺失

硬件正常（4x A100-PCIE-40GB），驱动已安装（nvidia-driver-580），但 `/dev/nvidia*` 设备节点不存在。

需要用 **sudo** 执行以下命令：

```bash
sudo nvidia-smi
```

如果 nvidia-smi 报错，手动创建设备节点：

```bash
sudo mknod -m 666 /dev/nvidia0 c 195 0
sudo mknod -m 666 /dev/nvidia1 c 195 1
sudo mknod -m 666 /dev/nvidia2 c 195 2
sudo mknod -m 666 /dev/nvidia3 c 195 3
sudo mknod -m 666 /dev/nvidiactl c 195 255
sudo mknod -m 666 /dev/nvidia-uvm c 243 0
sudo mknod -m 666 /dev/nvidia-uvm-tools c 243 1
```

验证 GPU 可用：

```bash
nvidia-smi  # 应显示 4 张 A100
python3 -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

## 5. 环境准备

```bash
cd /home/wangmeiyi/AuctionNet/medical

# 激活 Python 虚拟环境（已创建好的 Linux venv）
source .venv_linux/bin/activate

# 验证 CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"
# 期望输出: CUDA: True Devices: 4
```

如果 `.venv_linux` 的 torch 没有 CUDA 支持，需要安装 CUDA 版本的 torch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 6. 训练命令

### Stage 1: 冷启动训练（预计 2-3 小时）

```bash
cd /home/wangmeiyi/AuctionNet/medical
source .venv_linux/bin/activate

python exp3.15/scheme3_cspdt_v2/train_stage1_coldstart.py \
    --datadir /home/wangmeiyi/AuctionNet/medical/sepsis_mimiciii_2/data/phys45 \
    --logdir exp3.15/scheme3_cspdt_v2/checkpoints/stage1 \
    --max_steps 200000 \
    --save_interval 30000 \
    --log_interval 500
```

**预期输出**：
```
[Stage1] Starting step-based training: max_steps=200000, save_interval=30000
[Stage1] step 500/200000  pi=x.xxxx  O=x.xxxx
[Stage1] step 1000/200000  pi=x.xxxx  O=x.xxxx
...
[Stage1] Saved checkpoint at step 30000 -> .../checkpoints/stage1/step_30000
...
[Stage1] Training complete.
```

**检查点保存位置**：
```
exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_30000/{policy.pt, world_model.pt}
exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_60000/{policy.pt, world_model.pt}
exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_90000/{policy.pt, world_model.pt}
exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_120000/{policy.pt, world_model.pt}
exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_150000/{policy.pt, world_model.pt}
exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_180000/{policy.pt, world_model.pt}
exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_200000/{policy.pt, world_model.pt}
```

### Stage 2: 自博弈训练（预计 2-3 小时）

```bash
python exp3.15/scheme3_cspdt_v2/train_stage2_selfplay.py \
    --datadir /home/wangmeiyi/AuctionNet/medical/sepsis_mimiciii_2/data/phys45 \
    --logdir exp3.15/scheme3_cspdt_v2/checkpoints/stage2 \
    --policy_ckpt exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_200000/policy.pt \
    --world_model_ckpt exp3.15/scheme3_cspdt_v2/checkpoints/stage1/step_200000/world_model.pt \
    --max_steps 200000 \
    --save_interval 30000
```

### Stage 3: 评估（预计 30 分钟）

```bash
python exp3.15/scheme3_cspdt_v2/evaluate.py \
    --datadir /home/wangmeiyi/AuctionNet/medical/sepsis_mimiciii_2/data/phys45 \
    --policy_ckpt exp3.15/scheme3_cspdt_v2/checkpoints/stage2/step_200000/policy.pt \
    --world_model_ckpt exp3.15/scheme3_cspdt_v2/checkpoints/stage2/step_200000/world_model.pt \
    --output_json exp3.15/reports/raw_results/scheme3_v2_metrics.json
```

## 7. 一键运行脚本

也可以用 nohup 后台运行，避免断开 SSH 后训练中断：

```bash
cd /home/wangmeiyi/AuctionNet/medical
source .venv_linux/bin/activate

# Stage 1
nohup python -u exp3.15/scheme3_cspdt_v2/train_stage1_coldstart.py \
    --datadir /home/wangmeiyi/AuctionNet/medical/sepsis_mimiciii_2/data/phys45 \
    --logdir exp3.15/scheme3_cspdt_v2/checkpoints/stage1 \
    --max_steps 200000 --save_interval 30000 --log_interval 500 \
    > exp3.15/scheme3_cspdt_v2/stage1.log 2>&1 &

# 查看进度
tail -f exp3.15/scheme3_cspdt_v2/stage1.log
```

## 8. 预期效果对比

| 指标 | 原版 scheme3 | 修复后 (预期) |
|------|-------------|--------------|
| 高危 SAPS2 Δ | -6.90 | -7.5 ~ -8.5 |
| 中危 SAPS2 Δ | +0.47 | -0.5 ~ +1.0 |
| 全部 SAPS2 Δ | -0.19 | -1.5 ~ -3.0 |
| TF-Acc | 97.0% | 96% ~ 98% |

## 9. 文件结构

```
exp3.15/scheme3_cspdt_v2/
├── train_stage1_coldstart.py   # ✅ 已修复（第132行）
├── train_stage2_selfplay.py    # ✅ 无需修改
├── evaluate.py                  # ✅ 可用
├── config.py                    # ✅ 配置文件
├── models/
│   ├── policy.py
│   ├── world_model.py          # ✅ 无需修改
│   ├── GPT.py
│   └── MeDT.py
└── datasets/
    └── mimic_dataset.py

exp3.15/scheme3_cspdt_backup/   # 原始代码备份（未修改）
```
