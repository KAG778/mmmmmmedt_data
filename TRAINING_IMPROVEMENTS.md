# CSP-DT 训练流程改进说明

## 改进内容

### 1. Stage1 训练改进

**添加的功能：**
- ✅ **Validation loss 跟踪**：每个 epoch 结束后在验证集上评估模型
- ✅ **Best checkpoint 保存**：自动保存验证集上表现最好的 checkpoint
- ✅ **Metadata 记录**：保存 best checkpoint 的详细信息（epoch、loss、global_step）

**修改的文件：**
- `优化主要模型/scheme3_cspdt_v2/train_stage1.py`

**新增的输出：**
```
checkpoints/stage1/
├── best_checkpoint/          # 新增：最好的 checkpoint
│   ├── policy.pt
│   ├── world_model.pt
│   └── metadata.json         # 新增：checkpoint 元数据
├── epoch_10/
├── epoch_20/
...
└── epoch_100/
```

**训练日志示例：**
```
[Stage1] Epoch 50/100 complete (global_step=12500)
[Stage1] Validation: pi=0.8234, O=15.2341, total=16.0575
[Stage1] ✓ New best checkpoint! Epoch 50, val_loss=16.0575
```

### 2. Stage2 训练改进

**修改的功能：**
- ✅ **默认使用 best_checkpoint**：Stage2 默认加载 `checkpoints/stage1/best_checkpoint/` 而不是固定的 `epoch_100`
- ✅ **更清晰的参数说明**：添加了 help 信息

**修改的文件：**
- `优化主要模型/scheme3_cspdt_v2/train_stage2.py`

**新的默认参数：**
```python
--policy_ckpt checkpoints/stage1/best_checkpoint/policy.pt
--world_model_ckpt checkpoints/stage1/best_checkpoint/world_model.pt
```

### 3. 自动化训练脚本

**新增脚本：**
- `restart_cspdt_training.sh`：自动化 Stage1 → Stage2 训练流程

**功能：**
- ✅ 自动备份旧的 checkpoints
- ✅ 清理旧的日志
- ✅ 顺序执行 Stage1 → Stage2
- ✅ 自动检查 best_checkpoint 是否存在
- ✅ 显示训练进度和日志位置

## 使用方法

### 方法 1: 使用自动化脚本（推荐）

```bash
cd /home/wangmeiyi/AuctionNet/medical/重构之后的代码
bash restart_cspdt_training.sh
```

### 方法 2: 手动分步训练

**Step 1: 训练 Stage1**
```bash
cd /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2

python train_stage1.py \
    --epochs 100 \
    --save_interval_epochs 10 \
    --log_interval_steps 100 \
    --logdir checkpoints/stage1
```

**Step 2: 检查 best_checkpoint**
```bash
# 查看 best checkpoint 信息
cat checkpoints/stage1/best_checkpoint/metadata.json

# 示例输出：
# {
#   "epoch": 85,
#   "val_loss": 15.8234,
#   "val_pi_loss": 0.7891,
#   "val_O_loss": 15.0343,
#   "global_step": 21250
# }
```

**Step 3: 训练 Stage2（使用 best_checkpoint）**
```bash
python train_stage2.py \
    --policy_ckpt checkpoints/stage1/best_checkpoint/policy.pt \
    --world_model_ckpt checkpoints/stage1/best_checkpoint/world_model.pt \
    --epochs 50 \
    --selfplay_iterations 1000 \
    --logdir checkpoints/stage2
```

## 监控训练进度

```bash
# 查看 Stage1 训练日志
tail -f /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/logs/stage1.log

# 查看 Stage2 训练日志
tail -f /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/logs/stage2_main.log
```

## 验证改进效果

### 检查 Stage1 是否保存了 best_checkpoint

```bash
ls -lh /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1/best_checkpoint/

# 应该看到：
# policy.pt
# world_model.pt
# metadata.json
```

### 检查 Stage2 是否使用了 best_checkpoint

```bash
grep "Loaded stage1 checkpoints" /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/logs/stage2_main.log

# 应该看到：
# Loaded stage1 checkpoints from:
#   checkpoints/stage1/best_checkpoint/policy.pt
#   checkpoints/stage1/best_checkpoint/world_model.pt
```

## 与旧流程的对比

| 特性 | 旧流程 | 新流程 |
|------|--------|--------|
| Stage1 validation | ❌ 无 | ✅ 每个 epoch 评估 |
| Best checkpoint | ❌ 无，只保存固定间隔 | ✅ 自动保存最好的 |
| Stage2 checkpoint 选择 | ❌ 固定使用 epoch_100 | ✅ 使用 best_checkpoint |
| Checkpoint 元数据 | ❌ 无 | ✅ 保存详细信息 |
| 训练自动化 | ❌ 手动分步 | ✅ 一键启动 |

## 预期效果

1. **更好的模型性能**：Stage2 使用验证集上表现最好的 Stage1 checkpoint，而不是最后一个
2. **更清晰的训练过程**：可以看到每个 epoch 的验证 loss，了解模型收敛情况
3. **更方便的实验管理**：best_checkpoint 的 metadata.json 记录了详细信息，便于追踪

## 注意事项

1. **训练时间**：Stage1 现在会在每个 epoch 结束后运行 validation，会增加一些训练时间
2. **磁盘空间**：best_checkpoint 会额外占用一份 checkpoint 的空间
3. **兼容性**：如果需要使用特定的 epoch checkpoint，可以通过命令行参数指定：
   ```bash
   python train_stage2.py \
       --policy_ckpt checkpoints/stage1/epoch_80/policy.pt \
       --world_model_ckpt checkpoints/stage1/epoch_80/world_model.pt
   ```
