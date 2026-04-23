# Stage2 语义不匹配问题修复说明

## 问题描述

在原始的 Stage2 训练实现中，存在一个严重的语义不匹配问题：

**问题位置**: `train_stage2.py:223-230`

```python
# ❌ 原始实现的问题
task_emb = traj['task_embeddings'][t]
h_emb = traj['hindsight_embeddings'][t]      # 基于 doctor_action 的事后分析
f_emb = traj['foresight_embeddings'][t]      # 基于 doctor_action 的前瞻预测
sem_concat = torch.cat([task_emb, h_emb, f_emb], dim=-1)

# 但是用于训练 better_action！
# ❌ 逻辑错误：better_action 使用了 doctor_action 的语义
```

**核心矛盾**:
- `hindsight_embeddings` 和 `foresight_embeddings` 是基于 `doctor_action` 生成的语义描述
- 但在 Stage2 中，这些语义被用来训练 `better_action`（反事实动作）
- 这两个动作的语义含义应该是不同的！

## 解决方案

采用**方案1：为 better_action 重新生成语义嵌入**（最理论严谨）

### 修改内容

#### 1. 新增语义生成器模块

**文件**: `prompts/semantic_generator.py`

```python
class SemanticGenerator:
    """Generate semantic embeddings for actions at runtime."""
    
    def generate_for_action(
        self,
        prev_state: np.ndarray,
        curr_state: np.ndarray,
        next_acuities: np.ndarray,
        action_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate hindsight and foresight embeddings for a specific action.
        
        Returns:
            hindsight_emb: (1, 896) tensor
            foresight_emb: (1, 896) tensor
        """
```

这个模块复用了 Stage1 数据预处理中的 prompt 生成逻辑：
- `_build_history_text()`: 生成 hindsight 文本（动作 + 状态变化趋势）
- `_build_foresight_text()`: 生成 foresight 文本（风险评估 + 器官保护）

#### 2. 修改 train_stage2.py

**关键修改点**:

1. **导入语义生成器** (第 27-28 行):
```python
from prompts.semantic_generator import SemanticGenerator
from prompts.text_encoder import PromptTextEncoder
```

2. **初始化语义生成器** (第 177-183 行):
```python
# Initialize semantic generator for better_action embeddings
text_encoder = PromptTextEncoder(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device=device,
    max_length=256,
)
semantic_generator = SemanticGenerator(text_encoder, device=device)
```

3. **区分 doctor 和 better 的语义** (第 233-250 行):
```python
# Get task embedding (action-agnostic, can be reused)
task_emb = torch.FloatTensor(
    np.asarray(traj['task_embeddings'][t:t+1])).to(device)

# Get doctor action's semantic embeddings for search
doctor_h_emb = torch.FloatTensor(
    np.asarray(traj['hindsight_embeddings'][t:t+1])).to(device)
doctor_f_emb = torch.FloatTensor(
    np.asarray(traj['foresight_embeddings'][t:t+1])).to(device)

# Search for better action using doctor's semantics
better_action_idx, advantage, search_info = search_better_action(
    policy, world_model, state_t, doctor_action_idx,
    task_emb, doctor_h_emb, doctor_f_emb,  # 使用 doctor 语义进行搜索
    ...
)
```

4. **为 better_action 重新生成语义** (第 252-269 行):
```python
if advantage > 0:
    # ✅ FIX: Regenerate semantic embeddings for better_action
    if t > 0:
        prev_state_np = states_np[t-1]
    else:
        prev_state_np = states_np[t]
    
    curr_state_np = states_np[t]
    next_acuities_np = traj['acuities'][t+1] if t+1 < len(traj['acuities']) else traj['acuities'][t]
    
    # Generate better_action's hindsight and foresight embeddings
    better_h_emb, better_f_emb = semantic_generator.generate_for_action(
        prev_state_np, curr_state_np, next_acuities_np, better_action_idx
    )
    
    # Concatenate with task embedding (action-agnostic)
    sem_concat = torch.cat([task_emb, better_h_emb, better_f_emb], dim=-1)
```

5. **使用 better_action 的语义进行训练** (第 301-303 行):
```python
forward_kwargs = dict(
    ...
    task_embeddings=task_emb.unsqueeze(0),
    hindsight_embeddings=better_h_emb.unsqueeze(0),  # ✅ Use better_action semantics
    foresight_embeddings=better_f_emb.unsqueeze(0),  # ✅ Use better_action semantics
)
```

## 修复效果

### 修复前
- **搜索阶段**: 使用 doctor_action 的语义 ✓
- **训练阶段**: 使用 doctor_action 的语义 ❌（错误！）

### 修复后
- **搜索阶段**: 使用 doctor_action 的语义 ✓
- **训练阶段**: 使用 better_action 的语义 ✓（正确！）

## 理论依据

### 为什么搜索阶段使用 doctor 语义？
在搜索 better_action 时，我们需要评估候选动作相对于 doctor_action 的优势，因此使用 doctor_action 的语义作为基准是合理的。

### 为什么训练阶段必须使用 better 语义？
在训练阶段，我们希望策略学习到：
- **输入**: 当前状态 + better_action 的语义描述
- **输出**: better_action

如果使用 doctor_action 的语义，模型会学习到错误的映射关系，导致语义和动作不匹配。

## 性能影响

### 计算开销
- 每次找到 better_action 后，需要额外生成 2 个语义嵌入（hindsight + foresight）
- 使用 Qwen2.5-0.5B-Instruct 模型，单次生成耗时约 10-20ms
- 相比整体训练时间，开销可忽略不计

### 内存开销
- 语义生成器常驻内存，占用约 1GB（Qwen2.5-0.5B 模型）
- 每个 batch 额外生成的嵌入占用约 7KB（896 * 2 * 4 bytes）

## 使用方法

修复后的训练脚本使用方式不变：

```bash
python train_stage2.py \
    --policy_ckpt checkpoints/stage1/step_200000/policy.pt \
    --world_model_ckpt checkpoints/stage1/step_200000/world_model.pt \
    --train_data data/v3/train_Phys45_v3.pickle \
    --logdir checkpoints/stage2 \
    --epochs 50 \
    --selfplay_iterations 1000
```

## 验证建议

1. **对比实验**: 使用修复前后的代码分别训练，对比最终性能
2. **语义一致性检查**: 在训练过程中随机采样，验证 better_action 的语义描述是否与动作匹配
3. **消融实验**: 对比使用 doctor 语义 vs better 语义的训练效果

## 相关文件

- `prompts/semantic_generator.py` - 新增的语义生成器
- `train_stage2.py` - 修改的训练脚本
- `prompts/saps2_qualitative_prompts.py` - 复用的 prompt 生成逻辑
- `prompts/text_encoder.py` - 复用的文本编码器

## 修复日期

2026-04-23
