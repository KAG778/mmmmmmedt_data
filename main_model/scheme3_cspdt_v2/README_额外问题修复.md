# Stage2 额外问题修复说明

## 问题1: 数据归一化不一致

### 问题描述
在 `train_stage2.py:260-261` 中，传递给 `semantic_generator.generate_for_action()` 的状态数据存在归一化不一致问题：

```python
curr_state_np = states_np[t]  # 归一化后的状态
next_acuities_np = traj['acuities'][t+1]  # 原始 acuities
```

`_build_foresight_text()` 函数期望接收**原始的未归一化状态**，因为它需要使用临床阈值来分类指标（如 HR < 60 为心动过缓）。

### 影响
- 归一化后的状态值范围与临床阈值不匹配
- 导致生成的 foresight 文本分类错误（如正常心率被误判为异常）

### 修复方案

需要在 Stage2 训练时保留原始的未归一化状态数据。

**选项1**: 在数据集中同时保存归一化和原始状态
**选项2**: 在 Stage2 训练时动态反归一化（需要 normalization params）
**选项3**: 修改 `_build_foresight_text()` 使用归一化后的阈值

**推荐**: 选项2 - 动态反归一化

```python
# 在 train_stage2.py 初始化时加载归一化参数
norm_path = args.train_data.replace('.pickle', '_norm.pkl')
with open(norm_path, 'rb') as f:
    norm_params = pickle.load(f)
state_mean = norm_params['state_mean']
state_std = norm_params['state_std']

# 在生成语义时反归一化
curr_state_raw = curr_state_np * state_std + state_mean
prev_state_raw = prev_state_np * state_std + state_mean

better_h_emb, better_f_emb = semantic_generator.generate_for_action(
    prev_state_raw, curr_state_raw, next_acuities_np, better_action_idx
)
```

---

## 问题2: acuities 数据时间对齐错误

### 问题描述
在 `datasets/v3_semantic_dataset.py:144-150` 中，`acuities` 数据被前移了一步：

```python
# Shift acuities forward by one step
traj["acuities"] = np.concatenate(
    (traj["acuities"][1:, :],
     np.reshape(traj["acuities"][-1, :], (1, acuity_dim))),
    axis=0,
)
```

这意味着 `traj['acuities'][t]` 实际上是原始的 `acuities[t+1]`。

但在 `train_stage2.py:261` 中：

```python
next_acuities_np = traj['acuities'][t+1] if t+1 < len(traj['acuities']) else traj['acuities'][t]
```

这实际上访问的是 `t+2` 时刻的 acuities，而不是 `t+1`。

### 影响
- foresight 文本使用了错误时刻的 acuities 数据
- 预测的风险评估与实际状态不匹配

### 修复方案

```python
# 修复: 使用 traj['acuities'][t] 而不是 traj['acuities'][t+1]
# 因为数据集中的 acuities 已经前移了一步
next_acuities_np = traj['acuities'][t]
```

---

## 问题3: t=0 时的 prev_state 逻辑不当

### 问题描述
在 `train_stage2.py:255-258` 中：

```python
if t > 0:
    prev_state_np = states_np[t-1]
else:
    prev_state_np = states_np[t]  # Use current state for t=0
```

当 `t=0` 时，使用 `curr_state` 作为 `prev_state` 会导致：
- `_build_history_text()` 计算的所有状态变化 delta 都为 0
- 生成的 hindsight 文本不准确（如 "HR stable", "BP stable" 等）

### 影响
- t=0 时刻的 hindsight 语义不准确
- 可能影响模型对初始状态的理解

### 修复方案

**选项1**: 对 t=0 使用特殊的初始化文本（类似 Stage1 数据生成）

```python
if t == 0:
    # 使用初始化文本，不计算状态变化
    hindsight_text = build_hindsight_v7_first(curr_state_np)
    better_h_emb = encoder.encode_texts([hindsight_text], batch_size=1)
else:
    # 正常生成 hindsight
    better_h_emb, better_f_emb = semantic_generator.generate_for_action(...)
```

**选项2**: 跳过 t=0 的训练

```python
# 在采样时跳过 t=0
t = np.random.randint(1, T - 1)  # 从 1 开始而不是 0
```

**推荐**: 选项2 - 跳过 t=0，因为：
- t=0 没有 prev_state，无法计算有意义的 hindsight
- Stage1 已经在 t=0 上训练过了

---

## 修复优先级

1. **高优先级**: 问题2 (acuities 时间对齐) - 直接影响语义正确性
2. **中优先级**: 问题1 (数据归一化) - 影响语义质量
3. **低优先级**: 问题3 (t=0 逻辑) - 影响较小，可以通过跳过 t=0 简单解决

---

## 验证建议

修复后，建议进行以下验证：

1. **语义一致性检查**: 随机采样几个 better_action，打印生成的 hindsight/foresight 文本，人工检查是否合理
2. **数值范围检查**: 验证传递给 `_build_foresight_text()` 的状态值是否在合理的临床范围内
3. **时间对齐检查**: 验证 acuities 数据是否与对应时刻的状态匹配

---

## 修复日期
2026-04-23
