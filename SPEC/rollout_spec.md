# SPEC: `rollout.py` minimal refactor for multi-traj training

## Context

`src/rollout.py` 已经实现完整：`load_raw_h5` / `normalize_raw` / `run_onestep` /
`run_autoregressive` / `compute_baseline` / GIF 渲染 / 多实验对比图 / RMSE 曲线，
forward Euler 积分 (`integrate_accel`) 和 SDF (`compute_sdf_batch`) 都在 onestep 和 AR
两个 path 上一致使用。**不要重写这些。**

唯一**必须**变的是 **norm 来源** —— 训练改 global stats 后，rollout 不能再用
`NormStats(cfg["data"]["metadata_path"])` 读单条 traj 的 metadata 算 norm。
本 SPEC 描述这一个改动，外加一些不能回退的不变量。

## Goal

让 `rollout.py` 消费训练时缓存的 global stats（dataset SPEC 里写到
`global_stats.json` 的那一份），其它逻辑零改动。

## Scope

### In scope
- `NormStats` 构造方式从「读 per-traj metadata.json」改为「读 train run 的
  `global_stats.json`」
- 新增 CLI 参数 `--stats-path`（或等价的实验 config 字段）
- 调整 `main()` 里 `norm_stats = NormStats(...)` 那一行的来源

### Out of scope（不要碰）
- `run_onestep` / `run_autoregressive` 循环体
- `integrate_accel`（forward Euler，`dt=1`）—— 已和训练 convention 对齐，不改
- `compute_sdf_batch`（SDF 计算 + 内部 `/1000` 归一化）—— onestep / AR 共用，不改
- `compute_baseline` / `compute_frozen_baseline`（constant-velocity + frozen baselines
  已经实现，比 SPEC 推荐的还完整，保留）
- GIF / pkl / 多实验对比图 / RMSE 曲线
- CLI mode 划分（`onestep` / `autoregressive` / `both` / `raw_gt`）
- 单 traj-per-run 的 CLI 形态（`--raw-h5` 一次跑一条，**不要**改成多 traj 循环）

---

## Required change：stats 来源迁移

### 当前代码（`main()` 内）

```python
norm_stats = NormStats(cfg["data"]["metadata_path"])  # 单 traj 的 metadata
normed     = normalize_raw(raw_data, norm_stats)
```

`cfg["data"]["metadata_path"]` 在 dataset refactor 之后已经从 config 里删掉了，
这一行**必然报错**。

### 改动方案（两选一，agent 任选简单的实现）

**方案 A：CLI 显式传入（推荐，行为最透明）**

```python
parser.add_argument("--stats-path", default=None,
                    help="Path to training-run global_stats.json. "
                         "Defaults to <checkpoint_dir>/global_stats.json.")
```

加载时：

```python
from pathlib import Path

stats_path = Path(args.stats_path) if args.stats_path else (
    Path(args.checkpoint).parent / "global_stats.json"
)
if not stats_path.is_file():
    raise FileNotFoundError(
        f"Global stats not found: {stats_path}. "
        f"Pass --stats-path explicitly or place global_stats.json next to checkpoint."
    )
norm_stats = NormStats.from_global_stats(stats_path)
normed     = normalize_raw(raw_data, norm_stats)
```

**方案 B：实验 config 携带路径**

在 experiment yaml 里加 `data.stats_path`，rollout 读它。改动更小但耦合到 config，
重跑旧 checkpoint 时需要手工补字段。

**推荐方案 A**，因为这条 CLI 经常需要灵活指向不同 run 的 stats（比如比较两个
checkpoint 的输出），CLI 比 config 灵活。

### `NormStats.from_global_stats` 实现要求

`src/dataset.py` 里的 `NormStats` 类需要新增一个 classmethod（或顶层 factory）：

```python
@classmethod
def from_global_stats(cls, stats_path: str | Path) -> "NormStats":
    """Build NormStats from training-run global_stats.json.

    The file schema is the one written by load_or_compute_global_stats():
        {"key": [...], "fields": [...], "stats": {"velocity": {"mean":..., "std":...}, ...}}
    """
    ...
```

实现要点：

- 读取 `stats` 子字段，把每个 feature 的 `mean` / `std` 灌进 NormStats 内部存储，
  保持和原 `__init__` 走 metadata.json path **同样的内部表示**（`normalize` /
  `denormalize` 方法不变）。
- `_acc_scale` 字段：保留现有行为 —— 如果训练时用了 asinh transformation（看
  experiment cfg 的 `data.acc_scale`），rollout 也要拿到这个值。建议
  `from_global_stats` 多接一个可选 `acc_scale` 参数，由 caller 从 cfg 传入：

```python
norm_stats = NormStats.from_global_stats(
    stats_path,
    acc_scale=cfg["data"].get("acc_scale"),
)
```

这样 `_acc_scale` 的入口和旧版 `NormStats(metadata_path, acc_scale)` 注释里的
那段保持一致行为，rollout 的 asinh 分支（`if norm_stats._acc_scale is not None`）
继续生效，不需要改。

---

## Behavior contracts（必须成立）

1. **Stats 来自训练 run**：`global_stats.json` 必须存在，路径错或文件不存在 → raise，
   不要 silent fallback 到 metadata.json。
2. **`acc_scale` 不丢**：如果 experiment cfg 里有 `data.acc_scale`，rollout 加载到的
   `norm_stats._acc_scale` 必须等于这个值。否则 asinh 分支的 RMSE 报告会
   误算。
3. **onestep 和 AR 用同一 `norm_stats`**：当前已经是这样，改动后仍然如此。不要
   per-mode 各自构造。
4. **SDF normalization 不变**：`compute_sdf_batch` 内部的 `/ 1000.0` 保留；onestep
   和 AR 都通过同一个函数计算 SDF。
5. **CLI 单 traj 形态保留**：`--raw-h5` 一次一条，多条 traj = 多次调用脚本。

---

## Verification checklist

- [ ] `cfg["data"]["metadata_path"]` 在 `rollout.py` 中**不再**被读取（grep 确认）
- [ ] 不传 `--stats-path` 且 checkpoint dir 下没有 `global_stats.json` → 报
      `FileNotFoundError` 且消息里出现期望路径
- [ ] 传一个不存在的 `--stats-path` → 同样 `FileNotFoundError`
- [ ] 训练完一个 sc_03X 实验后，直接跑：
      ```
      python src/rollout.py \
          --checkpoint outputs/checkpoints/sc_03X/checkpoint-best.safetensors \
          --experiment configs/experiments/sc_03X.yaml \
          --raw-h5 /path/to/train_traj/output.h5 \
          --mode both
      ```
      可以正常跑通且 onestep pos_rmse **显著优于** baseline（这是已见过的 traj，
      验证 norm + integration pipeline 没坏）
- [ ] 换 `--raw-h5` 指向 held-out val traj 重跑，AR pos_rmse 至少**不差于**
      `rollout baseline`（constant-velocity）
- [ ] 打印加载到的 stats —— `norm_stats` 的 mean/std 值应该和 `global_stats.json`
      文件内容字面一致，而不是任何单条 traj 的 metadata.json 里那些数
- [ ] `--mode raw_gt` 仍然能跑（这个 mode 不依赖 model 也不依赖 stats，所以
      改动后必须仍然完整 work，作为最低保障的 smoke test）
- [ ] `print_summary` 里 GT acc / Pred acc 的 |mean| 同量级（不出现训练时
      诊断到过的 229g vs 1g 这种 collapse）

---

## 不要顺手做（明确 out of scope）

- **不要把 `--raw-h5` 改成 `--rollout-dirs` 列表**。单 traj-per-run 的形态对人工
  diagnostic 更友好，多 traj 评估写一个外层 shell 脚本循环调用即可。
- **不要把 pkl 输出改成 h5**。现有 `plot_multi_rmse` / `--compare-dirs` 全部依赖
  pkl，改了要连带改一堆下游。
- **不要参数化 `compute_sdf_batch` 里的 `barrier_angle_deg=-25.4` 和
  `barrier_anchor=(0, 2000)`**。这是 known limitation —— 跨不同 angle 工况
  时 SDF 会算错。但在当前 2 train + 1 val 都用同一 angle 的实验里不影响，
  留到真正扩到 angle 维度时再处理（届时 SDF 参数应从该条 traj 的
  metadata.json 读，而不是从全局 global_stats 读，因为它是几何属性不是 norm 属性）。
- **不要改 `integrate_accel`**。`dt = 1` 是与训练 forward-difference convention
  锁死的，任何 dt 改动都会让 rollout 和训练 mismatch。
- **不要删掉 `normalize_raw` 的 `acceleration` / `velocity` / `positions` 三件套**。
  即使 model 当前只消费 velocity，把三件全 normalize 留着方便 debug。

---

## Files expected to change

- `src/rollout.py` —— 只改 `main()` 里构造 `norm_stats` 那块（约 5–10 行）
  和新增一个 `--stats-path` argparse 项
- `src/dataset.py` —— 给 `NormStats` 加一个 `from_global_stats(...)` classmethod
  （约 10–15 行）

预计净改动量 **< 30 行**。

---

## Out-of-band note

在 dataset SPEC 里，训练 pipeline 会把 `global_stats.json` 写到训练 run 的输出
目录。本 SPEC 假设这个文件**和 checkpoint 在同一目录**（或父目录），所以默认值
是 `Path(checkpoint).parent / "global_stats.json"`。如果训练脚本实际把它写到
别处（比如 `outputs/runs/<exp>/global_stats.json` 而 checkpoint 在
`outputs/checkpoints/<exp>/`），让 agent 在写 default 之前先 grep 训练脚本里
`global_stats.json` 实际落盘位置，把默认路径对齐，否则用户每次都要手动
传 `--stats-path` 才能跑通。