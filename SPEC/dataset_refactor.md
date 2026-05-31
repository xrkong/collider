# SPEC: Dataset Multi-Trajectory Refactor

## Goal

把 dataset / dataloader pipeline 从「单 traj、file-path-based config」迁移到「多 traj、dir-based
config」，并且引入 **global normalization stats**（从 train trajs 算一次，train 和 val 共用），
为 2 train + 1 val 的实验做准备。

## Scope

### In scope
- Config schema 迁移：`*_paths` + `metadata_path` → `*_dirs`
- `build_dataloader` 签名改造：drop `split` 参数
- 新增 `_resolve_traj_dir` helper（fail-fast）
- Dataset 类支持多 traj 输入 + 外部 stats 覆盖
- Sliding window **不跨 traj 边界**
- Global stats 计算 + 缓存 + train/val 共用
- Per-traj `metadata.json` 里的 norm 字段在训练 pipeline 中**不再使用**

### Out of scope（不要顺手做）
- AdaLN / conditioning（下个实验再说）
- Element erosion / alive flag 处理
- Test split 引入
- Loss / model / 训练循环逻辑改动
- 自动 glob 目录（所有 dir 必须在 config 显式列出）

---

## Required changes

### 1. YAML config schema

**DELETE 旧 keys：**
- `data.train_paths`
- `data.val_paths`
- `data.metadata_path`

**ADD 新 keys：**
- `data.train_dirs`: `list[str]`，required，至少 1 个
- `data.val_dirs`: `list[str]`，required，至少 1 个
- `train.val_batch_size`: `int`，optional，default 1

每个 dir 内必须存在：
- `output.h5`
- `metadata.json`

示例：
```yaml
data.train_dirs:
  - "/home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_100km_50_5_dt"
  - "/home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_120km_50_5_dt"
data.val_dirs:
  - "/home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_80km_50_5_dt"
train.val_batch_size: 1
```

### 2. Helper `_resolve_traj_dir`

放在 dataset 模块顶层。

```python
from pathlib import Path

def _resolve_traj_dir(d: str | Path) -> dict:
    """Resolve a trajectory dir to {h5, metadata} paths. Fails loudly."""
    d = Path(d)
    if not d.is_dir():
        raise FileNotFoundError(f"Trajectory dir not found: {d}")
    h5, meta = d / "output.h5", d / "metadata.json"
    missing = [p.name for p in (h5, meta) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{d} missing required files: {missing}")
    return {"h5": str(h5), "metadata": str(meta)}
```

### 3. `build_dataloader` 新签名

**Drop `split` 参数。Drop legacy `bvc` 分支整段。** 由调用方显式传 dirs / shuffle / batch_size。

```python
def build_dataloader(
    cfg: dict,
    dirs: list[str] | str,
    *,
    shuffle: bool,
    batch_size: int,
    stats: dict | None = None,
) -> torch.utils.data.DataLoader:
    data_cfg = cfg.get("data", {})

    dataset_type = data_cfg.get("dataset_type", "bvc_sliced")
    dataset_cls  = _DATASET_MAP.get(dataset_type)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")

    if isinstance(dirs, str):
        dirs = [dirs]
    if not dirs:
        raise ValueError("build_dataloader needs at least one trajectory dir")

    trajectories = [_resolve_traj_dir(d) for d in dirs]
    ds_cfg = {**cfg, "data": {
        **data_cfg,
        "paths":          [t["h5"]       for t in trajectories],
        "metadata_paths": [t["metadata"] for t in trajectories],
    }}

    dataset = dataset_cls(ds_cfg, stats=stats)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = data_cfg.get("num_workers", 0),
        pin_memory  = data_cfg.get("pin_memory", True),
    )
```

### 4. 调用端 wiring（train script）

```python
from pathlib import Path

# 1. 防呆：train / val dir 不能重叠
train_set = {str(Path(d).resolve()) for d in cfg["data"]["train_dirs"]}
val_set   = {str(Path(d).resolve()) for d in cfg["data"]["val_dirs"]}
overlap   = train_set & val_set
if overlap:
    raise ValueError(f"Val dirs overlap with train dirs: {overlap}")

# 2. 算 / 读 global stats（仅基于 train trajs）
train_stats = load_or_compute_global_stats(
    train_dirs = cfg["data"]["train_dirs"],
    cache_path = Path(run_output_dir) / "global_stats.json",
    fields     = cfg["data"].get("norm_fields", ["acceleration", "velocity", "position"]),
)

# 3. 构建 loader（train 和 val 共用同一份 stats）
train_loader = build_dataloader(
    cfg, cfg["data"]["train_dirs"],
    shuffle=True,
    batch_size=cfg["train"]["batch_size"],
    stats=train_stats,
)
val_loader = build_dataloader(
    cfg, cfg["data"]["val_dirs"],
    shuffle=False,
    batch_size=cfg["train"].get("val_batch_size", 1),
    stats=train_stats,        # CRITICAL: val 用 train 算的 stats，不用自己的
)
```

### 5. Global stats 模块

新文件 / 新函数 `compute_global_stats`，one-pass Welford。

```python
import h5py, json
import numpy as np
from pathlib import Path

def compute_global_stats(traj_h5_paths: list[str], fields: list[str]) -> dict:
    """One-pass Welford mean/std over (frames × nodes) per field."""
    stats = {f: {"n": 0, "mean": 0.0, "M2": 0.0} for f in fields}
    for p in traj_h5_paths:
        with h5py.File(p, "r") as fh:
            for field in fields:
                x = np.asarray(fh[field][...]).reshape(-1).astype(np.float64)
                n_b = x.size
                if n_b == 0:
                    continue
                mean_b, var_b = float(x.mean()), float(x.var())
                s     = stats[field]
                n_a   = s["n"]
                delta = mean_b - s["mean"]
                n_new = n_a + n_b
                s["mean"] = (n_a * s["mean"] + n_b * mean_b) / n_new
                s["M2"]  += var_b * n_b + delta ** 2 * n_a * n_b / n_new
                s["n"]    = n_new
    return {f: {"mean": s["mean"], "std": (s["M2"] / s["n"]) ** 0.5}
            for f, s in stats.items()}


def load_or_compute_global_stats(
    train_dirs: list[str], cache_path: Path, fields: list[str]
) -> dict:
    """Cache key = sorted resolved train_dirs. Recompute if dirs change."""
    key = sorted(str(Path(d).resolve()) for d in train_dirs)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("key") == key and cached.get("fields") == sorted(fields):
            return cached["stats"]

    h5_paths = [_resolve_traj_dir(d)["h5"] for d in train_dirs]
    stats    = compute_global_stats(h5_paths, fields)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"key": key, "fields": sorted(fields), "stats": stats}, indent=2
    ))
    return stats
```

### 6. Dataset 类（`BVCSlicedDataset` 或现有等价类）

**Constructor 签名变化：**

```python
class BVCSlicedDataset:
    def __init__(self, cfg: dict, *, stats: dict | None = None):
        ...
```

**消费的新 keys：**
- `cfg["data"]["paths"]`: `list[str]`，每条 traj 的 h5 path（已 resolve）
- `cfg["data"]["metadata_paths"]`: `list[str]`，每条 traj 的 metadata path

**Norm 行为：**
- 如果 `stats` 传入了 → **以 stats 为准**，所有 traj 用同一份。
- 如果 `stats` 是 None → fallback 到第一条 traj 的 metadata norm（仅作单 traj 兼容，
  不要在多 traj 场景下走这条路径）。
- **不**再从每条 traj 的 `metadata.json` 各自读 norm 再分别标准化。

**Sliding window 行为（最重要）：**
- 每条 traj **独立**做 sliding window（5 input + 1 prediction，stride 默认 1）。
- **绝对不要**把多条 traj 的 frames concat 起来再 window —— 会产生跨 traj 的非法窗口。
- `__len__` = sum of per-traj window counts。
- `__getitem__(idx)` 把 global idx 映射到 `(traj_idx, local_window_idx)`，再从对应
  traj 取该 window。

**Trajectory 加载日志：**
- 构造时打印：每条 traj 的 dir 名（`Path(h5).parent.name`）+ window count + 该 traj
  对应的 metadata 中的工况标签（speed/mass/material/angle，如有）。

**保留但不使用：**
- metadata.json 里的 norm 字段保留读取（方便 sanity check / debug 比较），但
  **不**进入训练 forward path。

---

## Behavior contracts（必须成立）

1. Sliding window **永不跨 traj 边界**。
2. Val data 用 **train trajs 算出的 stats** 标准化，**绝不**用 val 自己 metadata 里的 norm。
3. `train_dirs` 与 `val_dirs` 不允许有 resolved path 重叠，重叠则 raise。
4. 任何一个 dir 缺 `output.h5` 或 `metadata.json` → `FileNotFoundError(dir 路径)`。
5. Global stats 命中 cache（train_dirs 不变）时直接读，不重算。
6. `build_dataloader` 调用处必须显式传 `shuffle` 和 `batch_size`，没有「is_train 推断」。

---

## Verification checklist

实现完成后逐条验证并报告结果：

- [ ] `build_dataloader` 调用处无 `split` 参数，全部显式传 `shuffle` / `batch_size`
- [ ] Config 用 `train_dirs` / `val_dirs`，全工程 grep `train_paths` / `val_paths` /
      `metadata_path` 应为空
- [ ] 构造 train_loader 后打印：(a) traj 数, (b) 总 window 数, (c) per-traj window 数
- [ ] 打印 train_loader 的一个 batch 的样本来源 traj_idx：**确认同一个 batch 内
      同时出现两条 train traj** —— 这是 window-level shuffle 生效的关键 sanity check
- [ ] 打印 val_loader 同样的信息
- [ ] 打印加载到的 global stats，对照源 metadata.json 里的 per-traj stats，确认
      global ≠ 任何单条 traj 的 stats（说明确实是 aggregate 的）
- [ ] 删掉某个 train_dir 下的 `metadata.json` 重跑 → 必须 raise `FileNotFoundError`
      并指出是哪个 dir
- [ ] 把某个 train_dir 同时塞进 val_dirs 重跑 → 必须 raise overlap 错误
- [ ] 全工程 grep 确认 legacy `bvc` 分支已删除：无 `dataset_type == "bvc"` 检查、
      无 `base_path / split / *_data_000.h5` 路径拼接
- [ ] 第二次跑相同的 train_dirs，确认 `global_stats.json` 被命中（日志打印
      "loaded cached stats" 之类）

---

## Gotchas / 千万别做

- 不要在 `stats` 传入时仍然 fallback 到 per-traj norm —— norm 来源必须**唯一确定**。
- 不要实现跨 traj 的 sliding window。
- 不要在 dataset 输出里加入 condition / 工况标签 channel —— 那是下一个实验。
- 不要改 loss、model、training loop。
- 不要改 `data.input_frames` 语义（仍是 5 input + 1 prediction）。
- 不要 glob 目录自动发现 traj —— 所有 dir 必须 config 显式列出。
- 不要新增 `test_dirs` / test split。
- 不要把多条 traj 的 frames concat 起来再 window，哪怕 "trajectory_id" 一起带着也不行。

---

## Files expected to change

- `dataset.py`（或 `BVCSlicedDataset` 所在文件）
- `build_dataloader.py` / data module 入口
- 训练主脚本（caller wiring + stats 计算 + overlap check + 日志）
- Config YAML（实验目录下对应文件）

新增：
- `compute_global_stats` / `load_or_compute_global_stats` 函数（可放在
  `dataset.py` 或独立 `stats.py`）
- `global_stats.json`（运行时产物，写到 `run_output_dir`）

---

## Out-of-band notes（给 agent 看，不一定写进代码）

- 本次 refactor 之后，metadata.json 里的 norm 字段在训练 path 上**完全 dead**，
  但保留方便对照。下个迭代如果上 AdaLN，会把 metadata 里的工况标签
  （speed/mass/material/angle）作为 condition 输入，norm 不参与。
- Global stats 的 fields list 默认 `["acceleration", "velocity", "position"]`，
  但应该和现有 metadata.json 里的字段保持一致——agent 实现前先 `cat` 一份
  metadata.json 确认字段名拼写。
- 如果 dataset 类的 sliding window 实现已经是 per-traj 的，不要为了"统一"
  改回 concat 风格；如果当前是 concat 风格（很可能是单 traj 时代留下的），
  这次必须改成 per-traj。