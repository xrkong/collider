# Spec: 重构 W&B 日志结构（train / rollout 关联与分组）

## 目标

把现在「每个 run 各自孤立」的结构,改成可以清晰追溯血缘的三层结构:

1. **同一次训练实验的所有 run 归入一个 `group`**
2. **用 `job_type` 区分 run 角色**(`train` / `rollout`)
3. **checkpoint 作为 W&B Artifact** 上传,rollout 通过 `use_artifact` 消费,从而自动建立 train → checkpoint → rollout 的血缘图(lineage)
4. **单个 rollout run 内的多条轨迹用 `wandb.Table` 汇总**,而不是每条轨迹再开一个 run

不要改动模型、训练逻辑、数据管线;只改 W&B 相关的初始化、日志、artifact 代码。

---

## 现状(待确认 / 待修改)

- 训练脚本和 `rollout.py` 各自调用 `wandb.init()`,run 之间没有任何关联字段。
- checkpoint 只存在本地磁盘;rollout 时直接从磁盘路径读取,W&B 上无血缘记录。
- GIF 直接 `log_artifact` 或单独上传,多条轨迹的结果散落在不同地方,难以对比。

> agent 执行前先全局搜索 `wandb.init`、`log_artifact`、`wandb.Video`、`wandb.log` 的调用点,列出清单后再改。

---

## 改动一:统一 group 名的传递

`group` 名必须让 train 和 rollout **两个脚本拿到同一个值**,否则关联失效。

要求实现如下机制(二选一,优先 A):

- **A(推荐)**:训练启动时生成一个 group 名(格式 `{tag}{number}`,如 `dg005`),写入该次实验的 checkpoint 目录下的 `meta.json`。`rollout.py` 从对应 checkpoint 目录读取 `meta.json` 拿到同一个 group 名。
`meta.json` 至少包含:

```json
{ "group": "dg005", "project": "barrier-vehicle-collision" } 
```

参考之前的命名规则，尽量保持一致的 tag(`dg` 等)和递增的 number,但不要求绝对连续(如跳过 `dg004` 直接到 `dg005` 没问题)。

---

## 改动二:训练脚本

### init

```python
run = wandb.init(
    project=meta["project"],
    group=meta["group"],
    job_type="train",
    name="train",
    config=cfg,  # 保留现有 config
)
```

### 每次保存 checkpoint 时,额外 log 成 artifact
比如
```python
art = wandb.Artifact(
    meta["group"],
    type="model",
    metadata={
        "epoch": 160,
        "step": 30400,
        "val_loss": 0.11319845554075743,
        "experiment": meta["group"]
    },
)
art.add_file(ckpt_path)              # e.g. ckpt/epoch_500.pt
run.log_artifact(art, aliases=[f"epoch_{epoch}"])
```

要求:
- artifact 名固定为group名如“dg003”，下面的不同版本命名规则参考outputs/checkpoints/dg003/下的结构。尽量保留原来逻辑。
- 你只测best,那不必把每个 checkpoint 都传(省存储)。
- metadata 里带上关键超参,方便后续在 UI 过滤。
- 不要删除本地保存逻辑;artifact 是额外动作。

---
## 改动三:rollout 脚本

### 前提:artifact 命名约定
每个rollout run都强制指定 checkpoint， experiment和raw-h5，这些都要写入 `meta.json` 供 rollout 脚本读取。

artifact **按 experiment 命名**:`checkpoint-{exp}`(如 `checkpoint-dg003`)。每个 experiment 内 train 出的多个 checkpoint 是该 artifact 的多个**版本**,最优 checkpoint(val_loss 最低)打 `best` alias。

每个 rollout run 只测一个 model,但要在**多个 test set** 上测,因此一个 model 会产出多个 GIF。

### init(同一个 group,job_type 改为 rollout)

```python
meta = load_meta(ckpt_dir)  # 读取改动一里的 meta.json;exp = meta["experiment"]
run = wandb.init(
    project=meta["project"],
    group=meta["group"],         # ★ 必须与 train 一致
    job_type="rollout",
    name=f"rollout_{exp}_best",
    config={"experiment": exp, "checkpoint": "best"},
)
```

### 用 use_artifact 消费 best checkpoint(建立血缘)

```python
ckpt = run.use_artifact(f"checkpoint-{exp}:best")
ckpt_path = ckpt.download()
model.load_state_dict(torch.load(f"{ckpt_path}/<ckpt文件名>.pt"))
```

要求:
- rollout 必须通过 `use_artifact` 加载 checkpoint(而非直接读本地磁盘),否则 W&B 不会画出血缘。如本地已有缓存,`download()` 不会重复下载。
- 始终引用 `:best`;不要硬编码具体 epoch。

### 多 test set × 多轨迹用一张 Table 汇总(不要每个 test set 或每条轨迹开 run)

一个 best model = 一个 rollout run。run 内对所有 test set、所有轨迹遍历,结果进同一张 Table,用 `test_set` 列作维度:
test_set, traj_id, mode(ar/os), pos_rmse(mm), vel_rmse(mm/dt), acc_rmse(mm/dt²), weight(kg), speed(km/h), angle(deg), concrete_type(defult "N"), 

table = wandb.Table(columns=[test_set, traj_id, mode(ar/os), pos_rmse(mm), vel_rmse(mm/dt), acc_rmse(mm/dt²), weight(kg), speed(km/h), angle(deg), concrete_type(defult "N"), ])

每个轨迹rollout都要考虑 always predict 0 情况下的指标，这个数值只需要计算一次。

```python
for ts in test_sets:                         # 多个 test set
    for tid in ts.trajectory_ids:            # 每个 test set 内的轨迹
        gif_path, metrics = rollout_one(model, ts, tid)   # 复用现有 rollout 逻辑
        table.add_data(
            ts.name,
            tid,
            "ar",  # or "os" depending on the mode
            metrics["pos_rmse"],
            metrics["vel_rmse"],
            metrics["acc_rmse"],
            metrics["weight"],
            metrics["speed"],
            metrics["angle"],
            metrics["concrete_type"] # 这个设置成固定值 "N" 就行,因为现在都是同一种混凝土;如果后续加了不同类型的混凝土再改这里
        )

run.log({"rollout_results": table})

# 按 test set 聚合的标量写 summary,方便跨 experiment 横向对比
for ts in test_sets:
    run.summary[f"mean_mse/{ts.name}"] = mean_mse_per_set[ts.name]
run.summary["mean_mse_overall"] = mean_mse_overall
```

要求:
- 一个 best model = 一个 rollout run;**所有 test set、所有轨迹进同一张 Table**,用 `test_set` 列区分(UI 里可按 `test_set` 过滤、按 MSE 排序)。
- GIF 通过 `wandb.Video` 进 Table,**不要**再单独 `log_artifact` 每个 GIF。
- 聚合指标写 `run.summary`:每个 test set 一个 `mean_mse/{set}` 字段 + 一个总体字段,字段名跨 run 保持一致(便于在 runs 表里按 `config.experiment` 横向比较不同 experiment 的 best model)。

---

## 验收标准

1. 训练后,W&B 项目中存在 `type="model"` 的 artifact `bvc-model`,带多个 epoch alias。
2. 运行多次 rollout 后,在 W&B UI:
   - 同一实验的 train + 所有 rollout run 在同一个 `group` 下;
   - 按 `group` → `job_type` 分组后呈现清晰两层结构;
   - 点开任一 checkpoint artifact,lineage 图能看到产出它的 train run 和消费它的所有 rollout run。
3. 每个 rollout run 里,`rollout_results` Table 可查看所有轨迹的 GIF + 指标,可按 MSE 排序。
4. 各 rollout run 的 `summary` 含统一命名的聚合指标,可在 runs 表里直接横向比较不同 checkpoint。
5. 训练 / rollout / 数据管线的非 W&B 逻辑无行为变化。

---

## 注意事项 / 边界

- 不要引入 `wandb.init(reinit=True)` 之类全局副作用;保持每个脚本单 run。
- 若 `wandb` 处于 disabled / offline 模式(如调试),所有新增逻辑要能安全跳过,不报错。
- group 名一旦写入 `meta.json` 不要在 rollout 阶段重新生成。
- 改动后在 README 或脚本注释里补一句:rollout 依赖 train 阶段产生的 `meta.json` 与 artifact。
