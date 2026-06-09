# SPEC / Agent Prompt: 可开关的 per-node node_type embedding

## 目标
给 `transolverplus_net` 增加一路 **per-node 类别标签(node_type / 粗化 part label)** 输入,
经 `nn.Embedding` 后 concat 进模型输入。是否启用通过 YAML 单一开关控制。
**默认关闭时,行为必须与当前代码字节级一致**(不破坏现有 single/multi-traj 实验)。

本次只实现 identity(节点属于哪个粗化 part)这一路类别特征。barrier material 作为
独立 per-node 通道是后续步骤,本 SPEC 不实现,但设计不得阻碍后续再 concat 第二路类别特征。

## 改动范围
仅以下 4 处。不要顺手重构其他逻辑、不要改 loss / SDF / push-forward / 归一化:
1. `models/.../transolverplus_net.py`(已提供)
2. `src/dataset.py`(未提供,需自行阅读后改)
3. `train.py`(已提供)
4. 模型 params yaml(`transolverplus_net.yaml`)

---

## 1. 配置(单一真相源)

在 `data` 段新增**唯一开关**:
```yaml
data:
  node_type: false          # 主开关。false = 完全当前行为
  node_type_field: <key>    # 仅当 node_type=true 时使用;h5/metadata 中 per-node int label 的字段名
```
在 `model` 段新增 embedding 超参(仅 node_type=true 时生效):
```yaml
model:
  num_node_types: 16        # 粗化后类别数(vocab size),需 > 数据中最大 id
  type_emb_dim: 8           # embedding 维度
```
**一致性断言**:`train.py` 启动时,若 `data.node_type` 为 true 则要求
`model.num_node_types > 0 and model.type_emb_dim > 0`,否则报错退出;反之 model 配了
embedding 但 `data.node_type` 为 false 也报错。二者必须同开同关。

---

## 2. dataset.py(先读后改)

先阅读 `src/dataset.py` 与一个样本 h5 的结构,搞清楚:
- per-node part / 类别标签存在哪个字段(`/metadata/` 下),dtype、shape、节点顺序。
  YAML 里的 `node_type_field` 用来指定它;不要硬编码字段名。
- **节点下采样 / 子集选择逻辑**:label 必须用与 position/velocity/acceleration **完全相同**的
  节点索引做下采样。这是最关键的正确性点 —— 若现有 dataloader 对节点做了任何
  subsample / reorder,label 必须同步,否则 type 与节点错位,静默产生错误结果。验证对齐。

改动:
- 当 `data.node_type=true` 时,在现有 batch 5-tuple
  `(x_vel, future_acc, input_pos, future_pos, v_last_phys)` **末尾追加第 6 个 tensor**
  `node_type`,shape `(N,)`(单样本)/ `(B, N)`(batch),dtype `torch.long`。
- `node_type` 是**静态**的(跨时间步不变),不参与任何 normalization,不做 asinh,原样 int。
- 当 `data.node_type=false` 时,batch 仍为原来的 5-tuple,长度不变。
- `global_stats` / Welford 归一化逻辑完全不动 —— int label 不进归一化。

---

## 3. transolverplus_net.py

设计:embedding 在 model 内部 concat,**`nnode_in_features` 保持不变**,`input_proj`
内部加宽。这样 yaml 的 `nnode_in_features` 仍只表示物理特征维度,用户无需手算。

`__init__`:
```python
self.num_node_types = m.get("num_node_types", 0)
type_emb_dim        = m.get("type_emb_dim", 0)
if self.num_node_types > 0:
    assert type_emb_dim > 0, "type_emb_dim must be > 0 when num_node_types > 0"
    self.type_embed = nn.Embedding(self.num_node_types, type_emb_dim)
else:
    self.type_embed = None
    type_emb_dim = 0
# input_proj 入维 = nnode_in + type_emb_dim
self._init_network(nnode_in + type_emb_dim, ...)
```
`forward(self, x, node_type=None)`:
- `node_type is None` 且 `self.type_embed is None` → 与当前实现完全一致(回归路径)。
- 否则:`node_type` 支持 `(N,)` / `(B, N)`,`.long()`,过 embedding 得 `(B, N, type_emb_dim)`,
  **concat 到 x 的最后一维**(concat-then-project,不要投影后相加),再过 `input_proj`。
- 若 `self.type_embed is not None` 但 `node_type is None` → 报错(assert)。
- 保持现有 `[N,C]` / `[B,N,C]` 的 squeeze/unsqueeze 逻辑不变。

embedding 用默认初始化即可,不需特殊处理。

---

## 4. train.py

- **batch 解包**:训练主循环、`run_validation`、以及第 380 行附近的
  诊断累加循环(`_, future_acc, _, _, _ = batch`)都按 5-tuple 写死。改成兼容
  可选第 6 元素(例如先判断 `len(batch)`,或统一 `batch[1]` 取 future_acc)。
  三处都要改,诊断循环别漏。
- node_type `.to(device)`,在训练主循环和 `run_validation` 里都传给 `model(x_in, node_type)`。
- **push-forward K 步**:`node_type` 静态,每一步传**同一个** tensor;不随窗口滑动改变,
  不参与 `push_forward_step`。
- 关闭时(5-tuple)调用保持 `model(x_in)`,不传 node_type。

---

## 验收
1. `data.node_type=false`:训练/验证数值与改动前完全一致(同 seed 同 loss)。
   现有 overfit sanity check 仍通过。
2. `data.node_type=true`:模型参数量增加 = `num_node_types * type_emb_dim` +
   `input_proj` 因入维 +`type_emb_dim` 增加的部分;前向无 shape error;
   embedding 的 grad 非零(确认 label 真的接入了计算图)。
3. label 与节点对齐:抽查若干节点,其 node_type 与该节点所属 part 一致(尤其下采样后)。
4. 一致性断言:data/model 开关不匹配时启动即报错。

## 不做(明确排除)
- barrier material 独立通道(下一步)
- 粗/细两层 embedding(下一步)
- 任何 loss / SDF / 归一化 / scheduler / push-forward 物理逻辑的改动
