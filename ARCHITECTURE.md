# FreDEA 白盒审查报告（White-Box Inspection Report）

> 本文档是在执行任何迭代开发之前完成的架构审查，旨在用客观的代码事实证明对系统的理解深度，而非主观的"我懂了"声明。

---

## 1. 模块职责清单（Module Responsibility List）

下表列出代码中**最重要的 3 个函数/类**及其单一职责：

| # | 类 / 函数 | 文件:行号 | 单一职责（一句话） |
|---|---|---|---|
| 1 | `ConditionalFreqDecomp` | `models/FreDEA.py:149–201` | 用可学习的 FFT 软掩码将输入序列分解为趋势分量与季节分量。 |
| 2 | `TEABlock` | `models/FreDEA.py:94–142` | 对季节分量执行时序外部注意力 + 门控通道注意力 + 前馈编码，产出等形状特征。 |
| 3 | `Exp_Main.train()` | `exp/exp_main.py:124–288` | 驱动完整的"前向 → 计算损失 → 反向 → 参数更新"训练循环，并托管早停与学习率调度。 |

---

## 2. 状态扭转机制（State Transition Mechanism）

### 可学习参数（经梯度下降更新）

| 变量 | 文件:行号 | 形状（低维 / 高维） | 更新时机 |
|---|---|---|---|
| `cutoff` | `FreDEA.py:171–175` | `scalar` / `[1,N,1]` | `exp_main.py:252` `loss.backward()` |
| `stepness` | `FreDEA.py:172–176` | `scalar` / `[1,N,1]` | 同上 |
| `fusion_logit` | `FreDEA.py:301` | `scalar` | 同上 |
| `gate_logit`（TEABlock） | `FreDEA.py:115` | `scalar` | 同上 |
| 所有 `nn.Linear` 权重 | 全局 | 各层维度 | Adam step（`exp_main.py:253`） |

### 非参数状态（每 forward 刷新）

| 变量 | 文件:行号 | 含义 |
|---|---|---|
| `RevIN.mean` / `.stdev` | `FreDEA.py:23–24` | 从当前 batch 提取，`detach()` 脱离梯度图 |
| `_moe_aux` | `FreDEA.py:315,358–360` | MoE 辅助损失字典，每次 forward 刷新 |

### 训练过程状态（跨 epoch 持久化）

| 变量 | 文件:行号 | 更新逻辑 |
|---|---|---|
| `EarlyStopping.counter` | `utils/tools.py:36–41` | 在 `exp_main.py:271` 由 `vali_loss` 驱动 |
| 最优模型 checkpoint | `utils/tools.py:61` | 验证损失下降时覆盖写入 `checkpoint.pth` |
| 学习率 | `exp_main.py:283` | `adjust_learning_rate()` 每 epoch 末调用 |

---

## 3. 扩展性评估（Extensibility Assessment）

### 卡点 A — 分解算法硬绑定 FFT（已修复）

**位置**：`FreDEA.py:149–201`（`ConditionalFreqDecomp`）、`FreDEA.py:257`（`Model.__init__`）

**原始问题**：若要将 FFT 分解替换为小波分解（Wavelet）或经验模态分解（EMD），
需直接修改 `ConditionalFreqDecomp` 的内部实现，
且 `Model.__init__` 硬编码了 `self.decomposition = ConditionalFreqDecomp(...)`。

**重构方案**（已实施）：
1. 新增 `models/base_components.py`，定义 `BaseDecomposer` 抽象基类，
   约定 `[B,N,T] → (seasonal, trend)` 的统一接口。
2. `ConditionalFreqDecomp` 继承 `BaseDecomposer`。
3. `Model.__init__` 新增 `decomposer_cls` 参数，支持注入任意 `BaseDecomposer` 子类：
   ```python
   class WaveletDecomp(BaseDecomposer):
       def forward(self, x):  # [B,N,T] -> (seasonal, trend)
           ...

   model = FreDEA.Model(configs, decomposer_cls=WaveletDecomp)
   ```

---

### 卡点 B — 季节编码器绑定 TEA/CEA（已修复）

**位置**：`FreDEA.py:94–142`（`TEABlock`）、`FreDEA.py:265–270`（`tea_blocks` 构建）

**原始问题**：若要用 RNN / CNN / Transformer 替换季节分支，
必须同时修改 `TEABlock` 类与 `Model` 内的构建逻辑（4D 张量约定）。

**重构方案**（已实施）：
1. `models/base_components.py` 定义 `BaseSeasonalProcessor` 抽象基类，
   约定 `[B,N,T,D] → [B,N,T,D]` 的统一接口。
2. `TEABlock` 继承 `BaseSeasonalProcessor`，保持现有行为不变。

---

### 卡点 C — 损失函数 if-elif 链（已修复）

**位置**：`exp/exp_main.py:57–71`（`_select_criterion`）

**原始问题**：新增损失类型须手动在 `_select_criterion` 中添加分支，无扩展点。

**重构方案**（已实施）：
1. 新增 `utils/loss_factory.py`，提供：
   - `_LOSS_REGISTRY` 全局注册表
   - `@register_loss('name')` 装饰器，供外部插件注册自定义损失
   - `build_criterion(name, **kwargs)` 工厂函数
2. `_select_criterion` 改为单行调用 `build_criterion(self.args.loss)`。

扩展示例：
```python
from utils.loss_factory import register_loss
import torch.nn as nn

@register_loss('quantile')
class QuantileLoss(nn.Module):
    def forward(self, pred, target):
        ...

# 在命令行中使用：--loss quantile
```

---

### 卡点 D — 数据集注册表为裸字典（已修复）

**位置**：`data_provider/data_factory.py:4–16`（`data_dict`）

**原始问题**：新增数据类型须修改源文件中的 `data_dict`，无 API 可在外部注册。

**重构方案**（已实施）：
1. `data_factory.py` 新增 `register_dataset(name, dataset_cls)` 函数，
   允许实验脚本在运行时动态注册新数据集：
   ```python
   from data_provider.data_factory import register_dataset

   class MyGraphDataset(torch.utils.data.Dataset):
       def __init__(self, root_path, data_path, flag, size,
                    features, target, timeenc, freq, train_only):
           ...

   register_dataset('my_graph', MyGraphDataset)
   # 之后可直接使用 --data my_graph
   ```

---

## 架构改进总结

| 卡点 | 改动文件 | 改动类型 |
|---|---|---|
| A — 分解算法 | `models/base_components.py`（新建）、`models/FreDEA.py` | 新增接口 + 继承 + 构造参数注入 |
| B — 季节编码 | `models/base_components.py`（新建）、`models/FreDEA.py` | 新增接口 + 继承 |
| C — 损失函数 | `utils/loss_factory.py`（新建）、`exp/exp_main.py` | 新增工厂 + 替换调用 |
| D — 数据注册 | `data_provider/data_factory.py` | 新增注册函数 |

所有改动均向后兼容：现有训练脚本无需任何修改即可继续正常运行。
