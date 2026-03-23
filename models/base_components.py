"""
Abstract base classes for swappable FreDEA model components.

白盒审查报告（White-Box Inspection Report）
============================================

1. 模块职责清单（Module Responsibility List）
   ─────────────────────────────────────────
   • ConditionalFreqDecomp（models/FreDEA.py:149-201）
     职责：利用 FFT 将输入序列自适应地分解为趋势分量与季节分量，
     通过可学习的 cutoff / stepness 参数控制分解边界。

   • TEABlock（models/FreDEA.py:94-142）
     职责：对季节分量执行"时序外部注意力 + 通道外部注意力 + 前馈"
     三阶段特征编码，产出与输入等形状的高级特征表示。

   • Exp_Main.train()（exp/exp_main.py:124-288）
     职责：驱动完整的"前向传播 → 计算损失 → 反向传播 → 参数更新"
     训练循环，并托管 EarlyStopping 与学习率调度。

2. 状态扭转机制（State Transition Mechanism）
   ─────────────────────────────────────────
   • cutoff / stepness（FreDEA.py:171-176）
     在每次 forward 中生成频率 mask（FreDEA.py:187-191），
     通过 loss.backward()（exp_main.py:252）经梯度下降持续调整。

   • gate_logit（FreDEA.py:115）、fusion_logit（FreDEA.py:301）
     同样为可学习 nn.Parameter，由同一反向传播流程更新。

   • RevIN mean / stdev（FreDEA.py:23-24）
     每次 forward 时从当前 batch 提取（detach），不进入梯度图。

   • EarlyStopping.counter（utils/tools.py:36-41）
     在 exp_main.py:271 被 vali_loss 驱动，决定是否终止训练。

3. 扩展性评估（Extensibility Assessment）
   ─────────────────────────────────────────
   卡点 A — 分解算法耦合 FFT（FreDEA.py:149-201）
     若需换成小波分解（Wavelet）或经验模态分解（EMD），需直接修改
     ConditionalFreqDecomp 内部逻辑，且 Model.__init__ 硬编码了
     self.decomposition = ConditionalFreqDecomp(...)（FreDEA.py:251）。
     重构方向：引入 BaseDecomposer 接口，通过构造参数注入具体实现。

   卡点 B — 季节编码器绑定 TEA/CEA（FreDEA.py:94-142, 265-270）
     若需用 RNN / Transformer / CNN 替换季节分支，必须同时修改
     TEABlock 类和 Model 内的构建逻辑（包括 4D 张量约定）。
     重构方向：引入 BaseSeasonalProcessor 接口，统一输入/输出约定。

   卡点 C — 损失函数 if-elif 链（exp_main.py:59-70）
     新增损失类型须手动在 _select_criterion 中添加分支，无扩展点。
     重构方向：使用工厂函数 + 注册表（utils/loss_factory.py）。

   卡点 D — 数据集注册表为裸字典（data_provider/data_factory.py:4-16）
     新增数据类型须修改源文件中的 data_dict，无 API 可在外部注册。
     重构方向：提供 register_dataset() 函数。

本模块（base_components.py）解决卡点 A 和 B，提供可供继承和替换的
BaseDecomposer 与 BaseSeasonalProcessor 抽象基类。
"""

import torch.nn as nn
from abc import ABC, abstractmethod


class BaseDecomposer(nn.Module, ABC):
    """时序分解模块的抽象基类。

    子类实现不同的分解策略（FFT、小波、移动平均、EMD 等），
    同时共享统一接口，使 Model 可通过构造参数注入任意分解器。

    构造函数约定
    -------------
    子类的 ``__init__`` 必须接受以下两个位置参数，以便 Model 统一实例化：

    .. code-block:: python

        def __init__(self, seq_len: int, enc_in: int):
            super().__init__()
            ...

    - ``seq_len`` : 输入序列长度
    - ``enc_in``  : 输入通道数

    输入/输出约定
    -------------
    输入 : x  形状 [B, N, T]  —— Batch × 通道数 × 时间步
    输出 : (x_seasonal, x_trend)，两者形状均为 [B, N, T]
    """

    @abstractmethod
    def forward(self, x):
        """将 x 分解为 (seasonal, trend) 分量。

        Parameters
        ----------
        x : torch.Tensor, shape [B, N, T]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (x_seasonal, x_trend)，每项形状均为 [B, N, T]
        """
        ...


class BaseSeasonalProcessor(nn.Module, ABC):
    """季节分支编码器的抽象基类。

    子类实现不同的序列编码策略（TEA、RNN、CNN、Transformer 等），
    同时共享统一接口，使 Model 可通过构造参数注入任意编码器。

    输入/输出约定
    -------------
    输入 : x  形状 [B, N, T, D]  —— Token 嵌入后的季节信号
    输出 : x  形状 [B, N, T, D]  —— 编码后的特征，形状不变
    """

    @abstractmethod
    def forward(self, x):
        """对输入特征进行编码，返回相同形状的输出。

        Parameters
        ----------
        x : torch.Tensor, shape [B, N, T, D]

        Returns
        -------
        torch.Tensor, shape [B, N, T, D]
        """
        ...
