"""
损失函数工厂模块（Loss Function Factory）。

解决白盒审查中识别的卡点 C：
  exp/exp_main.py._select_criterion 原本是一条僵硬的 if-elif 链，
  每增加新损失类型都必须修改源文件，缺乏扩展点。

本模块提供：
  • 内置损失的注册表（_LOSS_REGISTRY）
  • @register_loss 装饰器 —— 供外部插件注册自定义损失函数
  • build_criterion() 工厂函数 —— 统一的损失构建入口

用法示例
--------
  # 1. 使用内置损失
  criterion = build_criterion('mse')
  criterion = build_criterion('huber', delta=0.5)

  # 2. 注册并使用自定义损失
  @register_loss('my_loss')
  class MyLoss(nn.Module):
      def forward(self, pred, target):
          return ...

  criterion = build_criterion('my_loss')
"""

import functools
import torch.nn as nn

# ---------------------------------------------------------------------------
# 内部注册表：name -> callable（无参或接受 **kwargs）
# ---------------------------------------------------------------------------
_LOSS_REGISTRY: dict = {}


def register_loss(name: str):
    """装饰器：将自定义损失类注册到工厂注册表中。

    Parameters
    ----------
    name : str
        在 build_criterion() 中使用的损失名称（不区分大小写）。

    Returns
    -------
    decorator : callable
        原样返回被装饰的类，不修改其行为。

    Examples
    --------
    >>> @register_loss('my_loss')
    ... class MyLoss(nn.Module):
    ...     def forward(self, pred, target):
    ...         return (pred - target).abs().mean()
    """
    def decorator(cls):
        _LOSS_REGISTRY[name.lower()] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# 注册内置 PyTorch 损失（统一入口，无需特殊分支）
# ---------------------------------------------------------------------------
_LOSS_REGISTRY['mse'] = nn.MSELoss
_LOSS_REGISTRY['mae'] = nn.L1Loss
_LOSS_REGISTRY['smooth_l1'] = nn.SmoothL1Loss
# HuberLoss 统一注册为接受 delta 关键字的偏函数包装，与其他损失保持一致
_LOSS_REGISTRY['huber'] = functools.partial(nn.HuberLoss, delta=1.0)


def build_criterion(loss_name: str, **kwargs) -> nn.Module:
    """根据名称构建损失函数实例。

    内置支持：'mse'、'mae'、'huber'、'smooth_l1'。
    通过 @register_loss 注册的自定义损失也可直接使用。

    Parameters
    ----------
    loss_name : str
        损失函数名称（不区分大小写），例如 'mse'、'huber'。
    **kwargs
        透传给损失构造函数的关键字参数。
        例如 build_criterion('huber', delta=0.5) 会将 delta=0.5 传给
        nn.HuberLoss。

    Returns
    -------
    nn.Module
        已实例化的损失函数模块。

    Raises
    ------
    ValueError
        当 loss_name 不在注册表中时抛出，并列出所有已知名称供用户参考。

    Examples
    --------
    >>> criterion = build_criterion('mse')
    >>> criterion = build_criterion('huber', delta=0.5)
    >>> criterion = build_criterion('mae')
    """
    key = loss_name.lower()

    if key in _LOSS_REGISTRY:
        return _LOSS_REGISTRY[key](**kwargs)

    known = sorted(_LOSS_REGISTRY.keys())
    raise ValueError(
        f"未知损失函数 '{loss_name}'。已注册的损失：{known}。\n"
        "可通过 @register_loss('name') 装饰器注册自定义损失函数。"
    )
