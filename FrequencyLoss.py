import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WaveletConv1d(nn.Module):
    """
    使用 Conv1d 模拟单层离散小波变换 (DWT)，确保梯度可导。
    支持 Haar 小波 (默认) 或 db2 小波。
    """
    def __init__(self, in_channels, wavelet_name='haar'):
        super(WaveletConv1d, self).__init__()
        
        # 1. 定义小波滤波器系数
        if wavelet_name == 'haar':
            c = 1 / np.sqrt(2)
            dec_lo = [c, c]      # 低通 (Approximation)
            dec_hi = [-c, c]     # 高通 (Detail)
            k_size = 2
            padding = 0
        elif wavelet_name == 'db2':
            dec_lo = [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025]
            dec_hi = [-0.48296291314469025, 0.836516303737469, -0.22414386804185735, -0.12940952255092145]
            k_size = 4
            padding = 1
        else:
            raise ValueError("Unsupported wavelet type. Try 'haar' or 'db2'.")

        # 2. 构建卷积核权重 [out, in, k]
        dec_lo = torch.tensor(dec_lo, dtype=torch.float32).flip(0).view(1, 1, -1)
        dec_hi = torch.tensor(dec_hi, dtype=torch.float32).flip(0).view(1, 1, -1)

        # 注册为 buffer (不更新这些权重)
        self.register_buffer('filter_lo', dec_lo)
        self.register_buffer('filter_hi', dec_hi)
        
        self.in_channels = in_channels
        self.stride = 2
        self.padding = padding

    def forward(self, x):
        """Input: [Batch, Channel, Length], Output: (Approximation, Detail)"""
        weight_lo = self.filter_lo.repeat(self.in_channels, 1, 1)
        weight_hi = self.filter_hi.repeat(self.in_channels, 1, 1)

        x_approx = F.conv1d(x, weight_lo, stride=self.stride, padding=self.padding, groups=self.in_channels)
        x_detail = F.conv1d(x, weight_hi, stride=self.stride, padding=self.padding, groups=self.in_channels)
        return x_approx, x_detail


class FrequencyRegularizedLoss(nn.Module):
    """
    频域正则化损失函数。
    组合时域 MSE 损失与小波分解的频域正则化损失。
    """
    def __init__(self, reg_lambda=0.01, in_channels=1, wavelet='haar'):
        super().__init__()
        self.base_loss = nn.MSELoss()
        self.reg_lambda = reg_lambda
        self.dwt = WaveletConv1d(in_channels=in_channels, wavelet_name=wavelet)

    def forward(self, y_pred, y_true):
        # 维度调整：确保输入为 [Batch, Channel, Length]
        if y_pred.dim() == 3 and y_pred.shape[-1] == self.dwt.in_channels:
            y_pred_t = y_pred.permute(0, 2, 1)
            y_true_t = y_true.permute(0, 2, 1)
        else:
            y_pred_t = y_pred
            y_true_t = y_true

        # 时域损失
        loss_time = self.base_loss(y_pred, y_true)

        # 频域损失：小波分解
        pred_a, pred_d = self.dwt(y_pred_t)
        true_a, true_d = self.dwt(y_true_t)

        loss_low = F.mse_loss(pred_a, true_a)   # 低频一致性
        loss_high = F.mse_loss(pred_d, true_d)  # 高频分布匹配

        loss_freq = loss_low + loss_high
        
        # 返回总损失、时域损失、频域损失（用于日志记录）
        return loss_time + self.reg_lambda * loss_freq, loss_time, loss_freq
