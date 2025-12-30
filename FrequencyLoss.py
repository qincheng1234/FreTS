import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WaveletConv1d(nn.Module):
    """
    使用 Conv1d 模拟单层离散小波变换 (DWT)，确保梯度可导。
    支持 Haar, db2, db4 小波。
    """
    def __init__(self, in_channels, wavelet_name='haar'):
        super(WaveletConv1d, self).__init__()
        
        # 定义小波滤波器系数
        if wavelet_name == 'haar':
            c = 1 / np.sqrt(2)
            dec_lo = [c, c]      # 低通 (Approximation)
            dec_hi = [-c, c]     # 高通 (Detail)
            padding = 0
        elif wavelet_name == 'db2':
            dec_lo = [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025]
            dec_hi = [-0.48296291314469025, 0.836516303737469, -0.22414386804185735, -0.12940952255092145]
            padding = 1
        elif wavelet_name == 'db4':
            # db4 小波滤波器系数 (更长的支撑长度，更好的平滑性)
            dec_lo = [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114,
                      -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523]
            dec_hi = [-0.23037781330885523, 0.7148465705525415, -0.6308807679295904, -0.02798376941698385,
                      0.18703481171888114, 0.030841381835986965, -0.032883011666982945, -0.010597401784997278]
            padding = 3
        else:
            raise ValueError("Unsupported wavelet type. Try 'haar', 'db2', or 'db4'.")

        # 构建卷积核权重 [out, in, k]
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
        # 确保滤波器权重与输入在同一设备上
        weight_lo = self.filter_lo.to(x.device).repeat(self.in_channels, 1, 1)
        weight_hi = self.filter_hi.to(x.device).repeat(self.in_channels, 1, 1)

        x_approx = F.conv1d(x, weight_lo, stride=self.stride, padding=self.padding, groups=self.in_channels)
        x_detail = F.conv1d(x, weight_hi, stride=self.stride, padding=self.padding, groups=self.in_channels)
        return x_approx, x_detail


class MultiLevelWaveletDecompose(nn.Module):
    """
    多层小波分解模块
    返回: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    """
    def __init__(self, in_channels, wavelet_name='db4', levels=3):
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels
        # 为每一层创建独立的小波变换（共享相同的滤波器）
        self.dwt = WaveletConv1d(in_channels=in_channels, wavelet_name=wavelet_name)
    
    def forward(self, x):
        """
        Input: [Batch, Channel, Length]
        Output: dict with keys 'cA3', 'cD3', 'cD2', 'cD1' (for levels=3)
        """
        details = []
        approx = x
        
        for level in range(self.levels):
            approx, detail = self.dwt(approx)
            details.append(detail)  # cD_1, cD_2, cD_3 ...
        
        # 返回: cA_n (最终近似) 和 各层细节 cD_1 到 cD_n
        result = {'cA': approx}
        for i, d in enumerate(details):
            result[f'cD{i+1}'] = d  # cD1 是最高频, cD3 是中频
        
        return result


class FrequencyRegularizedLoss(nn.Module):
    """
    频域正则化损失函数 (v4.0 - 可学习小波稀疏策略)
    
    核心改进：
    1. 对每个频带 (cD1, cD2, cD3) 使用可学习的软阈值
    2. cD1 (极高频/噪声): 初始阈值较大，倾向于强过滤
    3. cD2 (次高频/细节): 初始阈值中等
    4. cD3 (中频/模式): 初始阈值较小，保留更多信息
    5. 稀疏损失：鼓励过滤后的高频系数趋近于零
    """
    def __init__(self, reg_lambda=0.01, in_channels=1, wavelet='db4', levels=3):
        super().__init__()
        self.base_loss = nn.MSELoss()
        self.reg_lambda = reg_lambda
        self.levels = levels
        self.wavelet = wavelet
        
        # 多层小波分解
        self.mwd = MultiLevelWaveletDecompose(in_channels=in_channels, wavelet_name=wavelet, levels=levels)
        
        # === 可学习的软阈值（替代模型内部的 softshrink）===
        # 不同频带使用不同的初始阈值
        self.threshold_cD1 = nn.Parameter(torch.tensor(0.05))  # 极高频 - 较大阈值
        self.threshold_cD2 = nn.Parameter(torch.tensor(0.02))  # 次高频 - 中等阈值
        self.threshold_cD3 = nn.Parameter(torch.tensor(0.01))  # 中频 - 较小阈值
        
        # 各层频带的损失权重
        self.detail_weights = {
            'cD1': 1.0,   # 极高频 - 强约束
            'cD2': 0.5,   # 次高频 - 中等约束
            'cD3': 0.1,   # 中频 - 弱约束
        }
        
        # Initial Scale Matching
        self.register_buffer('scale_factor', torch.tensor(1.0))
        self.scale_initialized = False
    
    def soft_threshold(self, x, threshold):
        """
        可学习的软阈值函数 (与 FreTS.py 中注释掉的函数一致)
        softshrink(x, λ) = sign(x) * max(|x| - λ, 0)
        """
        # 确保阈值非负
        threshold = F.relu(threshold)
        # 自定义 softshrink
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)

    def forward(self, y_pred, y_true):
        # 维度调整：确保输入为 [Batch, Channel, Length]
        if y_pred.dim() == 3 and y_pred.shape[-1] == self.mwd.in_channels:
            y_pred_t = y_pred.permute(0, 2, 1)
            y_true_t = y_true.permute(0, 2, 1)
        else:
            y_pred_t = y_pred
            y_true_t = y_true

        # 时域损失
        loss_time = self.base_loss(y_pred, y_true)

        # 多层小波分解
        pred_coeffs = self.mwd(y_pred_t)
        true_coeffs = self.mwd(y_true_t)
        
        # 获取各层阈值
        thresholds = {
            'cD1': self.threshold_cD1,
            'cD2': self.threshold_cD2,
            'cD3': self.threshold_cD3,
        }

        # === 可学习稀疏策略 ===
        # 对预测值的高频系数施加软阈值，然后与真实值比较
        loss_freq = torch.tensor(0.0, device=y_pred.device)
        
        for level in range(1, self.levels + 1):
            key = f'cD{level}'
            if key in self.detail_weights:
                weight = self.detail_weights[key]
                threshold = thresholds.get(key, self.threshold_cD1)
                
                # 对预测值的细节系数施加软阈值（去噪/稀疏化）
                pred_filtered = self.soft_threshold(pred_coeffs[key], threshold)
                # 对真实值也施加同样的阈值（保持一致性）
                true_filtered = self.soft_threshold(true_coeffs[key], threshold)
                
                # 计算过滤后的匹配损失
                level_loss = F.mse_loss(pred_filtered, true_filtered)
                loss_freq = loss_freq + weight * level_loss

        # Initial Scale Matching
        if not self.scale_initialized and self.training:
            with torch.no_grad():
                if loss_freq.item() > 1e-8:
                    self.scale_factor = loss_time.detach() / loss_freq.detach()
                else:
                    self.scale_factor = torch.tensor(1.0, device=y_pred.device)
                self.scale_initialized = True
                print(f"[Multi-Level DWT + Learnable Thresholds] wavelet={self.wavelet}, levels={self.levels}")
                print(f"  Initial thresholds: cD1={F.relu(self.threshold_cD1).item():.4f}, cD2={F.relu(self.threshold_cD2).item():.4f}, cD3={F.relu(self.threshold_cD3).item():.4f}")
                print(f"[Initial Scale Matching] L_time={loss_time.item():.6f}, L_freq={loss_freq.item():.6f}, scale_factor={self.scale_factor.item():.4f}")

        # 最终损失
        loss_total = loss_time + self.reg_lambda * self.scale_factor * loss_freq
        
        return loss_total

