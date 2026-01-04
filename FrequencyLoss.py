import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedBSPLoss(nn.Module):
    """
    改进版分箱谱功率损失 (Improved Binned Spectral Power Loss)
    
    核心改进：
    1. 相对误差替代对数误差：避免梯度爆炸/消失
    2. 均匀分箱 + 频率加权：低频高权重，高频低权重，渐进过渡
    3. 损失裁剪：防止异常值主导优化
    4. 可选的相位匹配：不仅匹配能量，还匹配相位信息
    """
    def __init__(self, n_bins=8, epsilon=1e-6, max_loss=10.0, use_phase=False):
        """
        Args:
            n_bins: 分箱数量（建议 8，过多会稀释信号）
            epsilon: 数值稳定性常数
            max_loss: 单个 bin 的最大损失值（裁剪）
            use_phase: 是否使用相位匹配
        """
        super().__init__()
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.max_loss = max_loss
        self.use_phase = use_phase

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [Batch, Length, Channel] 预测值
            y_true: [Batch, Length, Channel] 真实值
        
        Returns:
            bsp_loss: 标量损失值
        """
        # 1. 维度调整: [B, L, C] -> [B, C, L]
        if y_pred.dim() == 3:
            y_pred = y_pred.permute(0, 2, 1)
            y_true = y_true.permute(0, 2, 1)
        
        # 2. FFT 变换 (RFFT)
        pred_fft = torch.fft.rfft(y_pred, dim=-1, norm='ortho')
        true_fft = torch.fft.rfft(y_true, dim=-1, norm='ortho')
        
        # 3. 计算功率谱 (Power Spectrum) - 使用幅值而非功率，梯度更稳定
        pred_amp = torch.abs(pred_fft)
        true_amp = torch.abs(true_fft)
        
        n_freqs = pred_amp.shape[-1]
        device = y_pred.device
        
        if n_freqs <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 4. 均匀分箱
        actual_bins = min(self.n_bins, n_freqs)
        bin_size = n_freqs // actual_bins
        
        # 5. 计算频率加权（低频高权重，高频低权重）
        # 使用余弦衰减: weight = 0.5 * (1 + cos(pi * i / n_bins))
        bin_weights = []
        for i in range(actual_bins):
            weight = 0.5 * (1 + torch.cos(torch.tensor(3.14159 * i / actual_bins, device=device)))
            bin_weights.append(weight.clamp(min=0.1))  # 最小权重 0.1，保证高频也有贡献
        
        # 6. 计算每个 Bin 的相对误差
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_weight = 0.0
        
        for i in range(actual_bins):
            start = i * bin_size
            end = start + bin_size if i < actual_bins - 1 else n_freqs
            
            if start >= end:
                continue
            
            # 幅值聚合
            pred_bin_amp = torch.mean(pred_amp[..., start:end], dim=-1)
            true_bin_amp = torch.mean(true_amp[..., start:end], dim=-1)
            
            # 相对误差: |pred - true| / (true + epsilon)
            # 比对数误差更稳定，梯度更平滑
            relative_error = torch.abs(pred_bin_amp - true_bin_amp) / (true_bin_amp + self.epsilon)
            
            # 裁剪过大的误差
            relative_error = torch.clamp(relative_error, max=self.max_loss)
            
            # 加权累加
            bin_loss = torch.mean(relative_error)
            total_loss = total_loss + bin_weights[i] * bin_loss
            total_weight += bin_weights[i].item()
        
        # 7. 可选：相位匹配损失
        if self.use_phase:
            pred_phase = torch.angle(pred_fft)
            true_phase = torch.angle(true_fft)
            # 相位差的余弦距离
            phase_diff = 1 - torch.cos(pred_phase - true_phase)
            phase_loss = torch.mean(phase_diff)
            total_loss = total_loss + 0.1 * phase_loss  # 相位权重较小
        
        # 归一化
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        return total_loss


# 保留旧类名作为别名，但标记为废弃
class RobustBSPLoss(ImprovedBSPLoss):
    """[已废弃] 请使用 ImprovedBSPLoss"""
    def __init__(self, n_bins=16, epsilon=1e-8):
        super().__init__(n_bins=min(n_bins, 8), epsilon=epsilon)


class UniversalFrequencyLoss(nn.Module):
    """
    通用频域损失函数 (Universal Frequency Loss) - 改进版
    
    组合策略：
    Loss = L_time + reg_lambda * scale_factor * L_bsp
    
    改进点：
    - 使用 ImprovedBSPLoss 替代原始 RobustBSPLoss
    - EMA 动态调整 Scale Factor，适应训练过程中损失量级变化
    - 更保守的默认 reg_lambda (0.01)，避免频域损失主导优化
    """
    def __init__(self, reg_lambda=0.01, n_bins=8, in_channels=7, ema_momentum=0.99):
        """
        Args:
            reg_lambda: BSP 损失的权重（建议 0.001 ~ 0.1）
            n_bins: BSP 的分箱数量（建议 8）
            in_channels: 输入通道数
            ema_momentum: EMA 动量，用于动态调整 scale_factor
        """
        super().__init__()
        self.reg_lambda = reg_lambda
        self.in_channels = in_channels
        self.ema_momentum = ema_momentum
        
        # 时域损失
        self.time_loss = nn.MSELoss()
        
        # 改进的 BSP 损失
        self.bsp_loss = ImprovedBSPLoss(n_bins=n_bins)
        
        # 动态 Scale Factor (使用 EMA 更新)
        self.register_buffer('scale_factor', torch.tensor(1.0))
        self.register_buffer('ema_time', torch.tensor(0.0))
        self.register_buffer('ema_bsp', torch.tensor(0.0))
        self.scale_initialized = False
        self.step_count = 0
    
    def forward(self, y_pred, y_true):
        """
        计算组合损失
        
        Args:
            y_pred: [Batch, Length, Channel] 预测值
            y_true: [Batch, Length, Channel] 真实值
        
        Returns:
            total_loss: 加权组合的总损失
        """
        # 时域损失
        loss_time = self.time_loss(y_pred, y_true)
        
        # BSP 损失
        loss_bsp = self.bsp_loss(y_pred, y_true)
        
        # 动态 Scale Factor (EMA)
        if self.training:
            with torch.no_grad():
                if not self.scale_initialized:
                    # 首次初始化
                    self.ema_time = loss_time.detach()
                    self.ema_bsp = loss_bsp.detach().clamp(min=1e-8)
                    self.scale_factor = self.ema_time / self.ema_bsp
                    self.scale_initialized = True
                    print(f"[Universal Frequency Loss] n_bins={self.bsp_loss.n_bins}, reg_lambda={self.reg_lambda}")
                    print(f"[Initial Scale] L_time={loss_time.item():.6f}, L_bsp={loss_bsp.item():.6f}, scale_factor={self.scale_factor.item():.4f}")
                else:
                    # EMA 更新
                    self.ema_time = self.ema_momentum * self.ema_time + (1 - self.ema_momentum) * loss_time.detach()
                    self.ema_bsp = self.ema_momentum * self.ema_bsp + (1 - self.ema_momentum) * loss_bsp.detach().clamp(min=1e-8)
                    self.scale_factor = self.ema_time / self.ema_bsp
                
                self.step_count += 1
                # 每 500 步打印一次监控信息
                if self.step_count % 500 == 0:
                    print(f"[Step {self.step_count}] L_time={loss_time.item():.6f}, L_bsp={loss_bsp.item():.6f}, scale={self.scale_factor.item():.4f}")
        
        # 最终损失：时域为主，频域为辅
        loss_total = loss_time + self.reg_lambda * self.scale_factor * loss_bsp
        
        return loss_total


# 兼容性接口：保留旧的类名
class FrequencyRegularizedLoss(UniversalFrequencyLoss):
    """
    兼容旧代码的接口
    内部使用 UniversalFrequencyLoss (改进版 BSP) 实现
    """
    def __init__(self, reg_lambda=0.01, in_channels=1, wavelet='db4', levels=3):
        # wavelet 和 levels 参数保留但不使用（兼容性）
        super().__init__(reg_lambda=reg_lambda, n_bins=8, in_channels=in_channels)
