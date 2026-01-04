import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyGating(nn.Module):
    """
    基于 MMFNet 思想的频域门控模块
    
    核心思想：
    - 用自适应掩码替代固定阈值的 softshrink
    - 根据输入的幅值动态生成 (0,1) 之间的 Mask
    - 通过乘法门控实现平滑的频率滤波
    """
    def __init__(self, embed_size, reduction=4):
        """
        Args:
            embed_size: 嵌入维度
            reduction: 降维比例（用于减少参数量，类似 SE-Net）
        """
        super().__init__()
        self.embed_size = embed_size
        
        # Mask 生成器 (类似于 SE-Net 的 Squeeze-and-Excitation)
        # 输入是幅值 (实数)，输出是 Mask (0~1)
        hidden_size = max(embed_size // reduction, 16)  # 防止过小
        self.gate_generator = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, embed_size),
            nn.Sigmoid()  # 输出 (0, 1)，对应 MMFNet 的 σ
        )
        nn.init.constant_(self.gate_generator[-2].bias, 2.0)
    
    def forward(self, x_real, x_imag):
        """
        对实部和虚部分别应用自适应门控
        
        Args:
            x_real: [B, nd, freq, D] 实部
            x_imag: [B, nd, freq, D] 虚部
        
        Returns:
            gated_real, gated_imag: 门控后的实部和虚部
        """
        # 1. 计算幅值 (能量)
        # magnitude = sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)
        
        # 2. 生成自适应掩码
        # Mask = σ(W * |X| + b)
        mask = self.gate_generator(magnitude)
        
        # 3. 应用门控 (乘法滤波)
        # 相比 softshrink 的硬截断，乘法门控有更平滑的梯度
        gated_real = x_real * mask
        gated_imag = x_imag * mask
        
        return gated_real, gated_imag


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embed_size = 128  # embed_size
        self.hidden_size = 256  # hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in  # channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        
        # === [MMFNet] 替换 softshrink 为自适应频域门控 ===
        self.frequency_gating = FrequencyGating(self.embed_size, reduction=4)
        
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        # === [MMFNet] 使用自适应频域门控替代 softshrink ===
        o1_real, o1_imag = self.frequency_gating(o1_real, o1_imag)
        
        # 重组为复数
        y = torch.complex(o1_real, o1_imag)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == 1:
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x
