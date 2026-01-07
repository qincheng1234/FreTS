"""
FreTS-Lite: 基于 MTS-Mixers 思想的轻量化模型

架构设计：
1. RevIN: 实例归一化
2. FrequencyTemporalLearner: 全局周期性时序特征 (FFT)
3. TemporalMLP: 局部非周期性时序特征 (Linear) - MTS-Mixers 核心组件
4. ChannelOnlyLayer: 通道间相关性 (Attention)
5. Projection: 输出预测

理论依据：
- MTS-Mixers: "Attention is not necessary for temporal dependencies"
- 时间域用 FFT + MLP 替代 Attention，降低 O(T²) 到 O(T)
- 保留 Channel Attention 捕捉变量间相关性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization)"""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def normalize(self, x):
        self._get_statistics(x)
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
        x = x * self.stdev + self.mean
        return x


class ComplexLinear(nn.Module):
    """复数线性层 - 频域相位保护"""
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)

    def forward(self, x_real, x_imag):
        out_real = self.A(x_real) - self.B(x_imag)
        out_imag = self.B(x_real) + self.A(x_imag)
        return out_real, out_imag


class FrequencyTemporalLearner(nn.Module):
    """频域时序学习器 - 全局周期性特征提取"""
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.freq_dim = seq_len // 2 + 1
        self.complex_linear = ComplexLinear(self.freq_dim, self.freq_dim)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x):
        # x: [B, N, T, D]
        B, N, T, D = x.shape
        x = x.permute(0, 1, 3, 2)  # [B, N, D, T]
        
        x_fft = torch.fft.rfft(x, dim=-1)
        x_real, x_imag = x_fft.real, x_fft.imag
        
        out_real, out_imag = self.complex_linear(x_real, x_imag)
        
        x_out = torch.complex(out_real, out_imag)
        x_out = torch.fft.irfft(x_out, n=T, dim=-1)
        
        x_out = x_out.permute(0, 1, 3, 2) * self.scale  # [B, N, T, D]
        return x_out


class PositionalEncoding(nn.Module):
    """可学习位置编码"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x, seq_len):
        return x + self.pe[:, :seq_len, :]


class TemporalMLP(nn.Module):
    """
    [MTS-Mixers 核心组件] 时序 MLP
    
    作用：捕捉 FFT 难以处理的局部非线性变化和非周期性特征
    复杂度：O(T) 线性，远低于 Attention 的 O(T²)
    """
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        
        # 在 Time 维度做全连接
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout)
        )
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x):
        # x: [B, N, T, D]
        # 将 T 移到最后以便 Linear 处理
        x = x.permute(0, 1, 3, 2)  # [B, N, D, T]
        x = self.mlp(x)
        return x.permute(0, 1, 3, 2) * self.scale  # [B, N, T, D]


class SharedMultiHeadAttention(nn.Module):
    """共享权重的多头注意力"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B_G, L, D = x.shape
        q = self.W_q(x).view(B_G, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B_G, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B_G, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B_G, L, D)
        return self.out_proj(out)


class ChannelOnlyLayer(nn.Module):
    """
    仅通道注意力层 - 移除了 Sequence Attention
    
    基于 MTS-Mixers 论文：时序依赖由 FFT + MLP 处理
    这里只负责变量间相关性建模
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.shared_attn = SharedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm_channel = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def _channel_attention(self, x):
        """Channel MSA: 混合不同变量的信息"""
        B, N, T, D = x.shape
        x_c = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        x_c_norm = self.norm_channel(x_c)
        attn_out = self.shared_attn(x_c_norm)
        x_c = x_c + self.dropout(attn_out)
        return x_c.reshape(B, T, N, D).permute(0, 2, 1, 3)

    def forward(self, x):
        x = self._channel_attention(x)
        x_norm = self.norm_ff(x)
        x = x + self.ff(x_norm)
        return x


class Model(nn.Module):
    """
    FreTS-Lite: Frequency + TemporalMLP + ChannelAttention
    
    架构逻辑：
    - 时间域: FFT (全局周期) + MLP (局部非周期) = 完整时序建模
    - 空间域: Channel Attention = 变量间相关性
    - 复杂度: O(T log T) + O(T) + O(N²) << O(T²) + O(N²)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, 'd_model', 128)
        self.n_heads = getattr(configs, 'n_heads', 4)
        self.d_ff = getattr(configs, 'd_ff', 256)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # 1. RevIN
        self.revin = RevIN(self.enc_in, affine=True)
        
        # 2. Embedding
        self.embeddings = nn.Parameter(torch.randn(1, self.d_model))
        
        # 3. Position Encoding
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max(self.seq_len, 512))
        
        # 4. 时序处理: 双路融合
        # Path A: 频域 (全局周期性)
        self.freq_learner = FrequencyTemporalLearner(self.seq_len, self.d_model)
        # Path B: 时域 MLP (局部非周期性) - MTS-Mixers 核心
        self.time_mlp = TemporalMLP(self.seq_len, self.d_model, self.dropout)
        
        # 5. 空间处理: 仅 Channel Attention
        self.channel_layers = nn.ModuleList([
            ChannelOnlyLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.e_layers)
        ])
        
        # 6. Output
        self.final_norm = nn.LayerNorm(self.d_model)
        self.projection = nn.Linear(self.seq_len * self.d_model, self.pred_len)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def tokenEmb(self, x):
        return x.unsqueeze(3) * self.embeddings

    def forward(self, x):
        # x: [B, T, N]
        
        # 1. RevIN
        x = self.revin.normalize(x)
        x = x.permute(0, 2, 1)  # [B, N, T]
        
        # 2. Embedding
        x_enc = self.tokenEmb(x)  # [B, N, T, D]
        
        # 3. 双路时序特征融合 (核心改进)
        x_freq = self.freq_learner(x_enc)  # 全局周期
        x_time = self.time_mlp(x_enc)      # 局部非周期
        x_feat = x_enc + x_freq + x_time   # 融合
        
        # 4. Position Encoding
        B, N, T, D = x_feat.shape
        x_feat = x_feat.reshape(B * N, T, D)
        x_feat = self.pos_enc(x_feat, T)
        x_feat = x_feat.reshape(B, N, T, D)
        
        # 5. Channel Mixing Only
        for layer in self.channel_layers:
            x_feat = layer(x_feat)
        
        # 6. Output
        x_feat = self.final_norm(x_feat)
        B, N, T, D = x_feat.shape
        x_out = x_feat.reshape(B, N, -1)
        x_out = self.projection(x_out)
        
        x_out = x_out.permute(0, 2, 1)
        x_out = self.revin.denormalize(x_out)
        return x_out
