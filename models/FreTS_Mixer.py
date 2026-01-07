"""
FreTS-Mixer: 标准 Mixer Block 架构

架构设计（修正版）：
- FreTSMixerBlock = TimeMix + ChannelMix（作为一个整体堆叠 N 次）
- TimeMix = FrequencyLearner (全局周期) + TemporalMLP (局部非周期)
- ChannelMix = Channel Attention + FFN

修正的三个核心问题：
1. 时序模块放入 Block 循环，实现 (TimeMix + ChannelMix) × N
2. 使用 nn.Linear(1, d_model) 替代 Rank-1 Embedding
3. PE 放在 Embedding 之后、所有模块之前

理论依据：
- MTS-Mixers: "Interleaved Time and Channel Mixing"
- 标准 Mixer/Transformer Block 结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """可逆实例归一化"""
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
    """频域时序学习器 - 全局周期性特征"""
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.freq_dim = seq_len // 2 + 1
        self.complex_linear = ComplexLinear(self.freq_dim, self.freq_dim)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x):
        B, N, T, D = x.shape
        x = x.permute(0, 1, 3, 2)  # [B, N, D, T]
        
        x_fft = torch.fft.rfft(x, dim=-1)
        x_real, x_imag = x_fft.real, x_fft.imag
        out_real, out_imag = self.complex_linear(x_real, x_imag)
        
        x_out = torch.complex(out_real, out_imag)
        x_out = torch.fft.irfft(x_out, n=T, dim=-1)
        
        return x_out.permute(0, 1, 3, 2) * self.scale


class TemporalMLP(nn.Module):
    """时序 MLP - 局部非周期性特征"""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout)
        )
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)  # [B, N, D, T]
        x = self.mlp(x)
        return x.permute(0, 1, 3, 2) * self.scale


class PositionalEncoding(nn.Module):
    """可学习位置编码"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x, seq_len):
        return x + self.pe[:, :seq_len, :]


class SharedMultiHeadAttention(nn.Module):
    """共享权重多头注意力"""
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


class ChannelMixingLayer(nn.Module):
    """通道混合层：Attention + FFN"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = SharedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, T, D = x.shape
        x_c = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        
        # Attention + Residual
        x_c_norm = self.norm1(x_c)
        x_c = x_c + self.dropout(self.attn(x_c_norm))
        
        # FFN + Residual
        x_c_norm = self.norm2(x_c)
        x_c = x_c + self.ff(x_c_norm)
        
        return x_c.reshape(B, T, N, D).permute(0, 2, 1, 3)


class FreTSMixerBlock(nn.Module):
    """
    标准 Mixer Block：(TimeMix + ChannelMix) 作为一个整体
    
    结构：
    Input → [TimeMix: Freq + MLP] → [ChannelMix: Attn + FFN] → Output
    
    这解决了 "Time 1x vs Channel Nx" 的结构失衡问题
    """
    def __init__(self, seq_len, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Time Mixing (双路)
        self.freq_learner = FrequencyTemporalLearner(seq_len, d_model)
        self.time_mlp = TemporalMLP(seq_len, d_model, dropout)
        self.norm_time = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Channel Mixing
        self.channel_layer = ChannelMixingLayer(d_model, n_heads, d_ff, dropout)

    def forward(self, x):
        # x: [B, N, T, D]
        
        # === Time Mixing: Freq + MLP (Residual) ===
        x_freq = self.freq_learner(x)
        x_time = self.time_mlp(x)
        x = x + self.dropout(x_freq) + self.dropout(x_time)
        x = self.norm_time(x)
        
        # === Channel Mixing ===
        x = self.channel_layer(x)
        
        return x


class Model(nn.Module):
    """
    FreTS-Mixer: 标准 Block 堆叠架构
    
    架构: RevIN → Embedding → PE → (FreTSMixerBlock × N) → Projection
    
    修正点：
    1. 使用 nn.Linear(1, d_model) 替代 Rank-1 Embedding
    2. PE 位于 Embedding 之后
    3. TimeMix + ChannelMix 绑定在同一个 Block 中循环
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
        
        # 2. [修正] Full-Rank Embedding: nn.Linear(1, d_model)
        self.feature_embed = nn.Linear(1, self.d_model)
        
        # 3. Position Encoding (放在 Embedding 之后)
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max(self.seq_len, 512))
        
        # 4. [修正] 堆叠 FreTSMixerBlock (每个 Block 都包含 TimeMix + ChannelMix)
        self.layers = nn.ModuleList([
            FreTSMixerBlock(self.seq_len, self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.e_layers)
        ])
        
        # 5. Output
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

    def forward(self, x):
        # x: [B, T, N]
        
        # 1. RevIN
        x = self.revin.normalize(x)
        x = x.permute(0, 2, 1)  # [B, N, T]
        
        # 2. [修正] Full-Rank Embedding
        x = x.unsqueeze(-1)  # [B, N, T, 1]
        x = self.feature_embed(x)  # [B, N, T, D]
        
        # 3. Position Encoding (在 Embedding 之后)
        B, N, T, D = x.shape
        x = x.reshape(B * N, T, D)
        x = self.pos_enc(x, T)
        x = x.reshape(B, N, T, D)
        
        # 4. Deep Mixer Blocks
        for layer in self.layers:
            x = layer(x)
        
        # 5. Output
        x = self.final_norm(x)
        B, N, T, D = x.shape
        x = x.reshape(B, N, -1)  # [B, N, T*D]
        x = self.projection(x)   # [B, N, Pred]
        
        x = x.permute(0, 2, 1)   # [B, Pred, N]
        x = self.revin.denormalize(x)
        return x
