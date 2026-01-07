import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================
# 基础组件：RevIN, 分解, 频域模块
# =========================================================

class RevIN(nn.Module):
    """可逆实例归一化：用于缓解分布漂移"""
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

class MovingAvg(nn.Module):
    """移动平均：用于提取趋势项"""
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 填充以保持序列长度不变
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """序列分解：X = Trend + Seasonal"""
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x) # Trend
        res = x - moving_mean            # Seasonal
        return res, moving_mean

class ComplexLinear(nn.Module):
    """复数线性层：频域相位保护"""
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(input_dim, output_dim)
        self.fc_i = nn.Linear(input_dim, output_dim)

    def forward(self, x_real, x_imag):
        o_real = self.fc_r(x_real) - self.fc_i(x_imag)
        o_imag = self.fc_r(x_imag) + self.fc_i(x_real)
        return o_real, o_imag

class FrequencyTemporalLearner(nn.Module):
    """FreTS 核心：频域全局滤波 (带门控去噪)"""
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.complex_filter = ComplexLinear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, N, T, D]
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')
        real, imag = x_fft.real, x_fft.imag
        
        o_real, o_imag = self.complex_filter(real, imag)
        
        # Amplitude Gating (去噪)
        mag = torch.sqrt(o_real**2 + o_imag**2 + 1e-8)
        mask = self.gate(mag)
        o_real, o_imag = o_real * mask, o_imag * mask
        
        return torch.fft.irfft(torch.complex(o_real, o_imag), n=self.seq_len, dim=2, norm='ortho')

# =========================================================
# 混合层组件：TemporalMLP (Time) + Attention (Channel)
# =========================================================

class TemporalMLP(nn.Module):
    """
    [替换原 Sequence Attention]
    使用 MLP 混合时间维度信息，对固定周期模式捕捉效率更高
    """
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout)
        )
        # 初始化缩放因子
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x):
        # x: [B, N, T, D]
        # 转置为 [B, N, D, T] 以便 Linear 作用于 T 维度
        x = x.permute(0, 1, 3, 2) 
        x = self.mlp(x)
        return x.permute(0, 1, 3, 2) * self.scale

class SharedMultiHeadAttention(nn.Module):
    """共享权重注意力 (保留用于 Channel Mixing)"""
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
        # x: [B*T, N, D] (Channel Mixing 时)
        B_dimA, L, D = x.shape
        q = self.W_q(x).view(B_dimA, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B_dimA, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B_dimA, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B_dimA, L, D)
        return self.out_proj(out)

class HybridBlock(nn.Module):
    """
    [核心 Block] 
    Channel 维度使用 Attention (处理变量相关性)
    Time 维度使用 MLP (处理时间依赖，替换了原 Attention)
    """
    def __init__(self, seq_len, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 1. Channel Mixing (Attention)
        self.channel_attn = SharedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm_channel = nn.LayerNorm(d_model)
        
        # 2. Time Mixing (MLP - Replaced Sequence Attention)
        self.temporal_mlp = TemporalMLP(seq_len, d_model, dropout)
        self.norm_time = nn.LayerNorm(d_model)
        
        # 3. Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, T, D]
        B, N, T, D = x.shape
        
        # --- Channel Mixing (Attention) ---
        # Reshape to [B*T, N, D]
        x_c = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        x_c_norm = self.norm_channel(x_c)
        x_c = x_c + self.dropout(self.channel_attn(x_c_norm))
        # Restore to [B, N, T, D]
        x = x_c.reshape(B, T, N, D).permute(0, 2, 1, 3)
        
        # --- Time Mixing (MLP) ---
        x_t_norm = self.norm_time(x)
        x = x + self.dropout(self.temporal_mlp(x_t_norm))
        
        # --- FFN ---
        x_f_norm = self.norm_ff(x)
        x = x + self.ff(x_f_norm)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x, seq_len):
        return x + self.pe[:, :seq_len, :]

# =========================================================
# 主模型架构
# =========================================================

class Model(nn.Module):
    """
    FreTS-Hybrid: 
    Decomposition + FreTS Global Filter + Hybrid Mixer Backbone
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
        
        # 1. 序列分解 (对抗非平稳性)
        kernel_size = 25
        self.decomp = SeriesDecomp(kernel_size)
        
        # 2. Trend 分支处理 (简单的线性层)
        self.trend_proj = nn.Linear(self.seq_len, self.pred_len)
        
        # 3. Seasonal 分支处理 (主干网络)
        # 3.1 RevIN (只对 Seasonal 部分做)
        self.revin = RevIN(self.enc_in, affine=True)
        
        # 3.2 Embedding (升级为 Full-Rank Linear)
        self.feature_embed = nn.Linear(1, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max(self.seq_len, 512))
        
        # 3.3 FreTS 频域全局滤波 (FreTS 特色)
        self.freq_learner = FrequencyTemporalLearner(self.seq_len, self.d_model)
        
        # 3.4 Hybrid Backbone (MLP for Time, Attn for Channel)
        self.layers = nn.ModuleList([
            HybridBlock(self.seq_len, self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.e_layers)
        ])
        
        # 3.5 Output Projection
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
        
        # --- Step 1: Decomposition ---
        # 将原始数据分解为 Seasonal (短期波动) 和 Trend (长期趋势)
        seasonal_init, trend_init = self.decomp(x)
        
        # --- Step 2: Trend Branch ---
        # Trend 通常是非平稳的主要来源，直接用 Linear 映射
        trend_out = self.trend_proj(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        # --- Step 3: Seasonal Branch (Main Model) ---
        # 3.1 RevIN Normalize
        x_s = self.revin.normalize(seasonal_init)
        x_s = x_s.permute(0, 2, 1)  # [B, N, T]
        
        # 3.2 Embedding & Position
        x_s = x_s.unsqueeze(-1)     # [B, N, T, 1]
        x_s = self.feature_embed(x_s) # [B, N, T, D]
        
        # 3.3 Frequency Filtering (FreTS 的全局去噪)
        # 残差连接：x + Freq(x)
        x_freq = self.freq_learner(x_s)
        x_s = x_s + x_freq
        
        # Add Position Encoding (After freq filter)
        B, N, T, D = x_s.shape
        x_s = x_s.reshape(B * N, T, D)
        x_s = self.pos_enc(x_s, T)
        x_s = x_s.reshape(B, N, T, D)
        
        # 3.4 Deep Layers (Hybrid Mixing)
        for layer in self.layers:
            x_s = layer(x_s)
            
        # 3.5 Final Projection
        x_s = self.final_norm(x_s)
        x_s = x_s.reshape(B, N, -1)
        seasonal_out = self.projection(x_s) # [B, N, Pred]
        
        # 3.6 RevIN Denormalize
        seasonal_out = seasonal_out.permute(0, 2, 1) # [B, Pred, N]
        seasonal_out = self.revin.denormalize(seasonal_out)
        
        # --- Step 4: Final Fusion ---
        return seasonal_out + trend_out
