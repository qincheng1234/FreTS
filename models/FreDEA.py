import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """
    [必须启用] 实例归一化 (Reversible Instance Normalization)
    作用：消除时间序列在局部窗口内的分布漂移 (Non-stationarity)。
    """
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
    """
    复数域线性层 (用于频域特征提取)
    """
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        real = self.fc_r(x.real) - self.fc_i(x.imag)
        imag = self.fc_r(x.imag) + self.fc_i(x.real)
        return torch.complex(real, imag)


class FrequencyTemporalLearner(nn.Module):
    """
    [核心组件] 频域时序学习器 (FreTS)
    """
    def __init__(self, seq_len, d_model):
        super(FrequencyTemporalLearner, self).__init__()
        self.seq_len = seq_len
        self.complex_weight = ComplexLinear(seq_len // 2 + 1, seq_len // 2 + 1)

    def forward(self, x):
        B, N, T, D = x.shape
        x = x.permute(0, 1, 3, 2)
        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')
        x_freq = x_freq.permute(0, 1, 3, 2)
        x_freq = x_freq.reshape(B * N, x_freq.shape[2], D)
        weight = self.complex_weight.fc_r.weight.to(dtype=x_freq.dtype)
        x_freq = torch.einsum('bfd,kf->bkd', x_freq, weight)
        x_freq = x_freq.reshape(B, N, x_freq.shape[1], D)
        x_freq = x_freq.permute(0, 1, 3, 2)
        x_time = torch.fft.irfft(x_freq, n=T, dim=-1, norm='ortho')
        x_time = x_time.permute(0, 1, 3, 2)
        return x_time


class PositionalEncoding(nn.Module):
    """
    位置编码 (Positional Encoding)
    """
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x, seq_len):
        return x + self.pe[:, :seq_len, :]


class TemporalExternalAttention(nn.Module):
    """
    [核心创新] 时序外部注意力 (Temporal External Attention)
    """
    def __init__(self, seq_len, d_model, memory_size=64):
        super(TemporalExternalAttention, self).__init__()
        self.M_k = nn.Linear(seq_len, memory_size, bias=False)
        self.M_v = nn.Linear(memory_size, seq_len, bias=False)
        nn.init.xavier_uniform_(self.M_k.weight)
        nn.init.xavier_uniform_(self.M_v.weight)
    
    def forward(self, x):
        x_t = x.permute(0, 1, 3, 2)
        attn = self.M_k(x_t)
        attn = F.softmax(attn, dim=-1)
        out = self.M_v(attn)
        out = out.permute(0, 1, 3, 2)
        return out


class ChannelExternalAttention(nn.Module):
    """
    [核心创新] 通道外部注意力 (Channel External Attention)
    """
    def __init__(self, num_channels, d_model, memory_size=64, dropout=0.1):
        super(ChannelExternalAttention, self).__init__()
        self.M_k = nn.Linear(num_channels, memory_size, bias=False)
        self.M_v = nn.Linear(memory_size, num_channels, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.M_k.weight)
        nn.init.xavier_uniform_(self.M_v.weight)
    
    def forward(self, x):
        x_t = x.permute(0, 2, 3, 1)
        attn = self.M_k(x_t)
        attn = F.softmax(attn, dim=-1)
        out = self.M_v(attn)
        out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out


class TEABlock(nn.Module):
    """
    [V24] Parametric Sigmoid Gated TEABlock
    
    Gate 被 Sigmoid 约束在 [0, 1] 之间，且初始化接近 0。
    这允许模型自动决定是否需要通道混合：
    - Electricity (CI数据): Gate 保持接近 0
    - ETTm1 (CD数据): Gate 可学习增大
    """
    def __init__(self, d_model, seq_len, num_channels, d_ff, 
                 memory_size=64, dropout=0.1,
                 ablation_tea=0, ablation_cea=0):
        super(TEABlock, self).__init__()
        self.ablation_tea = ablation_tea
        self.ablation_cea = ablation_cea
        
        self.tea = TemporalExternalAttention(seq_len, d_model, memory_size)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cea = ChannelExternalAttention(num_channels, d_model, memory_size, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # [V24] Sigmoid Gate with Conservative Init
        # sigmoid(-5.0) ≈ 0.0067, 从几乎纯 CI 开始
        self.gate_logit = nn.Parameter(torch.tensor(-5.0))
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if not self.ablation_tea:
            residual = x
            x = self.norm1(x)
            x = residual + self.dropout(self.tea(x))
        
        if not self.ablation_cea:
            residual = x
            x = self.norm2(x)
            # [V24] Sigmoid Gate: 保证 Gate ∈ [0, 1]
            gate = torch.sigmoid(self.gate_logit)
            x = residual + gate * self.dropout(self.cea(x))
        
        residual = x
        x = self.norm3(x)
        x = residual + self.ff(x)
        return x


# =============================================================================
# V24 核心组件: 参数化频域分解 (Parametric Frequency Decomposition)
# =============================================================================

class ParametricFreqDecomp(nn.Module):
    """
    [V24 核心创新] 参数化低通掩码分解
    
    只学习一个参数：截止频率 (cutoff)。
    强制 Mask 为 Sigmoid 形状的平滑低通滤波器，防止过拟合到特定噪声频率。
    
    Mask(f) = Sigmoid(cutoff - f)
    
    优势：
    1. 参数量从 49 降为 1，消除过拟合风险
    2. 物理一致性：Trend 永远是低频，Seasonal 永远是高频
    3. 平滑过渡：避免硬截断的 Gibbs 振铃效应
    """
    def __init__(self, seq_len):
        super(ParametricFreqDecomp, self).__init__()
        self.seq_len = seq_len
        freq_len = seq_len // 2 + 1
        
        # 注册频率索引 [0, 1, 2, ..., freq_len-1]
        self.register_buffer('freq_indices', torch.arange(freq_len).float())
        
        # 可学习参数: 截止频率
        # 初始化为 3.0 (保留约 6% 低频作为 Trend)
        self.cutoff = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        # x: [B, N, T]
        B, N, T = x.shape
        
        # FFT: [B, N, Freq]
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 生成参数化 Mask
        # Sigmoid(cutoff - f):
        #   f < cutoff -> mask ≈ 1 (Trend)
        #   f > cutoff -> mask ≈ 0 (Seasonal)
        mask = torch.sigmoid(self.cutoff - self.freq_indices).view(1, 1, -1)
        
        # 频域分解
        x_trend_freq = x_freq * mask
        x_seasonal_freq = x_freq * (1 - mask)
        
        # iFFT
        x_trend = torch.fft.irfft(x_trend_freq, n=T, dim=-1)
        x_seasonal = torch.fft.irfft(x_seasonal_freq, n=T, dim=-1)
        
        return x_seasonal, x_trend


# =============================================================================
# V24 模型: Parametric Frequency-Adaptive Network (PFA-FreDEA)
# =============================================================================

class Model(nn.Module):
    """
    FreDEA V24_Refined: PFA-FreDEA (Parametric Frequency-Adaptive FreDEA)
    
    核心创新：
    1. Parametric Decomp: 只有 1 个可学习参数 (cutoff)，杜绝 Mask 过拟合
    2. Sigmoid Gate: [0,1] 约束，杜绝负值噪声，物理意义明确
    3. Trend Branch: 共享权重线性层，稳健趋势外推
    4. Seasonal Branch: TEA Encoder + Time Projection + Feature-Fusion MLP
    
    V24_Refined 优化：
    1. 自动 CI 策略：enc_in > 100 时自动关闭通道混合 (ablation_cea=1)
    2. 自适应 Dropout：pred_len >= 720 时增强正则化 (dropout_time=0.3)
    3. 可学习融合门控：sigmoid(fusion_logit) 控制 Trend/Seasonal 比例
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, 'd_model', 128)
        self.d_ff = getattr(configs, 'd_ff', 256)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.memory_size = getattr(configs, 'memory_size', 64)
        
        # Ablation study flags
        self.ablation_freq = getattr(configs, 'ablation_freq', 0)
        self.ablation_tea = getattr(configs, 'ablation_tea', 0)
        self.ablation_cea = getattr(configs, 'ablation_cea', 0)
        
        # -----------------------------------------------------------
        # [V24_Refined 优化 1] 自动 CI 策略
        # -----------------------------------------------------------
        # 高维数据 (如 Electricity 321 维) 自动关闭通道混合
        # 避免不同传感器之间的噪声相互干扰
        if self.enc_in > 100:
            self.ablation_cea = 1
        
        # 1. RevIN
        self.affine = getattr(configs, 'rev_affine', True) 
        self.revin = RevIN(self.enc_in, affine=self.affine)
        
        # 2. [V24] Parametric Frequency Decomposition
        # 只有 1 个可学习参数：cutoff，极其稳健
        self.decomposition = ParametricFreqDecomp(self.seq_len)
        
        # -----------------------------------------------------------
        # [Branch 1] Trend Modeling (Shared Linear)
        # -----------------------------------------------------------
        self.trend_model = nn.Linear(self.seq_len, self.pred_len)
        
        # -----------------------------------------------------------
        # [Branch 2] Seasonal Modeling (Deep Frequency Net)
        # -----------------------------------------------------------
        
        # A. Embedding
        self.embeddings = nn.Parameter(torch.randn(1, self.d_model))
        
        # B. Positional Encoding
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max(self.seq_len, 512))
        
        # C. Frequency Learner
        self.freq_learner = FrequencyTemporalLearner(self.seq_len, self.d_model)
        
        # D. [V24] Sigmoid Gated TEA Encoder
        self.tea_blocks = nn.ModuleList([
            TEABlock(self.d_model, self.seq_len, self.enc_in, self.d_ff, 
                     memory_size=self.memory_size, dropout=self.dropout,
                     ablation_tea=self.ablation_tea, ablation_cea=self.ablation_cea)
            for _ in range(self.e_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # E. Time Projection
        self.seasonal_time_proj = nn.Linear(self.seq_len, self.pred_len)
        
        # -----------------------------------------------------------
        # [V24_Refined 优化 2] 自适应 Dropout
        # -----------------------------------------------------------
        # 长序列预测 (720+) 极易过拟合，增强正则化强度
        if self.pred_len >= 720:
            time_dropout = max(0.3, self.dropout)  # 至少 0.3
        else:
            time_dropout = self.dropout
        self.dropout_time = nn.Dropout(time_dropout)
        
        # F. Feature-Fusion Output MLP
        self.seasonal_out_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, 1)
        )
        
        # -----------------------------------------------------------
        # [V24_Refined 优化 3] 可学习融合门控 (自适应初始化)
        # -----------------------------------------------------------
        # 根据预测长度动态初始化：
        # - 长预测 (720+): logit=2.0 → gate≈0.88 (88% Trend, 稳健)
        # - 中预测 (336+): logit=1.0 → gate≈0.73 (73% Trend)
        # - 短预测 (其他): logit=0.0 → gate=0.50 (平衡)
        # 让模型从合理的起点开始学习
        if self.pred_len >= 720:
            init_logit = 2.0  # 极依赖 Trend (88%)
        elif self.pred_len >= 336:
            init_logit = 1.0  # 偏向 Trend (73%)
        else:
            init_logit = 0.0  # 平衡 (50%)
        self.fusion_logit = nn.Parameter(torch.tensor(init_logit))
        
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
        
        # 1. RevIN Normalize
        x = self.revin.normalize(x)
        x = x.permute(0, 2, 1)  # [B, N, T]
        
        # 2. [V24] Parametric Frequency Decomposition
        x_seasonal, x_trend = self.decomposition(x)
        
        # ==========================================
        # Branch 1: Trend Prediction
        # ==========================================
        pred_trend = self.trend_model(x_trend)  # [B, N, Pred]
        
        # ==========================================
        # Branch 2: Seasonal Prediction (Deep)
        # ==========================================
        
        # A. Embedding
        x_enc = self.tokenEmb(x_seasonal)  # [B, N, T, D]
        
        # B. Frequency Filtering
        if not self.ablation_freq:
            x_freq = self.freq_learner(x_enc)
            x_feat = x_enc + x_freq
        else:
            x_feat = x_enc
        
        # C. Positional Encoding
        B, N, T, D = x_feat.shape
        x_feat = x_feat.reshape(B * N, T, D)
        x_feat = self.pos_enc(x_feat, T)
        x_feat = x_feat.reshape(B, N, T, D)
        
        # D. TEA Encoder (with Sigmoid Gate)
        for block in self.tea_blocks:
            x_feat = block(x_feat)
        
        x_feat = self.final_norm(x_feat)  # [B, N, T, D]
        
        # E. Projection Head
        # Time Dimension: T -> Pred
        x_feat_t = x_feat.permute(0, 1, 3, 2)  # [B, N, D, T]
        x_proj_t = self.seasonal_time_proj(x_feat_t)  # [B, N, D, Pred]
        x_proj_t = self.dropout_time(x_proj_t)
        
        # Feature Dimension: D -> 1 (via MLP)
        x_proj_t = x_proj_t.permute(0, 1, 3, 2)  # [B, N, Pred, D]
        pred_seasonal = self.seasonal_out_mlp(x_proj_t).squeeze(-1)  # [B, N, Pred]
        
        # ==========================================
        # [V24_Refined] 门控融合: Trend + Seasonal
        # ==========================================
        # gate ∈ [0, 1]，gate 越大越依赖 Trend（稳健）
        # 对于噪声数据 (Exchange)，模型可能学出 gate > 0.7
        gate = torch.sigmoid(self.fusion_logit)
        dec_out = gate * pred_trend + (1 - gate) * pred_seasonal
        
        # 3. RevIN Denormalize
        x_out = dec_out.permute(0, 2, 1)  # [B, Pred, N]
        x_out = self.revin.denormalize(x_out)
        
        return x_out
