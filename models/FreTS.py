import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """
    [必须启用] 实例归一化 (Reversible Instance Normalization)
    作用：消除时间序列在局部窗口内的分布漂移 (Non-stationarity)。
    保持 affine=True 以保留量纲恢复能力。
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
        # x: [B, T, N] -> 沿时间维度计算统计量
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
    [核心] 复数线性层 - 频域相位保护
    数学原理：z' = z * W = (xA - yB) + i(xB + yA)
    """
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(input_dim, output_dim)
        self.fc_i = nn.Linear(input_dim, output_dim)

    def forward(self, x_real, x_imag):
        o_real = self.fc_r(x_real) - self.fc_i(x_imag)
        o_imag = self.fc_r(x_imag) + self.fc_i(x_real)
        return o_real, o_imag


class FrequencyTemporalLearner(nn.Module):
    """
    [FreTS 核心] 频域滤波模块
    流程：FFT -> ComplexLinear (相位保护) -> Amplitude Gating (去噪) -> iFFT
    """
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
        
        # Complex Filtering
        o_real, o_imag = self.complex_filter(real, imag)
        
        # Amplitude Gating
        mag = torch.sqrt(o_real**2 + o_imag**2 + 1e-8)
        mask = self.gate(mag)
        o_real, o_imag = o_real * mask, o_imag * mask
        
        # iFFT
        return torch.fft.irfft(torch.complex(o_real, o_imag), n=self.seq_len, dim=2, norm='ortho')


# ============================================
# CSformer 核心组件: Shared Two-Stage Attention
# ============================================

class PositionalEncoding(nn.Module):
    """
    [新增] 可学习位置编码 - 为 Attention 提供时序位置信息
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x, seq_len):
        # x: [B, L, D]
        return x + self.pe[:, :seq_len, :]


class SharedMultiHeadAttention(nn.Module):
    """
    [CSformer 核心] 共享权重的多头注意力
    可处理 (Batch*Channel, Time, Dim) 或 (Batch*Time, Channel, Dim)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # Q, K, V 投影 (权重共享)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        # x: [Batch_Size * Dimension_A, Dimension_B, d_model]
        B_dimA, L, D = x.shape
        
        q = self.W_q(x).view(B_dimA, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B_dimA, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B_dimA, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention Score: (B*DimA, Heads, L, L)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B_dimA, L, D)
        return self.out_proj(out)


class TemporalMLP(nn.Module):
    """
    [新增] 时间维 MLP - 替代 Sequence Attention
    
    设计灵感：
    - iTransformer: 将时间步视为 token，在时间维上做 FFN
    - DLinear: 简单的线性层直接映射时间序列
    - TSMixer: Time-MLP 用于捕获时序依赖
    
    优势：
    1. 计算复杂度 O(T) vs Attention 的 O(T²)
    2. 无需位置编码，因为 MLP 隐式建模时序位置
    3. 参数量更少，更不容易过拟合
    """
    def __init__(self, seq_len, d_model, expansion_factor=2, dropout=0.1):
        super().__init__()
        hidden_dim = int(seq_len * expansion_factor)
        
        # 时间维 MLP: [B*N, T, D] -> 转置 -> [B*N, D, T] -> MLP -> 转置回来
        self.temporal_fc = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: [B, N, T, D]
        B, N, T, D = x.shape
        
        # Reshape and transpose for temporal mixing
        x_reshaped = x.reshape(B * N, T, D)  # [B*N, T, D]
        x_norm = self.norm(x_reshaped)
        
        # Transpose to apply MLP along time dimension
        x_t = x_norm.transpose(1, 2)  # [B*N, D, T]
        x_t = self.temporal_fc(x_t)   # [B*N, D, T]
        x_out = x_t.transpose(1, 2)   # [B*N, T, D]
        
        # Residual connection
        x_out = x_reshaped + x_out
        return x_out.reshape(B, N, T, D)


class CSformerLayer(nn.Module):
    """
    [改进版] CSformer 混合层 - 将 Sequence Attention 替换为 Temporal MLP
    
    支持两种混合顺序：
    - 'cm': Channel Attention → Temporal MLP (类似 CSformer 原设计)
    - 'mc': Temporal MLP → Channel Attention (先做时序汇聚再做通道交互)
    
    设计理念：
    - Channel Attention: 动态建模通道间的全局依赖关系
    - Temporal MLP: 高效的时序位置感知混合（类似 DLinear/iTransformer）
    
    优势：
    1. 降低计算复杂度: O(N²·T + N·T) vs O(N²·T + T²·N)
    2. 减少参数量和过拟合风险
    3. TimeMLP 对周期性模式的捕获更直接
    
    采用 Pre-Norm 结构，提升训练稳定性
    """
    def __init__(self, d_model, n_heads, d_ff, seq_len, dropout=0.1, attention_order='cm'):
        super().__init__()
        self.attention_order = attention_order
        
        # Channel Attention 模块
        self.channel_attn = SharedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm_channel = nn.LayerNorm(d_model)
        
        # Temporal MLP 模块 (替代 Sequence Attention)
        self.temporal_mlp = TemporalMLP(seq_len, d_model, expansion_factor=2, dropout=dropout)
        
        # Final Layer Norm
        self.norm_ff = nn.LayerNorm(d_model)
        
        # Feed Forward Network (通道维度)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def _channel_attention(self, x):
        """Channel MSA: 通道混合"""
        B, N, T, D = x.shape
        x_c = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        x_c_norm = self.norm_channel(x_c)
        attn_out = self.channel_attn(x_c_norm)
        x_c = x_c + self.dropout(attn_out)
        return x_c.reshape(B, T, N, D).permute(0, 2, 1, 3)
    
    def _temporal_mixing(self, x):
        """Temporal MLP: 时序混合"""
        return self.temporal_mlp(x)

    def forward(self, x):
        # Input x: [Batch, Channel(N), Time(T), D_model]
        
        if self.attention_order == 'cm':
            # Channel Attention → Temporal MLP
            # 适合通道相关性强的数据，先汇聚通道信息再做时序混合
            x = self._channel_attention(x)
            x = self._temporal_mixing(x)
        else:  # 'mc'
            # Temporal MLP → Channel Attention  
            # 适合时序模式先于通道交互的场景
            x = self._temporal_mixing(x)
            x = self._channel_attention(x)
        
        # Feed Forward (通道维度增强)
        x_norm = self.norm_ff(x)
        x = x + self.ff(x_norm)
        
        return x


class Model(nn.Module):
    """
    FreTS-CSformer Hybrid v1
    
    架构设计：
    - 前端 (FreTS): ComplexLinear + FFT 频域全局去噪和长周期特征提取
    - 后端 (CSformer): Two-Stage Attention 动态处理时间/通道依赖
    
    核心优势：
    1. 频域滤波 + 动态注意力的组合
    2. 共享权重降低过拟合风险
    3. Pre-Norm 结构提升训练稳定性
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
        # 混合顺序: 'cm' = Channel→MLP, 'mc' = MLP→Channel
        self.attention_order = getattr(configs, 'attention_order', 'cm')
        
        # 1. RevIN (保持 affine=True)
        self.revin = RevIN(self.enc_in, affine=True)
        
        # 2. Embedding
        self.embeddings = nn.Parameter(torch.randn(1, self.d_model))
        
        # 3. Position Encoding (为 Channel Attention 提供位置信息)
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max(self.seq_len, 512))
        
        # 4. Frequency Learner (FreTS Part)
        self.freq_learner = FrequencyTemporalLearner(self.seq_len, self.d_model)
        
        # 5. CSformer Layers (Channel Attention + Temporal MLP)
        self.csformer_layers = nn.ModuleList([
            CSformerLayer(self.d_model, self.n_heads, self.d_ff, self.seq_len, self.dropout, self.attention_order)
            for _ in range(self.e_layers)
        ])
        
        # 6. Final Norm
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # 7. Projection
        self.projection = nn.Linear(self.seq_len * self.d_model, self.pred_len)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def tokenEmb(self, x):
        # x: [B, N, T] -> [B, N, T, D]
        return x.unsqueeze(3) * self.embeddings

    def forward(self, x):
        # x: [B, T, N]
        
        # 1. RevIN Normalize
        x = self.revin.normalize(x)
        x = x.permute(0, 2, 1)  # [B, N, T]
        
        # 2. Embedding
        x_enc = self.tokenEmb(x)  # [B, N, T, D]
        
        # 3. Frequency Filtering (FreTS: 作为残差修正项)
        # 注意：先做频域滤波，再加 Position Encoding
        # 因为 FFT 对绝对位置不敏感，但 PE 是高频信息，可能被滤波改变
        x_freq = self.freq_learner(x_enc)
        x_feat = x_enc + x_freq
        
        # 4. Add Position Encoding (在 Attention 之前加入)
        B, N, T, D = x_feat.shape
        x_feat = x_feat.reshape(B * N, T, D)
        x_feat = self.pos_enc(x_feat, T)
        x_feat = x_feat.reshape(B, N, T, D)
        
        # 5. CSformer Layers (Dynamic Two-Stage Mixing)
        for layer in self.csformer_layers:
            x_feat = layer(x_feat)
        
        # 6. Final Norm
        x_feat = self.final_norm(x_feat)
        
        # 7. Projection
        B, N, T, D = x_feat.shape
        x_out = x_feat.reshape(B, N, -1)
        x_out = self.projection(x_out)  # [B, N, Pred]
        
        # 8. RevIN Denormalize
        x_out = x_out.permute(0, 2, 1)  # [B, Pred, N]
        x_out = self.revin.denormalize(x_out)
        
        return x_out
