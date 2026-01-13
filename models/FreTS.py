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


class RMSNorm(nn.Module):
    """
    [新增] RMSNorm: 比 LayerNorm 更稳定，适合 Linear Attention 的 QK 归一化
    用于防止高维输入导致的梯度爆炸/消失
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight



# ============================================
# TEA + Channel MLP 架构 (MEAformer 风格)
# ============================================

class TemporalExternalAttention(nn.Module):
    """
    时间外部注意力 (Temporal External Attention, TEA)
    
    论文参考: 
    - "Beyond Self-attention: External Attention using Two Linear Layers"
    - "MEAformer: An all-MLP transformer with temporal external attention"
    
    核心思想：
    - 用两个可学习的线性层 (M_k, M_v) 替代 Self-Attention 的 Q@K^T 和 A@V
    - 记忆单元 M 在整个数据集上共享，捕获跨样本的全局时序模式
    
    复杂度: O(L * S)，其中 S << L，实现线性复杂度
    """
    def __init__(self, d_model, memory_size=64):
        """
        Args:
            d_model: 输入特征维度
            memory_size (S): 外部记忆单元数量，控制模型的"记忆容量"
        """
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        
        # 外部记忆矩阵
        # M_k: 键记忆，用于计算注意力权重
        # M_v: 值记忆，用于特征重构
        self.M_k = nn.Linear(d_model, memory_size, bias=False)
        self.M_v = nn.Linear(memory_size, d_model, bias=False)
        
        # 初始化
        nn.init.xavier_uniform_(self.M_k.weight)
        nn.init.xavier_uniform_(self.M_v.weight)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, T, D] - 输入特征
        Returns:
            out: [B, N, T, D] - 输出特征
        """
        B, N, T, D = x.shape
        
        # Reshape: [B, N, T, D] -> [B*N, T, D]
        x = x.reshape(B * N, T, D)
        
        # Step 1: 计算注意力权重
        # attn = x @ M_k^T -> [B*N, T, S]
        attn = self.M_k(x)
        
        # Step 2: 双重归一化 (Double Normalization)
        # 这是 External Attention 的关键：先 Softmax，再 L1 归一化
        attn = F.softmax(attn, dim=-1)  # 对 S 维度 Softmax
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)  # 对 T 维度 L1 归一化
        
        # Step 3: 特征重构
        # out = attn @ M_v -> [B*N, T, D]
        out = self.M_v(attn)
        
        # Reshape back: [B*N, T, D] -> [B, N, T, D]
        out = out.reshape(B, N, T, D)
        
        return out


class ChannelMLP(nn.Module):
    """
    自适应 Channel MLP (Adaptive Channel MLP)
    
    根据通道数自动选择最优结构：
    - N > 100 (如 Traffic): Bottleneck 结构 (N -> N/16 -> N)
      利用 Low-Rank 属性，防止过拟合
    - N <= 100 (如 ETT, Weather): Expansion 结构 (N -> N*2 -> N)
      保留足够的表达能力
    
    这样可以同时适应高维和低维数据集
    """
    def __init__(self, num_channels, dropout=0.1):
        """
        Args:
            num_channels: 输入通道数 (N)
            dropout: Dropout 比率
        """
        super().__init__()
        self.num_channels = num_channels
        
        # 自适应选择隐藏层维度
        if num_channels > 100:
            # 高维数据 (如 Traffic N=862): 使用 Bottleneck
            # 瓶颈维度 = N/16，最小为 32
            self.hidden_dim = max(32, num_channels // 16)
            self.mode = 'bottleneck'
        else:
            # 低维数据 (如 ETT N=7, Weather N=21): 使用 Expansion
            # 扩展维度 = N*2，最小为 16
            self.hidden_dim = max(16, num_channels * 4)
            self.mode = 'expansion'
        
        # MLP 层
        self.fc1 = nn.Linear(num_channels, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_channels)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, T, D] - 输入特征
        Returns:
            out: [B, N, T, D] - 输出特征
        """
        B, N, T, D = x.shape
        
        # 转置：[B, N, T, D] -> [B, T, D, N]
        x = x.permute(0, 2, 3, 1)
        
        # MLP: N -> hidden -> N
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # 转置回来：[B, T, D, N] -> [B, N, T, D]
        x = x.permute(0, 3, 1, 2)
        
        return x



class TEABlock(nn.Module):
    """
    TEA + Adaptive Channel MLP 组合模块
    
    架构: Temporal External Attention -> Adaptive Channel MLP -> FFN
    
    Channel MLP 自动选择:
    - N > 100: Bottleneck (N/16)
    - N <= 100: Expansion (N*4)
    """
    def __init__(self, d_model, num_channels, d_ff, memory_size=64, dropout=0.1):
        super().__init__()
        
        # 1. Temporal External Attention
        self.tea = TemporalExternalAttention(d_model, memory_size=memory_size)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Adaptive Channel MLP (自动选择 Bottleneck 或 Expansion)
        self.channel_mlp = ChannelMLP(num_channels, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. Feed Forward Network
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
        """
        Args:
            x: [B, N, T, D] - 输入特征
        Returns:
            out: [B, N, T, D] - 输出特征
        """
        # 1. Temporal External Attention (Pre-Norm)
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.tea(x))
        
        # 2. Channel MLP (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.channel_mlp(x))
        
        # 3. Feed Forward (Pre-Norm)
        residual = x
        x = self.norm3(x)
        x = residual + self.ff(x)
        
        return x



class Model(nn.Module):
    """
    FreTS + TEA-MLP 混合架构
    
    架构设计：
    - 前端 (FreTS): ComplexLinear + FFT 频域全局去噪和长周期特征提取
    - 后端 (TEA + C-MLP): Temporal External Attention + Channel MLP
    
    核心优势：
    1. 频域滤波 + 外部记忆的组合
    2. TEA 以 O(L) 线性复杂度处理长序列
    3. ChannelMLP 高效处理多变量交互
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
        # TEA 外部记忆单元数量
        self.memory_size = getattr(configs, 'memory_size', 64)
        
        # 1. RevIN (可配置 affine, Traffic 建议设为 False)
        self.affine = getattr(configs, 'rev_affine', True) 
        self.revin = RevIN(self.enc_in, affine=self.affine)
        
        # 2. Embedding
        self.embeddings = nn.Parameter(torch.randn(1, self.d_model))
        
        # 3. Position Encoding (为时间维度提供位置信息)
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max(self.seq_len, 512))
        
        # 4. Frequency Learner (FreTS Part)
        self.freq_learner = FrequencyTemporalLearner(self.seq_len, self.d_model)
        
        # 5. TEA Blocks (Temporal External Attention + Channel MLP)
        self.tea_blocks = nn.ModuleList([
            TEABlock(self.d_model, self.enc_in, self.d_ff, memory_size=self.memory_size, dropout=self.dropout)
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
        
        # 5. TEA Blocks (Temporal EA + Channel MLP)
        for block in self.tea_blocks:
            x_feat = block(x_feat)
        
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
