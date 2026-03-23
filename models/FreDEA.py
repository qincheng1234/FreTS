import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.moe_denoiser import DenoiseMoE


class RevIN(nn.Module):
    """[标准组件] Reversible Instance Normalization"""
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


class PositionalEncoding(nn.Module):
    """位置编码"""
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
    """[核心创新] 时序外部注意力"""
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
    """[核心创新] 通道外部注意力"""
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
    [V25] TEABlock with Sigmoid Gated CEA
    
    Gate 被 Sigmoid 约束在 [0, 1] 之间，且初始化接近 0。
    这允许模型自动决定是否需要通道混合。
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
        
        # Sigmoid Gate: sigmoid(-5.0) ≈ 0.0067, 从几乎纯 CI 开始
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
            gate = torch.sigmoid(self.gate_logit)
            x = residual + gate * self.dropout(self.cea(x))
        
        residual = x
        x = self.norm3(x)
        x = residual + self.ff(x)
        return x


# =============================================================================
# V25_Refined: 条件分支频域分解 (带 Stepness 参数)
# =============================================================================

class ConditionalFreqDecomp(nn.Module):
    """
    [V25_Refined] 条件分支频域分解 (带 Stepness)
    
    改进：
    1. 条件分支：小维度共享 cutoff，高维度通道独立
    2. Stepness 参数：控制滤波器陡峭程度 sigmoid((cutoff-f)*stepness)
       - stepness 大：陡峭边界 (类似硬截断)
       - stepness 小：平滑过渡 (软滤波)
    """
    def __init__(self, seq_len, enc_in):
        super(ConditionalFreqDecomp, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.high_dim = enc_in > 100
        freq_len = seq_len // 2 + 1
        
        self.register_buffer('freq_indices', torch.arange(freq_len).float())
        
        # [V25_Refined] 条件分支 cutoff + stepness
        if self.high_dim:
            # Electricity: 通道独立
            self.cutoff = nn.Parameter(torch.ones(1, enc_in, 1) * 3.0)
            self.stepness = nn.Parameter(torch.ones(1, enc_in, 1) * 1.0)
        else:
            # ETTm1/Weather: 共享参数
            self.cutoff = nn.Parameter(torch.tensor(3.0))
            self.stepness = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [B, N, T]
        B, N, T = x.shape
        
        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')
        
        # 生成 Mask (带 stepness)
        # 公式: sigmoid((cutoff - idx) * stepness)
        if self.high_dim:
            delta = self.cutoff - self.freq_indices.view(1, 1, -1)
            mask = torch.sigmoid(delta * self.stepness)
        else:
            delta = self.cutoff - self.freq_indices
            mask = torch.sigmoid(delta * self.stepness).view(1, 1, -1)
        
        # 频域分解
        x_trend_freq = x_freq * mask
        x_seasonal_freq = x_freq * (1 - mask)
        
        # iFFT
        x_trend = torch.fft.irfft(x_trend_freq, n=T, dim=-1, norm='ortho')
        x_seasonal = torch.fft.irfft(x_seasonal_freq, n=T, dim=-1, norm='ortho')
        
        return x_seasonal, x_trend


# =============================================================================
# V25 模型: Channel-Adaptive FreDEA (Stabilized Single-Tower)
# =============================================================================

class Model(nn.Module):
    """
    FreDEA V25: Channel-Adaptive (Stabilized Single-Tower)
    
    核心设计：
    1. 回归 V24 的单塔架构 (避免 V26 双塔过拟合)
    2. 升级分解模块为通道自适应 (解决 Electricity 321 维问题)
    3. 保留所有 V24_Refined 优化 (自动 CI、自适应 Dropout、门控融合)
    
    这是折中的"黄金方案"：保持 ETTm 优势，同时修复 Electricity 短板。
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
        
        self.ablation_tea = getattr(configs, 'ablation_tea', 0)
        self.ablation_cea = getattr(configs, 'ablation_cea', 0)
        self.moe_enable = int(getattr(configs, 'moe_enable', 0))
        self.moe_position = getattr(configs, 'moe_position', 'post_decomp')
        self.moe_stats_enable = int(getattr(configs, 'moe_stats_enable', 1))

        if self.moe_enable and self.moe_position != 'post_decomp':
            raise ValueError(f"Unsupported moe_position={self.moe_position}. Only 'post_decomp' is supported.")
        
        # -----------------------------------------------------------
        # [自动 CI 策略] 高维数据自动关闭通道混合
        # -----------------------------------------------------------
        # 即使分解是自适应的，300+ 维的通道混合仍有噪声风险
        if self.enc_in > 100:
            self.ablation_cea = 1
        
        # 1. RevIN
        self.revin = RevIN(self.enc_in, affine=getattr(configs, 'rev_affine', True))
        
        # 2. [V25_Refined] Conditional Frequency Decomposition (带 Stepness)
        self.decomposition = ConditionalFreqDecomp(self.seq_len, self.enc_in)
        
        # -----------------------------------------------------------
        # [Branch 1] Trend Modeling (Simple Linear)
        # -----------------------------------------------------------
        # 保持简单，避免趋势分支过拟合
        self.trend_model = nn.Linear(self.seq_len, self.pred_len)
        
        # -----------------------------------------------------------
        # [Branch 2] Seasonal Modeling (Deep TEA Encoder)
        # -----------------------------------------------------------
        self.embeddings = nn.Parameter(torch.randn(1, self.d_model))
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max(self.seq_len, 512))
        
        self.tea_blocks = nn.ModuleList([
            TEABlock(self.d_model, self.seq_len, self.enc_in, self.d_ff, 
                     memory_size=self.memory_size, dropout=self.dropout,
                     ablation_tea=self.ablation_tea, ablation_cea=self.ablation_cea)
            for _ in range(self.e_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # Time Projection
        self.seasonal_time_proj = nn.Linear(self.seq_len, self.pred_len)
        
        # -----------------------------------------------------------
        # [自适应 Dropout] 长序列增强正则化 (修复版)
        # 336+ 步都需要更强的正则化
        # -----------------------------------------------------------
        if self.pred_len >= 720:
            time_dropout = max(0.35, self.dropout)
        elif self.pred_len >= 336:
            time_dropout = max(0.25, self.dropout)
        else:
            time_dropout = self.dropout
        self.dropout_time = nn.Dropout(time_dropout)
        
        # Feature MLP - [关键修复] 使用 time_dropout 而非 self.dropout
        self.seasonal_out_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(time_dropout),  # <-- 修复：长序列时也强正则化
            nn.Linear(self.d_model * 2, 1)
        )
        
        # -----------------------------------------------------------
        # [融合门控] 可学习的趋势/季节权重
        # fusion_init: 0.0 = 平衡, 3.0 = 偏向趋势 (sigmoid(3.0)≈0.95)
        # -----------------------------------------------------------
        fusion_init = getattr(configs, 'fusion_init', 0.0)
        self.fusion_logit = nn.Parameter(torch.tensor(fusion_init))

        if self.moe_enable and self.moe_position == 'post_decomp':
            self.denoise_moe = DenoiseMoE(
                seq_len=self.seq_len,
                num_channels=self.enc_in,
                num_experts=int(getattr(configs, 'moe_num_experts', 4)),
                topk=int(getattr(configs, 'moe_topk', 2)),
                gate_hidden=int(getattr(configs, 'moe_gate_hidden', 32)),
                gate_temp=float(getattr(configs, 'moe_gate_temp', 1.0)),
            )
        else:
            self.denoise_moe = None

        self._moe_aux = None
        
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

    def get_moe_aux_losses(self):
        if self.denoise_moe is None or self._moe_aux is None:
            zero = self.fusion_logit.new_zeros(())
            return zero, zero
        return self._moe_aux['lb_loss'], self._moe_aux['div_loss']

    def get_moe_stats(self):
        if self.denoise_moe is None:
            return {}
        return self.denoise_moe.get_aggregated_stats()

    def reset_moe_stats(self):
        if self.denoise_moe is not None:
            self.denoise_moe.reset_stats()

    def forward(self, x):
        # x: [B, T, N]
        
        # 1. RevIN Normalize
        x = self.revin.normalize(x)
        x = x.permute(0, 2, 1)  # [B, N, T]
        
        # 2. [V25_Refined] Decomposition (带 Stepness)
        x_seasonal, x_trend = self.decomposition(x)

        if self.denoise_moe is not None:
            x_seasonal, self._moe_aux = self.denoise_moe(x_seasonal)
        else:
            self._moe_aux = None
        
        # ==========================================
        # Branch 1: Trend Prediction (Simple Linear)
        # ==========================================
        pred_trend = self.trend_model(x_trend)  # [B, N, Pred]
        
        # ==========================================
        # Branch 2: Seasonal Prediction (Deep TEA)
        # ==========================================
        x_enc = self.tokenEmb(x_seasonal)  # [B, N, T, D]
        x_feat = x_enc
            
        B, N, T, D = x_feat.shape
        x_feat = x_feat.reshape(B * N, T, D)
        x_feat = self.pos_enc(x_feat, T)
        x_feat = x_feat.reshape(B, N, T, D)
        
        for block in self.tea_blocks:
            x_feat = block(x_feat)
        
        x_feat = self.final_norm(x_feat)  # [B, N, T, D]
        
        # Time Projection: T -> Pred
        x_feat_t = x_feat.permute(0, 1, 3, 2)  # [B, N, D, T]
        x_proj_t = self.seasonal_time_proj(x_feat_t)  # [B, N, D, Pred]
        x_proj_t = self.dropout_time(x_proj_t)
        
        # Feature MLP: D -> 1
        x_proj_t = x_proj_t.permute(0, 1, 3, 2)  # [B, N, Pred, D]
        pred_seasonal = self.seasonal_out_mlp(x_proj_t).squeeze(-1)  # [B, N, Pred]
        
        # ==========================================
        # Fusion with Learnable Gate
        # ==========================================
        gate = torch.sigmoid(self.fusion_logit)
        dec_out = gate * pred_trend + (1 - gate) * pred_seasonal
        
        
        # 3. RevIN Denormalize
        x_out = dec_out.permute(0, 2, 1)  # [B, Pred, N]
        x_out = self.revin.denormalize(x_out)
        
        return x_out
