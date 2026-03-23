import torch
import torch.nn as nn
import torch.nn.functional as F

# Registry that maps lower-case expert names to factory lambdas(num_channels, seq_len).
_EXPERT_MAP = {
    'wavelet': lambda nc, sl: WaveletLikeExpert(num_channels=nc),
    'ssa': lambda nc, sl: SSALikeExpert(seq_len=sl, rank=max(8, sl // 6)),
    'vmd': lambda nc, sl: VMDLikeExpert(seq_len=sl, num_modes=4),
    'nlm': lambda nc, sl: NLMLikeExpert(temperature=0.5),
}


class WaveletLikeExpert(nn.Module):
    """Wavelet-like denoiser via depthwise smoothing + soft threshold on high-frequency residual."""

    def __init__(self, num_channels: int, kernel_size: int = 5):
        super(WaveletLikeExpert, self).__init__()
        padding = kernel_size // 2
        self.smoother = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_channels,
            bias=False,
        )
        with torch.no_grad():
            self.smoother.weight.fill_(1.0 / kernel_size)
        self.log_threshold = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        # x: [B, N, T]
        low = self.smoother(x)
        high = x - low
        threshold = torch.exp(self.log_threshold)
        shrunk = torch.sign(high) * F.relu(torch.abs(high) - threshold)
        return low + shrunk


class SSALikeExpert(nn.Module):
    """SSA-like low-rank projector over temporal dimension."""

    def __init__(self, seq_len: int, rank: int = 16):
        super(SSALikeExpert, self).__init__()
        rank = max(4, min(rank, seq_len // 2))
        self.proj_down = nn.Linear(seq_len, rank, bias=False)
        self.proj_up = nn.Linear(rank, seq_len, bias=False)
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)

    def forward(self, x):
        # x: [B, N, T]
        return self.proj_up(self.proj_down(x))


class VMDLikeExpert(nn.Module):
    """VMD-like learnable narrow-band filtering in frequency domain."""

    def __init__(self, seq_len: int, num_modes: int = 4):
        super(VMDLikeExpert, self).__init__()
        self.seq_len = seq_len
        self.freq_len = seq_len // 2 + 1
        self.num_modes = num_modes
        self.mode_gates = nn.Parameter(torch.randn(num_modes, self.freq_len) * 0.02)
        self.mode_mix = nn.Parameter(torch.zeros(num_modes))

    def forward(self, x):
        # x: [B, N, T]
        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')  # [B,N,F]
        gates = torch.sigmoid(self.mode_gates).to(dtype=x_freq.real.dtype)  # [M,F]
        mix = torch.softmax(self.mode_mix, dim=0)  # [M]

        acc = 0.0
        for m in range(self.num_modes):
            filtered = x_freq * gates[m].view(1, 1, -1)
            mode_time = torch.fft.irfft(filtered, n=self.seq_len, dim=-1, norm='ortho')
            acc = acc + mix[m] * mode_time
        return acc


class NLMLikeExpert(nn.Module):
    """NLM-like denoiser using patch-wise temporal self-similarity on pooled signal."""

    def __init__(self, temperature: float = 0.5, blend_init: float = 0.7, patch_size: int = 5):
        super(NLMLikeExpert, self).__init__()
        self.temperature = temperature
        self.blend_logit = nn.Parameter(torch.tensor(blend_init))
        self.patch_size = patch_size

    def forward(self, x):
        # x: [B, N, T]
        pooled = x.mean(dim=1)  # [B,T]
        pad = self.patch_size // 2
        pooled_pad = F.pad(pooled, (pad, pad), mode='reflect')
        patches = pooled_pad.unfold(dimension=1, size=self.patch_size, step=1)  # [B,T,P]
        patches = F.normalize(patches, dim=-1)
        sim = torch.matmul(patches, patches.transpose(1, 2))  # [B,T,T]
        attn = F.softmax(sim / max(self.temperature, 1e-6), dim=-1)
        pooled_denoised = torch.matmul(attn, pooled.unsqueeze(-1)).squeeze(-1)  # [B,T]
        alpha = torch.sigmoid(self.blend_logit)
        return alpha * x + (1 - alpha) * pooled_denoised.unsqueeze(1)


class DenoiseGate(nn.Module):
    """
    Dual-track gate: backward-compatible manual statistics router (manual)
    and adaptive 1D-CNN dual-pool router (neural_v2).

    router_type='manual':
        Uses six hand-crafted statistics (mean, std, kurtosis, variance,
        spectral high-frequency ratio, etc.) derived from the input signal.
        Lightweight and interpretable; good baseline for ablation studies.

    router_type='neural_v2':
        Uses a two-layer 1D-CNN to extract latent features, then applies
        dual-channel pooling (Avg-pool for smooth global context suited to the
        VMD/periodic expert; Max-pool for high-frequency spike detection suited
        to the Wavelet/transient expert). Learns an adaptive routing policy
        at the cost of additional parameters.
    """

    def __init__(self, num_experts: int, num_channels: int = 1,
                 router_type: str = 'manual', hidden_dim: int = 32,
                 topk: int = 2, temperature: float = 1.0):
        super(DenoiseGate, self).__init__()
        self.topk = max(1, topk)
        self.temperature = temperature
        self.router_type = router_type

        if self.router_type == 'manual':
            # Backward-compatible [B, 6] instance-level statistics router.
            self.net = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_experts),
            )
        elif self.router_type == 'neural_v2':
            # Lightweight 1D-CNN with dual-channel pooling (Avg + Max).
            # Avg-pool captures smooth global context (suited for VMD/periodic experts).
            # Max-pool captures high-frequency spikes (suited for Wavelet/transient experts).
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=num_channels, out_channels=hidden_dim,
                          kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim,
                          kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_experts),
            )
        else:
            raise ValueError(f"Unsupported router_type: {router_type!r}. "
                             "Choose 'manual' or 'neural_v2'.")

    def _extract_features(self, x):
        # x: [B, N, T]
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2), unbiased=False)
        abs_mean = x.abs().mean(dim=(1, 2))
        centered = x - x.mean(dim=-1, keepdim=True)
        var = centered.pow(2).mean(dim=(1, 2)) + 1e-8
        kurt = centered.pow(4).mean(dim=(1, 2)) / (var.pow(2))

        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')
        power = x_freq.abs().pow(2)
        freq_len = power.shape[-1]
        cut = max(1, freq_len // 4)
        low_energy = power[..., :cut].mean(dim=(1, 2))
        high_energy = power[..., cut:].mean(dim=(1, 2))
        hf_ratio = high_energy / (low_energy + high_energy + 1e-8)

        feats = torch.stack([mean, std, abs_mean, kurt, var, hf_ratio], dim=-1)
        return feats

    def forward(self, x):
        # x: [B, N, T]
        if self.router_type == 'manual':
            feats = self._extract_features(x)    # [B, 6]
            logits = self.net(feats)              # [B, num_experts]
        else:
            feat_t = self.cnn(x)                  # [B, hidden_dim, T]
            pool_avg = feat_t.mean(dim=-1)        # [B, hidden_dim]
            pool_max = feat_t.max(dim=-1)[0]      # [B, hidden_dim]
            feats = torch.cat([pool_avg, pool_max], dim=-1)  # [B, hidden_dim*2]
            logits = self.fc(feats)               # [B, num_experts]

        logits = logits / max(self.temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)         # [B, num_experts]

        if self.topk >= probs.shape[-1]:
            return probs

        top_vals, top_idx = torch.topk(probs, k=self.topk, dim=-1)
        sparse = torch.zeros_like(probs)
        sparse.scatter_(1, top_idx, top_vals)
        sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse


class DenoiseMoE(nn.Module):
    """Differentiable MoE denoiser with load-balance and diversity auxiliaries."""

    def __init__(
        self,
        seq_len: int,
        num_channels: int,
        num_experts: int = 4,
        topk: int = 2,
        gate_hidden: int = 32,
        gate_temp: float = 1.0,
        router_type: str = 'manual',
        expert_names: str = '',
    ):
        super(DenoiseMoE, self).__init__()

        # Build expert list from name registry when provided; fall back to legacy ordering.
        if expert_names:
            names = [n.strip().lower() for n in expert_names.split(',') if n.strip()]
            for n in names:
                if n not in _EXPERT_MAP:
                    raise ValueError(
                        f"Unknown expert name {n!r}. "
                        f"Valid options: {list(_EXPERT_MAP.keys())}"
                    )
            self.experts = nn.ModuleList(
                [_EXPERT_MAP[n](num_channels, seq_len) for n in names]
            )
            self.expert_names = names
        else:
            # Legacy default ordering: Wavelet, SSA, VMD, NLM
            default_experts = [
                WaveletLikeExpert(num_channels=num_channels),
                SSALikeExpert(seq_len=seq_len, rank=max(8, seq_len // 6)),
                VMDLikeExpert(seq_len=seq_len, num_modes=4),
                NLMLikeExpert(temperature=0.5),
            ]
            if num_experts < len(default_experts):
                default_experts = default_experts[:num_experts]
            self.experts = nn.ModuleList(default_experts)
            self.expert_names = ['wavelet', 'ssa', 'vmd', 'nlm'][:len(self.experts)]

        self.num_experts = len(self.experts)
        self.gate = DenoiseGate(
            num_experts=self.num_experts,
            num_channels=num_channels,
            router_type=router_type,
            hidden_dim=gate_hidden,
            topk=topk,
            temperature=gate_temp,
        )
        self.output_scale = nn.Parameter(torch.ones(self.num_experts))
        self.gate_probs_store = None  # Populated in forward(); used for behavior logging.
        self._latest_stats = {}
        self.reset_stats()

    def reset_stats(self):
        self._stats_count = 0
        self._usage_sum = None
        self._usage_sq_sum = None
        self._entropy_sum = 0.0
        self._entropy_sq_sum = 0.0
        self._topk_sum = 0.0
        self._topk_sq_sum = 0.0
        self._latest_stats = {}

    @staticmethod
    def _mean_std_ci(sum_val, sq_sum_val, count):
        if count <= 0:
            return 0.0, 0.0, 0.0
        mean = sum_val / count
        var = max(sq_sum_val / count - mean * mean, 0.0)
        std = var ** 0.5
        ci95 = 1.96 * std / (count ** 0.5)
        return float(mean), float(std), float(ci95)

    def _load_balance_loss(self, weights):
        usage = weights.mean(dim=0)
        target = torch.full_like(usage, 1.0 / usage.numel())
        return F.mse_loss(usage, target)

    def _diversity_loss(self, expert_outputs):
        reps = []
        for out in expert_outputs:
            reps.append(out.mean(dim=0).reshape(-1))
        reps = torch.stack(reps, dim=0)  # [E, NT]
        reps = F.normalize(reps, dim=-1)
        sim = torch.matmul(reps, reps.t())
        eye = torch.eye(sim.shape[0], device=sim.device)
        off_diag = sim * (1 - eye)
        return off_diag.pow(2).mean()

    def forward(self, x):
        # x: [B,N,T]
        weights = self.gate(x)  # [B,E]
        # Store latest routing probabilities for external logging (e.g. test phase CSV).
        self.gate_probs_store = weights
        expert_outputs = [expert(x) for expert in self.experts]
        scaled_outputs = []
        for idx, out in enumerate(expert_outputs):
            scaled_outputs.append(self.output_scale[idx] * out)

        stack = torch.stack(scaled_outputs, dim=1)  # [B,E,N,T]
        y = (weights.unsqueeze(-1).unsqueeze(-1) * stack).sum(dim=1)

        lb_loss = self._load_balance_loss(weights)
        div_loss = self._diversity_loss(scaled_outputs)

        with torch.no_grad():
            mean_usage = weights.mean(dim=0)
            entropy = (-weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            topk_hit = (weights > 0).float().sum(dim=-1).float().mean()

            if self._usage_sum is None:
                self._usage_sum = torch.zeros_like(mean_usage)
                self._usage_sq_sum = torch.zeros_like(mean_usage)

            self._stats_count += 1
            self._usage_sum = self._usage_sum + mean_usage
            self._usage_sq_sum = self._usage_sq_sum + mean_usage.pow(2)
            self._entropy_sum += float(entropy.detach().cpu())
            self._entropy_sq_sum += float(entropy.detach().cpu()) ** 2
            self._topk_sum += float(topk_hit.detach().cpu())
            self._topk_sq_sum += float(topk_hit.detach().cpu()) ** 2

            util = (mean_usage > 1e-3).float().mean()
            self._latest_stats = {
                'expert_usage': mean_usage.detach().cpu(),
                'routing_entropy': float(entropy.detach().cpu()),
                'expert_utilization': float(util.detach().cpu()),
                'topk_active': float(topk_hit.detach().cpu()),
            }

        aux = {
            'lb_loss': lb_loss,
            'div_loss': div_loss,
            'weights': weights,
        }
        return y, aux

    def get_latest_stats(self):
        return self._latest_stats

    def get_aggregated_stats(self):
        if self._stats_count <= 0:
            return {}

        usage_mean = (self._usage_sum / self._stats_count).detach().cpu()
        usage_var = (self._usage_sq_sum / self._stats_count - (self._usage_sum / self._stats_count).pow(2)).clamp(min=0.0).detach().cpu()
        usage_std = usage_var.sqrt()
        usage_ci95 = 1.96 * usage_std / (self._stats_count ** 0.5)

        entropy_mean, entropy_std, entropy_ci95 = self._mean_std_ci(
            self._entropy_sum, self._entropy_sq_sum, self._stats_count
        )
        topk_mean, topk_std, topk_ci95 = self._mean_std_ci(
            self._topk_sum, self._topk_sq_sum, self._stats_count
        )

        util = float((usage_mean > 1e-3).float().mean())

        return {
            'batch_count': int(self._stats_count),
            'expert_usage_mean': usage_mean,
            'expert_usage_std': usage_std,
            'expert_usage_ci95': usage_ci95,
            'routing_entropy_mean': entropy_mean,
            'routing_entropy_std': entropy_std,
            'routing_entropy_ci95': entropy_ci95,
            'topk_active_mean': topk_mean,
            'topk_active_std': topk_std,
            'topk_active_ci95': topk_ci95,
            'expert_utilization': util,
        }
