"""Sparse spiking backbone and the action-conditioned SpikeWorld model.

The module mirrors the architecture used for the paper checkpoint.  It has one
shared temporal core, modality-specific input/output adapters, coupled semantic
readouts, and a two-token action-dynamics path through that same core.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STREAM_DIMS = {
    "audio_shd": 224,
    "audio_ssc": 285,
    "text": 40,
    "image": 40,
    "video": 896,
}
STREAM_MODALITY = {
    "audio_shd": "audio",
    "audio_ssc": "audio",
    "text": "text",
    "image": "image",
    "video": "video",
}
MODALITY_IDS = {"audio": 0, "text": 1, "image": 2, "video": 3}
SEMANTIC_CLASSES = {
    "audio_shd": 20,
    "audio_ssc": 35,
    "retinal": 10,
    "video": 42,
}
TASKS = ("reach-v3", "push-v3", "pick-place-v3", "door-open-v3")
STATE_DIM = 39
ACTION_DIM = 4


@dataclass(frozen=True)
class ModelConfig:
    sequence_length: int = 16
    width: int = 96
    layers: int = 2
    heads: int = 4
    ff_dim: int = 192
    dropout: float = 0.0
    beta: float = 0.85
    attention_topk: int = 4


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return (value > 0).to(value.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        return gradient / (10.0 * value.abs() + 1.0).square()


class LIFSequence(nn.Module):
    def __init__(self, size: int, beta: float = 0.85, threshold: float = 1.0):
        super().__init__()
        self.size = size
        self.beta = beta
        self.threshold = threshold

    def forward(
        self, current: torch.Tensor, return_trace: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = current.new_zeros(current.shape[0], current.shape[-1])
        spikes, trace = [], []
        for step in range(current.shape[1]):
            membrane = self.beta * membrane + current[:, step]
            spike = _Spike.apply(membrane - self.threshold)
            membrane = membrane - spike.detach() * self.threshold
            spikes.append(spike)
            if return_trace:
                trace.append(membrane)
        final = torch.stack(trace, dim=1) if return_trace else membrane
        return torch.stack(spikes, dim=1), final


class SpikeStats:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def add(self, name: str, value: torch.Tensor) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value.detach().sum())
        self.counts[name] = self.counts.get(name, 0.0) + value.numel()

    def scalar(self, name: str, value: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + value
        self.counts[name] = self.counts.get(name, 0.0) + 1.0

    def report(self) -> dict[str, float]:
        return {name: self.sums[name] / max(self.counts[name], 1.0) for name in self.sums}


class SparseSpikingAttention(nn.Module):
    """Causal local attention that materializes T x K rather than T x T QK pairs."""

    def __init__(self, width: int, heads: int, dropout: float, beta: float, topk: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.embed_dim = width
        self.heads = heads
        self.head_dim = width // heads
        self.attention_mode = "local_sparse_vectorized"
        self.attention_topk = topk
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.q_lif = LIFSequence(width, beta)
        self.k_lif = LIFSequence(width, beta)
        self.v_lif = LIFSequence(width, beta)
        self.out = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        _causal_mask: torch.Tensor,
        stats: SpikeStats | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_spike, q_mem = self.q_lif(self.q_proj(tokens), return_trace=True)
        k_spike, k_mem = self.k_lif(self.k_proj(tokens), return_trace=True)
        v_spike, v_mem = self.v_lif(self.v_proj(tokens), return_trace=True)
        if stats is not None:
            stats.add("q", q_spike)
            stats.add("k", k_spike)
            stats.add("v", v_spike)

        batch, steps, _ = q_spike.shape
        shape = (batch, steps, self.heads, self.head_dim)
        q = q_spike.view(shape).transpose(1, 2)
        k = k_spike.view(shape).transpose(1, 2)
        v = v_spike.view(shape).transpose(1, 2)
        q_gate = torch.sigmoid(q_mem).view(shape).mean(-1).transpose(1, 2)
        k_gate = torch.sigmoid(k_mem).view(shape).mean(-1).transpose(1, 2)

        keep = max(1, min(self.attention_topk, steps))
        query = torch.arange(steps, device=q.device)[:, None]
        offsets = torch.arange(keep - 1, -1, -1, device=q.device)[None, :]
        valid = query - offsets >= 0
        keys = F.pad(k, (0, 0, keep - 1, 0)).unfold(2, keep, 1)
        keys = keys.permute(0, 1, 2, 4, 3)
        values = F.pad(v, (0, 0, keep - 1, 0)).unfold(2, keep, 1)
        values = values.permute(0, 1, 2, 4, 3)
        selected_gate = F.pad(k_gate, (keep - 1, 0)).unfold(2, keep, 1)
        logits = (q.unsqueeze(-2) * keys).sum(-1) / math.sqrt(self.head_dim)
        logits = logits * (0.25 + q_gate.unsqueeze(-1) * selected_gate)
        logits = logits.masked_fill(~valid.view(1, 1, steps, keep), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        output = (self.drop(weights).unsqueeze(-1) * values).sum(-2)
        if stats is not None:
            stats.scalar("attention_density", float(valid.sum()) / (steps * steps))
        output = output.transpose(1, 2).contiguous().view(batch, steps, self.embed_dim)
        return self.out(output), (q_mem + k_mem + v_mem) / 3.0


class SpikingBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_norm = nn.LayerNorm(cfg.width)
        self.token_lif = LIFSequence(cfg.width, cfg.beta)
        self.attn = SparseSpikingAttention(
            cfg.width, cfg.heads, cfg.dropout, cfg.beta, cfg.attention_topk
        )
        self.ff_norm = nn.LayerNorm(cfg.width)
        self.ff_in = nn.Linear(cfg.width, cfg.ff_dim)
        self.ff_lif = LIFSequence(cfg.ff_dim, cfg.beta)
        self.ff_out = nn.Linear(cfg.ff_dim, cfg.width)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self, values: torch.Tensor, causal_mask: torch.Tensor, stats: SpikeStats | None
    ) -> torch.Tensor:
        token_spikes, _ = self.token_lif(self.token_norm(values))
        if stats is not None:
            stats.add("token", token_spikes)
        attention, _ = self.attn(token_spikes, causal_mask, stats)
        values = values + self.drop(attention)
        ff_spikes, _ = self.ff_lif(self.ff_in(self.ff_norm(values)))
        if stats is not None:
            stats.add("ff", ff_spikes)
        return values + self.drop(self.ff_out(ff_spikes))


class MultimodalBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.stems = nn.ModuleDict(
            {
                "audio": nn.Sequential(
                    nn.LayerNorm(STREAM_DIMS["audio_ssc"]),
                    nn.Linear(STREAM_DIMS["audio_ssc"], cfg.width),
                ),
                "retinal": nn.Sequential(
                    nn.LayerNorm(STREAM_DIMS["image"]),
                    nn.Linear(STREAM_DIMS["image"], cfg.width),
                ),
                "video": nn.Sequential(
                    nn.LayerNorm(STREAM_DIMS["video"]),
                    nn.Linear(STREAM_DIMS["video"], cfg.width),
                ),
            }
        )
        self.position = nn.Parameter(torch.zeros(1, cfg.sequence_length, cfg.width))
        self.modality = nn.Embedding(len(MODALITY_IDS), cfg.width)
        self.input_lif = LIFSequence(cfg.width, cfg.beta)
        self.shared_core = nn.ModuleList(SpikingBlock(cfg) for _ in range(cfg.layers))
        self.output_norm = nn.LayerNorm(cfg.width)
        self.next_heads = nn.ModuleDict(
            {
                "audio_shd": nn.Linear(cfg.width, STREAM_DIMS["audio_shd"]),
                "audio_ssc": nn.Linear(cfg.width, STREAM_DIMS["audio_ssc"]),
                "retinal": nn.Linear(cfg.width, STREAM_DIMS["image"]),
                "video": nn.Linear(cfg.width, STREAM_DIMS["video"]),
            }
        )
        self.video_semantic_head = nn.Linear(cfg.width, 64)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(cfg.sequence_length, cfg.sequence_length, dtype=torch.bool),
                diagonal=1,
            ),
        )
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.modality.weight, std=0.02)
        for head in self.next_heads.values():
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    @staticmethod
    def stem_name(stream: str) -> str:
        if stream.startswith("audio_"):
            return "audio"
        if stream in {"text", "image"}:
            return "retinal"
        if stream == "video":
            return "video"
        raise KeyError(stream)

    @staticmethod
    def head_name(stream: str) -> str:
        return "retinal" if stream in {"text", "image"} else stream

    def encode(
        self, values: torch.Tensor, stream: str, collect_stats: bool = False
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        if values.ndim != 3 or values.shape[1] > self.cfg.sequence_length:
            raise ValueError("expected [batch,time,features] with time <= sequence_length")
        source = values
        if stream == "audio_shd":
            source = F.pad(values, (0, STREAM_DIMS["audio_ssc"] - STREAM_DIMS[stream]))
        hidden = self.stems[self.stem_name(stream)](source)
        hidden = hidden + self.position[:, : hidden.shape[1]]
        modality = MODALITY_IDS[STREAM_MODALITY[stream]]
        hidden = hidden + self.modality.weight[modality].view(1, 1, -1)
        stats = SpikeStats() if collect_stats else None
        spikes, _ = self.input_lif(hidden)
        if stats is not None:
            stats.add("input", spikes)
        hidden = hidden + spikes
        mask = self.causal_mask[: hidden.shape[1], : hidden.shape[1]]
        for block in self.shared_core:
            hidden = block(hidden, mask, stats)
        hidden = self.output_norm(hidden)
        return hidden, None if stats is None else stats.report()

    def predict_next(
        self, values: torch.Tensor, stream: str, collect_stats: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
        hidden, stats = self.encode(values, stream, collect_stats)
        delta = self.next_heads[self.head_name(stream)](hidden[:, :-1])
        return values[:, :-1] + delta, hidden, stats


class PredictiveSystem(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.backbone = MultimodalBackbone(cfg)
        self.binding_projection = nn.Sequential(
            nn.LayerNorm(cfg.width),
            nn.Linear(cfg.width, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )

    def binding_representation(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.binding_projection(hidden.mean(1))


class FrozenLinearReadout(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self.linear = nn.Linear(input_dim, classes)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        standardized = ((values - self.mean) / self.scale).clamp(-20.0, 20.0)
        return self.linear(standardized)


class FrozenRawSemanticPath(nn.Module):
    def __init__(self, sequence_length: int):
        super().__init__()
        classes = {
            "audio_shd": 20,
            "audio_ssc": 35,
            "text": 10,
            "image": 10,
            "video": 42,
        }
        self.readouts = nn.ModuleDict(
            {
                stream: FrozenLinearReadout(sequence_length * STREAM_DIMS[stream], n)
                for stream, n in classes.items()
            }
        )

    def forward(self, values: torch.Tensor, stream: str) -> torch.Tensor:
        return self.readouts[stream](values.flatten(start_dim=1))


class SemanticWorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.predictive = PredictiveSystem(cfg)
        semantic_dim = cfg.sequence_length * cfg.width
        self.coupled_semantic_norm = nn.LayerNorm(semantic_dim)
        self.coupled_semantic_dropout = nn.Dropout(0.5)
        self.coupled_semantic_heads = nn.ModuleDict(
            {
                name: nn.Linear(semantic_dim, classes)
                for name, classes in SEMANTIC_CLASSES.items()
            }
        )
        self.stable_semantic = FrozenRawSemanticPath(cfg.sequence_length)

    @staticmethod
    def semantic_key(stream: str) -> str:
        return "retinal" if stream in {"image", "text"} else stream

    def coupled_semantic_logits(self, hidden: torch.Tensor, stream: str) -> torch.Tensor:
        features = self.coupled_semantic_norm(hidden.flatten(start_dim=1))
        return self.coupled_semantic_heads[self.semantic_key(stream)](
            self.coupled_semantic_dropout(features)
        )

    def semantic_logits(self, values: torch.Tensor, stream: str) -> torch.Tensor:
        return self.stable_semantic(values, stream)


class SpikeWorld(nn.Module):
    """Full paper model: multimodal state plus an action-conditioned transition head."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        delta_mean: np.ndarray | torch.Tensor | None = None,
        delta_scale: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = ModelConfig() if config is None else config
        self.config = config
        self.anchored = SemanticWorldModel(config)
        width = config.width
        self.state_stem = nn.Sequential(
            nn.LayerNorm(STATE_DIM), nn.Linear(STATE_DIM, width)
        )
        self.action_stem = nn.Sequential(
            nn.LayerNorm(ACTION_DIM), nn.Linear(ACTION_DIM, width)
        )
        self.task_embedding = nn.Embedding(len(TASKS), width)
        self.token_type = nn.Embedding(2, width)
        self.control_modality = nn.Parameter(torch.zeros(width))
        self.next_state_head = nn.Linear(width, STATE_DIM)
        mean = torch.zeros(STATE_DIM) if delta_mean is None else torch.as_tensor(delta_mean)
        scale = (
            torch.ones(STATE_DIM) if delta_scale is None else torch.as_tensor(delta_scale)
        )
        self.register_buffer("delta_mean", mean.float())
        self.register_buffer("delta_scale", scale.float())
        nn.init.normal_(self.control_modality, std=0.02)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.normal_(self.token_type.weight, std=0.02)
        nn.init.normal_(self.next_state_head.weight, std=1e-3)
        nn.init.zeros_(self.next_state_head.bias)

    @property
    def backbone(self) -> MultimodalBackbone:
        return self.anchored.predictive.backbone

    def control_hidden(
        self, state: torch.Tensor, action: torch.Tensor, task: torch.Tensor
    ) -> torch.Tensor:
        task_token = self.task_embedding(task)
        state_token = (
            self.state_stem(state)
            + task_token
            + self.token_type.weight[0]
            + self.control_modality
        )
        action_token = (
            self.action_stem(action)
            + task_token
            + self.token_type.weight[1]
            + self.control_modality
        )
        hidden = torch.stack((state_token, action_token), dim=1)
        hidden = hidden + self.backbone.position[:, :2]
        spikes, _ = self.backbone.input_lif(hidden)
        hidden = hidden + spikes
        mask = self.backbone.causal_mask[:2, :2]
        for block in self.backbone.shared_core:
            hidden = block(hidden, mask, None)
        return self.backbone.output_norm(hidden)[:, -1]

    def predict_standardized(
        self, state: torch.Tensor, action: torch.Tensor, task: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.control_hidden(state, action, task)
        return self.next_state_head(hidden), hidden

    def target_standardized(
        self, state: torch.Tensor, endpoint: torch.Tensor
    ) -> torch.Tensor:
        return (endpoint - state - self.delta_mean) / self.delta_scale

    def physical_endpoint(
        self, state: torch.Tensor, standardized: torch.Tensor
    ) -> torch.Tensor:
        return state + standardized * self.delta_scale + self.delta_mean


def tensor_hash(values: dict[str, torch.Tensor] | nn.Module) -> str:
    state = values.state_dict() if isinstance(values, nn.Module) else values
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def local_qk_pairs(sequence_length: int, topk: int) -> int:
    return sum(min(step + 1, topk) for step in range(sequence_length))
