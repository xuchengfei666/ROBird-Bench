from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from robird.models import ModelOutput


def _validate(features: Tensor, mask: Tensor) -> None:
    if features.ndim != 3:
        raise ValueError(f"features must have shape [B,V,D], got {tuple(features.shape)}")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError("mask must have shape [B,V]")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every observation must contain at least one real photo")


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    minimum = torch.finfo(values.dtype).min
    return values.masked_fill(~mask, minimum).max(dim=1).values


class RiskAwareReliabilityAggregator(nn.Module):
    """Permutation-invariant soft reliability pooling with stochastic risk training."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        keep_probability: float,
        anchor_probability: float,
    ) -> None:
        super().__init__()
        if not 0.0 < float(keep_probability) <= 1.0:
            raise ValueError("keep_probability must be in (0, 1]")
        if not 0.0 <= float(anchor_probability) <= 1.0:
            raise ValueError("anchor_probability must be in [0, 1]")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.keep_probability = float(keep_probability)
        self.anchor_probability = float(anchor_probability)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        reliability_input = self.feature_dim + self.num_classes + 3
        self.reliability = nn.Sequential(
            nn.LayerNorm(reliability_input),
            nn.Linear(reliability_input, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        gate_hidden = max(16, int(hidden_dim) // 4)
        self.conflict_gate = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, gate_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.constant_(self.conflict_gate[-1].bias, -1.5)

    def stochastic_mask(self, mask: Tensor) -> Tensor:
        """Sample a valid view mask without ever removing the final real view."""
        _validate(torch.zeros((*mask.shape, 1), device=mask.device), mask)
        sampled = torch.zeros_like(mask)
        for row in range(mask.shape[0]):
            indices = torch.nonzero(mask[row], as_tuple=False).flatten()
            if len(indices) == 1:
                sampled[row, indices[0]] = True
                continue
            if float(torch.rand((), device=mask.device)) < self.anchor_probability:
                chosen = indices[torch.randint(len(indices), (1,), device=mask.device)]
                sampled[row, chosen] = True
                continue
            keep = torch.rand(len(indices), device=mask.device) < self.keep_probability
            if not bool(keep.any()):
                keep[torch.randint(len(indices), (1,), device=mask.device)] = True
            sampled[row, indices[keep]] = True
        return sampled

    def _evidence(self, features: Tensor, mask: Tensor) -> dict[str, Tensor]:
        per_photo_logits = self.classifier(features)
        centered = per_photo_logits - per_photo_logits.mean(dim=-1, keepdim=True)
        normalized = F.normalize(centered, dim=-1, eps=1e-8)
        consensus = _masked_mean(normalized, mask)
        consensus = F.normalize(consensus, dim=-1, eps=1e-8)
        cosine = (normalized * consensus.unsqueeze(1)).sum(dim=-1)
        disagreement = 1.0 - cosine
        probabilities = torch.softmax(per_photo_logits, dim=-1)
        entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(math.log(float(self.num_classes)), 1e-8)
        top_values = torch.topk(per_photo_logits, k=min(2, self.num_classes), dim=-1).values
        margin = top_values[..., 0] if self.num_classes == 1 else top_values[..., 0] - top_values[..., 1]
        margin = torch.tanh(margin / 5.0)
        reliability_input = torch.cat(
            [features, per_photo_logits / 5.0, margin.unsqueeze(-1), entropy.unsqueeze(-1), disagreement.unsqueeze(-1)],
            dim=-1,
        )
        raw_reliability = self.reliability(reliability_input).squeeze(-1)
        raw_reliability = raw_reliability.masked_fill(~mask, torch.finfo(raw_reliability.dtype).min)
        weights = torch.softmax(raw_reliability, dim=1)
        mean_logits = _masked_mean(per_photo_logits, mask)
        weighted_logits = torch.einsum("bv,bvc->bc", weights, per_photo_logits)
        conflict_mean = _masked_mean(disagreement.unsqueeze(-1), mask).squeeze(-1)
        conflict_max = _masked_max(disagreement, mask)
        entropy_mean = _masked_mean(entropy.unsqueeze(-1), mask).squeeze(-1)
        view_count = mask.to(dtype=features.dtype).sum(dim=1).log1p()
        gate_input = torch.stack([conflict_mean, conflict_max, entropy_mean, view_count], dim=-1)
        gate = torch.sigmoid(self.conflict_gate(gate_input).squeeze(-1))
        logits = mean_logits + gate.unsqueeze(-1) * (weighted_logits - mean_logits)
        return {
            "logits": logits,
            "per_photo_logits": per_photo_logits,
            "view_weights": weights,
            "conflict": disagreement,
            "gate": gate,
            "mean_logits": mean_logits,
            "weighted_logits": weighted_logits,
        }

    def forward(
        self,
        features: Tensor,
        mask: Tensor,
        stochastic: bool | None = None,
    ) -> ModelOutput:
        _validate(features, mask)
        use_stochastic = self.training if stochastic is None else bool(stochastic)
        effective_mask = self.stochastic_mask(mask) if use_stochastic else mask
        evidence = self._evidence(features, effective_mask)
        return ModelOutput(
            logits=evidence["logits"],
            view_weights=evidence["view_weights"],
            auxiliary={
                key: value
                for key, value in evidence.items()
                if key not in {"logits", "view_weights"}
            },
        )


def load_classifier_state(model: RiskAwareReliabilityAggregator, state: dict[str, Any]) -> None:
    """Initialize the shared photo classifier from the frozen P1 linear head."""
    model_state = state.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint model_state is missing")
    weight = model_state.get("classifier.weight")
    bias = model_state.get("classifier.bias")
    if weight is None or bias is None:
        raise ValueError("Checkpoint lacks classifier.weight/classifier.bias")
    if tuple(weight.shape) != tuple(model.classifier.weight.shape) or tuple(bias.shape) != tuple(model.classifier.bias.shape):
        raise ValueError("Frozen classifier shape does not match P3 model")
    with torch.no_grad():
        model.classifier.weight.copy_(weight)
        model.classifier.bias.copy_(bias)
