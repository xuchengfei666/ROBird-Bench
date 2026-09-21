from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass
class ModelOutput:
    logits: Tensor
    view_weights: Tensor | None = None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)


def _validate_inputs(features: Tensor, mask: Tensor) -> None:
    if features.ndim != 3:
        raise ValueError(f"features must have shape [B,V,D], got {tuple(features.shape)}")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError("mask must have shape [B,V]")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every observation must contain at least one real photo")


def masked_mean(features: Tensor, mask: Tensor) -> Tensor:
    _validate_inputs(features, mask)
    weights = mask.unsqueeze(-1).to(dtype=features.dtype)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_max(features: Tensor, mask: Tensor) -> Tensor:
    _validate_inputs(features, mask)
    minimum = torch.finfo(features.dtype).min
    return features.masked_fill(~mask.unsqueeze(-1), minimum).max(dim=1).values


class ObservationClassifier(nn.Module):
    def forward(self, features: Tensor, mask: Tensor) -> ModelOutput:
        raise NotImplementedError


class MeanFeatureClassifier(ObservationClassifier):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: Tensor, mask: Tensor) -> ModelOutput:
        return ModelOutput(logits=self.classifier(masked_mean(features, mask)))


class MaxFeatureClassifier(ObservationClassifier):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: Tensor, mask: Tensor) -> ModelOutput:
        return ModelOutput(logits=self.classifier(masked_max(features, mask)))


class MeanLogitClassifier(ObservationClassifier):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: Tensor, mask: Tensor) -> ModelOutput:
        per_photo_logits = self.classifier(features)
        return ModelOutput(
            logits=masked_mean(per_photo_logits, mask),
            auxiliary={"per_photo_logits": per_photo_logits},
        )


class MaxLogitClassifier(ObservationClassifier):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: Tensor, mask: Tensor) -> ModelOutput:
        per_photo_logits = self.classifier(features)
        return ModelOutput(
            logits=masked_max(per_photo_logits, mask),
            auxiliary={"per_photo_logits": per_photo_logits},
        )


class DeepSetsClassifier(ObservationClassifier):
    def __init__(
        self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: Tensor, mask: Tensor) -> ModelOutput:
        encoded = self.phi(features)
        return ModelOutput(logits=self.rho(masked_mean(encoded, mask)))


class SetTransformerClassifier(ObservationClassifier):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: Tensor, mask: Tensor) -> ModelOutput:
        _validate_inputs(features, mask)
        encoded = self.encoder(self.input_projection(features), src_key_padding_mask=~mask)
        query = self.pool_query.expand(features.shape[0], -1, -1)
        pooled, weights = self.pool(
            query, encoded, encoded, key_padding_mask=~mask, need_weights=True
        )
        return ModelOutput(logits=self.classifier(pooled[:, 0]), view_weights=weights[:, 0])


class RivalConditionedAggregator(ObservationClassifier):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        rival_top_k: int,
        dropout: float,
        use_rival_conditioning: bool = True,
        use_mean_path: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("RCCA requires at least two classes")
        self.rival_top_k = min(int(rival_top_k), num_classes - 1)
        self.use_rival_conditioning = bool(use_rival_conditioning)
        self.use_mean_path = bool(use_mean_path)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.reliability = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.complement_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features: Tensor, mask: Tensor) -> ModelOutput:
        _validate_inputs(features, mask)
        per_photo_logits = self.classifier(features)
        provisional_logits = masked_mean(per_photo_logits, mask)
        ranked = provisional_logits.topk(self.rival_top_k + 1, dim=1).indices
        leaders = ranked[:, 0]
        rivals = ranked[:, 1:]
        class_weights = self.classifier.weight
        leader_direction = class_weights[leaders]
        rival_directions = torch.nn.functional.normalize(
            leader_direction.unsqueeze(1) - class_weights[rivals], dim=2
        )

        center = masked_mean(features, mask)
        residual = features - center.unsqueeze(1)
        complement = torch.einsum("bvd,bkd->bkv", residual, rival_directions)
        if not self.use_rival_conditioning:
            complement = torch.zeros_like(complement)
        reliability = self.reliability(features).squeeze(-1)
        scores = reliability.unsqueeze(1) + self.complement_scale * complement
        scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        rival_view_weights = torch.softmax(scores, dim=2)
        selected_by_rival = torch.einsum("bkv,bvd->bkd", rival_view_weights, features)
        rival_importance = torch.softmax(provisional_logits.gather(1, rivals), dim=1)
        selected = torch.einsum("bk,bkd->bd", rival_importance, selected_by_rival)
        view_weights = torch.einsum("bk,bkv->bv", rival_importance, rival_view_weights)
        gate = torch.sigmoid(self.residual_gate)
        aggregated = center + gate * (selected - center) if self.use_mean_path else selected
        logits = self.classifier(aggregated)
        return ModelOutput(
            logits=logits,
            view_weights=view_weights,
            auxiliary={
                "per_photo_logits": per_photo_logits,
                "provisional_logits": provisional_logits,
                "leaders": leaders,
                "rivals": rivals,
                "complement_scores": complement,
                "rival_view_weights": rival_view_weights,
                "rival_importance": rival_importance,
            },
        )


def build_model(config: Mapping[str, Any], num_classes: int) -> ObservationClassifier:
    name = str(config["name"]).lower()
    feature_dim = int(config["feature_dim"])
    if name == "mean_feature":
        return MeanFeatureClassifier(feature_dim, num_classes)
    if name == "max_feature":
        return MaxFeatureClassifier(feature_dim, num_classes)
    if name == "mean_logit":
        return MeanLogitClassifier(feature_dim, num_classes)
    if name == "max_logit":
        return MaxLogitClassifier(feature_dim, num_classes)
    if name == "deepsets":
        return DeepSetsClassifier(
            feature_dim, int(config["hidden_dim"]), num_classes, float(config["dropout"])
        )
    if name == "set_transformer":
        return SetTransformerClassifier(
            feature_dim,
            int(config["hidden_dim"]),
            num_classes,
            int(config["num_heads"]),
            int(config["num_layers"]),
            float(config["dropout"]),
        )
    if name == "rcca":
        return RivalConditionedAggregator(
            feature_dim,
            int(config["hidden_dim"]),
            num_classes,
            int(config["rival_top_k"]),
            float(config["dropout"]),
            bool(config.get("use_rival_conditioning", True)),
            bool(config.get("use_mean_path", True)),
        )
    raise ValueError(f"Unknown model name: {name}")
