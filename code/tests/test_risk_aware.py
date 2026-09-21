from __future__ import annotations

import torch

from robird.risk_aware import RiskAwareReliabilityAggregator


def _model() -> RiskAwareReliabilityAggregator:
    model = RiskAwareReliabilityAggregator(
        feature_dim=8,
        num_classes=3,
        hidden_dim=16,
        dropout=0.0,
        keep_probability=0.7,
        anchor_probability=0.2,
    )
    model.eval()
    return model


def test_risk_aware_is_permutation_invariant() -> None:
    torch.manual_seed(4)
    model = _model()
    features = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    first = model(features, mask, stochastic=False).logits
    permutation = torch.tensor([2, 0, 1])
    second = model(features[:, permutation], mask[:, permutation], stochastic=False).logits
    assert torch.allclose(first, second, atol=1e-6)


def test_risk_aware_padding_is_ignored() -> None:
    torch.manual_seed(5)
    model = _model()
    features = torch.randn(1, 2, 8)
    padded = torch.cat([features, torch.randn(1, 2, 8)], dim=1)
    first = model(features, torch.ones(1, 2, dtype=torch.bool), stochastic=False).logits
    second = model(padded, torch.tensor([[True, True, False, False]]), stochastic=False).logits
    assert torch.allclose(first, second, atol=1e-6)


def test_stochastic_mask_keeps_one_real_view() -> None:
    model = _model()
    mask = torch.tensor([[True, True, True], [True, False, False]])
    for _ in range(20):
        sampled = model.stochastic_mask(mask)
        assert bool(sampled.any(dim=1).all())
        assert bool((sampled & ~mask).sum() == 0)
