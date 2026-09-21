from __future__ import annotations

import pytest
import torch

from robird.models import build_model, masked_max, masked_mean


def _config(name: str) -> dict[str, object]:
    return {
        "name": name,
        "feature_dim": 4,
        "hidden_dim": 8,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "rival_top_k": 2,
    }


def test_masked_pooling_ignores_padding() -> None:
    features = torch.tensor([[[1.0, 2.0], [3.0, 0.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    torch.testing.assert_close(masked_mean(features, mask), torch.tensor([[2.0, 1.0]]))
    torch.testing.assert_close(masked_max(features, mask), torch.tensor([[3.0, 2.0]]))


@pytest.mark.parametrize(
    "name",
    ["mean_feature", "max_feature", "mean_logit", "max_logit", "deepsets", "set_transformer", "rcca"],
)
def test_aggregators_are_permutation_and_padding_invariant(name: str) -> None:
    torch.manual_seed(5)
    model = build_model(_config(name), num_classes=3).eval()
    features = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    permutation = torch.tensor([2, 0, 1])

    with torch.no_grad():
        reference = model(features, mask).logits
        permuted = model(features[:, permutation], mask[:, permutation]).logits
        padded_features = torch.cat([features, torch.randn(2, 2, 4)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
        padded = model(padded_features, padded_mask).logits

    torch.testing.assert_close(permuted, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(padded, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("use_rival_conditioning", "use_mean_path"),
    [(False, True), (True, False)],
)
def test_rcca_ablation_paths_are_executable_by_design(
    use_rival_conditioning: bool, use_mean_path: bool
) -> None:
    config = _config("rcca")
    config["use_rival_conditioning"] = use_rival_conditioning
    config["use_mean_path"] = use_mean_path
    model = build_model(config, num_classes=3).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))
    assert output.logits.shape == (2, 3)
    assert output.view_weights is not None
    torch.testing.assert_close(output.view_weights.sum(dim=1), torch.ones(2))
