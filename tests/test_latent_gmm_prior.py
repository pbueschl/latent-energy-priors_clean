import math

import torch

from lep.latent_gmm_prior import (
    GaussianMixtureLatentPrior,
    fit_gmm_prior_from_latents,
    load_gmm_prior,
    save_gmm_prior,
)


def test_diag_gmm_log_prob_matches_manual_standard_normal_mixture():
    prior = GaussianMixtureLatentPrior(
        weights=torch.tensor([0.25, 0.75]),
        means=torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        covariances=torch.ones(2, 2),
        covariance_type="diag",
    )
    z = torch.tensor([[0.0, 0.0], [2.0, 0.0]])

    log_prob = prior.log_prob(z)
    comp0_at_zero = math.log(0.25) - math.log(2.0 * math.pi)
    comp1_at_zero = math.log(0.75) - math.log(2.0 * math.pi) - 0.5 * 4.0
    expected_zero = torch.logsumexp(torch.tensor([comp0_at_zero, comp1_at_zero]), dim=0)
    comp0_at_two = math.log(0.25) - math.log(2.0 * math.pi) - 0.5 * 4.0
    comp1_at_two = math.log(0.75) - math.log(2.0 * math.pi)
    expected_two = torch.logsumexp(torch.tensor([comp0_at_two, comp1_at_two]), dim=0)

    assert torch.allclose(log_prob, torch.stack((expected_zero, expected_two)), atol=1e-6)


def test_gmm_prior_save_load_roundtrip(tmp_path):
    prior = GaussianMixtureLatentPrior(
        weights=torch.tensor([0.4, 0.6]),
        means=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        covariances=torch.full((2, 2), 0.5),
        covariance_type="diag",
    )
    path = tmp_path / "gmm_prior.pt"
    save_gmm_prior(str(path), prior, metadata={"components": 2})
    loaded, checkpoint = load_gmm_prior(str(path))

    z = torch.tensor([[0.25, -0.5]])
    assert torch.allclose(prior.log_prob(z), loaded.log_prob(z))
    assert checkpoint["metadata"]["components"] == 2


def test_fit_diag_gmm_prior_without_requiring_sklearn():
    latents = torch.tensor(
        [
            [-1.1, 0.0],
            [-0.9, 0.1],
            [0.9, -0.1],
            [1.1, 0.0],
        ],
        dtype=torch.float32,
    )
    prior, summary = fit_gmm_prior_from_latents(
        latents,
        components=2,
        covariance_type="diag",
        seed=7,
        max_iter=5,
    )

    assert prior.num_components == 2
    assert prior.latent_dim == 2
    assert summary["num_train_latents"] == 4
    assert torch.isfinite(prior.log_prob(torch.zeros(1, 2))).all()
