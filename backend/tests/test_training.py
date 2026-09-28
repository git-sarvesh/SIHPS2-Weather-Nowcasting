"""Tests for the multi-task loss and a CPU training smoke run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from app.models.network import MultiTaskNowcastNet, NowcastNetConfig
from app.training.losses import (
    LossWeights,
    MultiTaskLoss,
    class_balanced_weights,
    focal_bce_loss,
)

B, T_IN, H, W, K = 2, 3, 8, 8, 4
#: The ConvLSTM decoder emits one lead step per input frame, so the head output
#: length equals the input length; targets must be built with the same T.
T_OUT = T_IN


def _model_and_targets(seed: int = 0):
    """Tiny model plus targets in the dataset's own layout.

    The binary heads emit ``(B, T, 1, H, W)`` logits but the dataset yields
    ``(B, T, H, W)`` labels, so the loss must broadcast; using the dataset layout
    here exercises that path. ``rain_class`` is ``(B, T, H, W)`` because it
    indexes the class dimension.
    """
    torch.manual_seed(seed)
    model = MultiTaskNowcastNet(NowcastNetConfig.preset("lite"))
    x = torch.rand(B, T_IN, 12, H, W)
    terrain = torch.rand(B, 4, H, W)
    torch.manual_seed(seed + 1)
    targets = {
        "thunderstorm": (torch.rand(B, T_OUT, H, W) > 0.7).float(),
        "rain_class": torch.randint(0, K, (B, T_OUT, H, W)),
        "cloudburst": (torch.rand(B, T_OUT, H, W) > 0.85).float(),
        "flood": (torch.rand(B, T_OUT, H, W) > 0.8).float(),
        "flood_soft": torch.rand(B, T_OUT, H, W),
    }
    return model, x, terrain, targets


# --------------------------------------------------------------------------- #
# focal loss
# --------------------------------------------------------------------------- #
def test_focal_loss_reduces_to_weighted_bce_at_gamma_zero() -> None:
    logits = torch.randn(64, requires_grad=True)
    targets = (torch.rand(64) > 0.5).float()
    alpha = 0.5

    expected = F.binary_cross_entropy_with_logits(logits, targets)
    got = focal_bce_loss(logits, targets, gamma=0.0, alpha=alpha)

    # With alpha=0.5 the modulating factor is constant 0.5, so focal == 0.5 * BCE.
    assert float(got.detach()) == pytest.approx(0.5 * float(expected.detach()), rel=1e-5)


def test_focal_loss_downweights_easy_negatives() -> None:
    """Focal loss discounts easy negatives but keeps wrong predictions expensive."""
    target = torch.tensor([0.0])
    easy = torch.tensor([-6.0])    # sigmoid(-6) ~ 0.0025, target 0 -> easy negative
    wrong = torch.tensor([6.0])    # sigmoid(6)  ~ 0.9975, target 0 -> confidently wrong

    easy_g0 = float(focal_bce_loss(easy, target, gamma=0.0))
    easy_g2 = float(focal_bce_loss(easy, target, gamma=2.0))
    assert easy_g2 < easy_g0, "an easy negative should be discounted by the focal term"

    # A confidently wrong prediction is *not* discounted: its modulating factor
    # stays near 1, so the penalty remains close to the unweighted BCE.
    wrong_g0 = float(focal_bce_loss(wrong, target, gamma=0.0))
    wrong_g2 = float(focal_bce_loss(wrong, target, gamma=2.0))
    assert wrong_g2 == pytest.approx(wrong_g0, rel=0.05)
    assert wrong_g2 > 5 * easy_g2, "a wrong prediction must dominate an easy negative"


def test_focal_loss_is_differentiable() -> None:
    logits = torch.randn(16, requires_grad=True)
    targets = (torch.rand(16) > 0.5).float()
    focal_bce_loss(logits, targets, gamma=2.0).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_class_balanced_weights_emphasise_rare_classes() -> None:
    counts = torch.tensor([1000.0, 200.0, 50.0, 10.0])
    weights = class_balanced_weights(counts)

    assert weights.shape == (4,)
    # Rarer classes get larger weights; the ordering must be monotone.
    assert weights[0] < weights[1] < weights[2] < weights[3]
    assert float(weights.mean()) == pytest.approx(1.0, rel=1e-5)
    assert float(weights.min()) > 0.0
    # The dominant class is down-weighted below 1.
    assert float(weights[0]) < 1.0


def test_class_balanced_weights_clip_extreme_imbalance() -> None:
    """A 1-in-10000 class would get an unbounded weight without a clip."""
    weights = class_balanced_weights(torch.tensor([1e6, 1e4, 1e2, 1.0]))

    assert torch.isfinite(weights).all()
    assert float(weights.max()) / float(weights.min()) < 1e4
    assert weights[3] >= weights[0]


def test_class_balanced_weights_handles_absent_classes() -> None:
    weights = class_balanced_weights(torch.tensor([100.0, 0.0, 10.0, 5.0]))

    assert torch.isfinite(weights).all()
    # A class with no examples must not be silently ignored.
    assert float(weights[1]) >= float(weights[0])
    assert float(class_balanced_weights(torch.zeros(4)).sum()) == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# multi-task loss
# --------------------------------------------------------------------------- #
def test_multitask_loss_returns_finite_terms_and_gradients() -> None:
    model, x, terrain, targets = _model_and_targets()
    criterion = MultiTaskLoss(rain_class_weights=torch.ones(K))

    outputs = model(x, terrain)
    loss, terms = criterion(outputs, targets)

    assert set(terms) == {"thunderstorm", "rain_class", "cloudburst", "flood", "kl"}
    assert torch.isfinite(loss)
    for name, value in terms.items():
        assert np.isfinite(value), name
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_multitask_loss_total_equals_the_sum_of_its_terms() -> None:
    model, x, terrain, targets = _model_and_targets(seed=2)
    criterion = MultiTaskLoss()

    loss, terms = criterion(model(x, terrain), targets)
    assert float(loss.detach()) == pytest.approx(sum(terms.values()), rel=1e-5)


def test_multitask_loss_weights_scale_the_terms() -> None:
    model, x, terrain, targets = _model_and_targets(seed=3)
    outputs = model(x, terrain)

    base, base_terms = MultiTaskLoss()(outputs, targets)
    heavy, heavy_terms = MultiTaskLoss(weights=LossWeights(flood=5.0))(outputs, targets)

    # ``flood`` scales only the hard term; ``flood_soft`` keeps its own weight.
    hard_base = MultiTaskLoss(weights=LossWeights(flood=1.0, flood_soft=0.0)).flood_term(
        outputs["heads"].flood_logits, targets["flood"], targets["flood_soft"]
    )
    hard_heavy = MultiTaskLoss(weights=LossWeights(flood=5.0, flood_soft=0.0)).flood_term(
        outputs["heads"].flood_logits, targets["flood"], targets["flood_soft"]
    )
    assert float(hard_heavy.detach()) == pytest.approx(5.0 * float(hard_base.detach()), rel=1e-5)

    # Unrelated terms are unaffected.
    assert heavy_terms["thunderstorm"] == pytest.approx(base_terms["thunderstorm"], rel=1e-5)
    assert heavy_terms["rain_class"] == pytest.approx(base_terms["rain_class"], rel=1e-5)
    assert float(heavy.detach()) > float(base.detach())


def test_kl_term_is_zero_without_variational_bottleneck() -> None:
    model, x, terrain, targets = _model_and_targets(seed=4)
    criterion = MultiTaskLoss()

    loss, terms = criterion(model(x, terrain), targets)
    # The preset keeps the variational bottleneck on, so the KL term is positive.
    assert terms["kl"] > 0.0

    no_kl = MultiTaskLoss(weights=LossWeights(kl=0.0))
    _loss, zero_terms = no_kl(model(x, terrain), targets)
    assert zero_terms["kl"] == pytest.approx(0.0)


def test_flood_term_uses_the_soft_target() -> None:
    """The dense flood_soft target must reach the loss, not be ignored."""
    model, x, terrain, targets = _model_and_targets(seed=5)
    logits = model(x, terrain)["heads"].flood_logits.detach()
    criterion = MultiTaskLoss(weights=LossWeights(flood=1.0, flood_soft=1.0))

    zeroed = criterion.flood_term(logits, targets["flood"], torch.zeros_like(targets["flood"]))
    used = criterion.flood_term(logits, targets["flood"], targets["flood_soft"])

    assert float(used) != pytest.approx(float(zeroed), rel=1e-6)
    # The soft term is BCE against a [0, 1] target, so it is a positive, finite cost.
    soft_only = criterion.flood_term(logits, torch.zeros_like(targets["flood"]), targets["flood_soft"])
    assert float(soft_only) > 0.0
    assert torch.isfinite(soft_only)


def test_flood_term_accepts_both_target_layouts() -> None:
    """The dataset's ``(B, T, H, W)`` labels must work against ``(B, T, 1, H, W)`` logits."""
    model, x, terrain, targets = _model_and_targets(seed=7)
    logits = model(x, terrain)["heads"].flood_logits.detach()
    criterion = MultiTaskLoss()

    flat = criterion.flood_term(logits, targets["flood"], targets["flood_soft"])
    squeezed = criterion.flood_term(
        logits, targets["flood"].unsqueeze(2), targets["flood_soft"].unsqueeze(2)
    )
    assert float(flat) == pytest.approx(float(squeezed), rel=1e-6)


def test_cloudburst_term_reads_the_extreme_rainfall_logit() -> None:
    model, x, terrain, targets = _model_and_targets(seed=6)
    outputs = model(x, terrain)
    term = MultiTaskLoss().cloudburst_term(outputs["heads"].rain_logits, targets["cloudburst"])

    assert torch.isfinite(term)
    # P(extreme) = softmax over classes, index -1.
    probs = torch.softmax(outputs["heads"].rain_logits, dim=2)
    assert torch.allclose(probs[:, :, -1], outputs["heads"].cloudburst_prob, atol=1e-6)


# --------------------------------------------------------------------------- #
# training smoke run
# --------------------------------------------------------------------------- #
def test_training_smoke_run_produces_a_labelled_checkpoint(tmp_path: Path) -> None:
    """A real (tiny) training run on CPU: writes artefacts, labels them synthetic."""
    from app.training.train import TrainConfig, train

    config = TrainConfig(
        seed=3,
        n_events=2,
        seq_len=4,
        horizon=4,
        epochs=1,
        batch_size=1,
        max_steps=1,
        preset="lite",
        output_dir=str(tmp_path / "ckpt"),
    )
    result = train(config)

    assert result.epochs == 1
    assert result.n_parameters > 0
    assert result.history and result.history[0]["train"]["n_batches"] >= 1
    assert np.isfinite(result.history[0]["train"]["loss"])

    directory = Path(result.output_dir)
    for name in ("nowcast_model.pt", "model_config.json", "training_meta.json", "training_report.json"):
        assert (directory / name).exists(), name

    # The artefacts must declare the run as synthetic, never as validated.
    meta = json.loads((directory / "training_meta.json").read_text(encoding="utf-8"))
    assert meta["is_synthetic"] is True
    assert "SYNTHETIC DEMO" in meta["validation_status"]
    assert "no independent observational validation" in meta["validation_status"]
    assert meta["window"]["seq_len"] == 4 and meta["window"]["horizon"] == 4
    assert meta["dataset"]["is_synthetic"] is True
    assert meta["epochs_completed"] == 1


def test_training_is_reproducible_with_a_fixed_seed(tmp_path: Path) -> None:
    from app.training.train import TrainConfig, seed_everything, train

    def _run(name: str) -> list[float]:
        seed_everything(11)
        config = TrainConfig(
            seed=11,
            n_events=2,
            seq_len=4,
            horizon=4,
            epochs=1,
            batch_size=1,
            max_steps=1,
            output_dir=str(tmp_path / name),
        )
        return [entry["train"]["loss"] for entry in train(config).history]

    assert _run("a") == pytest.approx(_run("b"), rel=1e-6)


def test_training_rejects_a_window_longer_than_the_events(tmp_path: Path) -> None:
    from app.training.train import TrainConfig, train

    config = TrainConfig(
        n_events=1, seq_len=10, horizon=10, epochs=1, output_dir=str(tmp_path / "bad")
    )
    with pytest.raises(ValueError, match="window needs"):
        train(config)


def test_resolve_device_falls_back_to_cpu() -> None:
    from app.training.train import resolve_device

    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        assert resolve_device("cuda").type == "cpu"


# APPEND_MARKER
