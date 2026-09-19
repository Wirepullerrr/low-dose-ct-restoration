"""Tests for the residual CNN architecture and its input contract.

Fully synthetic and CPU-only. The architecture is small enough that every
structural claim the report makes - parameter count, layer count, receptive
field, zero-initialized identity - can simply be checked rather than asserted
in prose.

The identity property is the one that matters most. If the freshly
initialized network is not exactly the degraded image, the epoch-0 sanity
check stops being a check on the evaluation path and becomes a measurement of
an untrained network, which would hide a plumbing mismatch rather than expose
one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ct_restoration.config import load_config
from ct_restoration.models.adapter import raw_restore_array, restore_array, torch_restorer
from ct_restoration.models.cnn import (
    ALGORITHM_VERSION,
    CANONICAL_PARAMETER_COUNT,
    CANONICAL_RECEPTIVE_FIELD,
    ModelError,
    ResidualCNN,
    ResidualCnnConfig,
    build_model,
    validate_model_input,
)


@pytest.fixture
def config() -> ResidualCnnConfig:
    return ResidualCnnConfig()


@pytest.fixture
def model(config) -> ResidualCNN:
    return build_model(config)


@pytest.fixture
def image() -> torch.Tensor:
    """A deterministic textured [0, 1] batch, the shape the Dataset produces."""
    generator = torch.Generator().manual_seed(11)
    return torch.rand(3, 1, 32, 32, generator=generator, dtype=torch.float32)


def convolutions(model: ResidualCNN) -> list[nn.Conv2d]:
    return [layer for layer in model.layers if isinstance(layer, nn.Conv2d)]


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_the_committed_config_matches_the_declared_architecture() -> None:
    loaded = ResidualCnnConfig.from_mapping(load_config("cnn.yaml"))

    assert loaded.algorithm == ALGORITHM_VERSION
    assert (loaded.input_channels, loaded.output_channels) == (1, 1)
    assert (loaded.hidden_channels, loaded.depth) == (32, 5)
    assert (loaded.kernel_size, loaded.padding) == (3, 1)
    assert loaded.activation == "relu"
    assert loaded.batch_norm is False
    assert loaded.dropout == 0.0
    assert loaded.residual_correction is True
    assert loaded.zero_initialize_final_layer is True


def test_the_parameter_count_is_pinned(model) -> None:
    """320 + 3 * 9248 + 289 = 28,353, with bias on all five convolutions."""
    assert model.parameter_count() == CANONICAL_PARAMETER_COUNT == 28_353
    assert sum(p.numel() for p in model.parameters()) == 28_353


def test_the_parameter_count_is_reachable_by_hand(model) -> None:
    per_layer = [
        convolution.weight.numel() + convolution.bias.numel() for convolution in convolutions(model)
    ]

    assert per_layer == [320, 9248, 9248, 9248, 289]
    assert sum(per_layer) == CANONICAL_PARAMETER_COUNT


def test_there_are_exactly_five_convolutions(model) -> None:
    assert len(convolutions(model)) == 5
    assert [(c.in_channels, c.out_channels) for c in convolutions(model)] == [
        (1, 32),
        (32, 32),
        (32, 32),
        (32, 32),
        (32, 1),
    ]


def test_there_is_no_batch_norm_or_dropout(model) -> None:
    for module in model.modules():
        assert not isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d)
        assert not isinstance(module, nn.Dropout | nn.Dropout2d)


def test_there_is_no_pooling_striding_or_dilation(model) -> None:
    for module in model.modules():
        assert not isinstance(module, nn.MaxPool2d | nn.AvgPool2d)
    for convolution in convolutions(model):
        assert convolution.stride == (1, 1)
        assert convolution.dilation == (1, 1)


def test_relu_sits_between_convolutions_but_not_after_the_last(model) -> None:
    """The correction must be able to go negative."""
    kinds = [type(layer).__name__ for layer in model.layers]

    assert kinds == [
        "Conv2d",
        "ReLU",
        "Conv2d",
        "ReLU",
        "Conv2d",
        "ReLU",
        "Conv2d",
        "ReLU",
        "Conv2d",
    ]


def test_the_receptive_field_is_eleven_pixels(model, config) -> None:
    """1 + 5 * (3 - 1). A statement about pixels, nothing more."""
    assert model.receptive_field == CANONICAL_RECEPTIVE_FIELD == 11
    assert config.receptive_field == 11


def test_the_receptive_field_is_empirically_eleven() -> None:
    """Perturb one input pixel and measure how far the output moves.

    Checked rather than derived, because an accidental dilation or an extra
    layer would change the answer while the formula kept saying 11.
    """
    torch.manual_seed(3)
    model = build_model()
    # Two adjustments, both needed to make any gradient reach the input at
    # all. The final layer starts at zero by design, and on an all-zero image
    # every pre-activation would be the zero bias, where PyTorch's ReLU
    # subgradient is 0.
    nn.init.normal_(convolutions(model)[-1].weight, std=0.1)
    for convolution in convolutions(model):
        nn.init.constant_(convolution.bias, 0.1)

    generator = torch.Generator().manual_seed(5)
    image = torch.rand(1, 1, 41, 41, generator=generator, dtype=torch.float32)
    image.requires_grad_(True)
    model.correction(image)[0, 0, 20, 20].backward()
    influence = (image.grad[0, 0].abs() > 0).nonzero()

    rows = int(influence[:, 0].max() - influence[:, 0].min()) + 1
    columns = int(influence[:, 1].max() - influence[:, 1].min()) + 1
    assert (rows, columns) == (11, 11)


def test_padding_preserves_the_spatial_size(model) -> None:
    for height, width in [(16, 16), (32, 48), (256, 256), (17, 13)]:
        output = model.correction(torch.zeros(1, 1, height, width))

        assert tuple(output.shape) == (1, 1, height, width)


def test_the_correction_shape_matches_the_input(model, image) -> None:
    assert model.correction(image).shape == image.shape
    assert model.forward(image).shape == image.shape
    assert model.restore(image).shape == image.shape


# --------------------------------------------------------------------------
# Zero initialization: the untrained network is the identity
# --------------------------------------------------------------------------


def test_the_final_convolution_starts_all_zero(model) -> None:
    final = convolutions(model)[-1]

    assert torch.equal(final.weight, torch.zeros_like(final.weight))
    assert torch.equal(final.bias, torch.zeros_like(final.bias))


def test_the_earlier_convolutions_do_not_start_at_zero(model) -> None:
    """Kaiming init: zeroing everything would make the gradient path dead."""
    for convolution in convolutions(model)[:-1]:
        assert not torch.equal(convolution.weight, torch.zeros_like(convolution.weight))


def test_the_initial_correction_is_exactly_zero(model, image) -> None:
    with torch.no_grad():
        correction = model.correction(image)

    assert torch.equal(correction, torch.zeros_like(correction))


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (2, 1, 64, 64), (1, 1, 256, 256), (3, 1, 33, 17)])
def test_the_initialized_model_reproduces_its_input_byte_for_byte(shape) -> None:
    """The property the epoch-0 identity check depends on."""
    generator = torch.Generator().manual_seed(7)
    degraded = torch.rand(*shape, generator=generator, dtype=torch.float32)
    model = build_model()

    with torch.no_grad():
        restored = model.restore(degraded)
        raw = model.forward(degraded)

    assert restored.numpy().tobytes() == degraded.numpy().tobytes()
    assert raw.numpy().tobytes() == degraded.numpy().tobytes()


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_the_identity_holds_at_the_range_boundaries(value) -> None:
    degraded = torch.full((1, 1, 8, 8), value, dtype=torch.float32)

    with torch.no_grad():
        assert torch.equal(build_model().restore(degraded), degraded)


def test_the_model_does_not_modify_its_input(model, image) -> None:
    before = image.clone()

    with torch.no_grad():
        model.restore(image)

    assert torch.equal(image, before)


def test_a_trained_step_breaks_the_identity(model, image) -> None:
    """Otherwise the identity test above would pass on a model that cannot learn."""
    target = torch.zeros_like(image)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = torch.mean(torch.abs(model(image) - target))
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        assert not torch.equal(model.restore(image), image)


# --------------------------------------------------------------------------
# Clamping is the benchmark output, not the training output
# --------------------------------------------------------------------------


def test_restore_clamps_but_forward_does_not(image) -> None:
    model = build_model()
    with torch.no_grad():
        convolutions(model)[-1].bias.fill_(5.0)
        raw = model.forward(image)
        restored = model.restore(image)

    assert float(raw.max()) > 1.0
    assert float(restored.max()) <= 1.0
    assert float(restored.min()) >= 0.0


def test_the_final_layer_is_linear_not_squashed(model) -> None:
    kinds = [type(layer).__name__ for layer in model.layers]

    assert kinds[-1] == "Conv2d"
    assert "Sigmoid" not in kinds
    assert "Tanh" not in kinds


# --------------------------------------------------------------------------
# Input contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (1, 2, 8, 8)])
def test_a_wrong_channel_count_is_refused(model, shape) -> None:
    with pytest.raises(ModelError, match="channel"):
        model.correction(torch.zeros(*shape))


@pytest.mark.parametrize("shape", [(8, 8), (1, 8, 8), (1, 1, 1, 8, 8)])
def test_a_wrong_rank_is_refused(model, shape) -> None:
    with pytest.raises(ModelError, match="B, 1, H, W|B, C, H, W"):
        model.correction(torch.zeros(*shape))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_is_refused_by_the_validator(bad) -> None:
    tensor = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    tensor[0, 0, 2, 2] = bad

    with pytest.raises(ModelError, match="finite"):
        validate_model_input(tensor)


@pytest.mark.parametrize("value", [-0.5, 1.5])
def test_out_of_range_input_is_refused_by_the_validator(value) -> None:
    tensor = torch.full((1, 1, 4, 4), value, dtype=torch.float32)

    with pytest.raises(ModelError, match=r"\[0, 1\]"):
        validate_model_input(tensor)


def test_a_non_float32_input_is_refused_by_the_validator() -> None:
    with pytest.raises(ModelError, match="float32"):
        validate_model_input(torch.zeros(1, 1, 4, 4, dtype=torch.float64))


def test_a_tiny_overshoot_is_tolerated() -> None:
    """Only float32 round-trip error, not genuinely unnormalized input."""
    validate_model_input(torch.full((1, 1, 4, 4), 1.0 + 1e-7, dtype=torch.float32))


def test_the_validator_refuses_a_numpy_array() -> None:
    with pytest.raises(ModelError, match="torch.Tensor"):
        validate_model_input(np.zeros((1, 1, 4, 4), dtype=np.float32))


# --------------------------------------------------------------------------
# Configuration guards
# --------------------------------------------------------------------------


def test_an_unknown_algorithm_is_refused() -> None:
    with pytest.raises(ModelError, match="Unsupported model algorithm"):
        ResidualCnnConfig(algorithm="unet_v1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_norm", True),
        ("dropout", 0.1),
        ("activation", "gelu"),
        ("residual_correction", False),
        ("zero_initialize_final_layer", False),
    ],
)
def test_departures_from_the_declared_architecture_are_refused(field, value) -> None:
    with pytest.raises(ModelError):
        ResidualCnnConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1, 2.0, "32", True])
def test_a_malformed_channel_count_is_refused(value) -> None:
    with pytest.raises(ModelError, match="hidden_channels"):
        ResidualCnnConfig(hidden_channels=value)


def test_a_padding_that_would_shrink_the_image_is_refused() -> None:
    with pytest.raises(ModelError, match="preserve the spatial size"):
        ResidualCnnConfig(kernel_size=3, padding=0)


def test_config_refuses_missing_and_unknown_keys() -> None:
    document = dict(load_config("cnn.yaml")["model"])

    with pytest.raises(ModelError, match="unrecognised key"):
        ResidualCnnConfig.from_mapping({**document, "attention": True})
    with pytest.raises(ModelError, match="missing key"):
        ResidualCnnConfig.from_mapping({"algorithm": ALGORITHM_VERSION})


# --------------------------------------------------------------------------
# The benchmark adapter
# --------------------------------------------------------------------------


def test_the_restorer_takes_only_the_degraded_image(model) -> None:
    import inspect

    restore = torch_restorer(model)

    assert list(inspect.signature(restore).parameters) == ["degraded"]


def test_the_adapter_round_trips_a_numpy_slice(model) -> None:
    degraded = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)

    restored = restore_array(model, degraded)

    assert restored.dtype == np.float32
    assert restored.shape == degraded.shape
    # Zero-initialized: the adapter must not perturb the image either.
    assert restored.tobytes() == degraded.tobytes()


def test_the_adapter_clamps_but_the_raw_helper_does_not(model) -> None:
    with torch.no_grad():
        convolutions(model)[-1].bias.fill_(5.0)
    degraded = np.full((8, 8), 0.5, dtype=np.float32)

    assert float(restore_array(model, degraded).max()) <= 1.0
    assert float(raw_restore_array(model, degraded).max()) > 1.0


def test_the_adapter_refuses_a_non_2d_slice(model) -> None:
    with pytest.raises(ValueError, match="2-D"):
        restore_array(model, np.zeros((2, 8, 8), dtype=np.float32))


def test_the_adapter_leaves_the_model_in_eval_mode(model) -> None:
    model.train()
    restore_array(model, np.zeros((8, 8), dtype=np.float32))

    assert not model.training


def test_the_adapter_does_not_modify_the_caller_array(model) -> None:
    degraded = np.full((8, 8), 0.25, dtype=np.float32)
    before = degraded.copy()

    restore_array(model, degraded)

    assert np.array_equal(degraded, before)
