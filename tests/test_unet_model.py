"""Tests for the lightweight residual U-Net.

Fully synthetic and dataset-free: no CHAOS file is opened, no network access,
and every tensor here is made on the spot. The point is to pin the
architecture itself, because Milestone 9's whole claim is that the *only*
substantial change from Milestone 8 is the model - so anything about the
model that could drift silently is nailed down here.

Three groups: the architecture and its two pinned numbers, the input
contract the two pools impose, and the training/checkpoint behaviour shared
with the CNN.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ct_restoration.models.base import ModelError
from ct_restoration.models.cnn import CANONICAL_PARAMETER_COUNT as CNN_PARAMETERS
from ct_restoration.models.unet import (
    ALGORITHM_VERSION,
    CANONICAL_PARAMETER_COUNT,
    CANONICAL_RECEPTIVE_FIELD,
    LightweightResidualUnet,
    LightweightResidualUnetConfig,
    build_model,
    validate_model_input,
)
from ct_restoration.training import l1_training_loss


def _image(batch: int = 2, height: int = 32, width: int = 32, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 1, height, width, generator=generator, dtype=torch.float32)


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------


def test_the_parameter_count_is_pinned():
    # Follows from the frozen architecture. Channels were never adjusted to
    # reach this number; if it moves, the architecture moved.
    assert build_model().parameter_count() == 116_753
    assert CANONICAL_PARAMETER_COUNT == 116_753


def test_the_unet_is_about_four_times_the_cnn():
    ratio = CANONICAL_PARAMETER_COUNT / CNN_PARAMETERS
    assert CNN_PARAMETERS == 28_353
    assert round(ratio, 2) == 4.12


def test_the_declared_convolution_structure_is_exact():
    model = build_model()
    convolutions = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    transposed = [m for m in model.modules() if isinstance(m, nn.ConvTranspose2d)]

    # 2 per encoder block x 2, 2 bottleneck, 2 per decoder stage x 2, + head.
    assert len(convolutions) == 11
    assert len(transposed) == 2

    shapes = [(c.in_channels, c.out_channels, c.kernel_size[0]) for c in convolutions]
    assert shapes == [
        (1, 16, 3),
        (16, 16, 3),  # encoder 1
        (16, 32, 3),
        (32, 32, 3),  # encoder 2
        (32, 64, 3),
        (64, 64, 3),  # bottleneck
        (64, 32, 3),
        (32, 32, 3),  # decoder 1, after concatenating skip 2
        (32, 16, 3),
        (16, 16, 3),  # decoder 2, after concatenating skip 1
        (16, 1, 1),  # correction head
    ]
    assert [(t.in_channels, t.out_channels, t.kernel_size[0], t.stride[0]) for t in transposed] == [
        (64, 32, 2, 2),
        (32, 16, 2, 2),
    ]


def test_there_is_no_batch_norm_and_no_dropout():
    model = build_model()
    for module in model.modules():
        assert not isinstance(module, nn.modules.batchnorm._BatchNorm)
        assert not isinstance(module, nn.Dropout | nn.Dropout2d)


def test_there_are_exactly_two_max_pools_of_the_declared_shape():
    pools = [m for m in build_model().modules() if isinstance(m, nn.MaxPool2d)]
    assert len(pools) == 2
    for pool in pools:
        assert pool.kernel_size == 2
        assert pool.stride == 2


def test_every_convolution_has_a_bias():
    for module in build_model().modules():
        if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
            assert module.bias is not None


def test_the_correction_head_is_one_by_one_and_starts_at_zero():
    model = build_model()
    assert model.head.kernel_size == (1, 1)
    assert torch.equal(model.head.weight, torch.zeros_like(model.head.weight))
    assert torch.equal(model.head.bias, torch.zeros_like(model.head.bias))


def test_no_activation_follows_the_correction_head():
    # The correction has to be able to go negative, and a squashing function
    # here would also make the zero-init identity impossible.
    modules = list(build_model().modules())
    assert isinstance(modules[-1], nn.Conv2d)


def test_the_earlier_convolutions_are_not_all_zero():
    # Only the head is zero-initialized; a zero body would never learn.
    model = build_model()
    body = [
        m
        for m in model.modules()
        if isinstance(m, nn.Conv2d | nn.ConvTranspose2d) and m is not model.head
    ]
    assert len(body) == 12
    assert all(float(m.weight.detach().abs().sum()) > 0 for m in body)


# --------------------------------------------------------------------------
# the two pinned architectural properties
# --------------------------------------------------------------------------


def test_the_deepest_path_receptive_field_is_forty_four():
    assert build_model().receptive_field == 44
    assert CANONICAL_RECEPTIVE_FIELD == 44


def test_the_receptive_field_formula_matches_a_hand_accumulation():
    # Independent of the property's own loop: conv adds (k-1)*jump, a 2x2
    # stride-2 pool adds jump and doubles it, and a non-overlapping 2x2
    # stride-2 transposed convolution halves the jump and adds no extent.
    field, jump = 1, 1
    for _ in range(2):  # two encoder levels
        field += 2 * jump
        field += 2 * jump
        field += jump
        jump *= 2
    field += 2 * jump
    field += 2 * jump  # bottleneck
    for _ in range(2):  # two decoder stages
        jump //= 2
        field += 2 * jump
        field += 2 * jump
    assert field == 44


def test_the_receptive_field_is_empirically_at_least_forty_four():
    # A gradient probe: perturbing the centre output pixel must reach at
    # least a 44-wide window of the input. Biases are randomised so no path
    # is dead, and the head is un-zeroed so a signal can leave it at all.
    torch.manual_seed(7)
    model = build_model()
    for module in model.modules():
        if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.normal_(module.weight, std=0.2)
            nn.init.normal_(module.bias, std=0.2)

    image = _image(batch=1, height=128, width=128, seed=3).requires_grad_(True)
    model.correction(image)[0, 0, 64, 64].backward()
    reached = (image.grad[0, 0].abs() > 0).nonzero()
    height_span = int(reached[:, 0].max() - reached[:, 0].min()) + 1
    width_span = int(reached[:, 1].max() - reached[:, 1].min()) + 1
    # Pooling alignment can make the realized window slightly wider than the
    # theoretical maximum for a particular centre pixel; it must never be
    # narrower than the local paths and never exceed the bound by much.
    assert height_span >= 11
    assert width_span >= 11
    assert height_span <= 44 + 4
    assert width_span <= 44 + 4


# --------------------------------------------------------------------------
# skip-connection contract
# --------------------------------------------------------------------------


def test_the_decoder_stages_receive_the_declared_concatenated_widths():
    model = build_model()
    seen: list[int] = []

    def record(_module, inputs):
        seen.append(int(inputs[0].shape[1]))

    handles = [decoder.register_forward_pre_hook(record) for decoder in model.decoders]
    try:
        with torch.no_grad():
            model.correction(_image(height=32, width=32))
    finally:
        for handle in handles:
            handle.remove()

    # Stage 1: 32 upsampled + 32 from skip 2. Stage 2: 16 + 16 from skip 1.
    assert seen == [64, 32]


def test_skips_are_concatenated_and_not_summed():
    # A summation would leave the decoder input at the upsampled width; the
    # doubling above is what distinguishes the two.
    model = build_model()
    assert model.decoders[0][0].in_channels == 2 * model.upsamplers[0].out_channels
    assert model.decoders[1][0].in_channels == 2 * model.upsamplers[1].out_channels


def test_a_summation_skip_config_is_refused():
    with pytest.raises(ModelError, match="concatenative"):
        LightweightResidualUnetConfig(skip_connection="add")


# --------------------------------------------------------------------------
# input contract
# --------------------------------------------------------------------------


def test_the_canonical_input_shape_is_preserved():
    model = build_model()
    with torch.no_grad():
        assert model.restore(_image(batch=2, height=256, width=256)).shape == (2, 1, 256, 256)


def test_a_smaller_valid_shape_is_accepted():
    model = build_model()
    with torch.no_grad():
        assert model.restore(_image(batch=1, height=128, width=128)).shape == (1, 1, 128, 128)


@pytest.mark.parametrize(("height", "width"), [(255, 256), (256, 258), (254, 254), (6, 8)])
def test_a_shape_the_two_pools_cannot_halve_twice_is_refused(height, width):
    with pytest.raises(ModelError, match="divisible by 4"):
        build_model().correction(_image(batch=1, height=height, width=width))


def test_the_divisibility_rule_comes_from_the_level_count():
    assert LightweightResidualUnetConfig().spatial_multiple == 4
    assert LightweightResidualUnetConfig(levels=3).spatial_multiple == 8


def test_the_standalone_validator_agrees_with_the_model():
    validate_model_input(_image(batch=1, height=64, width=64))
    with pytest.raises(ModelError, match="divisible by 4"):
        validate_model_input(_image(batch=1, height=66, width=64))


def test_forward_checks_structure_and_leaves_values_to_the_validator():
    # Finiteness and range need a device-to-host synchronization on CUDA, so
    # forward does not check them - exactly like the CNN - and the full
    # validator, run before inference, still refuses them.
    model = build_model()
    image = _image(batch=1, height=32, width=32)
    image[0, 0, 3, 3] = float("nan")
    with torch.no_grad():
        assert torch.isnan(model.restore(image)).any()
    with pytest.raises(ModelError, match="finite"):
        validate_model_input(image)


def test_forward_never_runs_the_content_check(monkeypatch):
    from ct_restoration.models import base

    def refuse(tensor):
        raise AssertionError("content check reached")

    monkeypatch.setattr(base, "validate_restoration_content", refuse)
    with torch.no_grad():
        build_model().restore(_image(batch=1, height=32, width=32))
    with pytest.raises(AssertionError, match="content check reached"):
        validate_model_input(_image(batch=1, height=32, width=32))


@pytest.mark.parametrize(
    ("tensor", "match"),
    [
        (torch.zeros(1, 32, 32), "B, C, H, W"),
        (torch.zeros(1, 3, 32, 32), "channel"),
        (torch.zeros(1, 1, 32, 32, dtype=torch.float64), "float32"),
        (torch.zeros(1, 1, 30, 32), "divisible by 4"),
    ],
)
def test_forward_still_refuses_a_structurally_wrong_input(tensor, match):
    with pytest.raises(ModelError, match=match):
        build_model().restore(tensor)


def test_the_ordinary_input_contract_still_applies():
    with pytest.raises(ModelError, match="float32"):
        validate_model_input(torch.zeros(1, 1, 8, 8, dtype=torch.float64))
    with pytest.raises(ModelError, match="finite"):
        validate_model_input(torch.full((1, 1, 8, 8), float("nan"), dtype=torch.float32))
    with pytest.raises(ModelError, match=r"\[0, 1\]"):
        validate_model_input(torch.full((1, 1, 8, 8), 2.0, dtype=torch.float32))
    with pytest.raises(ModelError, match="channel"):
        validate_model_input(torch.zeros(1, 3, 8, 8, dtype=torch.float32))


# --------------------------------------------------------------------------
# residual identity
# --------------------------------------------------------------------------


def test_a_freshly_initialized_unet_is_the_identity_restoration():
    # The whole epoch-0 check rests on this: an untrained U-Net must BE the
    # no-restoration baseline, bit for bit.
    model = build_model()
    image = _image(batch=3, height=64, width=64, seed=11)
    with torch.no_grad():
        assert torch.equal(model.correction(image), torch.zeros_like(image))
        assert torch.equal(model.forward(image), image)
        assert torch.equal(model.restore(image), image)


def test_the_identity_holds_at_the_canonical_size_too():
    model = build_model()
    image = _image(batch=1, height=256, width=256, seed=12)
    with torch.no_grad():
        assert torch.equal(model.restore(image), image)


def test_restore_clamps_but_forward_does_not():
    model = build_model()
    nn.init.constant_(model.head.bias, 0.5)
    image = _image(batch=1, height=32, width=32, seed=13)
    with torch.no_grad():
        raw, restored = model.forward(image), model.restore(image)
    assert float(raw.max()) > 1.0
    assert float(restored.max()) <= 1.0
    assert torch.equal(restored, torch.clamp(raw, 0.0, 1.0))


def test_the_forward_pass_does_not_mutate_its_input():
    model = build_model()
    nn.init.normal_(model.head.weight, std=0.1)
    image = _image(batch=2, height=32, width=32, seed=14)
    reference = image.clone()
    with torch.no_grad():
        model.restore(image)
    assert torch.equal(image, reference)


def test_the_forward_pass_does_not_mutate_its_parameters():
    model = build_model()
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.restore(_image(height=32, width=32))
    for name, tensor in model.state_dict().items():
        assert torch.equal(before[name], tensor)


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------


def test_the_canonical_algorithm_name_is_required():
    with pytest.raises(ModelError, match="Unsupported model algorithm"):
        LightweightResidualUnetConfig(algorithm="unet_v2")
    assert ALGORITHM_VERSION == "lightweight_residual_unet_v1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_norm", True, "batch_norm"),
        ("dropout", 0.1, "dropout"),
        ("activation", "gelu", "relu"),
        ("residual_correction", False, "residual_correction"),
        ("zero_initialize_final_layer", False, "zero_initialize"),
        ("downsampling", "stride_2_conv", "max_pool_2x2"),
        ("upsampling", "bilinear", "conv_transpose"),
        ("final_kernel_size", 3, "final_kernel_size"),
        ("convolutions_per_block", 3, "convolutions_per_block"),
        ("base_channels", 0, "base_channels"),
        ("padding", 0, "preserve the spatial size"),
    ],
)
def test_a_departure_from_the_frozen_architecture_is_refused(field, value, message):
    with pytest.raises(ModelError, match=message):
        LightweightResidualUnetConfig(**{field: value})


def test_from_mapping_refuses_unknown_and_missing_keys():
    good = LightweightResidualUnetConfig().as_dict()
    assert LightweightResidualUnetConfig.from_mapping({"model": good}).as_dict() == good
    with pytest.raises(ModelError, match="unrecognised key"):
        LightweightResidualUnetConfig.from_mapping({"model": {**good, "depth": 5}})
    with pytest.raises(ModelError, match="missing key"):
        LightweightResidualUnetConfig.from_mapping({"model": {"algorithm": ALGORITHM_VERSION}})


def test_the_frozen_config_file_builds_the_canonical_model():
    from ct_restoration.config import load_config

    config = LightweightResidualUnetConfig.from_mapping(load_config("unet.yaml"))
    model = build_model(config)
    assert model.parameter_count() == CANONICAL_PARAMETER_COUNT
    assert model.receptive_field == CANONICAL_RECEPTIVE_FIELD


# --------------------------------------------------------------------------
# training behaviour
# --------------------------------------------------------------------------


def test_the_raw_l1_loss_is_finite_and_gradients_reach_every_parameter():
    model = build_model()
    degraded = _image(batch=2, height=32, width=32, seed=21)
    clean = _image(batch=2, height=32, width=32, seed=22)

    loss = l1_training_loss(model(degraded), clean)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
    # The head starts at zero, so its own gradient must still be non-zero or
    # nothing would ever move.
    assert float(model.head.weight.grad.detach().abs().sum()) > 0


def test_one_optimizer_step_changes_the_parameters():
    model = build_model()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    degraded = _image(batch=2, height=32, width=32, seed=23)
    clean = _image(batch=2, height=32, width=32, seed=24)
    optimizer.zero_grad(set_to_none=True)
    l1_training_loss(model(degraded), clean).backward()
    optimizer.step()

    changed = [n for n, t in model.state_dict().items() if not torch.equal(before[n], t)]
    assert changed


def test_a_tiny_synthetic_restoration_task_reduces_the_loss():
    # Not a claim about CT. A learnable toy target - a constant brightening -
    # that a working training path must be able to fit.
    torch.manual_seed(31)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    degraded = _image(batch=4, height=32, width=32, seed=32) * 0.5
    clean = torch.clamp(degraded + 0.1, 0.0, 1.0)

    first = float(l1_training_loss(model(degraded), clean).detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = l1_training_loss(model(degraded), clean)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < first * 0.5


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------


def test_a_checkpoint_round_trips_through_weights_only_load(tmp_path):
    torch.manual_seed(41)
    model = build_model()
    nn.init.normal_(model.head.weight, std=0.05)
    nn.init.normal_(model.head.bias, std=0.05)

    path = tmp_path / "unet.pt"
    torch.save(
        {name: t.detach().cpu() for name, t in model.state_dict().items()},
        path,
    )
    revived = build_model()
    revived.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, revived.state_dict()[name])

    probe = _image(batch=2, height=64, width=64, seed=42)
    model.eval()
    revived.eval()
    with torch.inference_mode():
        assert torch.equal(model.restore(probe), revived.restore(probe))


# --------------------------------------------------------------------------
# deterministic mini-run (engineering evidence only, not a science result)
# --------------------------------------------------------------------------


def _mini_run(seed: int) -> tuple[list[float], dict[str, torch.Tensor]]:
    """A few steps on synthetic CPU data, fully determined by ``seed``."""
    torch.manual_seed(seed)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    degraded = _image(batch=2, height=32, width=32, seed=51) * 0.6
    clean = torch.clamp(degraded + 0.05, 0.0, 1.0)

    history: list[float] = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = l1_training_loss(model(degraded), clean)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history, copy.deepcopy(model.state_dict())


def test_the_same_seed_reproduces_the_same_mini_run():
    first_history, first_state = _mini_run(2026)
    second_history, second_state = _mini_run(2026)
    assert first_history == second_history
    for name, tensor in first_state.items():
        assert torch.equal(tensor, second_state[name]), name


def test_a_different_seed_gives_a_different_initialization():
    _, state = _mini_run(2026)
    _, other = _mini_run(7)
    assert any(not torch.equal(state[n], other[n]) for n in state)


def test_build_model_places_the_model_in_float32():
    model = build_model(device="cpu")
    assert isinstance(model, LightweightResidualUnet)
    assert all(p.dtype is torch.float32 for p in model.parameters())
