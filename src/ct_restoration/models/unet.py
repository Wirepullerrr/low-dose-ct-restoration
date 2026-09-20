"""The lightweight residual U-Net: the second learned method in this benchmark.

Two downsampling levels, 16 base channels, two 3x3 convolutions per block,
concatenative skip connections, and a 1x1 correction head. 116,753 trainable
parameters, a 44x44 maximum receptive field through the deepest path.

Why this model, and why exactly like this
-----------------------------------------
The Milestone 8 residual CNN sees an 11x11 pixel neighbourhood and nothing
else. That is the one architectural property this milestone changes: a U-Net
pools its way down to a coarse representation, works there, and brings the
result back up, so the deepest path to an output pixel covers a far wider
window. Everything else - the frozen corruption, the Dataset, the sampler,
the loss, the optimizer, the seed, the epoch budget, the batch size and the
checkpoint-selection rule - is held identical to the CNN's, so a difference
in the measured result is attributable to architecture rather than to a
second change smuggled in beside it.

It is deliberately small. A large U-Net would probably score better and would
tell us less: the question is whether multi-scale context helps *at this
scale of model*, not whether a bigger network wins.

Encoder, bottleneck, decoder
----------------------------
::

    256x256x1  --conv,conv-->  256x256x16  ---------------skip 1--------+
                    |                                                   |
                 maxpool 2x2                                            |
                    v                                                   |
    128x128x16 --conv,conv-->  128x128x32  ------skip 2------+          |
                    |                                        |          |
                 maxpool 2x2                                 |          |
                    v                                        v          v
     64x64x32  --conv,conv-->   64x64x64  --up--> 128x128x32 cat -> ... up -> 256x256x16 cat -> ...
                                (bottleneck)

Spatial resolution falls by half at each pooling step while the channel count
doubles. That trade is the whole point of the shape: after pooling, one pixel
of the feature map summarises a larger patch of the image, so the same 3x3
convolution now relates things that were further apart in the original. The
extra channels are the capacity to describe what those larger patches
contain.

Skip connections are **concatenation**, not addition
----------------------------------------------------
Each encoder block's output is saved and joined to the matching decoder stage
along the channel dimension. Pooling coarsens the representation, and the
skip carries higher-resolution encoder features around that bottleneck: the
decoder gets the coarse, wide-context features it computed *and* features at
the finer scale that never passed through the coarsest representation.

A skip is not the raw input pixels and does not restore what pooling
removed. It is a feature map produced by that block's two convolutions -
already transformed, just not yet downsampled - so what it provides is
access to fine-scale feature information, not perfect preservation.

Concatenation rather than summation because the two are not the same kind of
quantity. Adding them forces the network to treat a fine-detail feature and a
context feature as interchangeable and commits to a fixed one-to-one mixing;
concatenating hands both to the next convolution and lets it learn the
mixing, at the cost of more channels to convolve over.

Receptive field
---------------
Through the deepest path - down two pools, across the bottleneck, and back up
- one output pixel depends on a **44x44** input window::

    conv,conv      1 ->  5   (jump 1)
    pool           5 ->  6   (jump 2)
    conv,conv      6 -> 14   (jump 2)
    pool          14 -> 16   (jump 4)
    conv,conv     16 -> 32   (jump 4)
    up-conv       32 -> 32   (jump 2, non-overlapping: no extra extent)
    conv,conv     32 -> 40   (jump 2)
    up-conv       40 -> 40   (jump 1)
    conv,conv     40 -> 44   (jump 1)
    1x1 head      44 -> 44

Two cautions about that number. It is the **maximum** over paths, not the
only path: the skip connections deliberately provide shallower routes that
carry small-scale local information, and a given output pixel's value mixes
contributions from all of them. And it is a statement about *pixels*. It is
not anatomical coverage, not clinical context, and the model has no access to
organ identity, slice position or anything outside the image.

Residual at the image level
---------------------------
Exactly as the CNN::

    raw_restored = degraded + UNet(degraded)
    restored     = clamp(raw_restored, 0, 1)

The U-Net predicts a correction, not a clean image, so the two methods share
the same high-level parameterization and differ only inside the box. The
final 1x1 convolution is zero-initialized, so before training the correction
is identically zero and the untrained model **is** the no-restoration
baseline - which makes epoch 0 a real check on the evaluation path rather
than a formality.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ct_restoration.models.base import (
    RANGE_TOLERANCE,
    ModelError,
    validate_restoration_input,
)
from ct_restoration.models.base import (
    require_positive_integer as _require_positive_integer,
)

__all__ = [
    "ALGORITHM_VERSION",
    "CANONICAL_PARAMETER_COUNT",
    "CANONICAL_RECEPTIVE_FIELD",
    "RANGE_TOLERANCE",
    "LightweightResidualUnet",
    "LightweightResidualUnetConfig",
    "ModelError",
    "build_model",
    "validate_model_input",
]

#: The only architecture this module implements. A config naming anything
#: else is refused rather than approximated.
ALGORITHM_VERSION = "lightweight_residual_unet_v1"

#: Trainable parameters of the canonical configuration. Follows from the
#: frozen architecture; it is pinned, never tuned towards.
CANONICAL_PARAMETER_COUNT = 116_753

#: Maximum input receptive field through the deepest encoder-decoder path, in
#: pixels. Shallower skip paths also contribute; see the module docstring.
CANONICAL_RECEPTIVE_FIELD = 44


def _double_convolution(in_channels: int, mid_channels: int, kernel_size: int, padding: int):
    """Two size-preserving convolutions, ReLU after each. The U-Net's unit."""
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=True),
        nn.ReLU(inplace=False),
        nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=True),
        nn.ReLU(inplace=False),
    )


@dataclass(frozen=True)
class LightweightResidualUnetConfig:
    """Architecture of one lightweight residual U-Net.

    Only the canonical shape is supported. A config that names a different
    algorithm, asks for batch norm, dropout, a non-ReLU activation, additive
    skips, bilinear upsampling or a non-residual head is refused rather than
    silently approximated, so a future v2 definition can never be executed by
    v1 code.
    """

    algorithm: str = ALGORITHM_VERSION
    input_channels: int = 1
    output_channels: int = 1
    base_channels: int = 16
    levels: int = 2
    convolutions_per_block: int = 2
    kernel_size: int = 3
    padding: int = 1
    activation: str = "relu"
    downsampling: str = "max_pool_2x2"
    upsampling: str = "conv_transpose_2x2_stride2"
    skip_connection: str = "concatenate"
    batch_norm: bool = False
    dropout: float = 0.0
    residual_correction: bool = True
    zero_initialize_final_layer: bool = True
    final_kernel_size: int = 1

    def __post_init__(self) -> None:
        if self.algorithm != ALGORITHM_VERSION:
            raise ModelError(
                f"Unsupported model algorithm {self.algorithm!r}; this module implements "
                f"{ALGORITHM_VERSION!r} only."
            )
        for name in ("input_channels", "output_channels", "base_channels", "levels"):
            _require_positive_integer(name, getattr(self, name))
        if _require_positive_integer("convolutions_per_block", self.convolutions_per_block) != 2:
            raise ModelError(
                f"convolutions_per_block must be 2 in this architecture, got "
                f"{self.convolutions_per_block}"
            )
        if _require_positive_integer("kernel_size", self.kernel_size) % 2 != 1:
            raise ModelError(
                f"kernel_size must be odd so padding can preserve size, got {self.kernel_size}"
            )
        if isinstance(self.padding, bool) or not isinstance(self.padding, int | np.integer):
            raise ModelError(f"padding must be an integer, got {self.padding!r}")
        if int(self.padding) != self.kernel_size // 2:
            raise ModelError(
                f"padding {self.padding} would not preserve the spatial size for kernel_size "
                f"{self.kernel_size}; expected {self.kernel_size // 2}."
            )
        if _require_positive_integer("final_kernel_size", self.final_kernel_size) != 1:
            raise ModelError(
                f"final_kernel_size must be 1 in this architecture, got {self.final_kernel_size}"
            )
        if self.activation != "relu":
            raise ModelError(f"only 'relu' is implemented, got {self.activation!r}")
        if self.downsampling != "max_pool_2x2":
            raise ModelError(
                f"only 'max_pool_2x2' downsampling is implemented, got {self.downsampling!r}"
            )
        if self.upsampling != "conv_transpose_2x2_stride2":
            raise ModelError(
                "only 'conv_transpose_2x2_stride2' upsampling is implemented, got "
                f"{self.upsampling!r}"
            )
        if self.skip_connection != "concatenate":
            raise ModelError(
                f"only concatenative skip connections are implemented, got "
                f"{self.skip_connection!r}. Summation is a different architecture."
            )
        if self.batch_norm is not False:
            raise ModelError("batch_norm must be false in this architecture")
        if isinstance(self.dropout, bool) or float(self.dropout) != 0.0:
            raise ModelError(f"dropout must be 0.0 in this architecture, got {self.dropout!r}")
        if self.residual_correction is not True:
            raise ModelError("residual_correction must be true in this architecture")
        if self.zero_initialize_final_layer is not True:
            raise ModelError("zero_initialize_final_layer must be true in this architecture")
        if self.input_channels != self.output_channels:
            raise ModelError(
                "a residual correction needs matching input and output channels, got "
                f"{self.input_channels} and {self.output_channels}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> LightweightResidualUnetConfig:
        """Build a config from a loaded YAML document.

        Accepts the whole file, which nests the architecture under ``model``,
        or that inner mapping alone. Values are passed through untouched so
        validation, not silent coercion, decides whether they are usable.
        """
        if not isinstance(mapping, Mapping):
            raise ModelError(f"U-Net config must be a mapping, got {type(mapping).__name__}")
        section = mapping.get("model", mapping)
        if not isinstance(section, Mapping):
            raise ModelError(f"'model' section must be a mapping, got {type(section).__name__}")

        fields = tuple(cls.__dataclass_fields__)
        missing = sorted(set(fields) - set(section))
        if missing:
            raise ModelError(f"U-Net model config is missing key(s): {missing}")
        unknown = sorted(set(section) - set(fields))
        if unknown:
            raise ModelError(
                f"U-Net model config has unrecognised key(s): {unknown}. "
                f"Known keys are {sorted(fields)}."
            )
        return cls(**{name: section[name] for name in fields})

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "input_channels": int(self.input_channels),
            "output_channels": int(self.output_channels),
            "base_channels": int(self.base_channels),
            "levels": int(self.levels),
            "convolutions_per_block": int(self.convolutions_per_block),
            "kernel_size": int(self.kernel_size),
            "padding": int(self.padding),
            "activation": self.activation,
            "downsampling": self.downsampling,
            "upsampling": self.upsampling,
            "skip_connection": self.skip_connection,
            "batch_norm": bool(self.batch_norm),
            "dropout": float(self.dropout),
            "residual_correction": bool(self.residual_correction),
            "zero_initialize_final_layer": bool(self.zero_initialize_final_layer),
            "final_kernel_size": int(self.final_kernel_size),
        }

    @property
    def channels_per_level(self) -> tuple[int, ...]:
        """Encoder widths, doubling per level: (16, 32) then 64 at the bottom."""
        return tuple(self.base_channels * 2**level for level in range(self.levels))

    @property
    def bottleneck_channels(self) -> int:
        return self.base_channels * 2**self.levels

    @property
    def spatial_multiple(self) -> int:
        """Input height and width must be a multiple of this.

        Each level halves the resolution, so a size not divisible by
        ``2 ** levels`` would be rounded down on the way in and the skip
        tensor would no longer match the upsampled one on the way out. The
        model refuses such an input rather than cropping or padding it.
        """
        return 2**self.levels

    @property
    def receptive_field(self) -> int:
        """Maximum input receptive field through the deepest path, in pixels.

        Accumulated forward: a size-preserving convolution adds
        ``(kernel - 1) * jump`` and leaves the jump alone; a 2x2 stride-2 pool
        adds ``jump`` and doubles it; a 2x2 stride-2 transposed convolution is
        non-overlapping, so each output pixel comes from exactly one input
        pixel - it halves the jump and adds no extent.

        Skip connections also provide shallower paths, so this is the maximum
        over paths and not a description of every contribution.
        """
        widening = self.kernel_size - 1
        field, jump = 1, 1
        for _ in range(self.levels):
            for _ in range(self.convolutions_per_block):
                field += widening * jump
            field += jump  # 2x2 stride-2 pool
            jump *= 2
        for _ in range(self.convolutions_per_block):  # bottleneck
            field += widening * jump
        for _ in range(self.levels):
            jump //= 2  # transposed convolution: no extent, half the jump
            for _ in range(self.convolutions_per_block):
                field += widening * jump
        return field  # the 1x1 head adds nothing


def validate_model_input(
    tensor: torch.Tensor, config: LightweightResidualUnetConfig | None = None
) -> None:
    """Check a tensor really is this model's input, or refuse.

    The U-Net adds one requirement the CNN does not have: height and width
    must both be divisible by ``2 ** levels``.
    """
    config = config or LightweightResidualUnetConfig()
    validate_restoration_input(
        tensor,
        input_channels=config.input_channels,
        spatial_multiple=config.spatial_multiple,
    )


class LightweightResidualUnet(nn.Module):
    """Predicts an additive correction to the degraded image, multi-scale.

    The module sees the degraded image and nothing else: no clean reference,
    no body mask, no HU slice, no patient identity. Those exist for
    supervision and evaluation, and a model that could read them would be
    scoring itself against information a deployment would not have.
    """

    def __init__(self, config: LightweightResidualUnetConfig | None = None) -> None:
        super().__init__()
        self.config = config or LightweightResidualUnetConfig()
        kernel, padding = self.config.kernel_size, self.config.padding
        widths = self.config.channels_per_level
        bottom = self.config.bottleneck_channels

        # Encoder: one double convolution per level, each followed by a pool.
        self.encoders = nn.ModuleList()
        in_channels = self.config.input_channels
        for width in widths:
            self.encoders.append(_double_convolution(in_channels, width, kernel, padding))
            in_channels = width
        self.pools = nn.ModuleList(nn.MaxPool2d(kernel_size=2, stride=2) for _ in widths)

        self.bottleneck = _double_convolution(in_channels, bottom, kernel, padding)

        # Decoder: upsample, concatenate the matching skip, double convolve.
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        below = bottom
        for width in reversed(widths):
            self.upsamplers.append(
                nn.ConvTranspose2d(below, width, kernel_size=2, stride=2, bias=True)
            )
            # After concatenation the decoder sees the upsampled tensor and
            # the skip side by side: width + width channels.
            self.decoders.append(_double_convolution(width * 2, width, kernel, padding))
            below = width

        self.head = nn.Conv2d(
            widths[0],
            self.config.output_channels,
            kernel_size=self.config.final_kernel_size,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-normal throughout, exact zeros for the 1x1 correction head."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d | nn.ConvTranspose2d) and module is not self.head:
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.config.zero_initialize_final_layer:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @property
    def receptive_field(self) -> int:
        return self.config.receptive_field

    def parameter_count(self) -> int:
        """Trainable parameters. 116,753 for the canonical configuration."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def correction(self, degraded: torch.Tensor) -> torch.Tensor:
        """The predicted additive correction, unbounded and unclamped."""
        validate_restoration_input(
            degraded,
            input_channels=self.config.input_channels,
            spatial_multiple=self.config.spatial_multiple,
        )
        skips: list[torch.Tensor] = []
        features = degraded
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            features = encoder(features)
            skips.append(features)
            features = pool(features)

        features = self.bottleneck(features)

        for upsampler, decoder, skip in zip(
            self.upsamplers, self.decoders, reversed(skips), strict=True
        ):
            features = upsampler(features)
            # Sizes match exactly for any input the validator accepted, so
            # there is nothing to crop. A mismatch here would be a bug, not
            # something to paper over.
            if features.shape[-2:] != skip.shape[-2:]:  # pragma: no cover - guarded upstream
                raise ModelError(
                    f"upsampled {tuple(features.shape[-2:])} does not match skip "
                    f"{tuple(skip.shape[-2:])}; the input size is not usable by this model."
                )
            features = torch.cat([features, skip], dim=1)
            features = decoder(features)

        return self.head(features)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        """The raw restored image, ``degraded + correction``, NOT clamped.

        Training loss is computed on this. Clamping here would create
        zero-gradient regions wherever the prediction left [0, 1], so a pixel
        that overshot would stop receiving any signal to come back.
        """
        return degraded + self.correction(degraded)

    def restore(self, degraded: torch.Tensor) -> torch.Tensor:
        """The benchmark output: the raw restoration clamped to [0, 1]."""
        return torch.clamp(self.forward(degraded), 0.0, 1.0)


def build_model(
    config: LightweightResidualUnetConfig | None = None,
    device: torch.device | str | None = None,
) -> LightweightResidualUnet:
    """Construct the model, optionally on a device, in float32."""
    model = LightweightResidualUnet(config)
    model = model.to(dtype=torch.float32)
    if device is not None:
        model = model.to(device=torch.device(device))
    return model
