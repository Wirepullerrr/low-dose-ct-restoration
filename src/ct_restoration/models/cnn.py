"""The small residual CNN: the first learned method in this benchmark.

Five 3x3 convolutions, 32 hidden channels, ReLU between them, nothing else.
No batch norm, no dropout, no pooling, no striding, no dilation, no attention,
and no skip connections other than one global residual. 28,353 trainable
parameters, an 11x11 receptive field.

It is deliberately the smallest thing that could plausibly work. A large model
that beats the degraded baseline would leave open whether the benchmark is
simply easy; a small one that does or does not is a cleaner first data point,
and it trains fast enough that the whole run is reproducible in minutes.

Residual, not direct
--------------------
The network predicts a **correction**, and the restored image is::

    restored_raw = degraded + CNN(degraded)
    restored     = clamp(restored_raw, 0, 1)

The alternative - predicting the clean image directly - would require the
network to reconstruct every pixel of the anatomy from scratch, including all
the structure the degraded image already carries perfectly well. Predicting
what to *change* means most of the image is already correct before the network
does anything, and the identity transform is reachable by outputting zero
rather than by learning to copy.

The correction is unbounded on purpose. No sigmoid or tanh on the final
convolution: squashing the output would bound the correction to a fixed range
and put a saturating nonlinearity in the gradient path for exactly the pixels
that need the largest change.

Zero-initialized final layer
----------------------------
The final convolution's weight and bias both start at zero, so at
initialization the correction is exactly 0 and ``restored_raw == degraded``
bit for bit. Because the degraded input is already in [0, 1], clamping changes
nothing, and the untrained network **is** the no-restoration baseline. Two
things follow: training starts from a known, already-measured operating point
rather than from noise, and epoch 0 is a real check on the evaluation path -
if it does not reproduce the frozen degraded baseline, something in the
learned-method plumbing disagrees with the benchmark.

The earlier convolutions use ordinary Kaiming-normal initialization for ReLU.

Receptive field
---------------
Five stacked 3x3 convolutions with stride 1 see an 11x11 pixel neighbourhood:
each layer adds one pixel on each side, so ``1 + 5 * 2 = 11``. That is a
statement about pixels, and nothing more. On the 256x256 benchmark image at
CHAOS in-plane spacing it is a small local window; it is not "anatomical
context", and the model has no access to organ identity, slice position or
anything outside those 11x11 pixels.
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

#: The only architecture this module implements. A config naming anything else
#: is refused rather than approximated.
ALGORITHM_VERSION = "residual_cnn_v1"

#: Trainable parameters of the canonical configuration, with bias on all five
#: convolutions: 320 + 3 * 9248 + 289.
CANONICAL_PARAMETER_COUNT = 28_353

#: Side of the square pixel neighbourhood one output pixel depends on.
CANONICAL_RECEPTIVE_FIELD = 11

#: ``ModelError``, ``RANGE_TOLERANCE`` and the input contract are shared with
#: the U-Net and live in :mod:`ct_restoration.models.base`. They are re-exported
#: here because this module was their original home and callers import them
#: from it.
__all__ = [
    "ALGORITHM_VERSION",
    "CANONICAL_PARAMETER_COUNT",
    "CANONICAL_RECEPTIVE_FIELD",
    "RANGE_TOLERANCE",
    "ModelError",
    "ResidualCNN",
    "ResidualCnnConfig",
    "build_model",
    "validate_model_input",
]


@dataclass(frozen=True)
class ResidualCnnConfig:
    """Architecture of one residual CNN.

    Only the canonical shape is supported: a config that names a different
    algorithm, asks for batch norm, dropout, a non-ReLU activation or a
    non-residual head is refused rather than silently approximated, so a
    future v2 definition can never be executed by v1 code.
    """

    algorithm: str = ALGORITHM_VERSION
    input_channels: int = 1
    output_channels: int = 1
    hidden_channels: int = 32
    depth: int = 5
    kernel_size: int = 3
    padding: int = 1
    activation: str = "relu"
    batch_norm: bool = False
    dropout: float = 0.0
    residual_correction: bool = True
    zero_initialize_final_layer: bool = True

    def __post_init__(self) -> None:
        if self.algorithm != ALGORITHM_VERSION:
            raise ModelError(
                f"Unsupported model algorithm {self.algorithm!r}; this module implements "
                f"{ALGORITHM_VERSION!r} only."
            )
        for name in ("input_channels", "output_channels", "hidden_channels", "depth"):
            _require_positive_integer(name, getattr(self, name))
        if self.depth < 2:
            raise ModelError(f"depth must be at least 2, got {self.depth}")
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
        if self.activation != "relu":
            raise ModelError(f"only 'relu' is implemented, got {self.activation!r}")
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
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ResidualCnnConfig:
        """Build a config from a loaded YAML document.

        Accepts the whole file, which nests the architecture under ``model``,
        or that inner mapping alone. Values are passed through untouched so
        validation, not silent coercion, decides whether they are usable.
        """
        if not isinstance(mapping, Mapping):
            raise ModelError(f"CNN config must be a mapping, got {type(mapping).__name__}")
        section = mapping.get("model", mapping)
        if not isinstance(section, Mapping):
            raise ModelError(f"'model' section must be a mapping, got {type(section).__name__}")

        fields = tuple(cls.__dataclass_fields__)
        missing = sorted(set(fields) - set(section))
        if missing:
            raise ModelError(f"CNN model config is missing key(s): {missing}")
        unknown = sorted(set(section) - set(fields))
        if unknown:
            raise ModelError(
                f"CNN model config has unrecognised key(s): {unknown}. "
                f"Known keys are {sorted(fields)}."
            )
        return cls(**{name: section[name] for name in fields})

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "input_channels": int(self.input_channels),
            "output_channels": int(self.output_channels),
            "hidden_channels": int(self.hidden_channels),
            "depth": int(self.depth),
            "kernel_size": int(self.kernel_size),
            "padding": int(self.padding),
            "activation": self.activation,
            "batch_norm": bool(self.batch_norm),
            "dropout": float(self.dropout),
            "residual_correction": bool(self.residual_correction),
            "zero_initialize_final_layer": bool(self.zero_initialize_final_layer),
        }

    @property
    def receptive_field(self) -> int:
        """Side of the pixel neighbourhood one output pixel depends on.

        Stride 1 and no dilation, so each layer widens the field by
        ``kernel_size - 1`` and the stack sees ``1 + depth * (kernel - 1)``.
        """
        return 1 + self.depth * (self.kernel_size - 1)

    @property
    def spatial_multiple(self) -> int:
        """Input height and width must be a multiple of this.

        One: every convolution here is stride 1 with size-preserving padding,
        so any spatial size works. The U-Net, which pools twice, needs 4.
        """
        return 1


def validate_model_input(tensor: torch.Tensor, config: ResidualCnnConfig | None = None) -> None:
    """Check a tensor really is this model's input, or refuse.

    A thin CNN-facing wrapper over the shared contract in
    :mod:`ct_restoration.models.base`, kept because this is where callers
    already import it from. The CNN is all stride-1 convolutions, so it
    imposes no spatial-divisibility requirement.
    """
    config = config or ResidualCnnConfig()
    validate_restoration_input(
        tensor,
        input_channels=config.input_channels,
        spatial_multiple=config.spatial_multiple,
    )


class ResidualCNN(nn.Module):
    """Predicts an additive correction to the degraded image.

    The module sees the degraded image and nothing else: no clean reference,
    no body mask, no HU slice, no patient identity. Those exist for
    supervision and evaluation, and a model that could read them would be
    scoring itself against information a deployment would not have.
    """

    def __init__(self, config: ResidualCnnConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResidualCnnConfig()

        widths = (
            [self.config.input_channels]
            + [self.config.hidden_channels] * (self.config.depth - 1)
            + [self.config.output_channels]
        )
        layers: list[nn.Module] = []
        for index in range(self.config.depth):
            layers.append(
                nn.Conv2d(
                    widths[index],
                    widths[index + 1],
                    kernel_size=self.config.kernel_size,
                    padding=self.config.padding,
                    bias=True,
                )
            )
            # ReLU between convolutions, never after the last one: the
            # correction has to be able to go negative.
            if index < self.config.depth - 1:
                layers.append(nn.ReLU(inplace=False))
        self.layers = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-normal for the ReLU stack, exact zeros for the final layer."""
        convolutions = [layer for layer in self.layers if isinstance(layer, nn.Conv2d)]
        for convolution in convolutions[:-1]:
            nn.init.kaiming_normal_(convolution.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(convolution.bias)
        if self.config.zero_initialize_final_layer:
            nn.init.zeros_(convolutions[-1].weight)
            nn.init.zeros_(convolutions[-1].bias)

    @property
    def receptive_field(self) -> int:
        return self.config.receptive_field

    def parameter_count(self) -> int:
        """Trainable parameters. 28,353 for the canonical configuration."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def correction(self, degraded: torch.Tensor) -> torch.Tensor:
        """The predicted additive correction, unbounded and unclamped."""
        if degraded.dim() != 4:
            raise ModelError(f"model input must be [B, C, H, W], got shape {tuple(degraded.shape)}")
        if degraded.shape[1] != self.config.input_channels:
            raise ModelError(
                f"model input must have {self.config.input_channels} channel(s), got "
                f"{degraded.shape[1]} in shape {tuple(degraded.shape)}"
            )
        return self.layers(degraded)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        """The raw restored image, ``degraded + correction``, NOT clamped.

        Training loss is computed on this. Clamping here would create
        zero-gradient regions wherever the prediction left [0, 1], so a pixel
        that overshot would stop receiving any signal to come back.
        """
        return degraded + self.correction(degraded)

    def restore(self, degraded: torch.Tensor) -> torch.Tensor:
        """The benchmark output: the raw restoration clamped to [0, 1].

        Every other method in this benchmark produces an image in [0, 1], and
        every metric assumes ``data_range = 1.0``, so the comparison is only
        meaningful if the learned method is held to the same representation.
        """
        return torch.clamp(self.forward(degraded), 0.0, 1.0)


def build_model(
    config: ResidualCnnConfig | None = None, device: torch.device | str | None = None
) -> ResidualCNN:
    """Construct the model, optionally on a device, in float32."""
    model = ResidualCNN(config)
    model = model.to(dtype=torch.float32)
    if device is not None:
        model = model.to(device=torch.device(device))
    return model
