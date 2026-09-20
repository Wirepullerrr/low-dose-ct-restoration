"""What every learned restoration method in this benchmark has in common.

Two architectures now share this project: the small residual CNN of
Milestone 8 and the lightweight residual U-Net of Milestone 9. They differ in
almost everything internal - depth, pooling, skip connections, parameter
count - and in nothing that the benchmark can see.

That is deliberate, and this module is where it is enforced. Both models:

* take the degraded image and nothing else - no clean reference, no body
  mask, no patient identity;
* take a finite ``float32`` ``[B, 1, H, W]`` tensor in [0, 1];
* predict an additive **correction** rather than a whole image;
* expose ``forward`` as the raw, unclamped restoration and ``restore`` as the
  clamped benchmark output;
* start life as the exact identity, because their final layer is
  zero-initialized.

Keeping the contract in one place is what makes a CNN-versus-U-Net comparison
a comparison of architectures. If each model brought its own input
validation, its own idea of what "restored" means, or its own clamping
convention, a measured difference between them could be any of those things
rather than the thing being studied.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

#: How far outside [0, 1] an input may stray before it is rejected. Only
#: float32 round-trip error is tolerated, never a genuinely out-of-range image.
RANGE_TOLERANCE = 1e-6


class ModelError(ValueError):
    """The model configuration or an input tensor is not usable as given."""


def require_positive_integer(name: str, value: Any) -> int:
    """Accept only a genuine positive integer, mirroring the config modules.

    A bool is not an integer here even though Python says it is, and a string
    is never parsed: a config that says ``"32"`` is a config someone should
    fix, not one this code should guess at.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ModelError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}. "
            "It is not rounded, truncated or parsed from a string."
        )
    if int(value) < 1:
        raise ModelError(f"{name} must be >= 1, got {value!r}")
    return int(value)


def validate_restoration_input(
    tensor: torch.Tensor, input_channels: int = 1, spatial_multiple: int = 1
) -> None:
    """Check a tensor really is the benchmark's model input, or refuse.

    Strict and non-repairing, like every other boundary in this project. The
    Dataset already guarantees this for training and evaluation data; the
    check is repeated where a model is actually fed, because a silently
    renormalized or non-finite batch would train and score perfectly happily
    on numbers that no longer mean what the report says.

    ``spatial_multiple`` is what differs between the two architectures. The
    CNN is all stride-1 convolutions and accepts any size; the U-Net pools
    twice, so an odd height would be silently rounded on the way down and the
    skip tensor would no longer line up on the way back. Rather than crop or
    pad to make that work, an incompatible size is refused.

    Raises:
        ModelError: not a finite float32 ``[B, C, H, W]`` tensor in [0, 1], or
            a spatial size this architecture cannot process exactly.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ModelError(f"model input must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dim() != 4:
        raise ModelError(f"model input must be [B, C, H, W], got shape {tuple(tensor.shape)}")
    if tensor.shape[1] != input_channels:
        raise ModelError(
            f"model input must have {input_channels} channel(s), got "
            f"{tensor.shape[1]} in shape {tuple(tensor.shape)}"
        )
    if tensor.dtype is not torch.float32:
        raise ModelError(f"model input must be float32, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise ModelError("model input must be finite; found NaN or infinite values")

    if spatial_multiple > 1:
        height, width = int(tensor.shape[2]), int(tensor.shape[3])
        if height % spatial_multiple or width % spatial_multiple:
            raise ModelError(
                f"model input height and width must both be divisible by {spatial_multiple} "
                f"for this architecture, got {height}x{width}. The model does not pad or "
                "crop to make an incompatible size fit."
            )

    # detach() before reducing to scalars: validation must never attach
    # itself to the autograd graph of the tensor it is inspecting.
    inspected = tensor.detach()
    minimum, maximum = float(inspected.min()), float(inspected.max())
    if minimum < -RANGE_TOLERANCE or maximum > 1.0 + RANGE_TOLERANCE:
        raise ModelError(
            f"model input must lie in [0, 1], got [{minimum:.6g}, {maximum:.6g}]. "
            "The model does not renormalize its input."
        )


@runtime_checkable
class RestorationModel(Protocol):
    """The surface the benchmark, the adapter and the training loop rely on.

    Structural, not inherited: a model satisfies this by having the methods,
    which keeps the two architectures independent of each other while making
    the shared harness honest about what it actually calls.
    """

    config: Any

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        """The raw restoration, ``degraded + correction``, NOT clamped."""
        ...

    def restore(self, degraded: torch.Tensor) -> torch.Tensor:
        """The benchmark output: the raw restoration clamped to [0, 1]."""
        ...
