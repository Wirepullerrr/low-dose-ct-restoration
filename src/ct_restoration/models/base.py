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

Two halves of the input contract, checked in different places
-------------------------------------------------------------
The contract has a **structural** half - a ``torch.Tensor``, rank 4, the right
channel count, ``float32``, a spatial size the architecture can process - and
a **content** half: finite, and within [0, 1].

The structural half reads only tensor metadata, which lives on the host. It
never touches a pixel value and never waits for the GPU, so it is cheap enough
to run inside ``forward``, and :func:`validate_restoration_structure` is what
a model's own forward calls.

The content half has to reduce over every value and bring the answer back to
the host to decide whether to raise. On CUDA each of those answers is a
device-to-host synchronization: the CPU stops and waits for the GPU. Inside
``forward`` that would put three stalls into every inference and would bias
any timing of the model towards whichever architecture ran them.
:func:`validate_restoration_content` therefore runs where data enters rather
than where it is computed on: at the Dataset boundary, in the benchmark
adapter before inference, and once before any timed region. It is never
dropped; :func:`validate_restoration_input` still runs both halves.
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


def validate_restoration_structure(
    tensor: torch.Tensor, input_channels: int = 1, spatial_multiple: int = 1
) -> None:
    """Check a tensor has the shape and type of the model input, or refuse.

    The structural half of the contract. It reads metadata only - type, rank,
    channel count, dtype, height and width - never a tensor value, so it
    never synchronizes with the GPU and is what a model's ``forward`` runs.
    Whether the values are finite and in [0, 1] is
    :func:`validate_restoration_content`'s job, done before inference.

    ``spatial_multiple`` is what differs between the two architectures. The
    CNN is all stride-1 convolutions and accepts any size; the U-Net pools
    twice, so an odd height would be silently rounded on the way down and the
    skip tensor would no longer line up on the way back. Rather than crop or
    pad to make that work, an incompatible size is refused.

    Raises:
        ModelError: not a float32 ``[B, C, H, W]`` tensor, or a spatial size
            this architecture cannot process exactly.
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

    if spatial_multiple > 1:
        height, width = int(tensor.shape[2]), int(tensor.shape[3])
        if height % spatial_multiple or width % spatial_multiple:
            raise ModelError(
                f"model input height and width must both be divisible by {spatial_multiple} "
                f"for this architecture, got {height}x{width}. The model does not pad or "
                "crop to make an incompatible size fit."
            )


def validate_restoration_content(tensor: torch.Tensor) -> None:
    """Check a tensor's values are finite and within [0, 1], or refuse.

    The content half of the contract. Deciding whether to raise needs three
    reductions brought back to the host - all finite, the minimum, the
    maximum - and on CUDA each one is a device-to-host synchronization. So
    this runs where data enters, never inside ``forward`` and never inside a
    timed region: at the Dataset boundary, in the benchmark adapter before
    inference, and once before a latency measurement starts.

    Raises:
        ModelError: a NaN or infinite value, or a value outside [0, 1] by more
            than :data:`RANGE_TOLERANCE`.
    """
    # detach() before reducing to scalars: validation must never attach
    # itself to the autograd graph of the tensor it is inspecting.
    inspected = tensor.detach()
    if not bool(torch.isfinite(inspected).all()):
        raise ModelError("model input must be finite; found NaN or infinite values")
    minimum, maximum = float(inspected.min()), float(inspected.max())
    if minimum < -RANGE_TOLERANCE or maximum > 1.0 + RANGE_TOLERANCE:
        raise ModelError(
            f"model input must lie in [0, 1], got [{minimum:.6g}, {maximum:.6g}]. "
            "The model does not renormalize its input."
        )


def validate_restoration_input(
    tensor: torch.Tensor, input_channels: int = 1, spatial_multiple: int = 1
) -> None:
    """Check a tensor really is the benchmark's model input, or refuse.

    The whole contract: :func:`validate_restoration_structure`, then
    :func:`validate_restoration_content`. Strict and non-repairing, like
    every other boundary in this project. The Dataset already guarantees this
    for training and evaluation data; the check is repeated where a model is
    fed from outside that path, because a silently renormalized or non-finite
    batch would train and score perfectly happily on numbers that no longer
    mean what the report says.

    It synchronizes with the GPU on a CUDA tensor, through its content half,
    so it belongs before inference, not inside it.

    Raises:
        ModelError: not a finite float32 ``[B, C, H, W]`` tensor in [0, 1], or
            a spatial size this architecture cannot process exactly.
    """
    validate_restoration_structure(
        tensor, input_channels=input_channels, spatial_multiple=spatial_multiple
    )
    validate_restoration_content(tensor)


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
