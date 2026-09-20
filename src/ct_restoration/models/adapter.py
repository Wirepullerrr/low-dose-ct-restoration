"""Bridging a torch model into the benchmark's one-argument method contract.

:mod:`ct_restoration.benchmark` scores a method as a
``Callable[[np.ndarray], np.ndarray]``: degraded image in, restored image out,
and nothing else in either direction. That contract is what guarantees no
method can reach the clean reference, the body mask or the patient identity,
and it is why the CNN is scored by the same loop, the same masks and the same
metric code as the degraded baseline and CLAHE - not by a parallel evaluation
path written for learned methods.

This module is the whole adapter. It is intentionally tiny and lives here
rather than inside ``benchmark.py``, so the evaluation harness stays
importable without touching torch and the Milestone 5 and 6 artifacts are
produced by byte-identical code.

It is written against the :class:`~ct_restoration.models.base.RestorationModel`
contract rather than against any one architecture, so the residual CNN and the
lightweight U-Net reach the benchmark through exactly the same code path. An
adapter that knew which model it was holding would be a place for the two
methods to diverge without anyone noticing.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ct_restoration.models.base import RestorationModel, validate_restoration_input


def _validate(model: RestorationModel, batch: torch.Tensor) -> None:
    """Apply the model's own input contract, whatever architecture it is."""
    validate_restoration_input(
        batch,
        input_channels=model.config.input_channels,
        spatial_multiple=getattr(model.config, "spatial_multiple", 1),
    )


def restore_array(
    model: RestorationModel, degraded: np.ndarray, device: torch.device | str = "cpu"
) -> np.ndarray:
    """Restore one 2-D degraded slice, clamped to [0, 1], as ``float32``.

    Wrapped in ``eval()`` and ``inference_mode()``: no dropout or batch-norm
    state to worry about in this architecture, but a model left in training
    mode is exactly the kind of difference that silently changes a reported
    number in a larger one.
    """
    array = np.asarray(degraded)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D degraded slice, got shape {array.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    batch = tensor.unsqueeze(0).unsqueeze(0).to(device=torch.device(device))
    _validate(model, batch)

    model.eval()
    with torch.inference_mode():
        restored = model.restore(batch)
    return restored.squeeze(0).squeeze(0).detach().to("cpu").numpy().astype(np.float32)


def torch_restorer(
    model: RestorationModel, device: torch.device | str = "cpu"
) -> Callable[[np.ndarray], np.ndarray]:
    """Bind a model into the harness's one-argument restoration contract.

    The returned callable takes only the degraded image, which is the point:
    the method cannot reach the clean reference, the mask or the patient.
    """

    def restore(degraded: np.ndarray) -> np.ndarray:
        return restore_array(model, degraded, device)

    return restore


def raw_restore_array(
    model: RestorationModel, degraded: np.ndarray, device: torch.device | str = "cpu"
) -> np.ndarray:
    """The **unclamped** restoration, for clipping diagnostics only.

    Never used to compute a reported metric: the benchmark output is the
    clamped image. This exists so the report can say how much work the final
    clamp is doing, rather than assuming it is a rare safety boundary.
    """
    array = np.asarray(degraded)
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    batch = tensor.unsqueeze(0).unsqueeze(0).to(device=torch.device(device))
    _validate(model, batch)

    model.eval()
    with torch.inference_mode():
        raw = model.forward(batch)
    return raw.squeeze(0).squeeze(0).detach().to("cpu").numpy().astype(np.float32)
