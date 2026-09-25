"""The Milestone 12 latency benchmark: one frozen protocol, measured once.

Milestone 11 fixed the image-quality result permanently. This module measures
the other half of the trade-off - how long each method takes to restore one
image - and nothing else. It reads no CT image of any split, recomputes no
metric, changes no checkpoint and trains nothing.

The frozen plan is :file:`configs/latency/benchmark_plan.yaml`, committed
before any latency is measured. This module refuses it unless it agrees with
the constants below, with the committed held-out plan and quality summary and
with the bytes of both checkpoints, so neither the plan nor the code can
drift alone.

What is timed
-------------
**Learned models: GPU model inference latency.** An already-on-GPU float32
``[1, 1, 256, 256]`` tensor in, ``model.restore`` out: the structural input
check, the forward pass, the residual addition and the clamp. Each sample is
one CUDA-event interval around exactly that call: a device-side interval, from
the GPU reaching the start event to it reaching the end event, not CPU
wall-clock time. Each measured inference is isolated by waiting for its end
event before the next iteration begins. The wait occurs outside the
CUDA-event timing interval and is therefore not included in reported
latency. The protocol targets isolated batch-1 GPU inference latency rather
than sustained throughput. There is no synchronization inside the interval:
the value checks that would need one run before the first warm-up call.

**CLAHE: CPU latency of the exact canonical method,** ``apply_clahe`` on the
same pixels as a 2-D float32 image, bracketed by ``time.perf_counter_ns``.

The two are on different hardware with different timers. They are reported
side by side and never as a speed ratio.

What is not timed
-----------------
DICOM reading, HU conversion, windowing, resizing, degradation, the
DataLoader, host-device transfers, checkpoint loading, model construction,
input value validation and metrics. The learned number is **not** end-to-end
CT processing latency, and it describes this machine, not the architectures
on other hardware.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ct_restoration.benchmark import write_csv, write_json
from ct_restoration.classical.clahe import ClaheConfig, apply_clahe
from ct_restoration.evaluation_integrity import file_sha256
from ct_restoration.holdout import (
    CheckReport,
    DicomOpenMonitor,
    GitState,
    array_sha256,
    text_sha256_lf,
)

# --------------------------------------------------------------------------
# The frozen protocol
# --------------------------------------------------------------------------

#: The frozen plan, relative to the repository root.
PLAN_PATH = Path("configs/latency/benchmark_plan.yaml")

#: Every file a complete measurement writes, and no other.
OUTPUT_FILES: tuple[str, ...] = (
    "latency_samples.csv",
    "latency_summary.json",
    "benchmark_receipt.json",
)

#: Columns of the per-sample table, in order.
SAMPLE_COLUMNS: tuple[str, ...] = ("method", "backend", "device", "iteration", "latency_ms")

#: Fields every method's summary carries.
SUMMARY_FIELDS: tuple[str, ...] = (
    "method",
    "backend",
    "device",
    "timer",
    "timed_call",
    "warmup",
    "iterations",
    "mean_ms",
    "sd_ms",
    "p50_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "parameter_count",
)

#: Fields a learned method's summary carries as well.
LEARNED_SUMMARY_FIELDS: tuple[str, ...] = (
    "model_name",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_seed",
)

#: The fields the benchmark receipt must contain.
RECEIPT_FIELDS: tuple[str, ...] = (
    "receipt_schema",
    "status",
    "protocol_version",
    "git",
    "benchmark_plan",
    "holdout_test_plan",
    "held_out_quality_reference",
    "environment",
    "cuda_backend",
    "timing_method",
    "input",
    "checkpoints",
    "timing_boundary",
    "measurement_started_utc",
    "completed_utc",
    "image_reads",
    "test_or_stress_identifiers_recorded",
    "preflight",
    "outputs",
    "latency_values_in_receipt",
)

#: Every boolean prohibition in the plan: each must be the YAML ``false``.
PROHIBITIONS: tuple[str, ...] = (
    "image_reads_allowed",
    "quality_recomputation_allowed",
    "retraining_allowed",
    "checkpoint_change_allowed",
    "checkpoint_selection_by_latency",
    "latency_averaged_across_seeds",
    "outlier_removal",
    "fastest_run_only",
    "clahe_to_gpu_speed_ratio",
    "significance_testing",
    "overwrite_allowed",
    "automatic_retry_allowed",
)

#: Plan keys that carry prose for a reader rather than a setting.
DOCUMENTATION_KEYS: frozenset[str] = frozenset({"claim_rules"})

#: ``(section, key)`` pairs inside a checked section that are prose.
DOCUMENTATION_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {("held_out_quality_reference", "meaning"), ("learned_comparison", "wording_template")}
)

#: The only file types a measurement may write.
PERMITTED_OUTPUT_SUFFIXES: frozenset[str] = frozenset({".csv", ".json"})


@dataclass(frozen=True)
class FrozenCheckpoint:
    """One learned architecture's timing checkpoint, as the plan must state it."""

    model_name: str
    config: str
    config_sha256: str
    checkpoint: str
    checkpoint_sha256: str
    parameter_count: int


@dataclass(frozen=True)
class FrozenLatencyProtocol:
    """The protocol constants a plan must agree with.

    The real protocol is :data:`FROZEN`. The class exists so the synthetic
    tests can run the whole machinery on a miniature input with a handful of
    iterations; the runner takes no argument that could substitute another.
    """

    milestone: int = 12
    stage: str = "protocol_frozen_measurement_pending"
    protocol_version: str = "m12_latency_v1"
    quality_result_commit: str = "639a79880cc12eae1e0c99e2309e02165e9ea23e"
    holdout_plan: str = "configs/holdout/test_plan.yaml"
    holdout_plan_sha256: str = "bbad8d1f738393b0712cc3e0ed04cbe5ad5e3aa71523d51afdd7ad4853cb4544"
    quality_source: str = "outputs/metrics/holdout/test/holdout_test_summary.json"
    quality_source_sha256: str = "2f8ce07f6e06350dc9f8933fda57089a603fb07925cf2e34c168d4d28030896b"
    quality_key: tuple[str, ...] = ("unet_vs_cnn", "per_metric", "full_psnr", "raw_delta", "mean")
    quality_value: float = 0.2435097969
    device: str = "cuda"
    gpu: str = "NVIDIA GeForce RTX 5070 Ti"
    input_seed: int = 12
    input_shape: tuple[int, int, int, int] = (1, 1, 256, 256)
    input_sha256: str = "007f7d94ab363d216a282148dd2d37b4e2c07688f5d4e635cb45d437bfa7ed55"
    clahe_view_sha256: str = "0e5985c8f3b264979c5d61def441dbacacd6f4d4bfb87ab910f2bb205e6adb8f"
    methods: tuple[str, ...] = ("clahe", "cnn", "unet")
    learned_methods: tuple[str, ...] = ("cnn", "unet")
    checkpoint_seed: int = 2026
    checkpoints: Mapping[str, FrozenCheckpoint] = field(
        default_factory=lambda: {
            "cnn": FrozenCheckpoint(
                model_name="residual_cnn_v1",
                config="configs/cnn.yaml",
                config_sha256="ac4bf06c3957b79990d3721d81bc7596c308042a12105d50e08226be0473bb08",
                checkpoint="outputs/checkpoints/cnn_seed2026_best.pt",
                checkpoint_sha256=(
                    "fe3cdc42c72dea163fd9b603023da76e64ebb9433f451744b4430b81291838fb"
                ),
                parameter_count=28_353,
            ),
            "unet": FrozenCheckpoint(
                model_name="lightweight_residual_unet_v1",
                config="configs/unet.yaml",
                config_sha256="8baba29199c516ac4309dc432eeae6c91a36f0d553d567cc8268aae29066ac3a",
                checkpoint="outputs/checkpoints/unet_seed2026_best.pt",
                checkpoint_sha256=(
                    "15e430f83ade635ccfc516b6db1bc04359d59dbd61ef19009abf90976df2a3a1"
                ),
                parameter_count=116_753,
            ),
        }
    )
    clahe_config: str = "configs/clahe.yaml"
    clahe_config_sha256_lf: str = "491573a15c476b30b7789244cf6c6d4de0d429cdb3d7c08860042d6b5cd3737f"
    clahe_clip_limit: float = 0.5
    clahe_tile_grid_size: tuple[int, int] = (4, 4)
    warmup: int = 100
    iterations: int = 1000
    data_root: str = "data/raw/chaos"
    output_root: str = "outputs/latency"
    staging_root: str = "outputs/latency.incomplete"


FROZEN = FrozenLatencyProtocol()

#: The CUDA backend settings the Milestone 11 quality evaluation ran under:
#: PyTorch's defaults, which the evaluation path never changes. Verified,
#: never set, so the kernels timed are the kernels that were scored.
CUDA_BACKEND: dict[str, Any] = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": False,
    "cudnn_allow_tf32": True,
    "matmul_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "deterministic_algorithms": False,
}


class LatencyProtocolError(RuntimeError):
    """The frozen protocol cannot run as it stands. Nothing was timed or written."""


class LatencyExecutionError(RuntimeError):
    """Measurement stopped after it started. The staging record is preserved.

    Never caught and retried: a latency run is not repeated until it gives a
    number someone prefers. Whether a failed run may be re-run is a decision
    for review, and is disclosed.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatencyProtocolError(message)


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def expected_plan(frozen: FrozenLatencyProtocol = FROZEN) -> dict[str, Any]:
    """Every setting the plan must state, derived from the frozen constants.

    Prose fields (:data:`DOCUMENTATION_KEYS`, :data:`DOCUMENTATION_FIELDS`)
    are left out and checked separately for presence only.
    """
    batch, channels, height, width = frozen.input_shape
    definitions: dict[str, Any] = {
        "clahe": {
            "backend": "opencv_cpu",
            "device": "cpu",
            "timer": "perf_counter_ns",
            "timed_call": "ct_restoration.classical.clahe.apply_clahe",
            "config": frozen.clahe_config,
            "config_sha256_lf": frozen.clahe_config_sha256_lf,
            "settings": {
                "algorithm": "opencv_clahe_v1",
                "input_quantization_bits": 8,
                "clip_limit": frozen.clahe_clip_limit,
                "tile_grid_size": list(frozen.clahe_tile_grid_size),
            },
            "parameter_count": 0,
        }
    }
    for method in frozen.learned_methods:
        entry = frozen.checkpoints[method]
        definitions[method] = {
            "backend": "pytorch_cuda",
            "device": frozen.device,
            "timer": "cuda_event",
            "timed_call": "model.restore",
            "model_name": entry.model_name,
            "config": entry.config,
            "config_sha256": entry.config_sha256,
            "checkpoint": entry.checkpoint,
            "checkpoint_sha256": entry.checkpoint_sha256,
            "checkpoint_seed": frozen.checkpoint_seed,
            "parameter_count": entry.parameter_count,
        }
    return {
        "milestone": frozen.milestone,
        "stage": frozen.stage,
        "protocol_version": frozen.protocol_version,
        "measures": "inference_latency_only",
        "quality_result_commit": frozen.quality_result_commit,
        "holdout_test_plan": {"path": frozen.holdout_plan, "sha256": frozen.holdout_plan_sha256},
        "held_out_quality_reference": {
            "source": frozen.quality_source,
            "sha256": frozen.quality_source_sha256,
            "key": list(frozen.quality_key),
            "value": frozen.quality_value,
        },
        "benchmark_device": frozen.device,
        "benchmark_gpu": frozen.gpu,
        "cuda_backend": dict(CUDA_BACKEND),
        "input": {
            "generator": "torch_cpu_generator_manual_seed_then_torch_rand",
            "seed": frozen.input_seed,
            "batch_size": batch,
            "channels": channels,
            "height": height,
            "width": width,
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "sha256": frozen.input_sha256,
            "clahe_view": "batch_0_channel_0_as_numpy_float32",
            "clahe_view_sha256": frozen.clahe_view_sha256,
            "content_validated_once_before_timing": True,
        },
        "methods": list(frozen.methods),
        "learned_methods": list(frozen.learned_methods),
        "method_definitions": definitions,
        "timing": {
            "warmup_iterations": frozen.warmup,
            "measured_iterations": frozen.iterations,
            "sample_unit": "milliseconds",
            "learned": {
                "timer": "torch.cuda.Event(enable_timing=True)",
                "eval_mode": True,
                "inference_mode": True,
                "events_created_before_timing": True,
                "synchronize_after_warmup": True,
                "synchronize_between_samples_outside_timed_interval": True,
                "synchronizing_calls_inside_timed_interval": 0,
            },
            "clahe": {"timer": "time.perf_counter_ns", "synchronous": True},
        },
        "timing_boundary": {
            "result_label": "gpu_model_inference_latency",
            "not_end_to_end_ct_processing_latency": True,
            "learned_included": [
                "model_forward_structural_input_check",
                "model_forward",
                "residual_addition",
                "clamp_to_unit_interval",
            ],
            "learned_excluded": [
                "dicom_loading",
                "pydicom",
                "hu_conversion",
                "ct_windowing",
                "resize",
                "synthetic_degradation",
                "dataloader",
                "host_to_device_transfer",
                "checkpoint_loading",
                "model_construction",
                "input_content_validation",
                "device_to_host_transfer",
                "metric_calculation",
            ],
            "clahe_included": [
                "input_validation_inside_apply_clahe",
                "float_to_uint8_quantization",
                "clahe_operator_construction",
                "clahe_apply",
                "uint8_to_float_conversion",
            ],
            "clahe_excluded": [
                "dicom_loading",
                "hu_conversion",
                "ct_windowing",
                "resize",
                "synthetic_degradation",
                "metric_calculation",
            ],
        },
        "statistics": {
            "per_method": ["mean_ms", "sd_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"],
            "sd_ddof": 1,
            "percentile_method": "linear",
        },
        "learned_comparison": {
            "latency_ratio": "unet_over_cnn",
            "primary_statistic": "p50",
            "secondary_statistic": "mean",
            "parameter_counts": {
                method: frozen.checkpoints[method].parameter_count
                for method in frozen.learned_methods
            },
            "parameter_ratio_is_not_a_latency_ratio": True,
        },
        "outputs": {
            "root": frozen.output_root,
            "staging": frozen.staging_root,
            "files": list(OUTPUT_FILES),
            "sample_columns": list(SAMPLE_COLUMNS),
        },
        "failure_policy": {
            "destination_exists": "refuse",
            "staging_exists": "refuse",
            "failure_after_measurement_started": "preserve_staging_and_record_failure",
            "rerun_to_improve_a_number": "forbidden",
        },
    }


#: Every top-level key the plan must carry, and no other.
PLAN_KEYS: frozenset[str] = frozenset({*expected_plan(), *PROHIBITIONS, *DOCUMENTATION_KEYS})


def strictly_equal(actual: Any, expected: Any) -> bool:
    """Equality that will not let ``true`` pass for ``1`` or ``"12"`` for ``12``."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(strictly_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list | tuple):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(strictly_equal(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, float):
        return isinstance(actual, float) and actual == expected
    if isinstance(expected, int):
        return isinstance(actual, int) and actual == expected
    return type(actual) is type(expected) and actual == expected


def validate_plan(plan: Any, frozen: FrozenLatencyProtocol = FROZEN) -> dict[str, Any]:
    """Refuse a plan that differs from the frozen protocol in any setting."""
    _require(isinstance(plan, Mapping), "the latency plan must be a mapping")
    missing = sorted(PLAN_KEYS - set(plan))
    unknown = sorted(set(plan) - PLAN_KEYS)
    _require(not missing, f"the latency plan is missing key(s): {missing}")
    _require(not unknown, f"the latency plan has unrecognised key(s): {unknown}")

    for name in PROHIBITIONS:
        _require(plan[name] is False, f"{name} must be the YAML boolean false, got {plan[name]!r}")

    for key, expected in expected_plan(frozen).items():
        actual = plan[key]
        prose = {field for section, field in DOCUMENTATION_FIELDS if section == key}
        if prose:
            _require(isinstance(actual, Mapping), f"plan {key} must be a mapping")
            for name in prose:
                _require(
                    isinstance(actual.get(name), str) and actual[name].strip(),
                    f"plan {key}.{name} must be a non-empty string",
                )
            actual = {name: value for name, value in actual.items() if name not in prose}
        _require(
            strictly_equal(actual, expected),
            f"plan {key} is {actual!r}, expected {expected!r}",
        )

    _require(
        "{ratio}" in plan["learned_comparison"]["wording_template"],
        "the wording template must leave the latency ratio as an unfilled {ratio}",
    )
    forbidden = (
        plan["claim_rules"].get("forbidden") if isinstance(plan["claim_rules"], Mapping) else None
    )
    _require(
        isinstance(forbidden, list)
        and forbidden
        and all(isinstance(item, str) for item in forbidden),
        "claim_rules.forbidden must be a non-empty list of strings",
    )
    return dict(plan)


def load_plan(path: Path, frozen: FrozenLatencyProtocol = FROZEN) -> dict[str, Any]:
    """Read and validate the plan. Returns it with its own SHA-256 attached."""
    _require(Path(path).is_file(), f"{Path(path).as_posix()} is missing; there is no protocol")
    plan = validate_plan(yaml.safe_load(Path(path).read_text(encoding="utf-8")), frozen)
    plan["_source"] = {"path": Path(path).as_posix(), "sha256": file_sha256(Path(path))}
    return plan


# --------------------------------------------------------------------------
# The input
# --------------------------------------------------------------------------


def benchmark_input(frozen: FrozenLatencyProtocol = FROZEN):
    """The one synthetic input, on the CPU: ``torch.rand`` from a seeded generator."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(frozen.input_seed)
    return torch.rand(frozen.input_shape, generator=generator, dtype=torch.float32)


def clahe_view(batch) -> np.ndarray:
    """The same pixels as CLAHE takes them: a 2-D float32 NumPy image, copied."""
    return np.array(batch.numpy()[0, 0], dtype=np.float32, copy=True)


def verify_input(batch, frozen: FrozenLatencyProtocol = FROZEN) -> str:
    """Both views of the input reproduce their frozen digests."""
    _require(
        array_sha256(batch.numpy()) == frozen.input_sha256,
        "the benchmark input does not reproduce its frozen SHA-256",
    )
    _require(
        array_sha256(clahe_view(batch)) == frozen.clahe_view_sha256,
        "the CLAHE view of the benchmark input does not reproduce its frozen SHA-256",
    )
    return "input and CLAHE view digests reproduced"


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def time_cuda_restore(model, batch, warmup: int, iterations: int, cuda=None) -> list[float]:
    """One CUDA-event interval per sample around ``model.restore(batch)``, in ms.

    The interval is ``start.record()``, the call, ``end.record()`` and nothing
    else, measured on the device: from the GPU reaching the start event to it
    reaching the end event. Events are created before timing starts. The host
    waits for the GPU once after the warm-up and once after every sample's end
    event, so each measured inference is isolated from the next. Every wait is
    outside every interval, so no wait is part of a sample, and there is no
    synchronization inside an interval. The elapsed times are read from the
    retained event pairs after the loop: each pair was complete at its own
    wait, and reading it later reads the same two device timestamps.

    ``cuda`` is ``torch.cuda``; the tests pass a recorder in its place.
    """
    import torch

    cuda = cuda or torch.cuda
    model.eval()
    starts = [cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [cuda.Event(enable_timing=True) for _ in range(iterations)]
    restored = None
    with torch.inference_mode():
        for _ in range(warmup):
            model.restore(batch)
        cuda.synchronize()
        for start, end in zip(starts, ends, strict=True):
            start.record()
            restored = model.restore(batch)
            end.record()
            cuda.synchronize()
    del restored
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]


def time_cpu_call(
    restore: Callable[[np.ndarray], Any],
    image: np.ndarray,
    warmup: int,
    iterations: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> list[float]:
    """One high-resolution wall-clock interval per sample around ``restore``, in ms."""
    for _ in range(warmup):
        restore(image)
    samples: list[float] = []
    for _ in range(iterations):
        started = clock()
        restore(image)
        finished = clock()
        samples.append((finished - started) / 1e6)
    return samples


def cuda_sync_probe(model, batch) -> None:
    """Run the timed call once with PyTorch's sync debug mode set to error.

    A synchronizing CUDA call inside ``model.restore`` - an ``.item()``, a
    ``bool()`` of a GPU tensor, a copy to the host - raises here, before any
    timing. One untimed call first, so one-time lazy initialization is not
    mistaken for a per-inference stall. The debug mode is a PyTorch prototype
    and does not see every synchronizing operation; it supplements the source
    audit rather than replacing it.
    """
    import torch

    with torch.inference_mode():
        model.restore(batch)
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        with warnings.catch_warnings():
            # The notice that the mode is a prototype; said once, above.
            warnings.filterwarnings("ignore", message="Synchronization debug mode is a prototype")
            torch.cuda.set_sync_debug_mode("error")
        try:
            model.restore(batch)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()


def summarise_samples(samples: list[float], iterations: int) -> dict[str, float]:
    """Mean, sample SD, p50, p95, min and max of one method's samples, in ms.

    Every sample is kept: no outlier removal, no fastest-run reporting. A
    missing, non-finite or non-positive sample means the measurement is
    broken, and it is refused rather than summarised.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.shape != (iterations,):
        raise LatencyExecutionError(f"expected {iterations} samples, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise LatencyExecutionError("a latency sample is NaN or infinite")
    if not np.all(values > 0):
        raise LatencyExecutionError("a latency sample is zero or negative")
    return {
        "mean_ms": float(values.mean()),
        "sd_ms": float(values.std(ddof=1)),
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


# --------------------------------------------------------------------------
# The runtime seam
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Runtime:
    """Everything that touches the GPU or a clock, behind one seam.

    The runner builds the real one with :func:`cuda_runtime`. The synthetic
    tests build a fake, so the protocol logic runs without a GPU and without
    producing a latency number anyone could mistake for a result.
    """

    cuda_available: bool
    gpu_name: str | None
    backend: Mapping[str, Any]
    load_device: str
    to_device: Callable[[Any], Any]
    model_device: Callable[[Any], str]
    sync_probe: Callable[[Any, Any], None]
    time_learned: Callable[[Any, Any, int, int], list[float]]
    time_cpu: Callable[[Callable[[np.ndarray], Any], np.ndarray, int, int], list[float]]
    environment: Callable[[], dict[str, Any]]


def cuda_backend_settings() -> dict[str, Any]:
    import torch

    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def cpu_description() -> str | None:
    """The CPU model name. No hostname, no serial number."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def gpu_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if result.returncode == 0 and lines else None


def environment_facts(device: str) -> dict[str, Any]:
    """Versions and hardware the receipt records. No hostname, no user name."""
    import cv2
    import torch

    from ct_restoration.reproducibility import describe_environment

    return {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "opencv_threads": int(cv2.getNumThreads()),
        "torch_cpu_threads": int(torch.get_num_threads()),
        "torch": describe_environment(device),
        "gpu_driver": gpu_driver_version(),
        "cpu": cpu_description(),
    }


def cuda_runtime(device: str = FROZEN.device) -> Runtime:
    """The real runtime: the frozen CUDA device and the two real timers."""
    import torch

    available = bool(torch.cuda.is_available())
    return Runtime(
        cuda_available=available,
        gpu_name=torch.cuda.get_device_name(0) if available else None,
        backend=cuda_backend_settings(),
        load_device=device,
        to_device=lambda tensor: tensor.to(device=torch.device(device)),
        model_device=lambda model: next(model.parameters()).device.type,
        sync_probe=cuda_sync_probe,
        time_learned=time_cuda_restore,
        time_cpu=time_cpu_call,
        environment=functools.partial(environment_facts, device),
    )


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


class LatencyCheckReport(CheckReport):
    """The Milestone 11 check report, refusing with this milestone's error."""

    def require_all(self) -> None:
        if not self.passed_all:
            lines = "\n  ".join(f"{name}: {detail}" for name, detail in self.failed)
            raise LatencyProtocolError(
                f"preflight refused - {len(self.failed)} of {len(self.checks)} checks failed; "
                f"nothing was timed or written:\n  {lines}"
            )


def read_git_state(root: Path, frozen: FrozenLatencyProtocol = FROZEN) -> GitState:
    """Ask git, read-only. ``git status`` counts untracked files too."""

    def git(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
        )

    commit = git("rev-parse", "HEAD").stdout.strip()
    porcelain = git("status", "--porcelain", "--untracked-files=all").stdout
    ancestor = git("merge-base", "--is-ancestor", frozen.quality_result_commit, "HEAD").returncode
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or None
    origin = git("rev-parse", "--verify", "--quiet", "refs/remotes/origin/main")
    published = (
        git("merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/main").returncode == 0
        if origin.returncode == 0
        else None
    )
    return GitState(commit, porcelain, ancestor == 0, branch, published)


def _equal(label: str, actual: Any, expected: Any) -> str:
    if actual != expected:
        raise LatencyProtocolError(f"{label} is {actual!r}, expected {expected!r}")
    return f"{label} verified"


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _lookup(document: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        document = document[key]
    return document


@dataclass
class PreflightResult:
    """The outcome of preflight. ``models`` exists only if every check passed."""

    report: LatencyCheckReport
    models: dict[str, Any] | None = None


#: ``(plan, root, device) -> {method: model}`` for the two timing checkpoints.
ModelLoader = Callable[[dict[str, Any], Path, str], dict[str, Any]]


def load_frozen_models(plan: dict[str, Any], root: Path, device: str) -> dict[str, Any]:
    """The two timing checkpoints, built, loaded ``weights_only`` and in eval mode."""
    import torch

    from ct_restoration.models import cnn, unet

    builders = {
        "cnn": (cnn.ResidualCnnConfig, cnn.build_model),
        "unet": (unet.LightweightResidualUnetConfig, unet.build_model),
    }
    models: dict[str, Any] = {}
    for method in plan["learned_methods"]:
        entry = plan["method_definitions"][method]
        config_class, build = builders[method]
        config = config_class.from_mapping(_read_yaml(root / entry["config"]))
        _equal(f"{method} config algorithm", config.algorithm, entry["model_name"])
        payload = torch.load(root / entry["checkpoint"], map_location="cpu", weights_only=True)
        model = build(config, device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models[method] = model
    return models


def run_preflight(
    plan: dict[str, Any],
    root: Path,
    git: GitState,
    runtime: Runtime,
    loader: ModelLoader,
    frozen: FrozenLatencyProtocol = FROZEN,
) -> PreflightResult:
    """Every check, before anything is timed. Times nothing and writes nothing."""
    report = LatencyCheckReport()
    report.add("git commit recorded", len(git.commit or "") == 40, git.commit or "no commit")
    dirty = [line for line in git.porcelain.splitlines() if line.strip()]
    report.add(
        "git working tree clean",
        not dirty,
        "clean" if not dirty else f"{len(dirty)} modified or untracked: {dirty[:5]}",
    )
    report.add(
        "quality result commit is an ancestor of HEAD",
        git.frozen_commit_is_ancestor,
        frozen.quality_result_commit,
    )

    def holdout_plan() -> str:
        return _equal(
            f"{frozen.holdout_plan} SHA-256",
            file_sha256(root / frozen.holdout_plan),
            frozen.holdout_plan_sha256,
        )

    def same_checkpoints_as_holdout() -> str:
        scored = _read_yaml(root / frozen.holdout_plan)["learned_checkpoints"]
        for method in frozen.learned_methods:
            entry = plan["method_definitions"][method]
            held = scored[method][frozen.checkpoint_seed]
            for key in ("config", "config_sha256", "checkpoint", "checkpoint_sha256"):
                _equal(f"{method} {key} against the held-out plan", entry[key], held[key])
        return "both timing checkpoints are the seed-2026 entries scored on the test split"

    def quality_reference() -> str:
        source = root / frozen.quality_source
        _equal(
            f"{frozen.quality_source} SHA-256", file_sha256(source), frozen.quality_source_sha256
        )
        value = _lookup(_read_json(source), frozen.quality_key)
        return _equal("held-out U-Net minus CNN full PSNR", value, frozen.quality_value)

    report.attempt("held-out test plan unchanged", holdout_plan)
    report.attempt(
        "timing checkpoints are the held-out seed-2026 entries", same_checkpoints_as_holdout
    )
    report.attempt("held-out quality reference unchanged", quality_reference)

    for method in frozen.learned_methods:
        entry = plan["method_definitions"][method]

        def config_bytes(entry=entry) -> str:
            return _equal(
                f"{entry['config']} SHA-256",
                file_sha256(root / entry["config"]),
                entry["config_sha256"],
            )

        def checkpoint_bytes(entry=entry) -> str:
            path = root / entry["checkpoint"]
            _require(
                path.is_file(),
                f"{entry['checkpoint']} is missing. It is git-ignored, so it must be the "
                "original file, whose SHA-256 is frozen.",
            )
            return _equal(
                f"{entry['checkpoint']} SHA-256", file_sha256(path), entry["checkpoint_sha256"]
            )

        report.attempt(f"config bytes {method}", config_bytes)
        report.attempt(f"checkpoint bytes {method}", checkpoint_bytes)

    def clahe_config() -> str:
        definition = plan["method_definitions"]["clahe"]
        path = root / definition["config"]
        _equal(
            f"{definition['config']} SHA-256 (LF)",
            text_sha256_lf(path),
            definition["config_sha256_lf"],
        )
        parsed = ClaheConfig.from_mapping(_read_yaml(path)).as_dict()
        return _equal("CLAHE settings", parsed, definition["settings"])

    report.attempt("CLAHE config unchanged", clahe_config)

    report.add("CUDA available for the frozen device", runtime.cuda_available, frozen.device)
    report.add(
        "benchmark GPU is the frozen GPU",
        runtime.gpu_name == frozen.gpu,
        f"{runtime.gpu_name!r}, expected {frozen.gpu!r}",
    )
    report.add(
        "CUDA backend settings are the frozen settings",
        dict(runtime.backend) == CUDA_BACKEND,
        f"{dict(runtime.backend)}",
    )
    report.attempt(
        "benchmark input reproduces its digests",
        lambda: verify_input(benchmark_input(frozen), frozen),
    )

    for key, location in (("root", frozen.output_root), ("staging", frozen.staging_root)):
        exists = (root / location).exists()
        report.add(
            f"output {key} absent", not exists, f"{location} {'exists' if exists else 'absent'}"
        )

    if not report.passed_all:
        return PreflightResult(report)

    try:
        models = loader(plan, root, runtime.load_device)
    except Exception as error:  # noqa: BLE001 - recorded as the check's failure
        report.add("timing checkpoints loaded", False, f"{type(error).__name__}: {error}")
        return PreflightResult(report)

    report.add(
        "exactly the frozen learned methods loaded",
        sorted(models) == sorted(frozen.learned_methods),
        f"{sorted(models)}",
    )
    batch = runtime.to_device(benchmark_input(frozen))
    for method in frozen.learned_methods:
        model = models.get(method)
        if model is None:
            continue
        expected = frozen.checkpoints[method].parameter_count
        report.add(
            f"{method} parameter count",
            model.parameter_count() == expected,
            f"{model.parameter_count()}, expected {expected}",
        )
        placed = runtime.model_device(model)
        report.add(
            f"{method} runs on the frozen device",
            placed == frozen.device,
            f"{placed!r}; a learned model is never timed on the CPU",
        )
        report.add(
            f"{method} in eval mode",
            not model.training,
            "eval" if not model.training else "training",
        )
        report.attempt(
            f"{method} timed call makes no synchronizing CUDA call",
            lambda model=model: runtime.sync_probe(model, batch) or "no synchronization detected",
        )
    return PreflightResult(report, models if report.passed_all else None)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def measure(
    plan: dict[str, Any],
    models: Mapping[str, Any],
    runtime: Runtime,
    root: Path,
    frozen: FrozenLatencyProtocol = FROZEN,
) -> dict[str, list[float]]:
    """Every method's samples, in the frozen order, on the frozen input."""
    from ct_restoration.models.base import validate_restoration_input

    batch = benchmark_input(frozen)
    verify_input(batch, frozen)
    image = clahe_view(batch)
    resident = runtime.to_device(batch)

    # The whole input contract, values included, against each learned
    # model's contract, before the first warm-up call of any method - once per
    # run, never per iteration. It synchronizes, so it is never inside a
    # timed interval.
    for method in frozen.learned_methods:
        config = models[method].config
        validate_restoration_input(
            resident,
            input_channels=config.input_channels,
            spatial_multiple=getattr(config, "spatial_multiple", 1),
        )

    clahe = ClaheConfig.from_mapping(
        _read_yaml(root / plan["method_definitions"]["clahe"]["config"])
    )
    samples: dict[str, list[float]] = {}
    for method in frozen.methods:
        if method == "clahe":
            restore = functools.partial(apply_clahe, config=clahe)
            samples[method] = runtime.time_cpu(restore, image, frozen.warmup, frozen.iterations)
        else:
            samples[method] = runtime.time_learned(
                models[method], resident, frozen.warmup, frozen.iterations
            )
    return samples


def sample_table(
    plan: dict[str, Any], samples: Mapping[str, list[float]], frozen: FrozenLatencyProtocol = FROZEN
) -> pd.DataFrame:
    rows = []
    for method in frozen.methods:
        definition = plan["method_definitions"][method]
        for iteration, value in enumerate(samples[method]):
            rows.append(
                {
                    "method": method,
                    "backend": definition["backend"],
                    "device": definition["device"],
                    "iteration": iteration,
                    "latency_ms": float(value),
                }
            )
    return pd.DataFrame(rows, columns=list(SAMPLE_COLUMNS))


def summarise(
    plan: dict[str, Any],
    samples: Mapping[str, list[float]],
    git: GitState,
    frozen: FrozenLatencyProtocol = FROZEN,
) -> dict[str, Any]:
    """The per-method statistics and the frozen learned-model comparison."""
    methods: dict[str, Any] = {}
    for method in frozen.methods:
        definition = plan["method_definitions"][method]
        entry = {
            "method": method,
            "backend": definition["backend"],
            "device": definition["device"],
            "timer": definition["timer"],
            "timed_call": definition["timed_call"],
            "warmup": frozen.warmup,
            "iterations": frozen.iterations,
            **summarise_samples(samples[method], frozen.iterations),
            "parameter_count": definition["parameter_count"],
        }
        if method in frozen.learned_methods:
            entry.update({name: definition[name] for name in LEARNED_SUMMARY_FIELDS})
        methods[method] = entry

    cnn, unet = methods["cnn"], methods["unet"]
    counts = plan["learned_comparison"]["parameter_counts"]
    return {
        "milestone": frozen.milestone,
        "result_class": "inference_latency",
        "benchmark_plan": plan["_source"],
        "commit": git.commit,
        "gpu": frozen.gpu,
        "input": {
            "shape": list(frozen.input_shape),
            "dtype": "float32",
            "seed": frozen.input_seed,
            "sha256": frozen.input_sha256,
        },
        "methods": methods,
        "learned_comparison": {
            "result_label": "gpu_model_inference_latency",
            "unet_over_cnn_latency_ratio_p50": unet["p50_ms"] / cnn["p50_ms"],
            "unet_over_cnn_latency_ratio_mean": unet["mean_ms"] / cnn["mean_ms"],
            "primary_statistic": "p50",
            "parameter_counts": dict(counts),
            "unet_over_cnn_parameter_ratio": counts["unet"] / counts["cnn"],
            "parameter_ratio_is_not_a_latency_ratio": True,
            "held_out_full_psnr_unet_minus_cnn_db": frozen.quality_value,
            "held_out_quality_source": frozen.quality_source,
        },
        "clahe_to_gpu_speed_ratio": None,
        "clahe_to_gpu_speed_ratio_reason": (
            "not computed: CLAHE is a CPU/OpenCV path and the learned models a GPU/PyTorch "
            "path, timed differently on different hardware. No hardware-controlled "
            "comparison exists between them."
        ),
        "not_end_to_end_ct_processing_latency": True,
        "claim_rules": plan["claim_rules"],
    }


def execute_protocol(
    plan: dict[str, Any],
    root: Path,
    git: GitState,
    runtime: Runtime,
    loader: ModelLoader,
    frozen: FrozenLatencyProtocol = FROZEN,
    clock: Callable[[], str] = utc_now,
) -> Path:
    """Preflight, measure once, write the three files, move them into place.

    The imaging root is sealed for the whole call: an open of any file under
    it raises before a byte is read, and the count of attempts is recorded.
    """
    staging = root / frozen.staging_root
    final = root / frozen.output_root
    with DicomOpenMonitor(root / frozen.data_root) as sealed:
        preflight = run_preflight(plan, root, git, runtime, loader, frozen)
        preflight.report.require_all()
        environment = runtime.environment()

        staging.mkdir(parents=True, exist_ok=False)
        started = clock()
        try:
            samples = measure(plan, preflight.models, runtime, root, frozen)
            summary = summarise(plan, samples, git, frozen)
            written = {
                "latency_samples.csv": write_csv(
                    sample_table(plan, samples, frozen), staging / "latency_samples.csv"
                ),
                "latency_summary.json": write_json(summary, staging / "latency_summary.json"),
            }
            reads = sealed.record()
            opened = sum(reads["file_opens_by_split"].values())
            refused = sum(reads["refused_opens_by_split"].values())
            if opened or refused:
                raise LatencyExecutionError(
                    f"{opened} file(s) opened and {refused} refused under the imaging root"
                )
            receipt = {
                "receipt_schema": "m12_latency_receipt_v1",
                "status": "complete",
                "protocol_version": frozen.protocol_version,
                "git": {
                    "commit": git.commit,
                    "working_tree_clean": not git.porcelain.strip(),
                    "branch": git.branch,
                    "head_in_origin_main": git.head_in_origin_main,
                    "quality_result_commit": frozen.quality_result_commit,
                },
                "benchmark_plan": plan["_source"],
                "holdout_test_plan": plan["holdout_test_plan"],
                "held_out_quality_reference": plan["held_out_quality_reference"],
                "environment": environment,
                "cuda_backend": dict(runtime.backend),
                "timing_method": {
                    "learned": plan["timing"]["learned"],
                    "clahe": plan["timing"]["clahe"],
                    "warmup_iterations": frozen.warmup,
                    "measured_iterations": frozen.iterations,
                },
                "input": plan["input"],
                "checkpoints": {
                    method: {
                        "path": plan["method_definitions"][method]["checkpoint"],
                        "sha256": plan["method_definitions"][method]["checkpoint_sha256"],
                    }
                    for method in frozen.learned_methods
                },
                "timing_boundary": {
                    **plan["timing_boundary"],
                    "host_to_device_excluded": True,
                    "device_to_host_excluded": True,
                    "preprocessing_excluded": True,
                    "input_content_validation_excluded": True,
                },
                "measurement_started_utc": started,
                "completed_utc": clock(),
                "image_reads": reads,
                "test_or_stress_identifiers_recorded": False,
                "preflight": preflight.report.as_dict(),
                "outputs": {
                    name: {"sha256": file_sha256(path)} for name, path in sorted(written.items())
                },
                "latency_values_in_receipt": False,
            }
            write_json(receipt, staging / "benchmark_receipt.json")
            present = sorted(path.name for path in staging.iterdir())
            if present != sorted(OUTPUT_FILES) or any(
                Path(name).suffix not in PERMITTED_OUTPUT_SUFFIXES for name in present
            ):
                raise LatencyExecutionError(f"unexpected output listing {present}")
        except Exception as error:
            write_json(
                {
                    "status": "failed",
                    "failed_utc": clock(),
                    "measurement_started_utc": started,
                    "error": f"{type(error).__name__}: {error}",
                    "rerun_policy": "no automatic retry; any re-run is decided by review",
                },
                staging / "failure_record.json",
            )
            raise LatencyExecutionError(
                f"latency measurement failed after it started: {type(error).__name__}: {error}. "
                f"The record is preserved in {frozen.staging_root}."
            ) from error
    os.rename(staging, final)
    return final


def parameter_ratio(frozen: FrozenLatencyProtocol = FROZEN) -> float:
    """U-Net over CNN trainable parameters. Context for latency, never a latency."""
    counts = frozen.checkpoints
    return counts["unet"].parameter_count / counts["cnn"].parameter_count


__all__ = [
    "CUDA_BACKEND",
    "FROZEN",
    "OUTPUT_FILES",
    "PLAN_PATH",
    "SAMPLE_COLUMNS",
    "FrozenCheckpoint",
    "FrozenLatencyProtocol",
    "LatencyExecutionError",
    "LatencyProtocolError",
    "Runtime",
    "cuda_runtime",
    "execute_protocol",
    "load_frozen_models",
    "load_plan",
    "read_git_state",
    "run_preflight",
    "summarise_samples",
    "time_cpu_call",
    "time_cuda_restore",
    "validate_plan",
]
