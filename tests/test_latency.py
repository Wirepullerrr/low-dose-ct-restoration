"""Tests for the Milestone 12 latency protocol.

Synthetic and dataset-free. No CT image is opened, no network is touched and
no latency number is produced that anyone could mistake for a result: the
protocol machinery runs on a miniature repository in ``tmp_path`` with a fake
runtime whose "timers" return made-up values, and the few checks that need a
real GPU are skipped without one and time an untrained model on a toy input.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from ct_restoration import latency
from ct_restoration.config import PROJECT_ROOT
from ct_restoration.evaluation import HeldOutSplitError
from ct_restoration.holdout import GitState, array_sha256, text_sha256_lf
from ct_restoration.latency import (
    CUDA_BACKEND,
    FROZEN,
    OUTPUT_FILES,
    SAMPLE_COLUMNS,
    FrozenCheckpoint,
    FrozenLatencyProtocol,
    LatencyExecutionError,
    LatencyProtocolError,
    Runtime,
)
from ct_restoration.models import cnn, unet

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

CLEAN_GIT = GitState("a" * 40, "", True, "main", True)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _documentation() -> dict:
    return {
        "claim_rules": {"forbidden": ["end-to-end CT processing latency"]},
    }


def _plan_document(frozen: FrozenLatencyProtocol) -> dict:
    """A complete plan document agreeing with ``frozen``."""
    plan = latency.expected_plan(frozen)
    plan["held_out_quality_reference"]["meaning"] = "paired U-Net minus CNN full PSNR"
    plan["learned_comparison"]["wording_template"] = "the U-Net needed {ratio}x the latency"
    plan.update({name: False for name in latency.PROHIBITIONS})
    plan.update(_documentation())
    return plan


# --------------------------------------------------------------------------
# a miniature repository
# --------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path
    for name in ("configs/cnn.yaml", "configs/unet.yaml", "configs/clahe.yaml"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / name, root / name)

    torch.manual_seed(0)
    checkpoints = {}
    for method, module, config in (
        ("cnn", cnn, "configs/cnn.yaml"),
        ("unet", unet, "configs/unet.yaml"),
    ):
        path = root / f"outputs/checkpoints/{method}_seed2026_best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": module.build_model().state_dict()}, path)
        checkpoints[method] = FrozenCheckpoint(
            model_name=module.ALGORITHM_VERSION,
            config=config,
            config_sha256=_sha(root / config),
            checkpoint=path.relative_to(root).as_posix(),
            checkpoint_sha256=_sha(path),
            parameter_count=module.CANONICAL_PARAMETER_COUNT,
        )

    holdout_plan = root / "configs/holdout/test_plan.yaml"
    holdout_plan.parent.mkdir(parents=True, exist_ok=True)
    holdout_plan.write_text(
        yaml.safe_dump(
            {
                "learned_checkpoints": {
                    method: {
                        2026: {
                            "config": entry.config,
                            "config_sha256": entry.config_sha256,
                            "checkpoint": entry.checkpoint,
                            "checkpoint_sha256": entry.checkpoint_sha256,
                        }
                    }
                    for method, entry in checkpoints.items()
                }
            }
        ),
        encoding="utf-8",
    )
    quality = root / "outputs/metrics/holdout/test/holdout_test_summary.json"
    quality.parent.mkdir(parents=True, exist_ok=True)
    quality.write_text(
        json.dumps({"unet_vs_cnn": {"per_metric": {"full_psnr": {"raw_delta": {"mean": 0.25}}}}}),
        encoding="utf-8",
    )
    (root / "data/raw/chaos").mkdir(parents=True)
    (root / "data/raw/chaos/sealed.dcm").write_bytes(b"not a real image")

    frozen = FrozenLatencyProtocol(
        gpu="Fake GPU",
        holdout_plan_sha256=_sha(holdout_plan),
        quality_source_sha256=_sha(quality),
        quality_value=0.25,
        input_shape=(1, 1, 16, 16),
        checkpoints=checkpoints,
        clahe_config_sha256_lf=text_sha256_lf(root / "configs/clahe.yaml"),
        warmup=2,
        iterations=5,
    )
    batch = latency.benchmark_input(frozen)
    frozen = dataclasses.replace(
        frozen,
        input_sha256=array_sha256(batch.numpy()),
        clahe_view_sha256=array_sha256(latency.clahe_view(batch)),
    )
    plan_path = root / "configs/latency/benchmark_plan.yaml"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(yaml.safe_dump(_plan_document(frozen), sort_keys=False), encoding="utf-8")
    return SimpleNamespace(root=root, frozen=frozen, plan=latency.load_plan(plan_path, frozen))


def _runtime(log: list | None = None, **overrides) -> Runtime:
    log = [] if log is None else log

    def time_learned(model, batch, warmup, iterations):
        log.append(("learned", model.config.algorithm, warmup, iterations))
        with torch.inference_mode():
            model.restore(batch)
        return [1.0 + 0.01 * i for i in range(iterations)]

    def time_cpu(restore, image, warmup, iterations):
        log.append(("cpu", warmup, iterations))
        restore(image)
        return [2.0 + 0.01 * i for i in range(iterations)]

    fields = {
        "cuda_available": True,
        "gpu_name": "Fake GPU",
        "backend": dict(CUDA_BACKEND),
        "load_device": "cpu",
        "to_device": lambda tensor: tensor,
        "model_device": lambda model: "cuda",
        "sync_probe": lambda model, batch: None,
        "time_learned": time_learned,
        "time_cpu": time_cpu,
        "environment": lambda: {"synthetic": True},
    }
    fields.update(overrides)
    return Runtime(**fields)


def _preflight(repo, runtime=None, git=CLEAN_GIT):
    return latency.run_preflight(
        repo.plan, repo.root, git, runtime or _runtime(), latency.load_frozen_models, repo.frozen
    )


def _failed(result) -> list[str]:
    return [name for name, _ in result.report.failed]


# --------------------------------------------------------------------------
# the committed plan
# --------------------------------------------------------------------------


def test_the_committed_plan_agrees_with_the_frozen_protocol():
    plan = latency.load_plan(PROJECT_ROOT / latency.PLAN_PATH)
    assert plan["stage"] == "protocol_frozen_measurement_pending"
    assert plan["quality_result_commit"] == "639a79880cc12eae1e0c99e2309e02165e9ea23e"


def test_the_frozen_timing_protocol_is_the_declared_one():
    assert (FROZEN.warmup, FROZEN.iterations) == (100, 1000)
    assert FROZEN.input_shape == (1, 1, 256, 256)
    assert FROZEN.methods == ("clahe", "cnn", "unet")
    assert FROZEN.device == "cuda"
    assert FROZEN.gpu == "NVIDIA GeForce RTX 5070 Ti"


def test_the_timing_checkpoints_are_the_seed_2026_entries_scored_on_test():
    scored = yaml.safe_load((PROJECT_ROOT / FROZEN.holdout_plan).read_text(encoding="utf-8"))
    for method in FROZEN.learned_methods:
        held = scored["learned_checkpoints"][method][2026]
        frozen = FROZEN.checkpoints[method]
        assert frozen.checkpoint == held["checkpoint"]
        assert frozen.checkpoint_sha256 == held["checkpoint_sha256"]
        assert frozen.config == held["config"]
        assert frozen.config_sha256 == held["config_sha256"]


def test_the_committed_input_digest_is_reproduced():
    assert (
        latency.verify_input(latency.benchmark_input()) == "input and CLAHE view digests reproduced"
    )


def test_parameter_counts_are_the_architectures_and_not_a_latency():
    assert (
        cnn.build_model().parameter_count() == FROZEN.checkpoints["cnn"].parameter_count == 28_353
    )
    assert unet.build_model().parameter_count() == FROZEN.checkpoints["unet"].parameter_count
    assert FROZEN.checkpoints["unet"].parameter_count == 116_753
    assert round(latency.parameter_ratio(), 2) == 4.12


# --------------------------------------------------------------------------
# plan refusals
# --------------------------------------------------------------------------


def _mutated(plan: dict, change) -> dict:
    document = copy.deepcopy({key: value for key, value in plan.items() if key != "_source"})
    change(document)
    return document


@pytest.mark.parametrize(
    ("label", "change"),
    [
        ("warm-up changed", lambda p: p["timing"].update(warmup_iterations=10)),
        ("iterations changed", lambda p: p["timing"].update(measured_iterations=999)),
        ("input height changed", lambda p: p["input"].update(height=512)),
        ("batch size changed", lambda p: p["input"].update(batch_size=2)),
        ("input dtype changed", lambda p: p["input"].update(dtype="float16")),
        ("wrong backend", lambda p: p["method_definitions"]["cnn"].update(backend="pytorch_cpu")),
        (
            "learned model on the CPU",
            lambda p: p["method_definitions"]["unet"].update(device="cpu"),
        ),
        ("benchmark device CPU", lambda p: p.update(benchmark_device="cpu")),
        ("a method missing", lambda p: p.update(methods=["clahe", "cnn"])),
        ("an extra method", lambda p: p.update(methods=["degraded", "clahe", "cnn", "unet"])),
        (
            "an extra method definition",
            lambda p: p["method_definitions"].update(degraded={"backend": "none"}),
        ),
        ("method order changed", lambda p: p.update(methods=["unet", "cnn", "clahe"])),
        (
            "another checkpoint",
            lambda p: p["method_definitions"]["unet"].update(checkpoint_seed=2028),
        ),
        (
            "checkpoint digest changed",
            lambda p: p["method_definitions"]["cnn"].update(checkpoint_sha256="0" * 64),
        ),
        (
            "primary statistic changed",
            lambda p: p["learned_comparison"].update(primary_statistic="min"),
        ),
        ("stage advanced", lambda p: p.update(stage="measured")),
        ("a prohibition lifted", lambda p: p.update(outlier_removal=True)),
        ("a prohibition as a string", lambda p: p.update(overwrite_allowed="false")),
        ("a bool standing in for an int", lambda p: p["input"].update(seed=True)),
        ("an unknown key", lambda p: p.update(warmup_override=5)),
        ("a missing key", lambda p: p.pop("cuda_backend")),
        ("cuDNN autotuning on", lambda p: p["cuda_backend"].update(cudnn_benchmark=True)),
        (
            "sync inside the interval",
            lambda p: p["timing"]["learned"].update(synchronizing_calls_inside_timed_interval=1),
        ),
        (
            "ratio pre-filled",
            lambda p: p["learned_comparison"].update(wording_template="the U-Net needed 3x"),
        ),
        ("no claim rules", lambda p: p.update(claim_rules={"forbidden": []})),
    ],
)
def test_a_plan_that_differs_from_the_frozen_protocol_is_refused(repo, label, change):
    with pytest.raises(LatencyProtocolError):
        latency.validate_plan(_mutated(repo.plan, change), repo.frozen)


def test_the_committed_plan_is_refused_by_a_different_protocol():
    plan = yaml.safe_load((PROJECT_ROOT / latency.PLAN_PATH).read_text(encoding="utf-8"))
    with pytest.raises(LatencyProtocolError, match="timing"):
        latency.validate_plan(plan, dataclasses.replace(FROZEN, iterations=10))


def test_a_missing_plan_is_refused(tmp_path):
    with pytest.raises(LatencyProtocolError, match="no protocol"):
        latency.load_plan(tmp_path / "absent.yaml")


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def test_preflight_passes_on_a_consistent_repository(repo):
    result = _preflight(repo)
    assert _failed(result) == []
    assert sorted(result.models) == ["cnn", "unet"]


def test_preflight_times_nothing_and_writes_nothing(repo):
    log: list = []
    before = sorted(path.relative_to(repo.root) for path in repo.root.rglob("*"))
    _preflight(repo, _runtime(log))
    assert log == []
    assert sorted(path.relative_to(repo.root) for path in repo.root.rglob("*")) == before


def test_a_wrong_checkpoint_is_refused(repo):
    path = repo.root / repo.frozen.checkpoints["cnn"].checkpoint
    path.write_bytes(path.read_bytes() + b"\0")
    result = _preflight(repo)
    assert "checkpoint bytes cnn" in _failed(result)
    assert result.models is None


def test_a_changed_config_is_refused(repo):
    path = repo.root / "configs/unet.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert "config bytes unet" in _failed(_preflight(repo))


def test_a_changed_heldout_plan_is_refused(repo):
    path = repo.root / repo.frozen.holdout_plan
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert "held-out test plan unchanged" in _failed(_preflight(repo))


def test_a_changed_quality_reference_is_refused(repo):
    path = repo.root / repo.frozen.quality_source
    path.write_text(path.read_text(encoding="utf-8").replace("0.25", "0.35"), encoding="utf-8")
    assert "held-out quality reference unchanged" in _failed(_preflight(repo))


def test_the_wrong_gpu_is_refused(repo):
    assert "benchmark GPU is the frozen GPU" in _failed(
        _preflight(repo, _runtime(gpu_name="Other"))
    )


def test_no_cuda_is_refused(repo):
    result = _preflight(repo, _runtime(cuda_available=False))
    assert "CUDA available for the frozen device" in _failed(result)


def test_a_learned_model_on_the_cpu_is_refused(repo):
    result = _preflight(repo, _runtime(model_device=lambda model: "cpu"))
    assert "cnn runs on the frozen device" in _failed(result)
    assert "unet runs on the frozen device" in _failed(result)
    assert result.models is None


def test_changed_backend_settings_are_refused(repo):
    backend = {**CUDA_BACKEND, "cudnn_benchmark": True}
    result = _preflight(repo, _runtime(backend=backend))
    assert "CUDA backend settings are the frozen settings" in _failed(result)


def test_a_synchronizing_timed_call_is_refused(repo):
    def probe(model, batch):
        raise RuntimeError("called a synchronizing CUDA operation")

    result = _preflight(repo, _runtime(sync_probe=probe))
    assert "unet timed call makes no synchronizing CUDA call" in _failed(result)


def test_a_dirty_tree_is_refused(repo):
    dirty = GitState("a" * 40, " M src/ct_restoration/models/unet.py\n", True, "main", True)
    assert "git working tree clean" in _failed(_preflight(repo, git=dirty))


def test_a_head_without_the_quality_commit_is_refused(repo):
    detached = GitState("a" * 40, "", False, "main", True)
    assert "quality result commit is an ancestor of HEAD" in _failed(_preflight(repo, git=detached))


@pytest.mark.parametrize("key", ["output_root", "staging_root"])
def test_an_existing_destination_is_refused(repo, key):
    (repo.root / getattr(repo.frozen, key)).mkdir(parents=True)
    kind = "root" if key == "output_root" else "staging"
    assert f"output {kind} absent" in _failed(_preflight(repo))


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def _execute(repo, runtime=None):
    return latency.execute_protocol(
        repo.plan,
        repo.root,
        CLEAN_GIT,
        runtime or _runtime(),
        latency.load_frozen_models,
        repo.frozen,
        clock=lambda: "2026-01-01T00:00:00+00:00",
    )


def test_a_complete_run_writes_exactly_the_frozen_files(repo):
    final = _execute(repo)
    assert final == repo.root / repo.frozen.output_root
    assert sorted(path.name for path in final.iterdir()) == sorted(OUTPUT_FILES)
    assert not (repo.root / repo.frozen.staging_root).exists()


def test_methods_are_timed_in_the_frozen_order_with_the_frozen_counts(repo):
    log: list = []
    _execute(repo, _runtime(log))
    warmup, iterations = repo.frozen.warmup, repo.frozen.iterations
    assert log == [
        ("cpu", warmup, iterations),
        ("learned", "residual_cnn_v1", warmup, iterations),
        ("learned", "lightweight_residual_unet_v1", warmup, iterations),
    ]


def test_the_sample_table_schema(repo):
    samples = pd.read_csv(_execute(repo) / "latency_samples.csv")
    assert tuple(samples.columns) == SAMPLE_COLUMNS
    assert len(samples) == 3 * repo.frozen.iterations
    assert samples.groupby("method")["iteration"].apply(list).to_dict() == {
        method: list(range(repo.frozen.iterations)) for method in ("clahe", "cnn", "unet")
    }
    backends = samples.groupby("method")[["backend", "device"]].first().to_dict("index")
    assert backends == {
        "clahe": {"backend": "opencv_cpu", "device": "cpu"},
        "cnn": {"backend": "pytorch_cuda", "device": "cuda"},
        "unet": {"backend": "pytorch_cuda", "device": "cuda"},
    }


def test_the_summary_schema(repo):
    summary = json.loads((_execute(repo) / "latency_summary.json").read_text(encoding="utf-8"))
    for method, entry in summary["methods"].items():
        assert set(latency.SUMMARY_FIELDS) <= set(entry)
        if method != "clahe":
            assert set(latency.LEARNED_SUMMARY_FIELDS) <= set(entry)
    assert summary["methods"]["clahe"]["parameter_count"] == 0
    assert summary["methods"]["unet"]["checkpoint_seed"] == 2026
    comparison = summary["learned_comparison"]
    assert comparison["primary_statistic"] == "p50"
    assert comparison["parameter_ratio_is_not_a_latency_ratio"] is True
    assert summary["clahe_to_gpu_speed_ratio"] is None
    assert summary["not_end_to_end_ct_processing_latency"] is True


def test_the_receipt_schema_and_boundary(repo):
    receipt = json.loads((_execute(repo) / "benchmark_receipt.json").read_text(encoding="utf-8"))
    assert tuple(receipt) == latency.RECEIPT_FIELDS
    assert receipt["latency_values_in_receipt"] is False
    assert receipt["test_or_stress_identifiers_recorded"] is False
    boundary = receipt["timing_boundary"]
    assert boundary["host_to_device_excluded"] is True
    assert boundary["device_to_host_excluded"] is True
    assert boundary["preprocessing_excluded"] is True
    assert sum(receipt["image_reads"]["file_opens_by_split"].values()) == 0
    assert set(receipt["outputs"]) == {"latency_samples.csv", "latency_summary.json"}


def test_an_existing_output_is_never_overwritten(repo):
    final = repo.root / repo.frozen.output_root
    final.mkdir(parents=True)
    (final / "marker.txt").write_text("earlier run", encoding="utf-8")
    with pytest.raises(LatencyProtocolError):
        _execute(repo)
    assert sorted(path.name for path in final.iterdir()) == ["marker.txt"]
    assert not (repo.root / repo.frozen.staging_root).exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_a_broken_timing_is_refused_and_the_record_preserved(repo, bad):
    def time_learned(model, batch, warmup, iterations):
        return [1.0] * (iterations - 1) + [bad]

    with pytest.raises(LatencyExecutionError):
        _execute(repo, _runtime(time_learned=time_learned))
    staging = repo.root / repo.frozen.staging_root
    assert sorted(path.name for path in staging.iterdir()) == ["failure_record.json"]
    assert not (repo.root / repo.frozen.output_root).exists()


def test_the_imaging_root_is_sealed_during_measurement(repo):
    def time_cpu(restore, image, warmup, iterations):
        (repo.root / "data/raw/chaos/sealed.dcm").read_bytes()
        return [1.0] * iterations

    with pytest.raises(LatencyExecutionError, match="HeldOutSplitError"):
        _execute(repo, _runtime(time_cpu=time_cpu))
    assert not (repo.root / repo.frozen.output_root).exists()


def test_the_imaging_root_is_sealed_during_preflight(repo):
    from ct_restoration.holdout import DicomOpenMonitor

    with DicomOpenMonitor(repo.root / repo.frozen.data_root), pytest.raises(HeldOutSplitError):
        (repo.root / "data/raw/chaos/sealed.dcm").read_bytes()


# --------------------------------------------------------------------------
# statistics and timers
# --------------------------------------------------------------------------


def test_the_summary_statistics():
    values = [float(value) for value in range(1, 101)]
    stats = latency.summarise_samples(values, 100)
    assert stats["mean_ms"] == pytest.approx(50.5)
    assert stats["sd_ms"] == pytest.approx(np.std(values, ddof=1))
    assert stats["p50_ms"] == pytest.approx(50.5)
    assert stats["p95_ms"] == pytest.approx(95.05)
    assert (stats["min_ms"], stats["max_ms"]) == (1.0, 100.0)


def test_percentiles_are_linear_interpolation():
    stats = latency.summarise_samples([4.0, 1.0, 3.0, 2.0], 4)
    assert stats["p50_ms"] == pytest.approx(2.5)
    assert stats["p95_ms"] == pytest.approx(3.85)


def test_a_short_sample_is_refused():
    with pytest.raises(LatencyExecutionError, match="expected 5 samples"):
        latency.summarise_samples([1.0, 2.0], 5)


class _RecordingCuda:
    """Stands in for ``torch.cuda`` and logs the order of every call."""

    def __init__(self, log: list):
        self.log = log
        self.clock = 0.0
        recorder = self

        class Event:
            def __init__(self, enable_timing: bool):
                assert enable_timing
                self.at = None

            def record(self):
                recorder.clock += 1.0
                self.at = recorder.clock
                recorder.log.append("record")

            def elapsed_time(self, end):
                return end.at - self.at

        self.Event = Event

    def synchronize(self):
        self.log.append("synchronize")


def test_the_cuda_timer_brackets_only_the_restore_call():
    log: list = []
    model = SimpleNamespace(eval=lambda: None, restore=lambda batch: log.append("restore"))
    samples = latency.time_cuda_restore(
        model, None, warmup=3, iterations=4, cuda=_RecordingCuda(log)
    )

    assert log[:4] == ["restore", "restore", "restore", "synchronize"]
    # Every sample: start, the call, end, and the wait - after the end event.
    assert log[4:] == ["record", "restore", "record", "synchronize"] * 4
    assert samples == [1.0] * 4


def test_the_cpu_timer_brackets_only_the_call():
    ticks = iter(range(0, 10**9, 250_000))
    calls: list = []
    samples = latency.time_cpu_call(
        calls.append, np.zeros((2, 2)), warmup=2, iterations=3, clock=lambda: next(ticks)
    )
    assert len(calls) == 5
    assert samples == [0.25, 0.25, 0.25]


# --------------------------------------------------------------------------
# the runner exposes no timing option
# --------------------------------------------------------------------------


def test_the_runner_takes_no_timing_option():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchmark_latency", PROJECT_ROOT / "scripts/benchmark_latency.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {"-h", "--help", "--preflight-only"}
    for flag in ("--warmup", "--iterations", "--device", "--checkpoint", "--batch-size"):
        with pytest.raises(SystemExit):
            parser.parse_args([flag, "1"])


# --------------------------------------------------------------------------
# on a real GPU: the timed call does not synchronize
# --------------------------------------------------------------------------


@CUDA
@pytest.mark.parametrize("module", [cnn, unet])
def test_the_timed_call_makes_no_synchronizing_cuda_call(module):
    model = module.build_model(device="cuda").eval()
    batch = torch.rand(1, 1, 16, 16, device="cuda")
    latency.cuda_sync_probe(model, batch)


@CUDA
def test_the_sync_probe_catches_a_host_readback():
    class Readback(torch.nn.Module):
        def restore(self, batch):
            return batch * float(batch.max())

    with pytest.raises(RuntimeError, match="synchroniz"):
        latency.cuda_sync_probe(Readback(), torch.rand(1, 1, 16, 16, device="cuda"))


@CUDA
def test_the_real_cuda_timer_returns_one_positive_sample_per_iteration():
    # An untrained model on a toy input: exercises the event API, measures nothing.
    model = cnn.build_model(device="cuda")
    samples = latency.time_cuda_restore(model, torch.rand(1, 1, 16, 16, device="cuda"), 2, 3)
    assert len(samples) == 3
    assert all(np.isfinite(samples)) and all(value > 0 for value in samples)
