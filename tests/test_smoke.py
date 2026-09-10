"""Milestone 0 smoke tests: package import, paths, config convention, environment."""

from __future__ import annotations

import pytest

import ct_restoration
from ct_restoration.config import CONFIGS_DIR, PROJECT_ROOT, ensure_dir, load_config


def test_package_exposes_version() -> None:
    assert isinstance(ct_restoration.__version__, str)
    assert ct_restoration.__version__


def test_project_root_is_the_repository() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert CONFIGS_DIR.parent == PROJECT_ROOT


def test_load_config_reads_a_yaml_mapping(tmp_path) -> None:
    config_file = tmp_path / "example.yaml"
    config_file.write_text(
        "seed: 42\nimage_size: 256\nwindow:\n  center: -600\n  width: 1500\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config == {"seed": 42, "image_size": 256, "window": {"center": -600, "width": 1500}}


def test_load_config_treats_an_empty_file_as_an_empty_mapping(tmp_path) -> None:
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")

    assert load_config(config_file) == {}


def test_load_config_rejects_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_rejects_a_non_mapping_document(tmp_path) -> None:
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(TypeError):
        load_config(config_file)


def test_ensure_dir_is_idempotent(tmp_path) -> None:
    target = tmp_path / "outputs" / "runs"

    assert ensure_dir(target).is_dir()
    assert ensure_dir(target).is_dir()


def test_torch_imports_and_computes_on_cpu() -> None:
    import torch

    result = torch.ones(2, 2) + torch.ones(2, 2)

    assert result.sum().item() == pytest.approx(8.0)


def test_report_cuda_availability() -> None:
    """Not a pass/fail requirement: records the device the benchmark will use."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; models would train and benchmark on CPU")

    assert torch.cuda.get_device_name(0)
