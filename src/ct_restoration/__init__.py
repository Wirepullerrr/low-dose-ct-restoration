"""Low-dose-like CT image restoration benchmark.

This package compares a no-restoration baseline, CLAHE, a small residual CNN,
and a lightweight U-Net on synthetically degraded CT slices. It is an
engineering image-restoration benchmark, not a clinical tool.
"""

from ct_restoration.config import PROJECT_ROOT, ensure_dir, load_config

__version__ = "0.1.0"

__all__ = ["PROJECT_ROOT", "ensure_dir", "load_config", "__version__"]
