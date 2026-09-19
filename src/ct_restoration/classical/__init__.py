"""Classical, non-learned restoration methods.

These are the comparison baselines the deep-learning methods have to beat.
They carry no trained parameters, so the only thing to choose is their
configuration, and that choice is made on validation and then frozen.
"""

from ct_restoration.classical.clahe import (
    ALGORITHM_VERSION,
    QUANTIZATION_BITS,
    QUANTIZATION_MAX,
    ClaheConfig,
    ClaheError,
    apply_clahe,
    clahe_restorer,
    from_uint8,
    to_uint8,
)

__all__ = [
    "ALGORITHM_VERSION",
    "QUANTIZATION_BITS",
    "QUANTIZATION_MAX",
    "ClaheConfig",
    "ClaheError",
    "apply_clahe",
    "clahe_restorer",
    "from_uint8",
    "to_uint8",
]
