"""
VecPosLib 共通定義

型定義、定数、ユーティリティ関数を提供
"""

# 標準ライブラリ
import sys
from typing import (
    Final,
    Literal,
    TypeVar,
    Union,
)

# サードパーティライブラリ
try:
    if sys.platform == "darwin": raise ImportWarning
    import cupy as xp  # type: ignore[import-untyped]
    gpu_count = cupy.cuda.runtime.getDeviceCount() # type: ignore[import-untyped]
    if gpu_count > 0:
        _USE_CUPY = True
    else:
        raise ImportWarning
except ImportWarning:
    import numpy as xp  # type: ignore[import-untyped]
    _USE_CUPY = False

# 型定義
Number = Union[float, int]
T = TypeVar("T", bound=Number)

VectorDimension = Literal[1, 2, 3, 4]
CoordinateName = Literal["x", "y", "z", "w"]
ArrayType = xp.ndarray

# 定数
_DEF_INT_KIND: Final = ("i",)
_DEFAULT_TOLERANCE: Final = 1e-8
_MAX_VECTOR_LENGTH: Final = 1024

# ユーティリティ関数
def _is_int(arr: ArrayType) -> bool:
    return getattr(arr, "dtype", None) is not None and arr.dtype.kind in _DEF_INT_KIND
