"""
VecPosLib 共通定義

型定義、定数、ユーティリティ関数を提供
"""

# 標準ライブラリ
import sys
from typing import Final, Literal, TypeVar, Union

# サードパーティライブラリ
try:
    if sys.platform == "darwin":
        raise ImportError
    import cupy # type: ignore[import-untyped]

    if cupy.cuda.runtime.getDeviceCount() > 0:  # type: ignore[import-untyped]
        xp = cupy
        _USE_CUPY = True
    else:
        raise ImportError
except ImportError:
    import numpy as xp  # type: ignore[import-untyped]
    _USE_CUPY = False

# 型定義
Number = Union[int, float]
T = TypeVar("T", bound=Number)

VectorDimension = Literal[1, 2, 3, 4]
CoordinateName = Literal['x', 'y', 'z', 'w']
ArrayType = xp.ndarray

# 定数
_DEFAULT_TOLERANCE: Final[float] = 1e-8
_MAX_VECTOR_LENGTH: Final[int] = 1024
