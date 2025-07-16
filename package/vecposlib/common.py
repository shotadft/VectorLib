# Common definitions for VecPosLib

# Standard library
import sys
from typing import Final, Literal, TypeVar, Union

# Third-party libraries
try:
    if sys.platform == "darwin":
        raise ImportError
    import cupy  # type: ignore[import-untyped]

    if cupy.cuda.runtime.getDeviceCount() > 0:  # type: ignore[import-untyped]
        xp = cupy
        _USE_CUPY = True
    else:
        raise ImportError
except ImportError:
    import numpy as xp  # type: ignore[import-untyped]
    _USE_CUPY = False

# Type definitions
Number = Union[int, float]
T = TypeVar("T", bound=Number)

VectorDimension = Literal[1, 2, 3, 4]
CoordinateName = Literal['x', 'y', 'z', 'w']
ArrayType = xp.ndarray

# Constants
_DEFAULT_TOLERANCE: Final[float] = 1e-8
_MAX_VECTOR_LENGTH: Final[int] = 1024
