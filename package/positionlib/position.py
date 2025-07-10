from typing import Union, List, Tuple, TypeVar, Generic
from numba import njit

Number = Union[float, int]
T = TypeVar("T", bound=Number)

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

def _validate_args(args: tuple, name: str) -> None:
    """引数のバリデーション（共通処理）"""
    if not 1 <= len(args) <= 4:
        raise TypeError(f"{name} takes 1 to 4 arguments, got {len(args)}")
    for i, v in enumerate(args):
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"Argument {i} must be int or float, got {type(v).__name__}"
            )

@njit(cache=True)
def _norm(arr: np.ndarray) -> float:
    s = 0.0
    for v in arr:
        s += v * v
    return s**0.5

@njit(cache=True)
def _is_zero(arr: np.ndarray) -> bool:
    for v in arr:
        if v != 0:
            return False
    return True

class Position(Generic[T]): # type: ignore
    def __init__(self, *args: T) -> None:
        _validate_args(args, "Position")
        dtype = float if any(isinstance(a, float) for a in args) else int
        arr = np.array(args, dtype=dtype)
        arr.setflags(write=False)
        self._coords = arr
        self._locked = True

    def __setattr__(self, name, value):
        if (
            hasattr(self, "_locked")
            and self._locked
            and name not in {"_coords", "_locked"}
        ):
            raise AttributeError("Position is immutable")
        super().__setattr__(name, value)

    def _get_coord(self, index: int, name: str) -> Number:
        """座標取得の共通処理"""
        if self._coords.size <= index:
            raise AttributeError(f"{name} is not defined for this dimension")
        return self._coords[index]

    @property
    def x(self) -> Number:
        return self._coords[0]

    @property
    def y(self) -> Number:
        return self._get_coord(1, "y")

    @property
    def z(self) -> Number:
        return self._get_coord(2, "z")

    @property
    def w(self) -> Number:
        return self._get_coord(3, "w")

    @property
    def ndim(self) -> int:
        return self._coords.size

    def to_list(self) -> List[Number]:
        return self._coords.tolist()

    def to_tuple(self) -> Tuple[Number, ...]:
        return tuple(self._coords.tolist())

    def is_zero(self) -> bool:
        return bool(_is_zero(self._coords))

    def normalize(self) -> "Position[float]":
        norm = _norm(self._coords)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        normalized = self._coords / norm
        return Position[float](*normalized.tolist())

    def __repr__(self) -> str:
        names = ["x", "y", "z", "w"]
        coords = [f"{names[i]}={self._coords[i]}" for i in range(self._coords.size)]
        return f"Position({', '.join(coords)})"
