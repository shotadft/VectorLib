from typing import Union, List, Tuple, TypeVar, Generic, Sequence, Literal
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
        self._coords: np.ndarray = arr
        self._locked: bool = True

    def __setattr__(self, name, value):
        if (
            hasattr(self, "_locked")
            and self._locked
            and name not in {"_coords", "_locked"}
        ):
            raise AttributeError("Position is immutable")
        super().__setattr__(name, value)

    @property
    def x(self) -> Number:
        return self._coords[0] if self._coords.dtype == int else float(self._coords[0])

    @property
    def y(self) -> Number:
        if self._coords.size <= 1:
            raise AttributeError("y is not defined for this dimension")
        return self._coords[1] if self._coords.dtype == int else float(self._coords[1])

    @property
    def z(self) -> Number:
        if self._coords.size <= 2:
            raise AttributeError("z is not defined for this dimension")
        return self._coords[2] if self._coords.dtype == int else float(self._coords[2])

    @property
    def w(self) -> Number:
        if self._coords.size <= 3:
            raise AttributeError("w is not defined for this dimension")
        return self._coords[3] if self._coords.dtype == int else float(self._coords[3])

    def __getitem__(self, key: Literal["x", "y", "z", "w"]) -> Number:
        names: Tuple[str, ...] = ("x", "y", "z", "w")
        if key not in names:
            raise KeyError(f"Invalid coordinate name: {key}")
        idx = names.index(key)
        if self._coords.size <= idx:
            raise KeyError(f"'{key}' is not defined for this dimension")
        v = self._coords[idx]
        return v if self._coords.dtype == int else float(v)

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
