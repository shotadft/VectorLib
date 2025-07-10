from typing import Union, List, Tuple, TypeVar, Generic, Sequence
from numba import njit

Number = Union[float, int]
T = TypeVar("T", bound=Number)

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

@njit(cache=True)
def _norm(arr: xp.ndarray) -> float:
    s = 0.0
    for v in arr:
        s += v * v
    return s**0.5

@njit(cache=True)
def _dot(a: xp.ndarray, b: xp.ndarray) -> float:
    s = 0.0
    for i in range(a.size):
        s += a[i] * b[i]
    return s

class Vector(Generic[T]): # type: ignore
    def __init__(self, data: Sequence[T]) -> None:
        if not 1 <= len(data) <= 1024:
            raise ValueError("Vector length must be 1 to 1024")
        dtype = float if any(isinstance(x, float) for x in data) else int
        arr = xp.array(data, dtype=dtype)
        arr.setflags(write=False)
        self._vec = arr
        self._locked = True

    def __setattr__(self, name, value):
        if hasattr(self, "_locked") and self._locked and name not in {"_vec", "_locked"}:
            raise AttributeError("Vector is immutable")
        super().__setattr__(name, value)

    @property
    def ndim(self) -> int:
        return self._vec.size

    def to_list(self) -> List[Number]:
        return self._vec.tolist()

    def to_tuple(self) -> Tuple[Number, ...]:
        return tuple(self._vec.tolist())

    def norm(self) -> float:
        return _norm(self._vec)

    def normalize(self) -> "Vector[float]":
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector[float]((self._vec / n).tolist())

    def dot(self, other: "Vector[T]") -> float:
        return _dot(self._vec, other._vec)

    def __add__(self, other: "Vector[T]") -> "Vector[T]":
        return Vector((self._vec + other._vec).tolist())

    def __sub__(self, other: "Vector[T]") -> "Vector[T]":
        return Vector((self._vec - other._vec).tolist())

    def __repr__(self) -> str:
        return f"Vector({self.to_list()})"

    def __getitem__(self, idx: int) -> Number:
        v = self._vec[idx]
        return v if self._vec.dtype == int else float(v)

    def __len__(self) -> int:
        return self._vec.size

    def __iter__(self):
        return iter(self._vec.tolist())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return (self._vec == other._vec).all()

    def __mul__(self, scalar: Number) -> "Vector[float]":
        return Vector[float]((self._vec * scalar).tolist())

    def __rmul__(self, scalar: Number) -> "Vector[float]":
        return self.__mul__(scalar)
