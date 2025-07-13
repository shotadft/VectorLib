# 標準ライブラリ
from typing import (
    Final,
    Generic,
    Iterator,
    List,
    Tuple,
    TypeVar,
    Union,
    cast,
)

# サードパーティライブラリ
from numba import njit

try:
    import cupy as xp  # type: ignore

    _USE_CUPY = True
except ImportError:
    import numpy as xp

    _USE_CUPY = False

# 型定義
Number = Union[float, int]
T = TypeVar("T", bound=Number)

_DEF_INT_KIND: Final = ("i",)


def _is_int(arr) -> bool:
    return getattr(arr, "dtype", None) is not None and arr.dtype.kind in _DEF_INT_KIND


def _to_number(v, is_int: bool) -> Number:
    return int(v) if is_int else float(v)


def _validate(args: tuple, name: str):
    """引数のバリデーション（共通処理）"""
    if not 1 <= len(args) <= 4:
        raise TypeError(f"{name} takes 1 to 4 arguments, got {len(args)}")
    for i, v in enumerate(args):
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"Argument {i} must be int or float, got {type(v).__name__}"
            )


@njit(cache=True)
def _norm_fast(arr) -> float:
    s = 0.0
    for v in arr:
        s += v * v
    return s**0.5


@njit(cache=True)
def _is_zero_fast(arr) -> bool:
    for v in arr:
        if v != 0:
            return False
    return True


def _norm(arr) -> float:
    if not _USE_CUPY and hasattr(arr, "dtype"):
        return _norm_fast(arr)
    else:
        return float((arr * arr).sum() ** 0.5)


def _is_zero(arr) -> bool:
    if not _USE_CUPY and hasattr(arr, "dtype"):
        return _is_zero_fast(arr)
    else:
        return bool((arr == 0).all())


class Position(Generic[T]):
    def __init__(self, *args: T):
        _validate(args, "Position")
        dtype = float if any(isinstance(a, float) for a in args) else int
        arr = xp.array(args, dtype=dtype)
        arr.setflags(write=False)
        self._coords: xp.ndarray = arr
        self._locked: bool = True
        self._is_int: bool = _is_int(arr)

    def __setattr__(self, name, value):
        if (
            hasattr(self, "_locked")
            and self._locked
            and name not in {"_coords", "_locked", "_is_int"}
        ):
            raise AttributeError("Position is immutable")
        super().__setattr__(name, value)

    def __len__(self) -> int:
        return self._coords.size

    def __iter__(self) -> Iterator[T]:
        is_int = self._is_int
        target_type = int if is_int else float
        for v in self._coords:
            if target_type == int:
                yield cast(T, int(v))
            else:
                yield cast(T, float(v))

    def __getitem__(self, key) -> T:
        names: Tuple[str, ...] = ("x", "y", "z", "w")
        if isinstance(key, int):
            if key < 0 or key >= self._coords.size:
                raise IndexError("Position index out of range")
            v = self._coords[key]
            target_type = int if self._is_int else float
            if target_type == int:
                return cast(T, int(v))
            else:
                return cast(T, float(v))
        elif isinstance(key, str):
            idx = names.index(key)
            if self._coords.size <= idx:
                raise KeyError(f"'{key}' is not defined for this dimension")
            v = self._coords[idx]
            target_type = int if self._is_int else float
            if target_type == int:
                return cast(T, int(v))
            else:
                return cast(T, float(v))
        else:
            raise TypeError("Key must be int or one of 'x', 'y', 'z', 'w'")

    @property
    def x(self) -> T:
        v = self._coords[0]
        target_type = int if self._is_int else float
        if target_type == int:
            return cast(T, int(v))
        else:
            return cast(T, float(v))

    @property
    def y(self) -> T:
        if self._coords.size <= 1:
            raise IndexError("y is not defined for this dimension")
        v = self._coords[1]
        target_type = int if self._is_int else float
        if target_type == int:
            return cast(T, int(v))
        else:
            return cast(T, float(v))

    @property
    def z(self) -> T:
        if self._coords.size <= 2:
            raise IndexError("z is not defined for this dimension")
        v = self._coords[2]
        target_type = int if self._is_int else float
        if target_type == int:
            return cast(T, int(v))
        else:
            return cast(T, float(v))

    @property
    def w(self) -> T:
        if self._coords.size <= 3:
            raise IndexError("w is not defined for this dimension")
        v = self._coords[3]
        target_type = int if self._is_int else float
        if target_type == int:
            return cast(T, int(v))
        else:
            return cast(T, float(v))

    @property
    def ndim(self) -> int:
        return self._coords.size

    def to_list(self) -> List[T]:
        is_int = self._is_int
        target_type = int if is_int else float
        if target_type == int:
            return [cast(T, int(v)) for v in self._coords]
        else:
            return [cast(T, float(v)) for v in self._coords]

    def to_tuple(self) -> Tuple[T, ...]:
        is_int = self._is_int
        target_type = int if is_int else float
        if target_type == int:
            return tuple(cast(T, int(v)) for v in self._coords)
        else:
            return tuple(cast(T, float(v)) for v in self._coords)

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
        return f"{self.__class__.__name__}({', '.join(coords)})"
