# 標準ライブラリ
from typing import Final, Generic, Iterator, List, Literal, Sequence, Tuple, TypeVar, Union

# サードパーティライブラリ
from numba import njit, prange

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

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
    if hasattr(arr, "dtype") and arr.__class__.__module__.startswith("numpy"):
        return _norm_fast(arr)
    else:
        return float((arr * arr).sum() ** 0.5)


def _is_zero(arr) -> bool:
    if hasattr(arr, "dtype") and arr.__class__.__module__.startswith("numpy"):
        return _is_zero_fast(arr)
    else:
        return bool((arr == 0).all())


@njit(cache=True, parallel=True, fastmath=True)
def batch_norm(arrs: xp.ndarray) -> xp.ndarray:
    # arrs: (N, D)
    N = arrs.shape[0]
    out = xp.empty(N, dtype=xp.float64)
    for i in prange(N):
        s = 0.0
        for v in arrs[i]:
            s += v * v
        out[i] = s ** 0.5
    return out

@njit(cache=True, parallel=True, fastmath=True)
def batch_is_zero(arrs: xp.ndarray) -> xp.ndarray:
    # arrs: (N, D)
    N = arrs.shape[0]
    out = xp.empty(N, dtype=xp.bool_)
    for i in prange(N):
        is_zero = True
        for v in arrs[i]:
            if v != 0:
                is_zero = False
                break
        out[i] = is_zero
    return out


class Position(Generic[T]):  # type: ignore
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

    def __iter__(self) -> Iterator[Number]:
        is_int = self._is_int
        for v in self._coords:
            yield _to_number(v, is_int)

    def __getitem__(self, key) -> Number:
        names: Tuple[str, ...] = ("x", "y", "z", "w")
        if isinstance(key, int):
            if key < 0 or key >= self._coords.size:
                raise IndexError("Position index out of range")
            v = self._coords[key]
            return _to_number(v, self._is_int)
        elif isinstance(key, str):
            idx = names.index(key)
            if self._coords.size <= idx:
                raise KeyError(f"'{key}' is not defined for this dimension")
            v = self._coords[idx]
            return _to_number(v, self._is_int)
        else:
            raise TypeError("Key must be int or one of 'x', 'y', 'z', 'w'")

    @property
    def x(self) -> Number:
        v = self._coords[0]
        return _to_number(v, self._is_int)

    @property
    def y(self) -> Number:
        if self._coords.size <= 1:
            raise IndexError("y is not defined for this dimension")
        v = self._coords[1]
        return _to_number(v, self._is_int)

    @property
    def z(self) -> Number:
        if self._coords.size <= 2:
            raise IndexError("z is not defined for this dimension")
        v = self._coords[2]
        return _to_number(v, self._is_int)

    @property
    def w(self) -> Number:
        if self._coords.size <= 3:
            raise IndexError("w is not defined for this dimension")
        v = self._coords[3]
        return _to_number(v, self._is_int)

    @property
    def ndim(self) -> int:
        return self._coords.size

    def to_list(self) -> List[Number]:
        is_int = self._is_int
        return [_to_number(v, is_int) for v in self._coords]

    def to_tuple(self) -> Tuple[Number, ...]:
        is_int = self._is_int
        return tuple(_to_number(v, is_int) for v in self._coords)

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
