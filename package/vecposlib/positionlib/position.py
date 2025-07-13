# 標準ライブラリ
from typing import Final, Generic, Iterator, List, Tuple, Union, cast

# サードパーティライブラリ
from numba import njit

# ローカルモジュール
from ..common import (
    T,
    ArrayType,
    CoordinateName,
    VectorDimension,
    _USE_CUPY,
    xp,
)

# 定数
_DEF_INT_KIND = ("i",)


@njit(cache=True)
def _norm_f(arr: ArrayType) -> float:
    s = 0.0
    for v in arr:
        s += v * v
    return s**0.5


@njit(cache=True)
def _is_zero_f(arr: ArrayType) -> bool:
    for v in arr:
        if v != 0:
            return False
    return True


def _norm(arr: ArrayType) -> float:
    return (
        _norm_f(arr)
        if not _USE_CUPY and hasattr(arr, "dtype")
        else float((arr * arr).sum() ** 0.5)
    )


def _is_zero(arr: ArrayType) -> bool:
    return (
        _is_zero_f(arr)
        if not _USE_CUPY and hasattr(arr, "dtype")
        else bool((arr == 0).all())
    )


class Position(Generic[T]):
    def __init__(self, *args: T):
        if not 1 <= len(args) <= 4:
            raise TypeError(f"Position takes 1 to 4 arguments, got {len(args)}")
        for i, v in enumerate(args):
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"Argument {i} must be int or float, got {type(v).__name__}"
                )
        dtype = float if any(isinstance(a, float) for a in args) else int
        arr = xp.array(args, dtype=dtype)
        arr.setflags(write=False)
        is_int = (
            getattr(arr, "dtype", None) is not None and arr.dtype.kind in _DEF_INT_KIND
        )
        self._coords: ArrayType = arr
        self._locked: bool = True
        self._is_int: bool = is_int

    def __setattr__(self, name: str, value: object):
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
        return iter(self._cast_coords(self._coords))

    def __eq__(self, other: object) -> bool:
        return (
            False
            if not isinstance(other, Position)
            else (self._coords == other._coords).all()
        )

    def __getitem__(self, key: Union[int, CoordinateName]) -> T:
        names: Tuple[CoordinateName, ...] = ("x", "y", "z", "w")
        if isinstance(key, int):
            if key < 0 or key >= self._coords.size:
                raise IndexError("Position index out of range")
            v = key
        elif isinstance(key, str):
            idx = names.index(key)
            if self._coords.size <= idx:
                raise KeyError(f"'{key}' is not defined for this dimension")
            v = idx

        return self._get_coord(v)

    def _get_coord(self, index: int) -> T:
        v = self._coords[index]
        target_type: Final = int if self._is_int else float
        return cast(T, int(v) if target_type == int else float(v))

    @property
    def x(self) -> T:
        return self._get_coord(0)

    @property
    def y(self) -> T:
        if self._coords.size <= 1:
            raise IndexError("y is not defined for this dimension")
        return self._get_coord(1)

    @property
    def z(self) -> T:
        if self._coords.size <= 2:
            raise IndexError("z is not defined for this dimension")
        return self._get_coord(2)

    @property
    def w(self) -> T:
        if self._coords.size <= 3:
            raise IndexError("w is not defined for this dimension")
        return self._get_coord(3)

    @property
    def ndim(self) -> int:
        return self._coords.size

    @property
    def dimension(self) -> VectorDimension:
        return cast(VectorDimension, self._coords.size)

    def _cast_coords(self, coords: ArrayType) -> List[T]:
        target_type: Final = int if self._is_int else float
        return [cast(T, int(v) if target_type == int else float(v)) for v in coords]

    def to_list(self) -> List[T]:
        return self._cast_coords(self._coords)

    def to_tuple(self) -> Tuple[T, ...]:
        return tuple(self._cast_coords(self._coords))

    def is_zero(self) -> bool:
        return bool(_is_zero(self._coords))

    def normalize(self) -> "Position[float]":
        norm = _norm(self._coords)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return Position[float](* (self._coords / norm).tolist())

    def __repr__(self) -> str:
        names = ["x", "y", "z", "w"]
        coords = [f"{names[i]}={self._coords[i]}" for i in range(self._coords.size)]
        return f"{self.__class__.__name__}({', '.join(coords)})"
