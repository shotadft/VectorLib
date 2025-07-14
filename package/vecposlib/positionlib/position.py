"""座標情報管理用モジュール"""

# 標準ライブラリ
from typing import Generic, Iterator, List, Tuple, Union, cast

# サードパーティライブラリ
from numba import njit

# プロジェクト共通
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
    """座標情報クラス（1～4次元対応）"""

    def __init__(self, *args: T):
        """座標初期化"""
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
        """属性設定（イミュータブル制御）"""
        if (
            hasattr(self, "_locked")
            and self._locked
            and name not in {"_coords", "_locked", "_is_int"}
        ):
            assert False, "Position is immutable"
        super().__setattr__(name, value)

    def __len__(self) -> int:
        """要素数返却"""
        return self._coords.size

    def __iter__(self) -> Iterator[T]:
        """イテレータ返却"""
        return iter(self._cast_coords(self._coords))

    def __eq__(self, other: object) -> bool:
        """等価判定"""
        return (
            False
            if not isinstance(other, Position)
            else (self._coords == other._coords).all()
        )

    def __getitem__(self, key: Union[int, CoordinateName]) -> T:
        """インデックスまたは座標名で値取得"""
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
        else:
            raise TypeError(f"Invalid key type: {type(key).__name__}")
        return self._get_coord(v)

    def _get_coord(self, index: int) -> T:
        """指定インデックスの座標値取得"""
        v = self._coords[index]
        target_type = int if self._is_int else float
        return cast(T, int(v) if target_type == int else float(v))

    @property
    def x(self) -> T:
        """x座標値返却"""
        return self._get_coord(0)

    @property
    def y(self) -> T:
        """y座標値返却"""
        if self._coords.size <= 1:
            raise IndexError("y is not defined for this dimension")
        return self._get_coord(1)

    @property
    def z(self) -> T:
        """z座標値返却"""
        if self._coords.size <= 2:
            raise IndexError("z is not defined for this dimension")
        return self._get_coord(2)

    @property
    def w(self) -> T:
        """w座標値返却"""
        if self._coords.size <= 3:
            raise IndexError("w is not defined for this dimension")
        return self._get_coord(3)

    @property
    def ndim(self) -> int:
        """次元数返却"""
        return self._coords.size

    @property
    def dimension(self) -> VectorDimension:
        """次元数（型安全）返却"""
        return cast(VectorDimension, self._coords.size)

    def _cast_coords(self, coords: ArrayType) -> List[T]:
        """配列を型Tのリスト変換"""
        target_type = int if self._is_int else float
        return [cast(T, int(v) if target_type == int else float(v)) for v in coords]

    def to_list(self) -> List[T]:
        """座標値リスト返却"""
        return self._cast_coords(self._coords)

    def to_tuple(self) -> Tuple[T, ...]:
        """座標値タプル返却"""
        return tuple(self._cast_coords(self._coords))

    def is_zero(self) -> bool:
        """全要素ゼロ判定"""
        return bool(_is_zero(self._coords))

    def normalize(self) -> "Position[float]":
        """正規化座標返却"""
        norm = _norm(self._coords)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return Position[float](*(self._coords / norm).tolist())

    def __repr__(self) -> str:
        """文字列表現返却"""
        names = ["x", "y", "z", "w"]
        coords = [f"{names[i]}={v}" for i, v in enumerate(self.to_list())]
        return f"{self.__class__.__name__}({', '.join(coords)})"
