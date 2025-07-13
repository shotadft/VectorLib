# 標準ライブラリ
from typing import (
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

# サードパーティライブラリ
from numba import njit, prange

# ローカルモジュール
from ..common import (
    ArrayType,
    CoordinateName,
    Number,
    T,
    VectorDimension,
    _DEFAULT_TOLERANCE,
    _MAX_VECTOR_LENGTH,
    _USE_CUPY,
    _is_int,
    xp,
)
from positionlib.position import Position


def _refl(a: Number, b: Number, c: Number, factor: Number = 2) -> float:
    return float(a - b * factor * c)


def _acos(dot: float, norm1: float, norm2: float) -> float:
    cos_theta = dot / (norm1 * norm2)
    return xp.acos(xp.clip(cos_theta, -1.0, 1.0))


def _refl_vec(incident: "Vector", normal: "Vector", dot_product: float) -> "Vector":
    return incident - normal * 2 * dot_product


def _proj_vec(normal: "Vector", dot_product: float) -> "Vector":
    return normal * dot_product


def _refl_coords(
    coords: Sequence[Number], normal_coords: Sequence[Number], dot_product: float
) -> List[float]:
    return [
        _refl(coord, dot_product, n_coord)
        for coord, n_coord in zip(coords, normal_coords)
    ]


def _proj_coords(normal_coords: Sequence[Number], dot_product: float) -> List[float]:
    return [float(n_coord * dot_product) for n_coord in normal_coords]


def _inv_coords(coords: Sequence[Number]) -> List[float]:
    return [float(-coord) for coord in coords]


@njit(cache=True, parallel=True, fastmath=True)
def _norm_f(arr) -> float:
    s = 0.0
    for i in prange(arr.size):
        s += arr[i] * arr[i]
    return s**0.5


def _norm(arr) -> float:
    return (
        _norm_f(arr)
        if not _USE_CUPY and hasattr(arr, "dtype") and hasattr(arr, "sum")
        else float((arr * arr).sum() ** 0.5)
    )


@njit(cache=True, parallel=True, fastmath=True)
def _dot_f(a, b) -> float:
    s = 0.0
    for i in prange(a.size):
        s += a[i] * b[i]
    return s


def _dot(a, b) -> float:
    return (
        _dot_f(a, b) if not _USE_CUPY and hasattr(a, "dtype") else float((a * b).sum())
    )


class Vector(Generic[T]):
    def __init__(self, data: Union[Sequence[T], Position[T]]):
        if isinstance(data, Position):
            data = data.to_tuple()

        if not 1 <= len(data) <= _MAX_VECTOR_LENGTH:
            raise ValueError(f"Vector length must be 1 to {_MAX_VECTOR_LENGTH}")

        has_float = any(isinstance(x, float) for x in data)
        dtype = float if has_float else int
        arr = xp.array(data, dtype=dtype)
        arr.setflags(write=False)

        self._vec, self._locked, self._is_int = arr, True, _is_int(arr)

    @property
    def dimension(self) -> VectorDimension:
        return cast(VectorDimension, self._vec.size)

    def _create(self, data: Sequence[Number]) -> "Vector[float]":
        return Vector[float](data)

    def _get_coord(self, index: int) -> T:
        if index < 0 or index >= self._vec.size:
            raise IndexError(f"Coordinate index {index} out of range")
        v = self._vec[index]
        target_type = int if self._is_int else float
        return cast(T, int(v) if target_type == int else float(v))

    def _from_arr(self, arr: ArrayType) -> "Vector[float]":
        return self._create(arr.tolist())

    def __setattr__(self, name, value):
        if (
            hasattr(self, "_locked")
            and self._locked
            and name not in {"_vec", "_locked", "_is_int"}
        ):
            raise AttributeError("Vector is immutable")
        super().__setattr__(name, value)

    @property
    def ndim(self) -> int:
        return self._vec.size

    def _cast_coords(self, coords: ArrayType) -> List[T]:
        target_type = int if self._is_int else float
        return [cast(T, int(v) if target_type == int else float(v)) for v in coords]

    def to_list(self) -> List[T]:
        return self._cast_coords(self._vec)

    def to_tuple(self) -> Tuple[T, ...]:
        return tuple(self._cast_coords(self._vec))

    def norm(self) -> float:
        return _norm(self._vec)

    def normalize(self) -> "Vector[float]":
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize zero vector")
        return self._from_arr(self._vec / n)

    def dot(self, other: "Vector[T]") -> float:
        return _dot(self._vec, other._vec)

    def __add__(self, other: "Vector[T]") -> "Vector[float]":
        return self._create((self._vec + other._vec).tolist())

    def __sub__(self, other: "Vector[T]") -> "Vector[float]":
        return self._create((self._vec - other._vec).tolist())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_list()})"

    def __getitem__(self, idx: int) -> T:
        return self._get_coord(idx)

    def __len__(self) -> int:
        return self._vec.size

    def __iter__(self) -> Iterator[T]:
        return iter(self._cast_coords(self._vec))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return (self._vec == other._vec).all()

    def __mul__(self, scalar: Number) -> "Vector[float]":
        return Vector[float]((self._vec * scalar).tolist())

    def __rmul__(self, scalar: Number) -> "Vector[float]":
        return self.__mul__(scalar)

    def distance(self, other: "Vector[T]") -> float:
        diff = self._vec - other._vec
        return float((diff * diff).sum() ** 0.5)

    def _cast_val(self, value: float) -> T:
        target_type = int if self._is_int else float
        return cast(T, int(value) if target_type == int else float(value))

    def sum(self) -> T:
        return self._cast_val(self._vec.sum())

    def prod(self) -> T:
        return self._cast_val(self._vec.prod())

    def min(self) -> T:
        return self._cast_val(self._vec.min())

    def max(self) -> T:
        return self._cast_val(self._vec.max())

    def manhattan(self, other: "Vector[T]") -> T:
        diff = abs(self._vec - other._vec)
        return self._cast_val(diff.sum())

    def lerp(self, other: "Vector[T]", t: float) -> "Vector[float]":
        return self._from_arr(self._vec * (1 - t) + other._vec * t)

    def clamp(self, min_val: Number, max_val: Number) -> "Vector[float]":
        arr = xp.clip(self._vec, min_val, max_val)
        return self._from_arr(arr)

    def abs(self) -> "Vector[float]":
        arr = xp.abs(self._vec)
        return self._from_arr(arr)

    def is_unit(self, tol: float = _DEFAULT_TOLERANCE) -> bool:
        return abs(self.norm() - 1.0) < tol

    def inverse(self) -> "Vector[float]":
        return self._from_arr(-self._vec)

    def astype(self, dtype: type) -> "Vector[float]":
        arr = self._vec.astype(dtype)
        return self._from_arr(arr)

    def reflect(self, normal: "Vector[T]") -> "Vector[float]":
        s, n = self.astype(float), normal.normalize()
        d = s.dot(n)
        return _refl_vec(s, n, d).normalize() * s.norm()

    def project(self, other: "Vector[T]") -> "Vector[float]":
        s, n = self.astype(float), other.normalize()
        d = s.dot(n)
        return _proj_vec(n, d)

    def angle_between(self, other: "Vector[T]") -> float:
        dot, norm1, norm2 = self.dot(other), self.norm(), other.norm()
        return _acos(dot, norm1, norm2)

    def get_coordinate(self, name: CoordinateName) -> T:
        coord_map = {"x": 0, "y": 1, "z": 2, "w": 3}
        if name not in coord_map:
            raise ValueError(f"Invalid coordinate name: {name}")
        idx = coord_map[name]
        if idx >= self._vec.size:
            raise IndexError(f"Coordinate '{name}' is not defined for this dimension")
        return self._get_coord(idx)

    def _inv_coords(self) -> List[float]:
        coords = [self._get_coord(i) for i in range(self._vec.size)]
        return _inv_coords(coords)

    def _refl_coords(self, normal: "Vector[T]") -> List[float]:
        n_vec = normal.normalize()
        n_coords = [n_vec[i] for i in range(self._vec.size)]
        d = sum(self._get_coord(i) * n_coords[i] for i in range(self._vec.size))
        coords = [self._get_coord(i) for i in range(self._vec.size)]
        return _refl_coords(coords, n_coords, d)

    def _proj_coords(self, other: "Vector[T]") -> List[float]:
        n_vec = other.normalize()
        n_coords = [n_vec[i] for i in range(self._vec.size)]
        d = sum(self._get_coord(i) * n_coords[i] for i in range(self._vec.size))
        return _proj_coords(n_coords, d)


class Vec2(Vector[T]):
    @overload
    def __init__(self, x: Position[T], y: None = None): ...
    @overload
    def __init__(self, x: T, y: T): ...

    def __init__(self, x: Union[T, Position[T]], y: Optional[T] = None):
        if isinstance(x, Position):
            if len(x) != 2:
                raise ValueError
            super().__init__(x)
        else:
            if y is None:
                raise TypeError("y must not be None when x is not Position")
            super().__init__([x, y])

    def _create(self, data: Sequence[Number]) -> "Vec2[float]":
        return Vec2[float](data[0], data[1])

    @property
    def x(self) -> T:
        return self.get_coordinate("x")

    @property
    def y(self) -> T:
        return self.get_coordinate("y")

    def cross(self, other: "Vec2[T]") -> T:
        result = self.x * other.y - self.y * other.x
        return self._cast_val(result)

    def angle(self, other: "Vec2[T]") -> float:
        dot = self.x * other.x + self.y * other.y
        norm1, norm2 = xp.sqrt(self.x**2 + self.y**2), xp.sqrt(other.x**2 + other.y**2)
        return _acos(dot, norm1, norm2)

    def inverse(self) -> "Vec2[float]":
        inverse_coords = self._inv_coords()
        return Vec2[float](*inverse_coords)

    def reflect(self, normal: "Vec2[T]") -> "Vec2[float]":  # type: ignore[override]
        reflection_coords = self._refl_coords(normal)
        return Vec2[float](*reflection_coords)

    def project(self, other: "Vec2[T]") -> "Vec2[float]":  # type: ignore[override]
        projection_coords = self._proj_coords(other)
        return Vec2[float](*projection_coords)


class Vec3(Vector[T]):
    @overload
    def __init__(self, x: Position[T], y: None = None, z: None = None): ...
    @overload
    def __init__(self, x: T, y: T, z: T): ...
    def __init__(
        self, x: Union[T, Position[T]], y: Optional[T] = None, z: Optional[T] = None
    ):
        if isinstance(x, Position):
            if len(x) != 3:
                raise ValueError
            super().__init__(x)
        else:
            if y is None or z is None:
                raise TypeError("y and z must not be None when x is not Position")
            super().__init__([x, y, z])

    def _create(self, data: Sequence[Number]) -> "Vec3[float]":
        return Vec3[float](data[0], data[1], data[2])

    @property
    def x(self) -> T:
        return self.get_coordinate("x")

    @property
    def y(self) -> T:
        return self.get_coordinate("y")

    @property
    def z(self) -> T:
        return self.get_coordinate("z")

    def cross(self, other: "Vec3[T]") -> "Vec3[float]":
        cx = self.y * other.z - self.z * other.y
        cy = self.z * other.x - self.x * other.z
        cz = self.x * other.y - self.y * other.x
        return Vec3[float](float(cx), float(cy), float(cz))

    def angle(self, other: "Vec3[T]") -> float:
        dot = self.x * other.x + self.y * other.y + self.z * other.z
        norm1, norm2 = xp.sqrt(self.x**2 + self.y**2 + self.z**2), xp.sqrt(
            other.x**2 + other.y**2 + other.z**2
        )
        return _acos(dot, norm1, norm2)

    def inverse(self) -> "Vec3[float]":
        inverse_coords = self._inv_coords()
        return Vec3[float](*inverse_coords)

    def reflect(self, normal: "Vec3[T]") -> "Vec3[float]":  # type: ignore[override]
        reflection_coords = self._refl_coords(normal)
        return Vec3[float](*reflection_coords)

    def project(self, other: "Vec3[T]") -> "Vec3[float]":  # type: ignore[override]
        projection_coords = self._proj_coords(other)
        return Vec3[float](*projection_coords)


class Vec4(Vector[T]):
    @overload
    def __init__(
        self, x: Position[T], y: None = None, z: None = None, w: None = None
    ): ...
    @overload
    def __init__(self, x: T, y: T, z: T, w: T): ...
    def __init__(
        self,
        x: Union[T, Position[T]],
        y: Optional[T] = None,
        z: Optional[T] = None,
        w: Optional[T] = None,
    ):
        if isinstance(x, Position):
            if len(x) != 4:
                raise ValueError
            super().__init__(x)
        else:
            if y is None or z is None or w is None:
                raise TypeError("y, z, w must not be None when x is not Position")
            super().__init__([x, y, z, w])

    def _create(self, data: Sequence[Number]) -> "Vec4[float]":
        return Vec4[float](data[0], data[1], data[2], data[3])

    @property
    def x(self) -> T:
        return self.get_coordinate("x")

    @property
    def y(self) -> T:
        return self.get_coordinate("y")

    @property
    def z(self) -> T:
        return self.get_coordinate("z")

    @property
    def w(self) -> T:
        return self.get_coordinate("w")

    def inverse(self) -> "Vec4[float]":
        inverse_coords = self._inv_coords()
        return Vec4[float](*inverse_coords)

    def reflect(self, normal: "Vec4[T]") -> "Vec4[float]":  # type: ignore[override]
        reflection_coords = self._refl_coords(normal)
        return Vec4[float](*reflection_coords)

    def project(self, other: "Vec4[T]") -> "Vec4[float]":  # type: ignore[override]
        projection_coords = self._proj_coords(other)
        return Vec4[float](*projection_coords)
