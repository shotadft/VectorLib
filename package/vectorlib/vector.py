from typing import Union, List, Tuple, TypeVar, Generic, Sequence, Final
from numba import njit, prange

Number = Union[float, int]
T = TypeVar("T", bound=Number)

try:
    import cupy as xp  # type: ignore
except ImportError:
    import numpy as xp

_DEF_INT_KIND: Final = ("i",)


def _is_int_dtype(arr) -> bool:
    return getattr(arr, "dtype", None) is not None and arr.dtype.kind in _DEF_INT_KIND


def _as_number(v, is_int: bool) -> Number:
    return int(v) if is_int else float(v)


@njit(cache=True, parallel=True, fastmath=True)
def _norm_numba(arr):
    s = 0.0
    for i in prange(arr.size):
        s += arr[i] * arr[i]
    return s**0.5


def _norm(arr) -> float:
    if (
        hasattr(arr, "dtype")
        and hasattr(arr, "sum")
        and arr.__class__.__module__.startswith("numpy")
    ):
        return _norm_numba(arr)
    else:
        return float((arr * arr).sum() ** 0.5)


@njit(cache=True, parallel=True, fastmath=True)
def _dot_numba(a, b):
    s = 0.0
    for i in prange(a.size):
        s += a[i] * b[i]
    return s


def _dot(a, b) -> float:
    if hasattr(a, "dtype") and a.__class__.__module__.startswith("numpy"):
        return _dot_numba(a, b)
    else:
        return float((a * b).sum())


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
def batch_dot(arrs1: xp.ndarray, arrs2: xp.ndarray) -> xp.ndarray:
    # arrs1, arrs2: (N, D)
    N = arrs1.shape[0]
    out = xp.empty(N, dtype=xp.float64)
    for i in prange(N):
        s = 0.0
        for j in range(arrs1.shape[1]):
            s += arrs1[i, j] * arrs2[i, j]
        out[i] = s
    return out


class Vector(Generic[T]):  # type: ignore
    def __init__(self, data: Sequence[T]) -> None:
        if not 1 <= len(data) <= 1024:
            raise ValueError("Vector length must be 1 to 1024")
        dtype = float if any(isinstance(x, float) for x in data) else int
        arr = xp.array(data, dtype=dtype)
        arr.setflags(write=False)
        self._vec = arr
        self._locked = True
        self._is_int = _is_int_dtype(arr)

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

    def to_list(self) -> List[Number]:
        is_int = self._is_int
        return [_as_number(v, is_int) for v in self._vec]

    def to_tuple(self) -> Tuple[Number, ...]:
        is_int = self._is_int
        return tuple(_as_number(v, is_int) for v in self._vec)

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
        return _as_number(v, self._is_int)

    def __len__(self) -> int:
        return self._vec.size

    def __iter__(self):
        is_int = self._is_int
        return (_as_number(v, is_int) for v in self._vec)

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

    def manhattan(self, other: "Vector[T]") -> Number:
        diff = abs(self._vec - other._vec)
        s = diff.sum()
        return _as_number(s, self._is_int)

    def lerp(self, other: "Vector[T]", t: float) -> "Vector[float]":
        return Vector[float]((self._vec * (1 - t) + other._vec * t).tolist())

    def clamp(self, min_val: Number, max_val: Number) -> "Vector[T]":
        arr = xp.clip(self._vec, min_val, max_val)
        return Vector[T](arr.tolist())

    def abs(self) -> "Vector[T]":
        arr = xp.abs(self._vec)
        return Vector[T](arr.tolist())

    def sum(self) -> Number:
        s = self._vec.sum()
        return _as_number(s, self._is_int)

    def prod(self) -> Number:
        p = self._vec.prod()
        return _as_number(p, self._is_int)

    def min(self) -> Number:
        m = self._vec.min()
        return _as_number(m, self._is_int)

    def max(self) -> Number:
        m = self._vec.max()
        return _as_number(m, self._is_int)

    def is_unit(self, tol: float = 1e-8) -> bool:
        return abs(self.norm() - 1.0) < tol

    def inverse(self) -> "Vector[float]":
        return Vector[float]((-self._vec).tolist())

    def astype(self, dtype: type) -> "Vector[float]":
        arr = self._vec.astype(dtype)
        return Vector[float](arr.tolist())

    def reflect(self, normal: "Vector[T]") -> "Vector[float]":
        s = self.astype(float)
        n = normal.normalize()
        d = s.dot(n)
        return (s - n * 2 * d).normalize() * s.norm()

    def project(self, other: "Vector[T]") -> "Vector[float]":
        s = self.astype(float)
        n = other.normalize()
        d = s.dot(n)
        return n * d

    def angle_between(self, other: "Vector[T]") -> float:
        dot = self.dot(other)
        norm1 = self.norm()
        norm2 = other.norm()
        cos_theta = dot / (norm1 * norm2)
        return xp.acos(xp.clip(cos_theta, -1.0, 1.0))


class Vec2(Vector[T]):
    def __init__(self, x: T, y: T) -> None:
        super().__init__([x, y])

    @property
    def x(self) -> Number:
        v = self._vec[0]
        return _as_number(v, self._is_int)

    @property
    def y(self) -> Number:
        v = self._vec[1]
        return _as_number(v, self._is_int)

    def cross(self, other: "Vec2[T]") -> Number:
        return self.x * other.y - self.y * other.x

    def angle(self, other: "Vec2[T]") -> float:
        dot = self.x * other.x + self.y * other.y
        norm1 = xp.sqrt(self.x**2 + self.y**2)
        norm2 = xp.sqrt(other.x**2 + other.y**2)
        cos_theta = dot / (norm1 * norm2)
        return xp.acos(xp.clip(cos_theta, -1.0, 1.0))

    def distance(self, other: "Vec2[T]") -> float:  # type: ignore[override]
        return super().distance(other)

    def manhattan(self, other: "Vec2[T]") -> Number:  # type: ignore[override]
        return super().manhattan(other)

    def lerp(self, other: "Vec2[T]", t: float) -> "Vec2[float]":  # type: ignore[override]
        return Vec2[float](
            float(self.x * (1 - t) + other.x * t), float(self.y * (1 - t) + other.y * t)
        )

    def clamp(self, min_val: Number, max_val: Number) -> "Vec2[float]":  # type: ignore[override]
        arr = xp.clip(self._vec, min_val, max_val)
        return Vec2[float](float(arr[0]), float(arr[1]))

    def abs(self) -> "Vec2[float]":  # type: ignore[override]
        arr = xp.abs(self._vec)
        return Vec2[float](float(arr[0]), float(arr[1]))

    def sum(self) -> Number:
        return super().sum()

    def prod(self) -> Number:
        return super().prod()

    def min(self) -> Number:
        return super().min()

    def max(self) -> Number:
        return super().max()

    def is_unit(self, tol: float = 1e-8) -> bool:
        return super().is_unit(tol)

    def inverse(self) -> "Vec2[float]":
        return Vec2[float](-float(self.x), -float(self.y))

    def reflect(self, normal: "Vec2[T]") -> "Vec2[float]":  # type: ignore[override]
        n_vec = normal.normalize()
        n = Vec2[float](n_vec[0], n_vec[1])
        d = self.x * n.x + self.y * n.y
        rx = self.x - 2 * d * n.x
        ry = self.y - 2 * d * n.y
        return Vec2[float](float(rx), float(ry))

    def project(self, other: "Vec2[T]") -> "Vec2[float]":  # type: ignore[override]
        n_vec = other.normalize()
        n = Vec2[float](n_vec[0], n_vec[1])
        d = self.x * n.x + self.y * n.y
        return Vec2[float](float(n.x * d), float(n.y * d))

    def angle_between(self, other: "Vec2[T]") -> float:  # type: ignore[override]
        return super().angle_between(other)

    def astype(self, dtype: type) -> "Vec2[float]":
        arr = self._vec.astype(dtype)
        return Vec2[float](float(arr[0]), float(arr[1]))


class Vec3(Vector[T]):
    def __init__(self, x: T, y: T, z: T) -> None:
        super().__init__([x, y, z])

    @property
    def x(self) -> Number:
        v = self._vec[0]
        return _as_number(v, self._is_int)

    @property
    def y(self) -> Number:
        v = self._vec[1]
        return _as_number(v, self._is_int)

    @property
    def z(self) -> Number:
        v = self._vec[2]
        return _as_number(v, self._is_int)

    def cross(self, other: "Vec3[T]") -> "Vec3[float]":
        cx = self.y * other.z - self.z * other.y
        cy = self.z * other.x - self.x * other.z
        cz = self.x * other.y - self.y * other.x
        return Vec3[float](float(cx), float(cy), float(cz))

    def angle(self, other: "Vec3[T]") -> float:
        dot = self.x * other.x + self.y * other.y + self.z * other.z
        norm1 = xp.sqrt(self.x**2 + self.y**2 + self.z**2)
        norm2 = xp.sqrt(other.x**2 + other.y**2 + other.z**2)
        cos_theta = dot / (norm1 * norm2)
        return xp.acos(xp.clip(cos_theta, -1.0, 1.0))

    def distance(self, other: "Vec3[T]") -> float:  # type: ignore[override]
        return super().distance(other)

    def manhattan(self, other: "Vec3[T]") -> Number:  # type: ignore[override]
        return super().manhattan(other)

    def lerp(self, other: "Vec3[T]", t: float) -> "Vec3[float]":  # type: ignore[override]
        return Vec3[float](
            float(self.x * (1 - t) + other.x * t),
            float(self.y * (1 - t) + other.y * t),
            float(self.z * (1 - t) + other.z * t),
        )

    def clamp(self, min_val: Number, max_val: Number) -> "Vec3[float]":  # type: ignore[override]
        arr = xp.clip(self._vec, min_val, max_val)
        return Vec3[float](float(arr[0]), float(arr[1]), float(arr[2]))

    def abs(self) -> "Vec3[float]":  # type: ignore[override]
        arr = xp.abs(self._vec)
        return Vec3[float](float(arr[0]), float(arr[1]), float(arr[2]))

    def sum(self) -> Number:
        return super().sum()

    def prod(self) -> Number:
        return super().prod()

    def min(self) -> Number:
        return super().min()

    def max(self) -> Number:
        return super().max()

    def is_unit(self, tol: float = 1e-8) -> bool:
        return super().is_unit(tol)

    def inverse(self) -> "Vec3[float]":
        return Vec3[float](-float(self.x), -float(self.y), -float(self.z))

    def reflect(self, normal: "Vec3[T]") -> "Vec3[float]":  # type: ignore[override]
        n_vec = normal.normalize()
        n = Vec3[float](n_vec[0], n_vec[1], n_vec[2])
        d = self.x * n.x + self.y * n.y + self.z * n.z
        rx = self.x - 2 * d * n.x
        ry = self.y - 2 * d * n.y
        rz = self.z - 2 * d * n.z
        return Vec3[float](float(rx), float(ry), float(rz))

    def project(self, other: "Vec3[T]") -> "Vec3[float]":  # type: ignore[override]
        n_vec = other.normalize()
        n = Vec3[float](n_vec[0], n_vec[1], n_vec[2])
        d = self.x * n.x + self.y * n.y + self.z * n.z
        return Vec3[float](float(n.x * d), float(n.y * d), float(n.z * d))

    def angle_between(self, other: "Vec3[T]") -> float:  # type: ignore[override]
        return super().angle_between(other)

    def astype(self, dtype: type) -> "Vec3[float]":
        arr = self._vec.astype(dtype)
        return Vec3[float](float(arr[0]), float(arr[1]), float(arr[2]))


class Vec4(Vector[T]):
    def __init__(self, x: T, y: T, z: T, w: T) -> None:
        super().__init__([x, y, z, w])

    @property
    def x(self) -> Number:
        v = self._vec[0]
        return _as_number(v, self._is_int)

    @property
    def y(self) -> Number:
        v = self._vec[1]
        return _as_number(v, self._is_int)

    @property
    def z(self) -> Number:
        v = self._vec[2]
        return _as_number(v, self._is_int)

    @property
    def w(self) -> Number:
        v = self._vec[3]
        return _as_number(v, self._is_int)

    def distance(self, other: "Vec4[T]") -> float:  # type: ignore[override]
        return super().distance(other)

    def manhattan(self, other: "Vec4[T]") -> Number:  # type: ignore[override]
        return super().manhattan(other)

    def lerp(self, other: "Vec4[T]", t: float) -> "Vec4[float]":  # type: ignore[override]
        return Vec4[float](
            float(self.x * (1 - t) + other.x * t),
            float(self.y * (1 - t) + other.y * t),
            float(self.z * (1 - t) + other.z * t),
            float(self.w * (1 - t) + other.w * t),
        )

    def clamp(self, min_val: Number, max_val: Number) -> "Vec4[float]":  # type: ignore[override]
        arr = xp.clip(self._vec, min_val, max_val)
        return Vec4[float](float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def abs(self) -> "Vec4[float]":  # type: ignore[override]
        arr = xp.abs(self._vec)
        return Vec4[float](float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def sum(self) -> Number:
        return super().sum()

    def prod(self) -> Number:
        return super().prod()

    def min(self) -> Number:
        return super().min()

    def max(self) -> Number:
        return super().max()

    def is_unit(self, tol: float = 1e-8) -> bool:
        return super().is_unit(tol)

    def inverse(self) -> "Vec4[float]":
        return Vec4[float](
            -float(self.x), -float(self.y), -float(self.z), -float(self.w)
        )

    def reflect(self, normal: "Vec4[T]") -> "Vec4[float]":  # type: ignore[override]
        n_vec = normal.normalize()
        n = Vec4[float](n_vec[0], n_vec[1], n_vec[2], n_vec[3])
        d = self.x * n.x + self.y * n.y + self.z * n.z + self.w * n.w
        rx = self.x - 2 * d * n.x
        ry = self.y - 2 * d * n.y
        rz = self.z - 2 * d * n.z
        rw = self.w - 2 * d * n.w
        return Vec4[float](float(rx), float(ry), float(rz), float(rw))

    def project(self, other: "Vec4[T]") -> "Vec4[float]":  # type: ignore[override]
        n_vec = other.normalize()
        n = Vec4[float](n_vec[0], n_vec[1], n_vec[2], n_vec[3])
        d = self.x * n.x + self.y * n.y + self.z * n.z + self.w * n.w
        return Vec4[float](
            float(n.x * d), float(n.y * d), float(n.z * d), float(n.w * d)
        )

    def angle_between(self, other: "Vec4[T]") -> float:  # type: ignore[override]
        return super().angle_between(other)

    def astype(self, dtype: type) -> "Vec4[float]":
        arr = self._vec.astype(dtype)
        return Vec4[float](float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
