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
        return [int(v) if self._vec.dtype == int else float(v) for v in self._vec]

    def to_tuple(self) -> Tuple[Number, ...]:
        return tuple(int(v) if self._vec.dtype == int else float(v) for v in self._vec)

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
        return int(v) if self._vec.dtype == int else float(v)

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

    def distance(self, other: "Vector[T]") -> float:
        # ユークリッド距離
        diff = self._vec - other._vec
        return float((diff * diff).sum() ** 0.5)

    def manhattan(self, other: "Vector[T]") -> Number:
        # マンハッタン距離
        diff = abs(self._vec - other._vec)
        s = diff.sum()
        return int(s) if self._vec.dtype == int else float(s)

    def lerp(self, other: "Vector[T]", t: float) -> "Vector[float]":
        # 線形補間
        return Vector[float]((self._vec * (1 - t) + other._vec * t).tolist())

    def clamp(self, min_val: Number, max_val: Number) -> "Vector[T]":
        # 成分ごとの範囲制限
        arr = xp.clip(self._vec, min_val, max_val)
        return Vector[T](arr.tolist())

    def abs(self) -> "Vector[T]":
        arr = xp.abs(self._vec)
        return Vector[T](arr.tolist())

    def sum(self) -> Number:
        s = self._vec.sum()
        return int(s) if self._vec.dtype == int else float(s)

    def prod(self) -> Number:
        p = self._vec.prod()
        return int(p) if self._vec.dtype == int else float(p)

    def min(self) -> Number:
        m = self._vec.min()
        return int(m) if self._vec.dtype == int else float(m)

    def max(self) -> Number:
        m = self._vec.max()
        return int(m) if self._vec.dtype == int else float(m)

    def is_unit(self, tol: float = 1e-8) -> bool:
        return abs(self.norm() - 1.0) < tol

    def inverse(self) -> "Vector[float]":
        return Vector[float]((-self._vec).tolist())

    def astype(self, dtype: type) -> "Vector[float]":
        arr = self._vec.astype(dtype)
        return Vector[float](arr.tolist())

    def reflect(self, normal: "Vector[T]") -> "Vector[float]":
        # 反射ベクトル
        s = self.astype(float)
        n = normal.normalize()
        d = s.dot(n)
        return (s - n * 2 * d).normalize() * s.norm()

    def project(self, other: "Vector[T]") -> "Vector[float]":
        # otherへの射影
        s = self.astype(float)
        n = other.normalize()
        d = s.dot(n)
        return n * d

    def angle_between(self, other: "Vector[T]") -> float:
        # なす角（ラジアン）
        import math
        dot = self.dot(other)
        norm1 = self.norm()
        norm2 = other.norm()
        cos_theta = dot / (norm1 * norm2)
        return math.acos(max(-1.0, min(1.0, cos_theta)))

class Vec2(Vector[T]):
    def __init__(self, x: T, y: T) -> None:
        super().__init__([x, y])

    @property
    def x(self) -> Number:
        v = self._vec[0]
        return int(v) if self._vec.dtype == int else float(v)

    @property
    def y(self) -> Number:
        v = self._vec[1]
        return int(v) if self._vec.dtype == int else float(v)

    def cross(self, other: "Vec2[T]") -> Number:
        # 2次元ベクトルの外積はスカラー
        return self.x * other.y - self.y * other.x

    def angle(self, other: "Vec2[T]") -> float:
        # 2ベクトルのなす角（ラジアン）
        import math
        dot = self.x * other.x + self.y * other.y
        norm1 = (self.x ** 2 + self.y ** 2) ** 0.5
        norm2 = (other.x ** 2 + other.y ** 2) ** 0.5
        cos_theta = dot / (norm1 * norm2)
        return math.acos(max(-1.0, min(1.0, cos_theta)))

    def distance(self, other: "Vec2[T]") -> float:
        return super().distance(other)
    def manhattan(self, other: "Vec2[T]") -> Number:
        return super().manhattan(other)
    def lerp(self, other: "Vec2[T]", t: float) -> "Vec2[float]":
        return Vec2[float](float(self.x * (1 - t) + other.x * t),
                           float(self.y * (1 - t) + other.y * t))
    def clamp(self, min_val: Number, max_val: Number) -> "Vec2[float]":
        arr = xp.clip(self._vec, min_val, max_val)
        return Vec2[float](float(arr[0]), float(arr[1]))
    def abs(self) -> "Vec2[float]":
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
    def reflect(self, normal: "Vec2[T]") -> "Vec2[float]":
        n_vec = normal.normalize()
        n = Vec2[float](n_vec[0], n_vec[1])
        d = self.x * n.x + self.y * n.y
        rx = self.x - 2 * d * n.x
        ry = self.y - 2 * d * n.y
        return Vec2[float](float(rx), float(ry))
    def project(self, other: "Vec2[T]") -> "Vec2[float]":
        n_vec = other.normalize()
        n = Vec2[float](n_vec[0], n_vec[1])
        d = self.x * n.x + self.y * n.y
        return Vec2[float](float(n.x * d), float(n.y * d))
    def angle_between(self, other: "Vec2[T]") -> float:
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
        return int(v) if self._vec.dtype == int else float(v)

    @property
    def y(self) -> Number:
        v = self._vec[1]
        return int(v) if self._vec.dtype == int else float(v)

    @property
    def z(self) -> Number:
        v = self._vec[2]
        return int(v) if self._vec.dtype == int else float(v)

    def cross(self, other: "Vec3[T]") -> "Vec3[float]":
        # 3次元ベクトルの外積
        cx = self.y * other.z - self.z * other.y
        cy = self.z * other.x - self.x * other.z
        cz = self.x * other.y - self.y * other.x
        return Vec3[float](float(cx), float(cy), float(cz))

    def angle(self, other: "Vec3[T]") -> float:
        # 2ベクトルのなす角（ラジアン）
        import math
        dot = self.x * other.x + self.y * other.y + self.z * other.z
        norm1 = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        norm2 = (other.x ** 2 + other.y ** 2 + other.z ** 2) ** 0.5
        cos_theta = dot / (norm1 * norm2)
        return math.acos(max(-1.0, min(1.0, cos_theta)))

    def distance(self, other: "Vec3[T]") -> float:
        return super().distance(other)
    def manhattan(self, other: "Vec3[T]") -> Number:
        return super().manhattan(other)
    def lerp(self, other: "Vec3[T]", t: float) -> "Vec3[float]":
        return Vec3[float](float(self.x * (1 - t) + other.x * t),
                           float(self.y * (1 - t) + other.y * t),
                           float(self.z * (1 - t) + other.z * t))
    def clamp(self, min_val: Number, max_val: Number) -> "Vec3[float]":
        arr = xp.clip(self._vec, min_val, max_val)
        return Vec3[float](float(arr[0]), float(arr[1]), float(arr[2]))
    def abs(self) -> "Vec3[float]":
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
    def reflect(self, normal: "Vec3[T]") -> "Vec3[float]":
        n_vec = normal.normalize()
        n = Vec3[float](n_vec[0], n_vec[1], n_vec[2])
        d = self.x * n.x + self.y * n.y + self.z * n.z
        rx = self.x - 2 * d * n.x
        ry = self.y - 2 * d * n.y
        rz = self.z - 2 * d * n.z
        return Vec3[float](float(rx), float(ry), float(rz))
    def project(self, other: "Vec3[T]") -> "Vec3[float]":
        n_vec = other.normalize()
        n = Vec3[float](n_vec[0], n_vec[1], n_vec[2])
        d = self.x * n.x + self.y * n.y + self.z * n.z
        return Vec3[float](float(n.x * d), float(n.y * d), float(n.z * d))
    def angle_between(self, other: "Vec3[T]") -> float:
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
        return int(v) if self._vec.dtype == int else float(v)

    @property
    def y(self) -> Number:
        v = self._vec[1]
        return int(v) if self._vec.dtype == int else float(v)

    @property
    def z(self) -> Number:
        v = self._vec[2]
        return int(v) if self._vec.dtype == int else float(v)

    @property
    def w(self) -> Number:
        v = self._vec[3]
        return int(v) if self._vec.dtype == int else float(v)

    def distance(self, other: "Vec4[T]") -> float:
        return super().distance(other)
    def manhattan(self, other: "Vec4[T]") -> Number:
        return super().manhattan(other)
    def lerp(self, other: "Vec4[T]", t: float) -> "Vec4[float]":
        return Vec4[float](float(self.x * (1 - t) + other.x * t),
                           float(self.y * (1 - t) + other.y * t),
                           float(self.z * (1 - t) + other.z * t),
                           float(self.w * (1 - t) + other.w * t))
    def clamp(self, min_val: Number, max_val: Number) -> "Vec4[float]":
        arr = xp.clip(self._vec, min_val, max_val)
        return Vec4[float](float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    def abs(self) -> "Vec4[float]":
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
        return Vec4[float](-float(self.x), -float(self.y), -float(self.z), -float(self.w))
    def reflect(self, normal: "Vec4[T]") -> "Vec4[float]":
        n_vec = normal.normalize()
        n = Vec4[float](n_vec[0], n_vec[1], n_vec[2], n_vec[3])
        d = self.x * n.x + self.y * n.y + self.z * n.z + self.w * n.w
        rx = self.x - 2 * d * n.x
        ry = self.y - 2 * d * n.y
        rz = self.z - 2 * d * n.z
        rw = self.w - 2 * d * n.w
        return Vec4[float](float(rx), float(ry), float(rz), float(rw))
    def project(self, other: "Vec4[T]") -> "Vec4[float]":
        n_vec = other.normalize()
        n = Vec4[float](n_vec[0], n_vec[1], n_vec[2], n_vec[3])
        d = self.x * n.x + self.y * n.y + self.z * n.z + self.w * n.w
        return Vec4[float](float(n.x * d), float(n.y * d), float(n.z * d), float(n.w * d))
    def angle_between(self, other: "Vec4[T]") -> float:
        return super().angle_between(other)
    def astype(self, dtype: type) -> "Vec4[float]":
        arr = self._vec.astype(dtype)
        return Vec4[float](float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
