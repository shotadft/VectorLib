"""N-dimensional vector operations"""

# Standard library
from typing import (
    Final,
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

# Third-party libraries
from numba import njit, prange

# Project common
from ..common import (
    T,
    ArrayType,
    CoordinateName,
    Number,
    VectorDimension,
    _DEFAULT_TOLERANCE,
    _MAX_VECTOR_LENGTH,
    _USE_CUPY,
    xp,
)

# Local modules
from ..positionlib.position import Position

# Constants
_DEF_INT_KIND = ("i",)


@njit(cache=True, parallel=True, fastmath=True)
def _norm_f(arr: ArrayType) -> float:
    """Euclidean norm"""
    s = 0.0
    for i in prange(arr.size):  # pylint: disable=not-an-iterable
        s += arr[i] * arr[i]
    return s**0.5


def _norm(arr: ArrayType) -> float:
    """Euclidean norm"""
    return (
        _norm_f(arr)
        if not _USE_CUPY and hasattr(arr, "dtype") and hasattr(arr, "sum")
        else float((arr * arr).sum() ** 0.5)
    )


@njit(cache=True, parallel=True, fastmath=True)
def _dot_f(a: ArrayType, b: ArrayType) -> float:
    """Dot product"""
    s = 0.0
    for i in prange(a.size):  # pylint: disable=not-an-iterable
        s += a[i] * b[i]
    return s


def _dot(a: ArrayType, b: ArrayType) -> float:
    """Dot product"""
    return (
        _dot_f(a, b) if not _USE_CUPY and hasattr(a, "dtype") else float((a * b).sum())
    )


def _refl_vec(incident: "Vector", normal: "Vector", dot_product: float) -> "Vector":
    """Reflected vector"""
    return incident - normal * 2 * dot_product


def _proj_vec(normal: "Vector", dot_product: float) -> "Vector":
    """Projected vector"""
    return normal * dot_product


def _acos(dot: float, norm1: float, norm2: float) -> float:
    """Angle arccosine"""
    return xp.acos(xp.clip((dot / (norm1 * norm2)), -1.0, 1.0))


class Vector(Generic[T]):
    """N-dimensional vector"""

    # --- Initialization ---
    def __init__(self, data: Union[Sequence[T], Position[T]]):
        """N-dimensional vector."""
        if isinstance(data, Position):
            data = data.to_tuple()

        if not 1 <= len(data) <= _MAX_VECTOR_LENGTH:
            raise ValueError(f"Vector length must be 1 to {_MAX_VECTOR_LENGTH}")

        has_float = any(isinstance(x, float) for x in data)
        dtype = float if has_float else int
        arr: ArrayType = xp.array(data, dtype=dtype)
        arr.setflags(write=False)
        is_int = (
            getattr(arr, "dtype", None) is not None and arr.dtype.kind in _DEF_INT_KIND
        )
        self._vec, self._locked, self._is_int = arr, True, is_int

    # --- Class methods ---
    @classmethod
    def from_seq(cls, data: Sequence[T]) -> "Vector[T]":
        """Create vector from sequence"""
        return cls(data)

    # --- Properties ---
    @property
    def dimension(self) -> VectorDimension:
        """Return dimension."""
        return cast(VectorDimension, self._vec.size)

    @property
    def ndim(self) -> int:
        """Return dimension."""
        return self._vec.size

    @property
    def x(self) -> T:
        """x coordinate (width) value."""
        return self.get_coordinate("x")

    @property
    def y(self) -> T:
        """y coordinate (height) value."""
        return self.get_coordinate("y")

    @property
    def z(self) -> T:
        """z coordinate (depth) value."""
        return self.get_coordinate("z")

    @property
    def w(self) -> T:
        """w coordinate (time?) value."""
        return self.get_coordinate("w")

    # --- Internal helpers ---
    def _create(self, data: Sequence[T]) -> "Vector[T]":
        """New vector from sequence"""
        return self.__class__(data)

    def _get_coord(self, index: int) -> T:
        """Coordinate by index"""
        assert 0 <= index < self._vec.size, f"Coordinate index {index} out of range"
        v = self._vec[index]
        target_type = int if self._is_int else float
        return cast(T, int(v) if target_type == int else float(v))

    def _from_arr(self, arr: ArrayType) -> "Vector[T]":
        """New vector from array"""
        return self._create(self._cast_coords(arr))

    def __setattr__(self, name: str, value: object):
        """Set attribute (immutable)"""
        if (
            hasattr(self, "_locked")
            and self._locked
            and name not in {"_vec", "_locked", "_is_int"}
        ):
            assert False, "Vector is immutable"
        super().__setattr__(name, value)

    def _target_type(self):
        """Target type"""
        return int if self._is_int else float

    def _cast(self, v):
        """Cast to target type (int/float only)"""
        t = self._target_type()
        if t not in (int, float):
            raise TypeError(f"Unsupported type for Vector: {t}")
        return cast(T, int(v) if t == int else float(v))

    def _cast_coords(self, coords: ArrayType) -> List[T]:
        """Cast array to list"""
        return [self._cast(v) for v in coords]

    def _cast_val(self, value: float) -> T:
        """Cast float to T (int/float only)"""
        target_type = int if self._is_int else float
        if target_type not in (int, float):
            raise TypeError(f"Unsupported type for Vector: {target_type}")
        return cast(T, int(value) if target_type == int else float(value))

    def get_vec(self) -> ArrayType:
        """Internal array"""
        return self._vec

    def _inv_coords(self) -> List[float]:
        """Inverse coordinates"""
        return (-self._vec).tolist()

    def _refl_coords(self, normal: "Vector[T]") -> List[float]:
        """Reflected coordinates"""
        n = normal.normalize().get_vec()
        d = float((self._vec * n).sum())
        return (self._vec - n * d * 2).tolist()

    def _proj_coords(self, other: "Vector[T]") -> List[float]:
        """Projected coordinates"""
        n = other.normalize().get_vec()
        d = float((self._vec * n).sum())
        return (n * d).tolist()

    def _norm(self, arr: ArrayType) -> float:
        """Norm of array"""
        return _norm(arr)

    def _dot(self, a: ArrayType, b: ArrayType) -> float:
        """Dot of arrays"""
        return _dot(a, b)

    # --- Conversion ---
    def to_list(self) -> List[T]:
        """Return coordinates as list."""
        return self._cast_coords(self._vec)

    def to_tuple(self) -> Tuple[T, ...]:
        """Return coordinates as tuple."""
        return tuple(self._cast_coords(self._vec))

    def astype(self, dtype: type) -> "Vector[float]":
        """Return vector with type conversion."""
        return Vector[float](self._vec.astype(dtype).tolist())

    # --- Arithmetic operations ---
    def __add__(self, other: "Vector[T]") -> "Vector[T]":
        """Vector addition."""
        return self._create(self._cast_coords(self._vec + other.get_vec()))

    def __sub__(self, other: "Vector[T]") -> "Vector[T]":
        """Vector subtraction."""
        return self._create(self._cast_coords(self._vec - other.get_vec()))

    def __mul__(self, scalar: Number) -> "Vector[T]":
        """Scalar multiplication."""
        return self._create(self._cast_coords(self._vec * scalar))

    def __rmul__(self, scalar: Number) -> "Vector[T]":
        """Scalar multiplication (right-hand side)."""
        return self.__mul__(scalar)

    def clamp(self, min_val: Number, max_val: Number) -> "Vector[T]":
        """Limit values to specified range."""
        return self._create(self._cast_coords(xp.clip(self._vec, min_val, max_val)))

    def abs(self) -> "Vector[T]":
        """Return vector of absolute values."""
        return self._create(self._cast_coords(xp.abs(self._vec)))

    def inverse(self) -> "Vector[T]":
        """Return inverse vector."""
        return self._create(self._cast_coords(-self._vec))

    def normalize(self) -> "Vector[float]":
        """Return normalized vector."""
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector[float]((self._vec / n).tolist())

    def reflect(self, normal: "Vector[T]") -> "Vector[float]":
        """Reflect by normal vector."""
        s = self.astype(float)
        n = normal.normalize()
        d = s.dot(n)
        return _refl_vec(s, n, d).normalize() * s.norm()

    def project(self, other: "Vector[T]") -> "Vector[float]":
        """Project onto another vector."""
        s = self.astype(float)
        n = other.normalize()
        d = s.dot(n)
        return _proj_vec(n, d)

    def lerp(self, other: "Vector[T]", t: float) -> "Vector[float]":
        """Linear interpolation."""
        return Vector[float]((self._vec * (1 - t) + other.get_vec() * t).tolist())

    # --- Mathematical operations ---
    def dot(self, other: "Vector[T]") -> float:
        """Calculate dot product."""
        return _dot(self._vec, other.get_vec())

    def norm(self) -> float:
        """Return norm."""
        return _norm(self._vec)

    def distance(self, other: "Vector[T]") -> float:
        """Calculate Euclidean distance."""
        return float(((self._vec - other.get_vec()) ** 2).sum() ** 0.5)

    def manhattan(self, other: "Vector[T]") -> T:
        """Calculate Manhattan distance."""
        return self._cast_val(abs(self._vec - other.get_vec()).sum())

    def angle_between(self, other: "Vector[T]") -> float:
        """Calculate angle."""
        return _acos(self.dot(other), self.norm(), other.norm())

    # --- Comparison and utility ---
    def __eq__(self, other: object) -> bool:
        """Check for equivalence."""
        return (
            False
            if not isinstance(other, Vector)
            else (self._vec == other.get_vec()).all()
        )

    def is_unit(self, tol: float = _DEFAULT_TOLERANCE) -> bool:
        """Check if vector is a unit vector."""
        return abs(self.norm() - 1.0) < tol

    def get_coordinate(self, name: CoordinateName) -> T:
        """Get value by coordinate name."""
        coord_map = {"x": 0, "y": 1, "z": 2, "w": 3}
        if name not in coord_map:
            raise ValueError(f"Invalid coordinate name: {name}")
        idx = coord_map[name]
        if idx >= self._vec.size:
            raise IndexError(f"Coordinate '{name}' is not defined for this dimension")
        return self._get_coord(idx)

    def __getitem__(self, idx: int) -> T:
        """Index access."""
        return self._get_coord(idx)

    def __len__(self) -> int:
        """Return number of elements."""
        return self._vec.size

    def __iter__(self) -> Iterator[T]:
        """Return iterator."""
        return iter(self._cast_coords(self._vec))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}({self.to_list()})"


# --- Vec2 ---
class Vec2(Vector[T]):
    """Two-dimensional vector."""

    @overload
    def __init__(self, x: Position[T], y: None = None): ...
    @overload
    def __init__(self, x: T, y: T): ...

    def __init__(self, x: Union[T, Position[T]], y: Optional[T] = None):
        """Two-dimensional vector."""
        if isinstance(x, Position):
            if len(x) != 2:
                raise ValueError
            super().__init__(x)
        else:
            if y is None:
                raise TypeError("y must not be None when x is not Position")
            super().__init__([x, y])

    def cross(self, other: "Vec2[T]") -> T:
        """Calculate cross product."""
        result: Final = self.x * other.y - self.y * other.x
        return self._cast_val(result)


# --- Vec3 ---
class Vec3(Vector[T]):
    """Three-dimensional vector."""

    @overload
    def __init__(self, x: Position[T], y: None = None, z: None = None): ...
    @overload
    def __init__(self, x: T, y: T, z: T): ...
    def __init__(
        self, x: Union[T, Position[T]], y: Optional[T] = None, z: Optional[T] = None
    ):
        """Three-dimensional vector."""
        if isinstance(x, Position):
            if len(x) != 3:
                raise ValueError
            super().__init__(x)
        else:
            missing: Final = [name for name, val in [("y", y), ("z", z)] if val is None]
            if missing:
                raise TypeError(
                    f"{', '.join(missing)} must not be None when x is not Position"
                )
            super().__init__([x, y, z])  # type: ignore[arg-type]

    def cross(self, other: "Vec3[T]") -> "Vec3[float]":
        """Calculate cross product."""
        cx: Final[float] = self.y * other.z - self.z * other.y
        cy: Final[float] = self.z * other.x - self.x * other.z
        cz: Final[float] = self.x * other.y - self.y * other.x
        return Vec3[float](cx, cy, cz)


# --- Vec4 ---
class Vec4(Vector[T]):
    """Four-dimensional vector."""

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
        """Four-dimensional vector."""
        if isinstance(x, Position):
            if len(x) != 4:
                raise ValueError
            super().__init__(x)
        else:
            missing: Final = [
                name for name, val in [("y", y), ("z", z), ("w", w)] if val is None
            ]
            if missing:
                raise TypeError(
                    f"{', '.join(missing)} must not be None when x is not Position"
                )
            super().__init__([x, y, z, w])  # type: ignore[arg-type]
