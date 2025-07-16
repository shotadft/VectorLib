# Module for managing coordinate information

# Standard library
from typing import Generic, Iterator, List, Tuple, Union, cast

# Third-party libraries
from numba import njit

# Project common
from ..common import (
    T,
    ArrayType,
    CoordinateName,
    VectorDimension,
    _USE_CUPY,
    xp,
)

# Constants
_DEF_INT_KIND = ("i",)


class Position(Generic[T]):
    """Position class."""

    # --- Initialization ---
    def __init__(self, *args: T):
        """Initialize coordinates."""
        if not 1 <= len(args) <= 4:
            raise TypeError(f"Position takes 1 to 4 arguments, got {len(args)}")
        for i, v in enumerate(args):
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"Argument {i} must be int or float, got {type(v).__name__}"
                )
        dtype = float if any(isinstance(a, float) for a in args) else int
        arr: ArrayType = xp.array(args, dtype=dtype)
        arr.setflags(write=False)
        is_int = (
            getattr(arr, "dtype", None) is not None and arr.dtype.kind in _DEF_INT_KIND
        )
        self._coords: ArrayType = arr
        self._locked: bool = True
        self._is_int: bool = is_int

    # --- Properties ---
    @property
    def ndim(self) -> int:
        """Return dimension."""
        return self._coords.size

    @property
    def dimension(self) -> VectorDimension:
        """Return dimension."""
        return cast(VectorDimension, self._coords.size)

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

    # --- Internal helpers ---
    def _target_type(self):
        return int if self._is_int else float

    def _cast(self, v):
        t = self._target_type()
        return cast(T, int(v) if t == int else float(v))

    def _cast_coords(self, coords: ArrayType) -> List[T]:
        return [self._cast(v) for v in coords]

    def _validate_index(self, idx: int):
        if idx < 0 or idx >= self._coords.size:
            raise IndexError("Position index out of range")

    def _get_coord(self, index: int) -> T:
        v = self._coords[index]
        return self._cast(v)

    # --- Conversion ---
    def to_list(self) -> List[T]:
        return self._cast_coords(self._coords)

    def to_tuple(self) -> Tuple[T, ...]:
        return tuple(self._cast_coords(self._coords))

    # --- Arithmetic operations ---
    def normalize(self) -> "Position[float]":
        norm = (self._coords * self._coords).sum() ** 0.5
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return Position[float](*(self._coords / norm).tolist())

    # --- Comparison and utility ---
    def __setattr__(self, name: str, value: object):
        """Set attribute."""
        if (
            hasattr(self, "_locked")
            and self._locked
            and name not in {"_coords", "_locked", "_is_int"}
        ):
            assert False, "Position is immutable"
        super().__setattr__(name, value)

    def __len__(self) -> int:
        """Return number of elements."""
        return self._coords.size

    def __iter__(self) -> Iterator[T]:
        """Return iterator."""
        return iter(self._cast_coords(self._coords))

    def __eq__(self, other: object) -> bool:
        """Check for equivalence."""
        return (
            False
            if not isinstance(other, Position)
            else (self._coords == other._coords).all()
        )

    def __getitem__(self, key: Union[int, CoordinateName]) -> T:
        names: Tuple[CoordinateName, ...] = ('x', 'y', 'z', 'w')
        if isinstance(key, int):
            self._validate_index(key)
            v = key
        elif isinstance(key, str):
            idx = names.index(key)
            if self._coords.size <= idx:
                raise KeyError(f"'{key}' is not defined for this dimension")
            v = idx
        else:
            raise TypeError(f"Invalid key type: {type(key).__name__}")
        return self._get_coord(v)

    def is_zero(self) -> bool:
        return bool((self._coords == 0).all())

    def __repr__(self) -> str:
        names = ['x', 'y', 'z', 'w']
        coords = [f"{names[i]}={v}" for i, v in enumerate(self.to_list())]
        return f"{self.__class__.__name__}({', '.join(coords)})"
