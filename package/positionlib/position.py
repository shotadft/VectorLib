from typing import Union, List, Tuple, TypeVar, Generic
import numpy as np
from numba import njit

Number = Union[float, int]
T = TypeVar('T', bound=Number)

@njit(cache=True)
def _norm(arr: np.ndarray) -> float:
    s = 0.0
    for v in arr:
        s += v * v
    return s ** 0.5

@njit(cache=True)
def _is_zero(arr: np.ndarray) -> bool:
    for v in arr:
        if v != 0:
            return False
    return True

class Position(Generic[T]):
    def __init__(self, *args: T) -> None:
        if not 1 <= len(args) <= 4:
            raise TypeError(f"Position takes 1 to 4 arguments, got {len(args)}")
        for i, v in enumerate(args):
            if not isinstance(v, (int, float)):
                raise TypeError(f"Argument {i} must be int or float, got {type(v).__name__}")
        dtype = float if any(isinstance(a, float) for a in args) else int
        arr = np.array(args, dtype=dtype)
        arr.setflags(write=False)
        self._coords = arr
        self._locked = True

    def __setattr__(self, name, value):
        if hasattr(self, '_locked') and self._locked and name not in {'_coords', '_locked'}:
            raise AttributeError("Position is immutable")
        super().__setattr__(name, value)

    @property
    def x(self) -> Number:
        return self._coords[0]
    @property
    def y(self) -> Number:
        if self._coords.size < 2:
            raise AttributeError('y is not defined for this dimension')
        return self._coords[1]
    @property
    def z(self) -> Number:
        if self._coords.size < 3:
            raise AttributeError('z is not defined for this dimension')
        return self._coords[2]
    @property
    def w(self) -> Number:
        if self._coords.size < 4:
            raise AttributeError('w is not defined for this dimension')
        return self._coords[3]

    @property
    def ndim(self) -> int:
        return self._coords.size

    def to_list(self) -> List[Number]:
        return self._coords.tolist()

    def to_tuple(self) -> Tuple[Number, ...]:
        return tuple(self._coords.tolist())

    def is_zero(self) -> bool:
        return bool(_is_zero(self._coords))

    def normalize(self) -> 'Position[float]':
        norm = _norm(self._coords)
        if norm == 0:
            raise ValueError('Cannot normalize zero vector')
        normalized = self._coords / norm
        return Position[float](*normalized.tolist())

    def __repr__(self) -> str:
        names = ['x', 'y', 'z', 'w']
        coords = [f"{names[i]}={self._coords[i]}" for i in range(self._coords.size)]
        return f"Position({', '.join(coords)})"
