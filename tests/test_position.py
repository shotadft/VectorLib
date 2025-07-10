import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package.positionlib.position import Position

def test_position_int():
    p = Position(1, 2, 3)
    assert p.x == 1 and isinstance(p.x, int)
    assert p.y == 2 and isinstance(p.y, int)
    assert p.z == 3 and isinstance(p.z, int)
    assert p.to_list() == [1, 2, 3]
    assert all(isinstance(x, int) for x in p.to_list())
    assert p.to_tuple() == (1, 2, 3)
    assert all(isinstance(x, int) for x in p.to_tuple())
    assert p.ndim == 3
    assert p['x'] == 1
    assert p['y'] == 2
    assert p['z'] == 3
    assert p.is_zero() is False
    p2 = Position(0, 0, 0)
    assert p2.is_zero() is True
    p3 = p.normalize()
    assert isinstance(p3.x, float)
    assert pytest.approx(p3.to_tuple()[0]**2 + p3.to_tuple()[1]**2 + p3.to_tuple()[2]**2, 1e-8) == 1.0

def test_position_float():
    p = Position(1.0, 2, 3)
    assert p.x == 1.0 and isinstance(p.x, float)
    assert p.y == 2.0 and isinstance(p.y, float)
    assert all(isinstance(x, float) for x in p.to_list())
    assert all(isinstance(x, float) for x in p.to_tuple())
    p2 = p.normalize()
    assert all(isinstance(x, float) for x in p2.to_list())

def test_position_invalid():
    with pytest.raises(TypeError):
        Position()
    with pytest.raises(TypeError):
        Position(1, 'a', 3)  # type: ignore
    with pytest.raises(KeyError):
        _ = Position(1)['z']
    with pytest.raises(KeyError):
        _ = Position(1)['y']
    with pytest.raises(KeyError):
        _ = Position(1)['foo']
    with pytest.raises(ValueError):
        Position(0,0,0).normalize()

def test_position_immutable():
    p = Position(1,2,3)
    with pytest.raises(AttributeError):
        p.__setattr__('ndim', 10) 