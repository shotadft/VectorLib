import pytest
from package.vectorlib.vector import Vector, Number

def test_vector_int():
    v = Vector([1, 2, 3])
    assert v[0] == 1 and isinstance(v[0], int)
    assert v[1] == 2 and isinstance(v[1], int)
    assert v.to_list() == [1, 2, 3]
    assert all(isinstance(x, int) for x in v.to_list())
    assert v.to_tuple() == (1, 2, 3)
    assert all(isinstance(x, int) for x in v.to_tuple())
    assert v.ndim == 3
    assert v == Vector([1, 2, 3])
    assert v != Vector([1, 2, 4])
    assert (v + Vector([1, 1, 1])).to_list() == [2, 3, 4]
    assert (v - Vector([1, 1, 1])).to_list() == [0, 1, 2]
    assert v.dot(Vector([1, 0, 0])) == 1
    assert isinstance(v.norm(), float)
    v2 = v.normalize()
    assert isinstance(v2[0], float)
    assert pytest.approx(v2.norm(), 1e-8) == 1.0
    v3 = v * 2
    assert v3.to_list() == [2, 4, 6]
    v4 = 2 * v
    assert v4.to_list() == [2, 4, 6]

def test_vector_float():
    v = Vector([1.0, 2, 3])
    assert v[0] == 1.0 and isinstance(v[0], float)
    assert v[1] == 2.0 and isinstance(v[1], float)
    assert all(isinstance(x, float) for x in v.to_list())
    assert all(isinstance(x, float) for x in v.to_tuple())
    v2 = v * 0.5
    assert all(isinstance(x, float) for x in v2.to_list())

def test_vector_invalid_length():
    with pytest.raises(ValueError):
        Vector([])
    with pytest.raises(ValueError):
        Vector([0]*2000)

def test_vector_immutable():
    v = Vector([1,2,3])
    with pytest.raises(AttributeError):
        v.__setattr__('ndim', 10) 