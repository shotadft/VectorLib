import pytest
from package.vectorlib.vector import Vector

@pytest.mark.benchmark(group="vector_create")
def test_vector_create_int(benchmark):
    benchmark(lambda: Vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

@pytest.mark.benchmark(group="vector_create")
def test_vector_create_float(benchmark):
    benchmark(lambda: Vector([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

@pytest.mark.benchmark(group="vector_access")
def test_vector_getitem(benchmark):
    v = Vector([i for i in range(100)])
    benchmark(lambda: v[50])

@pytest.mark.benchmark(group="vector_to_list")
def test_vector_to_list(benchmark):
    v = Vector([i for i in range(100)])
    benchmark(v.to_list)

@pytest.mark.benchmark(group="vector_to_tuple")
def test_vector_to_tuple(benchmark):
    v = Vector([i for i in range(100)])
    benchmark(v.to_tuple)

@pytest.mark.benchmark(group="vector_add")
def test_vector_add(benchmark):
    v1 = Vector([i for i in range(100)])
    v2 = Vector([i for i in range(100)])
    benchmark(lambda: v1 + v2)

@pytest.mark.benchmark(group="vector_sub")
def test_vector_sub(benchmark):
    v1 = Vector([i for i in range(100)])
    v2 = Vector([i for i in range(100)])
    benchmark(lambda: v1 - v2)

@pytest.mark.benchmark(group="vector_dot")
def test_vector_dot(benchmark):
    v1 = Vector([i for i in range(100)])
    v2 = Vector([i for i in range(100)])
    benchmark(lambda: v1.dot(v2))

@pytest.mark.benchmark(group="vector_norm")
def test_vector_norm(benchmark):
    v = Vector([i for i in range(100)])
    benchmark(v.norm)

@pytest.mark.benchmark(group="vector_normalize")
def test_vector_normalize(benchmark):
    v = Vector([i+1 for i in range(100)])
    benchmark(v.normalize)
