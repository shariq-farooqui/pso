import numpy as np
import pytest

from pso.fitness import quadratic_1d, quadratic_2d, sphere


def test_quadratic_1d():
    assert quadratic_1d(np.array([2])) == 4
    assert quadratic_1d(np.array([0])) == 0

    with pytest.raises(IndexError):
        quadratic_1d(np.array([]))

    with pytest.raises(TypeError):
        quadratic_1d(2)


def test_quadratic_2d():
    assert quadratic_2d(np.array([2, 3])) == 13
    assert quadratic_2d(np.array([0, 0])) == 0

    with pytest.raises(IndexError):
        quadratic_2d(np.array([1]))

    with pytest.raises(TypeError):
        quadratic_2d(2)


def test_sphere():
    assert sphere(np.array([2, 2, 2])) == 12
    assert sphere(np.array([0, 0])) == 0
    assert sphere(np.array([])) == 0

    with pytest.raises(TypeError):
        sphere(2)
